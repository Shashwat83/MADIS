from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Sequence

from ..coordinator import HeuristicCoordinator
from ..models import HOTSPOTS, Coord, move_toward_target


class TeacherPolicy(ABC):
    @abstractmethod
    def decide_targets(
        self,
        observation: Mapping[str, Any],
        *,
        env: Optional[Any] = None,
    ) -> Dict[str, Coord]:
        """Return one target coordinate per drone."""

    def decide_actions(
        self,
        observation: Mapping[str, Any],
        *,
        env: Optional[Any] = None,
    ) -> Dict[str, str]:
        drone_positions = observation["drone_positions"]
        targets = self.decide_targets(observation, env=env)
        actions: Dict[str, str] = {}
        for drone_id, position in drone_positions.items():
            _, action_id = move_toward_target(tuple(position), targets[drone_id])
            actions[drone_id] = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}[action_id]
        return actions


class ActionFormatTeacherPolicy(TeacherPolicy):
    """Deterministic valid-action teacher focused on output schema correctness."""

    def decide_targets(
        self,
        observation: Mapping[str, Any],
        *,
        env: Optional[Any] = None,
    ) -> Dict[str, Coord]:
        del env
        drone_positions = observation["drone_positions"]
        return {drone_id: tuple(position) for drone_id, position in drone_positions.items()}


class HeuristicTeacherPolicy(TeacherPolicy):
    def __init__(self) -> None:
        self.policy = HeuristicCoordinator()

    def decide_targets(
        self,
        observation: Mapping[str, Any],
        *,
        env: Optional[Any] = None,
    ) -> Dict[str, Coord]:
        del env
        return self.policy.decide(observation).targets


class OracleTeacherPolicy(TeacherPolicy):
    """Privileged teacher that can use hidden simulator state during data generation."""

    def decide_targets(
        self,
        observation: Mapping[str, Any],
        *,
        env: Optional[Any] = None,
    ) -> Dict[str, Coord]:
        if env is None:
            raise ValueError("OracleTeacherPolicy requires the live environment instance.")

        drone_positions = {
            drone_id: tuple(position) for drone_id, position in observation["drone_positions"].items()
        }
        assigned: Dict[str, Coord] = {}
        assigned_cells: set[Coord] = set()

        active_events = [
            event
            for event in env.state.active_events
            if not event["detected"] and event["start_time"] <= env.timestep < event["end_time"]
        ]
        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        prioritized_events = sorted(
            active_events,
            key=lambda event: (
                severity_order.get(str(event["severity"]), 99),
                int(event["deadline_step"]) - env.timestep,
            ),
        )

        remaining_frontier = [
            tuple(cell) for cell in observation.get("team_frontier_cells", [])
            if tuple(cell) not in assigned_cells
        ]
        hotspot_anchors = [tuple(hotspot["center"]) for hotspot in HOTSPOTS]

        for drone_id, position in drone_positions.items():
            target = self._select_visible_or_hidden_event_target(position, prioritized_events, assigned_cells)
            if target is None:
                target = self._select_best_hotspot_or_frontier(position, remaining_frontier, hotspot_anchors, assigned_cells)
            assigned[drone_id] = target
            assigned_cells.add(target)
            if target in remaining_frontier:
                remaining_frontier.remove(target)

        return assigned

    @staticmethod
    def _distance(left: Coord, right: Coord) -> int:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])

    def _select_visible_or_hidden_event_target(
        self,
        position: Coord,
        prioritized_events: Sequence[Mapping[str, Any]],
        assigned_cells: set[Coord],
    ) -> Coord | None:
        candidates = [
            tuple(event["location"])
            for event in prioritized_events
            if tuple(event["location"]) not in assigned_cells
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda location: self._distance(position, location))

    def _select_best_hotspot_or_frontier(
        self,
        position: Coord,
        frontier_cells: Sequence[Coord],
        hotspot_anchors: Sequence[Coord],
        assigned_cells: set[Coord],
    ) -> Coord:
        frontier_candidates = [cell for cell in frontier_cells if cell not in assigned_cells]
        if frontier_candidates:
            return min(frontier_candidates, key=lambda cell: self._distance(position, cell))

        hotspot_candidates = [anchor for anchor in hotspot_anchors if anchor not in assigned_cells]
        if hotspot_candidates:
            return min(hotspot_candidates, key=lambda cell: self._distance(position, cell))

        return position
