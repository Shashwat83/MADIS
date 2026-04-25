from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


Coord = Tuple[int, int]
VALID_MOVES = {0, 1, 2, 3, 4}
ACTION_LABELS = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}


@dataclass
class Event:
    id: int
    location: Coord
    start_time: int
    duration: int
    detected: bool = False
    detection_time: Optional[int] = None

    @property
    def end_time(self) -> int:
        return self.start_time + self.duration

    def is_active(self, timestep: int) -> bool:
        return self.start_time <= timestep < self.end_time


@dataclass
class Drone:
    id: str
    position: Coord
    visited_cells: set[Coord] = field(default_factory=set)
    revisit_count: int = 0
    last_action: int = 4
    steps_taken: int = 0

    def move(self, action: int, grid_size: int) -> None:
        x, y = self.position
        if action == 0:
            y -= 1
        elif action == 1:
            y += 1
        elif action == 2:
            x -= 1
        elif action == 3:
            x += 1
        elif action != 4:
            raise ValueError(f"Invalid action {action}; expected one of {sorted(VALID_MOVES)}.")

        self.position = (int(np.clip(x, 0, grid_size - 1)), int(np.clip(y, 0, grid_size - 1)))
        self.last_action = int(action)
        self.steps_taken += 1
        if self.position in self.visited_cells:
            self.revisit_count += 1
        self.visited_cells.add(self.position)


class DroneActions(Action):
    actions: Dict[str, int] = Field(
        ...,
        description="Per-agent actions keyed by drone id. 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY.",
    )


class DisasterObservation(Observation):
    timestep: int
    agents: Dict[str, Dict[str, Any]]
    metrics: Dict[str, Any]


class DisasterState(State):
    timestep: int
    active_events: List[Dict[str, Any]]
    drone_positions: Dict[str, Coord]
    per_drone_stats: Dict[str, Dict[str, Any]]
    metrics: Dict[str, Any]


def compute_fov(position: Coord, grid_size: int, radius: int = 2) -> set[Coord]:
    xd, yd = position
    cells: set[Coord] = set()
    radius_sq = radius * radius
    for x in range(max(0, xd - radius), min(grid_size, xd + radius + 1)):
        for y in range(max(0, yd - radius), min(grid_size, yd + radius + 1)):
            if (x - xd) ** 2 + (y - yd) ** 2 <= radius_sq:
                cells.add((x, y))
    return cells


def get_fov_cells(drone: Drone, grid_size: int, radius: int = 2) -> set[Coord]:
    return compute_fov(drone.position, grid_size, radius)


def spawn_event(
    rng: np.random.Generator,
    timestep: int,
    next_event_id: int,
    grid_size: int,
    p_spawn: float,
) -> Optional[Event]:
    if rng.random() >= p_spawn:
        return None

    x = int(rng.integers(0, grid_size))
    y = int(rng.integers(0, grid_size))
    duration = int(rng.integers(5, 11))
    return Event(id=next_event_id, location=(x, y), start_time=timestep, duration=duration)


def compute_fov_overlap_penalty(fov_sets: Mapping[str, set[Coord]]) -> Tuple[float, Dict[str, Any]]:
    overlap_pairs: List[Tuple[str, str]] = []
    overlapping_cells: set[Coord] = set()
    pairwise_overlap_cells = 0
    drone_ids = list(fov_sets)

    for index, left_id in enumerate(drone_ids):
        for right_id in drone_ids[index + 1 :]:
            intersection = fov_sets[left_id] & fov_sets[right_id]
            if intersection:
                overlap_pairs.append((left_id, right_id))
                overlapping_cells.update(intersection)
                pairwise_overlap_cells += len(intersection)

    return -1.0 * pairwise_overlap_cells, {
        "fov_overlap": bool(overlap_pairs),
        "overlap_pairs": overlap_pairs,
        "overlap_cells": len(overlapping_cells),
        "pairwise_overlap_cells": pairwise_overlap_cells,
    }


def compute_overlap_penalty(fov_sets: Mapping[str, set[Coord]]) -> Tuple[float, Dict[str, Any]]:
    return compute_fov_overlap_penalty(fov_sets)


def compute_coverage_reward(
    fov_sets: Mapping[str, set[Coord]],
    visited_cells: set[Coord],
    reward_per_new_cell: float = 0.5,
) -> Tuple[float, set[Coord], set[Coord]]:
    current_visible_cells = set().union(*fov_sets.values()) if fov_sets else set()
    new_cells = current_visible_cells - visited_cells
    return reward_per_new_cell * len(new_cells), current_visible_cells, new_cells


def build_observation(
    *,
    timestep: int,
    grid_size: int,
    drones: Mapping[str, Drone],
    events: Iterable[Event],
    fovs: Mapping[str, set[Coord]],
    metrics: Mapping[str, Any],
    reward: float,
    done: bool,
) -> DisasterObservation:
    agent_obs: Dict[str, Dict[str, Any]] = {}
    active_events = [event for event in events if event.is_active(timestep) and not event.detected]
    total_cells = grid_size * grid_size

    for drone_id, drone in drones.items():
        visible_events = [
            {
                "id": event.id,
                "location": event.location,
                "start_time": event.start_time,
                "duration": event.duration,
                "end_time": event.end_time,
            }
            for event in active_events
            if event.location in fovs[drone_id]
        ]
        frontier_cells = sorted(cell for cell in fovs[drone_id] if cell not in drone.visited_cells)
        agent_obs[drone_id] = {
            "position": drone.position,
            "visible_cells": sorted(fovs[drone_id]),
            "visible_events": visible_events,
            "local_visited_cells": sorted(cell for cell in fovs[drone_id] if cell in drone.visited_cells),
            "frontier_cells": frontier_cells,
            "exploration": {
                "steps_taken": drone.steps_taken,
                "visited_cell_count": len(drone.visited_cells),
                "revisit_count": drone.revisit_count,
                "coverage_ratio": len(drone.visited_cells) / float(total_cells),
                "last_action": ACTION_LABELS[drone.last_action],
            },
            "policy_constraints": {
                "communication": "disabled",
                "reward_sharing": "global_shared",
            },
        }

    return DisasterObservation(
        timestep=timestep,
        agents=agent_obs,
        metrics=dict(metrics),
        reward=reward,
        done=done,
    )


def compute_reward(
    *,
    detected_count: int,
    missed_count: int,
    fovs: Mapping[str, set[Coord]],
    visited_cells: set[Coord],
    reward_per_new_cell: float = 0.5,
) -> Tuple[float, Dict[str, Any]]:
    overlap_penalty, overlap_info = compute_overlap_penalty(fovs)
    coverage_reward, current_visible_cells, new_cells = compute_coverage_reward(
        fov_sets=fovs,
        visited_cells=visited_cells,
        reward_per_new_cell=reward_per_new_cell,
    )
    reward = -1.0 + 20.0 * detected_count - 20.0 * missed_count + overlap_penalty + coverage_reward

    return reward, {
        **overlap_info,
        "overlap_penalty": overlap_penalty,
        "coverage_reward": coverage_reward,
        "current_visible_cells": len(current_visible_cells),
        "new_cells_covered": len(new_cells),
        "new_cells": sorted(new_cells),
    }


def compute_grid_coverage(visited_cells: set[Coord], grid_size: int) -> float:
    return len(visited_cells) / float(grid_size * grid_size)


def build_team_metrics(
    *,
    drones: Mapping[str, Drone],
    grid_size: int,
    metrics: Mapping[str, Any],
    team_visited_cells: Optional[set[Coord]] = None,
) -> Dict[str, Any]:
    total_cells = grid_size * grid_size
    movement_visited_cells = set().union(*(drone.visited_cells for drone in drones.values())) if drones else set()
    surveilled_cells = team_visited_cells if team_visited_cells is not None else movement_visited_cells
    return {
        **dict(metrics),
        "coordination_mode": "decentralized_no_communication",
        "reward_mode": metrics.get("reward_mode", "shared_global_overlap_and_coverage_shaping"),
        "team_unique_cells_visited": len(movement_visited_cells),
        "team_coverage_ratio": len(movement_visited_cells) / float(total_cells),
        "unique_cells_visited": len(surveilled_cells),
        "grid_coverage": len(surveilled_cells) / float(total_cells),
        "grid_coverage_percent": 100.0 * len(surveilled_cells) / float(total_cells),
        "per_drone_exploration": {
            drone_id: {
                "visited_cell_count": len(drone.visited_cells),
                "coverage_ratio": len(drone.visited_cells) / float(total_cells),
                "revisit_count": drone.revisit_count,
                "steps_taken": drone.steps_taken,
                "last_action": ACTION_LABELS[drone.last_action],
            }
            for drone_id, drone in drones.items()
        },
    }


def normalize_actions(
    action: DroneActions | Mapping[str, int] | Sequence[int],
    agent_ids: Sequence[str],
) -> Dict[str, int]:
    if isinstance(action, DroneActions):
        actions = action.actions
    elif isinstance(action, Mapping):
        actions = dict(action)
    else:
        if len(action) != len(agent_ids):
            raise ValueError(f"Expected {len(agent_ids)} actions, got {len(action)}.")
        actions = dict(zip(agent_ids, action))

    missing = set(agent_ids) - set(actions)
    extra = set(actions) - set(agent_ids)
    if missing or extra:
        raise ValueError(f"Action keys must match {list(agent_ids)}; missing={sorted(missing)}, extra={sorted(extra)}.")

    for drone_id, value in actions.items():
        if value not in VALID_MOVES:
            raise ValueError(f"{drone_id} action must be one of {sorted(VALID_MOVES)}; got {value}.")

    return {drone_id: int(value) for drone_id, value in actions.items()}
