from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


Coord = Tuple[int, int]
VALID_MOVES = {0, 1, 2, 3, 4}
ACTION_LABELS = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}

SEVERITY_CONFIG: Dict[str, Dict[str, float]] = {
    "LOW": {"base_reward": 10.0, "deadline": 10.0, "miss_penalty": -10.0},
    "MEDIUM": {"base_reward": 20.0, "deadline": 6.0, "miss_penalty": -25.0},
    "HIGH": {"base_reward": 40.0, "deadline": 3.0, "miss_penalty": -60.0},
}

SEVERITY_DISTRIBUTION: Tuple[Tuple[str, float], ...] = (
    ("LOW", 0.6),
    ("MEDIUM", 0.3),
    ("HIGH", 0.1),
)

HOTSPOTS: Tuple[Dict[str, Any], ...] = (
    {"center": (2, 2), "radius": 2, "weight": 0.4},
    {"center": (7, 7), "radius": 2, "weight": 0.4},
)


@dataclass
class Event:
    id: int
    location: Coord
    start_time: int
    duration: int
    severity: str
    detected: bool = False
    detection_time: Optional[int] = None

    @property
    def end_time(self) -> int:
        return self.start_time + self.duration

    @property
    def deadline(self) -> int:
        return int(SEVERITY_CONFIG[self.severity]["deadline"])

    @property
    def deadline_step(self) -> int:
        return self.start_time + self.deadline

    @property
    def base_reward(self) -> float:
        return float(SEVERITY_CONFIG[self.severity]["base_reward"])

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
    detected_event_history: List[Dict[str, Any]] = field(default_factory=list)
    remembered_detected_event_ids: set[int] = field(default_factory=set)

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


def sample_event_severity(rng: np.random.Generator) -> str:
    severities = [name for name, _ in SEVERITY_DISTRIBUTION]
    probabilities = [probability for _, probability in SEVERITY_DISTRIBUTION]
    return str(rng.choice(severities, p=probabilities))


def _cells_in_hotspot(center: Coord, radius: int, grid_size: int) -> list[Coord]:
    cx, cy = center
    cells: list[Coord] = []
    for x in range(max(0, cx - radius), min(grid_size, cx + radius + 1)):
        for y in range(max(0, cy - radius), min(grid_size, cy + radius + 1)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius:
                cells.append((x, y))
    return cells


def sample_event_location_with_hotspots(
    rng: np.random.Generator,
    grid_size: int,
    hotspots: Sequence[Mapping[str, Any]] = HOTSPOTS,
) -> Coord:
    hotspot_weight = sum(float(hotspot["weight"]) for hotspot in hotspots)
    if hotspots and rng.random() < hotspot_weight:
        normalized_weights = np.array([float(hotspot["weight"]) for hotspot in hotspots], dtype=float)
        normalized_weights = normalized_weights / normalized_weights.sum()
        hotspot_index = int(rng.choice(len(hotspots), p=normalized_weights))
        hotspot = hotspots[hotspot_index]
        cells = _cells_in_hotspot(
            center=tuple(hotspot["center"]),
            radius=int(hotspot["radius"]),
            grid_size=grid_size,
        )
        if cells:
            return tuple(cells[int(rng.integers(0, len(cells)))])

    return int(rng.integers(0, grid_size)), int(rng.integers(0, grid_size))


def sample_event(
    rng: np.random.Generator,
    timestep: int,
    next_event_id: int,
    grid_size: int,
    p_spawn: float,
    hotspots: Sequence[Mapping[str, Any]] = HOTSPOTS,
) -> Optional[Event]:
    if rng.random() >= p_spawn:
        return None

    severity = sample_event_severity(rng)
    location = sample_event_location_with_hotspots(rng, grid_size, hotspots)
    duration = int(rng.integers(5, 11))
    return Event(
        id=next_event_id,
        location=location,
        start_time=timestep,
        duration=duration,
        severity=severity,
    )


def spawn_event(
    rng: np.random.Generator,
    timestep: int,
    next_event_id: int,
    grid_size: int,
    p_spawn: float,
    hotspots: Sequence[Mapping[str, Any]] = HOTSPOTS,
) -> Optional[Event]:
    return sample_event(rng, timestep, next_event_id, grid_size, p_spawn, hotspots)


def compute_event_reward(event: Event, detection_time: int) -> float:
    latency = detection_time - event.start_time
    time_left = event.deadline - latency
    if latency <= event.deadline:
        early_bonus = max(0, time_left) * 2.0
        return event.base_reward + early_bonus
    return event.base_reward * 0.3


def compute_miss_penalty(event: Event) -> float:
    return float(SEVERITY_CONFIG[event.severity]["miss_penalty"])


def compute_episode_bonus(
    metrics: Mapping[str, Any],
    latency_threshold: float = 3.0,
) -> Tuple[float, Dict[str, Any]]:
    reward = 0.0
    breakdown = {
        "all_high_detected_on_time_bonus": 0.0,
        "high_priority_miss_penalty": 0.0,
        "low_latency_team_bonus": 0.0,
    }

    high_total = int(metrics.get("events_spawned_by_severity", {}).get("HIGH", 0))
    high_on_time = int(metrics.get("detected_on_time_by_severity", {}).get("HIGH", 0))
    high_missed = int(metrics.get("missed_by_severity", {}).get("HIGH", 0))
    mean_latency = metrics.get("mean_detection_latency")

    if high_total > 0 and high_on_time == high_total:
        reward += 50.0
        breakdown["all_high_detected_on_time_bonus"] = 50.0

    if high_missed > 0:
        reward -= 50.0
        breakdown["high_priority_miss_penalty"] = -50.0

    if mean_latency is not None and float(mean_latency) < latency_threshold:
        reward += 20.0
        breakdown["low_latency_team_bonus"] = 20.0

    return reward, breakdown


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
    event_memory_limit: int = 5,
) -> DisasterObservation:
    agent_obs: Dict[str, Dict[str, Any]] = {}
    active_events = [event for event in events if event.is_active(timestep) and not event.detected]
    total_cells = grid_size * grid_size

    for drone_id, drone in drones.items():
        visible_events = [
            {
                "id": event.id,
                "location": event.location,
                "severity": event.severity,
                "start_time": event.start_time,
                "duration": event.duration,
                "end_time": event.end_time,
                "time_remaining": event.end_time - timestep,
                "deadline_remaining": event.deadline_step - timestep,
            }
            for event in active_events
            if event.location in fovs[drone_id]
        ]
        frontier_cells = sorted(cell for cell in fovs[drone_id] if cell not in drone.visited_cells)
        detected_event_history = [
            {
                **entry,
                "time_since_detection": timestep - int(entry["detected_at"]),
            }
            for entry in drone.detected_event_history[-event_memory_limit:]
        ]
        agent_obs[drone_id] = {
            "position": drone.position,
            "visible_cells": sorted(fovs[drone_id]),
            "visible_events": visible_events,
            "local_visited_cells": sorted(cell for cell in fovs[drone_id] if cell in drone.visited_cells),
            "frontier_cells": frontier_cells,
            "detected_event_history": detected_event_history,
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
        "detection_reward": 20.0 * detected_count,
        "miss_penalty": -20.0 * missed_count,
        "episode_bonus": 0.0,
        "overlap_penalty": overlap_penalty,
        "coverage_reward": coverage_reward,
        "current_visible_cells": len(current_visible_cells),
        "new_cells_covered": len(new_cells),
        "new_cells": sorted(new_cells),
        "reward_breakdown": {
            "time_penalty": -1.0,
            "detection_reward": 20.0 * detected_count,
            "miss_penalty": -20.0 * missed_count,
            "overlap_penalty": overlap_penalty,
            "coverage_reward": coverage_reward,
            "episode_bonus": 0.0,
        },
    }


def compute_level5_reward(
    *,
    detected_events: Sequence[Event],
    missed_events: Sequence[Event],
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

    detection_rewards = {
        event.id: compute_event_reward(event, int(event.detection_time))
        for event in detected_events
        if event.detection_time is not None
    }
    miss_penalties = {event.id: compute_miss_penalty(event) for event in missed_events}
    total_detection_reward = float(sum(detection_rewards.values()))
    total_miss_penalty = float(sum(miss_penalties.values()))

    reward = -1.0 + total_detection_reward + total_miss_penalty + overlap_penalty + coverage_reward
    return reward, {
        **overlap_info,
        "detection_reward": total_detection_reward,
        "miss_penalty": total_miss_penalty,
        "episode_bonus": 0.0,
        "overlap_penalty": overlap_penalty,
        "coverage_reward": coverage_reward,
        "current_visible_cells": len(current_visible_cells),
        "new_cells_covered": len(new_cells),
        "new_cells": sorted(new_cells),
        "detected_event_rewards": detection_rewards,
        "missed_event_penalties": miss_penalties,
        "reward_breakdown": {
            "time_penalty": -1.0,
            "detection_reward": total_detection_reward,
            "miss_penalty": total_miss_penalty,
            "overlap_penalty": overlap_penalty,
            "coverage_reward": coverage_reward,
            "episode_bonus": 0.0,
        },
    }


def compute_baseline_reward(
    *,
    detected_count: int,
    missed_count: int,
    fovs: Mapping[str, set[Coord]],
    visited_cells: set[Coord],
) -> Tuple[float, Dict[str, Any]]:
    reward = -1.0 + 20.0 * detected_count - 20.0 * missed_count
    _, overlap_info = compute_overlap_penalty(fovs)
    _, current_visible_cells, new_cells = compute_coverage_reward(
        fov_sets=fovs,
        visited_cells=visited_cells,
        reward_per_new_cell=0.0,
    )
    return reward, {
        **overlap_info,
        "detection_reward": 20.0 * detected_count,
        "miss_penalty": -20.0 * missed_count,
        "episode_bonus": 0.0,
        "overlap_penalty": 0.0,
        "coverage_reward": 0.0,
        "current_visible_cells": len(current_visible_cells),
        "new_cells_covered": len(new_cells),
        "new_cells": sorted(new_cells),
        "reward_breakdown": {
            "time_penalty": -1.0,
            "detection_reward": 20.0 * detected_count,
            "miss_penalty": -20.0 * missed_count,
            "overlap_penalty": 0.0,
            "coverage_reward": 0.0,
            "episode_bonus": 0.0,
        },
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
                "detected_event_history_size": len(drone.detected_event_history),
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
