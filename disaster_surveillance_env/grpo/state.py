from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, MutableMapping

import numpy as np

from ..models import Drone, Event
from ..server.disaster_surveillance_environment import DisasterSurveillanceEnvironment


@dataclass(slots=True)
class EnvironmentSnapshot:
    episode_id: str | None
    level: int
    grid_size: int
    episode_length: int
    timestep: int
    next_event_id: int
    next_report_id: int
    rng_state: Dict[str, Any]
    drones: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    reported_events: List[Dict[str, Any]]
    visited_cells: List[tuple[int, int]]
    metrics: Dict[str, Any]
    last_reward: float
    episode_bonus_applied: bool
    recently_observed_cells: List[tuple[int, int]]
    last_assigned_targets: Dict[str, tuple[int, int]]
    last_coordinator_observation: Dict[str, Any]


def snapshot_environment(env: DisasterSurveillanceEnvironment) -> Dict[str, Any]:
    return asdict(
        EnvironmentSnapshot(
            episode_id=env.episode_id,
            level=env.level,
            grid_size=env.grid_size,
            episode_length=env.episode_length,
            timestep=env.timestep,
            next_event_id=env.next_event_id,
            next_report_id=env.next_report_id,
            rng_state=dict(env.rng.bit_generator.state),
            drones=[
                {
                    "id": drone.id,
                    "position": drone.position,
                    "visited_cells": sorted(drone.visited_cells),
                    "revisit_count": drone.revisit_count,
                    "last_action": drone.last_action,
                    "steps_taken": drone.steps_taken,
                    "detected_event_history": list(drone.detected_event_history),
                    "remembered_detected_event_ids": sorted(drone.remembered_detected_event_ids),
                    "current_target": drone.current_target,
                }
                for drone in env.drones.values()
            ],
            events=[asdict(event) for event in env.events],
            reported_events=[dict(report) for report in env.reported_events],
            visited_cells=sorted(env.visited_cells),
            metrics=_deep_copy_metrics(env.metrics),
            last_reward=float(env._last_reward),
            episode_bonus_applied=bool(env._episode_bonus_applied),
            recently_observed_cells=list(env._recently_observed_cells),
            last_assigned_targets=dict(env.last_assigned_targets),
            last_coordinator_observation=dict(env.last_coordinator_observation),
        )
    )


def restore_environment(
    snapshot: Mapping[str, Any],
    *,
    grid_size: int | None = None,
    episode_length: int | None = None,
    level: int | None = None,
) -> DisasterSurveillanceEnvironment:
    env = DisasterSurveillanceEnvironment(
        grid_size=int(snapshot.get("grid_size", grid_size or DisasterSurveillanceEnvironment.GRID_SIZE)),
        episode_length=int(snapshot.get("episode_length", episode_length or DisasterSurveillanceEnvironment.EPISODE_LENGTH)),
        level=int(snapshot.get("level", level or 6)),
        seed=0,
    )
    env.episode_id = snapshot.get("episode_id")
    env.timestep = int(snapshot["timestep"])
    env.next_event_id = int(snapshot["next_event_id"])
    env.next_report_id = int(snapshot.get("next_report_id", 1))
    env.rng = np.random.default_rng()
    env.rng.bit_generator.state = dict(snapshot["rng_state"])
    env.drones = {
        drone_payload["id"]: Drone(
            id=drone_payload["id"],
            position=tuple(drone_payload["position"]),
            visited_cells={tuple(cell) for cell in drone_payload["visited_cells"]},
            revisit_count=int(drone_payload["revisit_count"]),
            last_action=int(drone_payload["last_action"]),
            steps_taken=int(drone_payload["steps_taken"]),
            detected_event_history=list(drone_payload["detected_event_history"]),
            remembered_detected_event_ids={int(item) for item in drone_payload["remembered_detected_event_ids"]},
            current_target=tuple(drone_payload["current_target"]) if drone_payload["current_target"] is not None else None,
        )
        for drone_payload in snapshot["drones"]
    }
    env.events = [Event(**{**event_payload, "location": tuple(event_payload["location"])}) for event_payload in snapshot["events"]]
    env.reported_events = [
        {
            **dict(report_payload),
            "location": tuple(report_payload["location"]),
        }
        for report_payload in snapshot.get("reported_events", [])
    ]
    env.visited_cells = {tuple(cell) for cell in snapshot["visited_cells"]}
    env.metrics = _deep_copy_metrics(snapshot["metrics"])
    investigated_report_ids = env.metrics.get("_investigated_false_report_ids")
    if isinstance(investigated_report_ids, list):
        env.metrics["_investigated_false_report_ids"] = set(str(item) for item in investigated_report_ids)
    env._last_reward = float(snapshot["last_reward"])
    env._episode_bonus_applied = bool(snapshot["episode_bonus_applied"])
    env._recently_observed_cells = [tuple(cell) for cell in snapshot["recently_observed_cells"]]
    env.last_assigned_targets = {
        drone_id: tuple(target) for drone_id, target in snapshot["last_assigned_targets"].items()
    }
    env.last_coordinator_observation = dict(snapshot["last_coordinator_observation"])
    return env


def _deep_copy_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    copied: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, Mapping):
            copied[key] = _deep_copy_metrics(value)
        elif isinstance(value, list):
            copied[key] = list(value)
        elif isinstance(value, set):
            copied[key] = sorted(value)
        else:
            copied[key] = value
    return copied
