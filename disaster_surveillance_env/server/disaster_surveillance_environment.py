from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence
from uuid import uuid4

import numpy as np
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from ..models import (
    ACTION_LABELS,
    Coord,
    DisasterObservation,
    DisasterState,
    Drone,
    DroneActions,
    Event,
    build_observation,
    build_team_metrics,
    compute_grid_coverage,
    compute_reward,
    get_fov_cells,
    normalize_actions,
    spawn_event,
)


class DisasterSurveillanceEnvironment(
    Environment[DroneActions, DisasterObservation, DisasterState]
):
    """Multi-agent drone surveillance grid world for event detection."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    GRID_SIZE = 10
    EPISODE_LENGTH = 50
    N_DRONES = 3
    FOV_RADIUS = 2
    P_SPAWN = 0.2
    COVERAGE_REWARD_PER_NEW_CELL = 0.5

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        episode_length: int = EPISODE_LENGTH,
        n_drones: int = N_DRONES,
        fov_radius: int = FOV_RADIUS,
        p_spawn: float = P_SPAWN,
        coverage_reward_per_new_cell: float = COVERAGE_REWARD_PER_NEW_CELL,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.episode_length = episode_length
        self.agent_ids = [f"drone_{index + 1}" for index in range(n_drones)]
        self.action_space: Dict[str, Any] = {
            "type": "multi_agent_discrete",
            "agents": self.agent_ids,
            "n": 5,
            "actions": ACTION_LABELS,
        }
        self.observation_space: Dict[str, Any] = {
            "type": "per_agent_partial_observation",
            "coordination": "decentralized",
            "communication": "none",
            "reward_mode": "shared_global_overlap_and_coverage_shaping",
            "agents": self.agent_ids,
            "position": {"shape": [2], "dtype": "int"},
            "visible_cells": {"shape": ["variable", 2], "dtype": "int"},
            "visible_events": {
                "shape": ["variable"],
                "fields": ["id", "location", "start_time", "duration", "end_time"],
            },
            "local_visited_cells": {"shape": ["variable", 2], "dtype": "int"},
            "frontier_cells": {"shape": ["variable", 2], "dtype": "int"},
            "exploration": {
                "fields": ["steps_taken", "visited_cell_count", "revisit_count", "coverage_ratio", "last_action"],
            },
        }
        self.fov_radius = fov_radius
        self.p_spawn = p_spawn
        self.coverage_reward_per_new_cell = coverage_reward_per_new_cell
        self.rng = np.random.default_rng(seed)
        self.episode_id: Optional[str] = None
        self.timestep = 0
        self.next_event_id = 1
        self.drones: Dict[str, Drone] = {}
        self.events: list[Event] = []
        self.visited_cells: set[Coord] = set()
        self.metrics: Dict[str, Any] = {}
        self._last_reward = 0.0
        self.reset(seed=seed)

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="disaster-surveillance-grid",
            description="Level 4 decentralized coordination environment with implicit FOV overlap and coverage reward shaping.",
            version="0.2.0",
        )

    @property
    def state(self) -> DisasterState:
        return DisasterState(
            episode_id=self.episode_id,
            step_count=self.timestep,
            timestep=self.timestep,
            active_events=[
                {
                    "id": event.id,
                    "location": event.location,
                    "start_time": event.start_time,
                    "duration": event.duration,
                    "end_time": event.end_time,
                    "detected": event.detected,
                }
                for event in self.events
            ],
            drone_positions={drone_id: drone.position for drone_id, drone in self.drones.items()},
            per_drone_stats={
                drone_id: {
                    "visited_cell_count": len(drone.visited_cells),
                    "revisit_count": drone.revisit_count,
                    "steps_taken": drone.steps_taken,
                    "last_action": ACTION_LABELS[drone.last_action],
                }
                for drone_id, drone in self.drones.items()
            },
            metrics=build_team_metrics(
                drones=self.drones,
                grid_size=self.grid_size,
                metrics=self.metrics,
                team_visited_cells=self.visited_cells,
            ),
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DisasterObservation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.episode_id = episode_id or str(uuid4())
        self.timestep = 0
        self.next_event_id = 1
        self.events = []
        self.visited_cells = set()
        self.drones = {
            drone_id: Drone(
                id=drone_id,
                position=(
                    int(self.rng.integers(0, self.grid_size)),
                    int(self.rng.integers(0, self.grid_size)),
                ),
            )
            for drone_id in self.agent_ids
        }
        for drone in self.drones.values():
            drone.visited_cells.add(drone.position)
        self.metrics = {
            "level": 4,
            "coordination_mode": "decentralized_no_communication",
            "reward_mode": "shared_global_overlap_and_coverage_shaping",
            "total_events_spawned": 0,
            "events_detected": 0,
            "events_missed": 0,
            "detection_latencies": [],
            "mean_detection_latency": None,
            "total_reward": 0.0,
            "total_overlap_penalty": 0.0,
            "total_coverage_reward": 0.0,
            "unique_cells_visited": 0,
            "grid_coverage": 0.0,
            "grid_coverage_percent": 0.0,
        }
        self._mark_initial_cells_visited()
        self._sync_coverage_metrics()
        self._last_reward = 0.0
        return self._build_current_observation(reward=0.0, done=False)

    def step(
        self,
        action: DroneActions | Mapping[str, int] | Sequence[int],
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DisasterObservation:
        del timeout_s, kwargs

        if self.timestep >= self.episode_length:
            return self._build_current_observation(reward=0.0, done=True)

        actions = normalize_actions(action, self.agent_ids)

        new_event = spawn_event(
            self.rng,
            self.timestep,
            self.next_event_id,
            self.grid_size,
            self.p_spawn,
        )
        if new_event is not None:
            self.events.append(new_event)
            self.next_event_id += 1
            self.metrics["total_events_spawned"] += 1

        for drone_id, drone in self.drones.items():
            drone.move(actions[drone_id], self.grid_size)

        fovs = self._compute_fovs()
        detected_count = self._detect_events(fovs)
        missed_count = self._remove_detected_and_expired()

        reward, reward_info = compute_reward(
            detected_count=detected_count,
            missed_count=missed_count,
            fovs=fovs,
            visited_cells=self.visited_cells,
            reward_per_new_cell=self.coverage_reward_per_new_cell,
        )
        self._apply_reward_metrics(reward_info)
        self.metrics["total_reward"] += reward
        self._last_reward = reward

        self.timestep += 1
        done = self.timestep >= self.episode_length
        observation = self._build_current_observation(reward=reward, done=done)
        observation.metadata.update(reward_info)
        observation.metadata["last_step_detected"] = detected_count
        observation.metadata["last_step_missed"] = missed_count
        observation.metadata["coordination_mode"] = "decentralized_no_communication"
        observation.metadata["reward_mode"] = "shared_global_overlap_and_coverage_shaping"
        return observation

    def render_ascii(self) -> str:
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for x, y in self.visited_cells:
            grid[y][x] = "v"
        for event in self.events:
            if event.is_active(self.timestep) and not event.detected:
                x, y = event.location
                grid[y][x] = "E"
        for index, drone in enumerate(self.drones.values(), start=1):
            x, y = drone.position
            grid[y][x] = str(index)
        return "\n".join(" ".join(row) for row in grid)

    def _compute_fovs(self) -> Dict[str, set[Coord]]:
        return {
            drone_id: get_fov_cells(drone, self.grid_size, self.fov_radius)
            for drone_id, drone in self.drones.items()
        }

    def _mark_initial_cells_visited(self) -> None:
        initial_fovs = self._compute_fovs()
        initial_visible_cells = set().union(*initial_fovs.values()) if initial_fovs else set()
        self.visited_cells.update(initial_visible_cells)

    def _apply_reward_metrics(self, reward_info: Mapping[str, Any]) -> None:
        self.visited_cells.update(reward_info["new_cells"])
        self.metrics["total_overlap_penalty"] += reward_info["overlap_penalty"]
        self.metrics["total_coverage_reward"] += reward_info["coverage_reward"]
        self._sync_coverage_metrics()

    def _sync_coverage_metrics(self) -> None:
        self.metrics["unique_cells_visited"] = len(self.visited_cells)
        self.metrics["grid_coverage"] = compute_grid_coverage(self.visited_cells, self.grid_size)
        self.metrics["grid_coverage_percent"] = 100.0 * self.metrics["grid_coverage"]

    def _detect_events(self, fovs: Mapping[str, set[Coord]]) -> int:
        detected_count = 0
        visible_cells = set().union(*fovs.values()) if fovs else set()
        for event in self.events:
            if event.is_active(self.timestep) and not event.detected and event.location in visible_cells:
                event.detected = True
                event.detection_time = self.timestep
                detected_count += 1
                self.metrics["events_detected"] += 1
                latency = self.timestep - event.start_time
                self.metrics["detection_latencies"].append(latency)

        latencies = self.metrics["detection_latencies"]
        self.metrics["mean_detection_latency"] = float(np.mean(latencies)) if latencies else None
        return detected_count

    def _remove_detected_and_expired(self) -> int:
        missed_count = 0
        kept_events: list[Event] = []
        for event in self.events:
            if event.detected:
                continue
            if self.timestep >= event.end_time:
                missed_count += 1
                self.metrics["events_missed"] += 1
                continue
            kept_events.append(event)
        self.events = kept_events
        return missed_count

    def _build_current_observation(self, reward: float, done: bool) -> DisasterObservation:
        return build_observation(
            timestep=self.timestep,
            grid_size=self.grid_size,
            drones=self.drones,
            events=self.events,
            fovs=self._compute_fovs(),
            metrics=build_team_metrics(
                drones=self.drones,
                grid_size=self.grid_size,
                metrics=self.metrics,
                team_visited_cells=self.visited_cells,
            ),
            reward=reward,
            done=done,
        )


def run_random_episode(seed: int = 7, render: bool = False) -> Dict[str, Any]:
    env = DisasterSurveillanceEnvironment(seed=seed)
    observation = env.reset(seed=seed)
    print("Initial observation:")
    print(observation.model_dump())

    while not observation.done:
        actions = {agent_id: int(env.rng.integers(0, 5)) for agent_id in env.agent_ids}
        observation = env.step(DroneActions(actions=actions))
        if render:
            print(f"\nt={env.timestep} reward={observation.reward} actions={actions}")
            print(env.render_ascii())

    print("\nFinal metrics:")
    for key, value in env.metrics.items():
        print(f"{key}: {value}")
    print(f"\nTotal reward: {env.metrics['total_reward']}")
    print(f"Overlap penalty: {env.metrics['total_overlap_penalty']}")
    print(f"Coverage %: {env.metrics['grid_coverage_percent']:.1f}")
    return env.metrics


def main() -> None:
    run_random_episode(seed=42, render=False)


if __name__ == "__main__":
    main()
