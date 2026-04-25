from __future__ import annotations

import argparse
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
    HOTSPOTS,
    SEVERITY_CONFIG,
    build_observation,
    build_team_metrics,
    compute_baseline_reward,
    compute_episode_bonus,
    compute_grid_coverage,
    compute_level5_reward,
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
    EVENT_MEMORY_LIMIT = 5
    LATENCY_BONUS_THRESHOLD = 3.0

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        episode_length: int = EPISODE_LENGTH,
        n_drones: int = N_DRONES,
        fov_radius: int = FOV_RADIUS,
        p_spawn: float = P_SPAWN,
        coverage_reward_per_new_cell: float = COVERAGE_REWARD_PER_NEW_CELL,
        level: int = 4,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if level not in {3, 4, 5}:
            raise ValueError(f"level must be 3, 4, or 5; got {level}.")

        self.level = level
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
            "reward_mode": self.reward_mode,
            "agents": self.agent_ids,
            "position": {"shape": [2], "dtype": "int"},
            "visible_cells": {"shape": ["variable", 2], "dtype": "int"},
            "visible_events": {
                "shape": ["variable"],
                "fields": [
                    "id",
                    "location",
                    "severity",
                    "start_time",
                    "duration",
                    "end_time",
                    "time_remaining",
                    "deadline_remaining",
                ],
            },
            "local_visited_cells": {"shape": ["variable", 2], "dtype": "int"},
            "frontier_cells": {"shape": ["variable", 2], "dtype": "int"},
            "detected_event_history": {
                "shape": ["variable"],
                "fields": ["id", "severity", "location", "detected_at", "time_since_detection"],
            },
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
        self._episode_bonus_applied = False
        self.reset(seed=seed)

    def get_metadata(self) -> EnvironmentMetadata:
        descriptions = {
            3: "Level 3 decentralized RL environment with shared global reward and no communication.",
            4: "Level 4 decentralized coordination environment with implicit FOV overlap and coverage reward shaping.",
            5: "Level 5 long-horizon disaster surveillance with urgency, prioritization, delayed rewards, and hotspot-biased event spawning.",
        }
        return EnvironmentMetadata(
            name="disaster-surveillance-grid",
            description=descriptions[self.level],
            version="0.4.0",
        )

    @property
    def reward_mode(self) -> str:
        if self.level == 3:
            return "shared_global_baseline"
        if self.level == 4:
            return "shared_global_overlap_and_coverage_shaping"
        return "shared_global_long_horizon_priority_shaping"

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
                    "severity": event.severity,
                    "deadline": event.deadline,
                    "deadline_step": event.deadline_step,
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
                    "detected_event_history_size": len(drone.detected_event_history),
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
        self._episode_bonus_applied = False
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

        zero_by_severity = {severity: 0 for severity in SEVERITY_CONFIG}
        latency_lists = {severity: [] for severity in SEVERITY_CONFIG}
        self.metrics = {
            "level": self.level,
            "coordination_mode": "decentralized_no_communication",
            "reward_mode": self.reward_mode,
            "hotspots": list(HOTSPOTS),
            "total_events_spawned": 0,
            "events_detected": 0,
            "events_missed": 0,
            "detection_latencies": [],
            "mean_detection_latency": None,
            "total_reward": 0.0,
            "total_overlap_penalty": 0.0,
            "total_coverage_reward": 0.0,
            "total_detection_reward": 0.0,
            "total_miss_penalty": 0.0,
            "total_episode_bonus": 0.0,
            "unique_cells_visited": 0,
            "grid_coverage": 0.0,
            "grid_coverage_percent": 0.0,
            "events_spawned_by_severity": dict(zero_by_severity),
            "events_detected_by_severity": dict(zero_by_severity),
            "missed_by_severity": dict(zero_by_severity),
            "detected_on_time_by_severity": dict(zero_by_severity),
            "avg_latency_by_severity": {severity: None for severity in SEVERITY_CONFIG},
            "on_time_detection_rate": 0.0,
            "high_priority_miss_rate": 0.0,
            "episode_rescue_score": 0.0,
            "reward_breakdown": {
                "time_penalty": 0.0,
                "detection_reward": 0.0,
                "miss_penalty": 0.0,
                "overlap_penalty": 0.0,
                "coverage_reward": 0.0,
                "episode_bonus": 0.0,
            },
            "_latencies_by_severity": latency_lists,
        }
        self._mark_initial_cells_visited()
        self._sync_coverage_metrics()
        self._sync_priority_metrics()
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
        self._spawn_step_event()

        for drone_id, drone in self.drones.items():
            drone.move(actions[drone_id], self.grid_size)

        fovs = self._compute_fovs()
        detected_events = self._detect_events(fovs)
        missed_events = self._remove_detected_and_expired()
        self._update_drone_detection_memory(detected_events, fovs)

        reward, reward_info = self._compute_step_reward(
            detected_events=detected_events,
            missed_events=missed_events,
            fovs=fovs,
        )
        self._apply_reward_metrics(reward_info)

        self.timestep += 1
        done = self.timestep >= self.episode_length
        if done:
            episode_bonus, episode_bonus_info = self._apply_episode_end_bonus()
            reward += episode_bonus
            reward_info["episode_bonus"] = episode_bonus
            reward_info["reward_breakdown"]["episode_bonus"] = episode_bonus
            reward_info["episode_bonus_breakdown"] = episode_bonus_info

        self.metrics["total_reward"] += reward
        self.metrics["episode_rescue_score"] = self.metrics["total_reward"]
        self._last_reward = reward

        observation = self._build_current_observation(reward=reward, done=done)
        observation.metadata.update(reward_info)
        observation.metadata["last_step_detected"] = len(detected_events)
        observation.metadata["last_step_missed"] = len(missed_events)
        observation.metadata["coordination_mode"] = "decentralized_no_communication"
        observation.metadata["reward_mode"] = self.reward_mode
        return observation

    def render_ascii(self) -> str:
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for x, y in self.visited_cells:
            grid[y][x] = "v"
        for event in self.events:
            if event.is_active(self.timestep) and not event.detected:
                x, y = event.location
                label = event.severity[0]
                grid[y][x] = label
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

    def _spawn_step_event(self) -> None:
        new_event = spawn_event(
            self.rng,
            self.timestep,
            self.next_event_id,
            self.grid_size,
            self.p_spawn,
        )
        if new_event is None:
            return

        self.events.append(new_event)
        self.next_event_id += 1
        self.metrics["total_events_spawned"] += 1
        self.metrics["events_spawned_by_severity"][new_event.severity] += 1
        self._sync_priority_metrics()

    def _compute_step_reward(
        self,
        *,
        detected_events: Sequence[Event],
        missed_events: Sequence[Event],
        fovs: Mapping[str, set[Coord]],
    ) -> tuple[float, Dict[str, Any]]:
        if self.level == 3:
            return compute_baseline_reward(
                detected_count=len(detected_events),
                missed_count=len(missed_events),
                fovs=fovs,
                visited_cells=self.visited_cells,
            )

        if self.level == 4:
            return compute_reward(
                detected_count=len(detected_events),
                missed_count=len(missed_events),
                fovs=fovs,
                visited_cells=self.visited_cells,
                reward_per_new_cell=self.coverage_reward_per_new_cell,
            )

        return compute_level5_reward(
            detected_events=detected_events,
            missed_events=missed_events,
            fovs=fovs,
            visited_cells=self.visited_cells,
            reward_per_new_cell=self.coverage_reward_per_new_cell,
        )

    def _apply_reward_metrics(self, reward_info: Mapping[str, Any]) -> None:
        self.visited_cells.update(reward_info["new_cells"])
        self.metrics["total_overlap_penalty"] += reward_info["overlap_penalty"]
        self.metrics["total_coverage_reward"] += reward_info["coverage_reward"]
        self.metrics["total_detection_reward"] += reward_info["detection_reward"]
        self.metrics["total_miss_penalty"] += reward_info["miss_penalty"]
        for key, value in reward_info["reward_breakdown"].items():
            if key == "episode_bonus":
                continue
            self.metrics["reward_breakdown"][key] += float(value)
        self._sync_coverage_metrics()
        self._sync_priority_metrics()

    def _sync_coverage_metrics(self) -> None:
        self.metrics["unique_cells_visited"] = len(self.visited_cells)
        self.metrics["grid_coverage"] = compute_grid_coverage(self.visited_cells, self.grid_size)
        self.metrics["grid_coverage_percent"] = 100.0 * self.metrics["grid_coverage"]

    def _sync_priority_metrics(self) -> None:
        latencies_by_severity = self.metrics["_latencies_by_severity"]
        for severity, values in latencies_by_severity.items():
            self.metrics["avg_latency_by_severity"][severity] = (
                float(np.mean(values)) if values else None
            )

        total_detected = self.metrics["events_detected"]
        on_time_total = sum(self.metrics["detected_on_time_by_severity"].values())
        self.metrics["on_time_detection_rate"] = (
            on_time_total / float(total_detected) if total_detected else 0.0
        )

        spawned_high = self.metrics["events_spawned_by_severity"]["HIGH"]
        missed_high = self.metrics["missed_by_severity"]["HIGH"]
        self.metrics["high_priority_miss_rate"] = (
            missed_high / float(spawned_high) if spawned_high else 0.0
        )

    def _detect_events(self, fovs: Mapping[str, set[Coord]]) -> list[Event]:
        detected_events: list[Event] = []
        visible_cells = set().union(*fovs.values()) if fovs else set()
        for event in self.events:
            if event.is_active(self.timestep) and not event.detected and event.location in visible_cells:
                event.detected = True
                event.detection_time = self.timestep
                detected_events.append(event)
                self.metrics["events_detected"] += 1
                self.metrics["events_detected_by_severity"][event.severity] += 1

                latency = self.timestep - event.start_time
                self.metrics["detection_latencies"].append(latency)
                self.metrics["_latencies_by_severity"][event.severity].append(latency)
                if self.timestep <= event.deadline_step:
                    self.metrics["detected_on_time_by_severity"][event.severity] += 1

        latencies = self.metrics["detection_latencies"]
        self.metrics["mean_detection_latency"] = float(np.mean(latencies)) if latencies else None
        self._sync_priority_metrics()
        return detected_events

    def _remove_detected_and_expired(self) -> list[Event]:
        missed_events: list[Event] = []
        kept_events: list[Event] = []
        for event in self.events:
            if event.detected:
                continue
            if self.timestep >= event.end_time:
                missed_events.append(event)
                self.metrics["events_missed"] += 1
                self.metrics["missed_by_severity"][event.severity] += 1
                continue
            kept_events.append(event)
        self.events = kept_events
        self._sync_priority_metrics()
        return missed_events

    def _update_drone_detection_memory(
        self,
        detected_events: Sequence[Event],
        fovs: Mapping[str, set[Coord]],
    ) -> None:
        for drone_id, drone in self.drones.items():
            for event in detected_events:
                if event.location not in fovs[drone_id]:
                    continue
                if event.id in drone.remembered_detected_event_ids:
                    continue
                drone.detected_event_history.append(
                    {
                        "id": event.id,
                        "severity": event.severity,
                        "location": event.location,
                        "detected_at": int(event.detection_time),
                    }
                )
                drone.remembered_detected_event_ids.add(event.id)
                if len(drone.detected_event_history) > self.EVENT_MEMORY_LIMIT:
                    removed = drone.detected_event_history.pop(0)
                    drone.remembered_detected_event_ids.discard(int(removed["id"]))

    def _apply_episode_end_bonus(self) -> tuple[float, Dict[str, Any]]:
        if self._episode_bonus_applied:
            return 0.0, {}

        self._episode_bonus_applied = True
        episode_bonus, breakdown = compute_episode_bonus(
            self.metrics,
            latency_threshold=self.LATENCY_BONUS_THRESHOLD,
        )
        self.metrics["total_episode_bonus"] += episode_bonus
        self.metrics["reward_breakdown"]["episode_bonus"] += episode_bonus
        self.metrics["episode_bonus_breakdown"] = breakdown
        return episode_bonus, breakdown

    def _build_current_observation(self, reward: float, done: bool) -> DisasterObservation:
        public_metrics = build_team_metrics(
            drones=self.drones,
            grid_size=self.grid_size,
            metrics={key: value for key, value in self.metrics.items() if not key.startswith("_")},
            team_visited_cells=self.visited_cells,
        )
        return build_observation(
            timestep=self.timestep,
            grid_size=self.grid_size,
            drones=self.drones,
            events=self.events,
            fovs=self._compute_fovs(),
            metrics=public_metrics,
            reward=reward,
            done=done,
            event_memory_limit=self.EVENT_MEMORY_LIMIT,
        )


def _print_severity_metrics(metrics: Mapping[str, Any]) -> None:
    print("Severity metrics:")
    for severity in SEVERITY_CONFIG:
        print(
            "  {severity}: spawned={spawned} detected={detected} on_time={on_time} missed={missed} avg_latency={latency}".format(
                severity=severity,
                spawned=metrics["events_spawned_by_severity"][severity],
                detected=metrics["events_detected_by_severity"][severity],
                on_time=metrics["detected_on_time_by_severity"][severity],
                missed=metrics["missed_by_severity"][severity],
                latency=metrics["avg_latency_by_severity"][severity],
            )
        )


def _print_reward_breakdown(metrics: Mapping[str, Any]) -> None:
    print("Reward breakdown:")
    for key, value in metrics["reward_breakdown"].items():
        print(f"  {key}: {value}")
    if "episode_bonus_breakdown" in metrics:
        print("Episode bonus details:")
        for key, value in metrics["episode_bonus_breakdown"].items():
            print(f"  {key}: {value}")


def run_random_episode(
    seed: int = 7,
    render: bool = False,
    level: int = 4,
    verbose: bool = True,
) -> Dict[str, Any]:
    env = DisasterSurveillanceEnvironment(seed=seed, level=level)
    observation = env.reset(seed=seed)
    if verbose:
        print("Initial observation:")
        print(observation.model_dump())

    while not observation.done:
        actions = {agent_id: int(env.rng.integers(0, 5)) for agent_id in env.agent_ids}
        observation = env.step(DroneActions(actions=actions))
        if render:
            print(f"\nt={env.timestep} reward={observation.reward} actions={actions}")
            print(env.render_ascii())

    public_metrics = {key: value for key, value in env.metrics.items() if not key.startswith("_")}
    if verbose:
        print("\nFinal metrics:")
        for key, value in public_metrics.items():
            print(f"{key}: {value}")
        _print_severity_metrics(public_metrics)
        _print_reward_breakdown(public_metrics)
        print(f"\nTotal reward: {public_metrics['total_reward']}")
        print(f"Coverage %: {public_metrics['grid_coverage_percent']:.1f}")
    return public_metrics


def run_random_episodes(
    *,
    episodes: int = 1,
    seed: int = 7,
    level: int = 4,
    render: bool = False,
) -> list[Dict[str, Any]]:
    if episodes < 1:
        raise ValueError(f"episodes must be >= 1; got {episodes}.")

    all_metrics: list[Dict[str, Any]] = []
    for episode_index in range(episodes):
        episode_seed = seed + episode_index
        metrics = run_random_episode(
            seed=episode_seed,
            render=render and episodes == 1,
            level=level,
            verbose=episodes == 1,
        )
        all_metrics.append(metrics)
        if episodes > 1:
            print(
                "episode={episode} seed={seed} level={level} total_reward={reward:.1f} "
                "detected={detected} missed={missed} high_miss_rate={high_miss:.2f} "
                "coverage={coverage:.1f}%".format(
                    episode=episode_index + 1,
                    seed=episode_seed,
                    level=level,
                    reward=metrics["total_reward"],
                    detected=metrics["events_detected"],
                    missed=metrics["events_missed"],
                    high_miss=metrics["high_priority_miss_rate"],
                    coverage=metrics["grid_coverage_percent"],
                )
            )

    if episodes > 1:
        mean_reward = float(np.mean([metrics["total_reward"] for metrics in all_metrics]))
        mean_coverage = float(np.mean([metrics["grid_coverage_percent"] for metrics in all_metrics]))
        mean_high_miss = float(np.mean([metrics["high_priority_miss_rate"] for metrics in all_metrics]))
        print("\nAggregate metrics:")
        print(f"episodes: {episodes}")
        print(f"level: {level}")
        print(f"mean_total_reward: {mean_reward:.2f}")
        print(f"mean_coverage_percent: {mean_coverage:.2f}")
        print(f"mean_high_priority_miss_rate: {mean_high_miss:.2f}")

    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random rollouts for the disaster surveillance environment.")
    parser.add_argument(
        "--level",
        type=int,
        choices=[3, 4, 5],
        default=4,
        help="Environment level: 3 baseline, 4 shaped coordination, or 5 long-horizon urgency.",
    )
    parser.add_argument("--episodes", "-k", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed. Episode i uses seed + i.")
    parser.add_argument("--render", action="store_true", help="Render ASCII grid per step. Only enabled for a single episode.")
    args = parser.parse_args()
    run_random_episodes(
        episodes=args.episodes,
        seed=args.seed,
        level=args.level,
        render=args.render,
    )


if __name__ == "__main__":
    main()
