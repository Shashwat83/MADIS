from __future__ import annotations

import argparse
from typing import Any, Dict, Mapping, Optional, Sequence
from uuid import uuid4

import numpy as np
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from ..coordinator import CoordinatorAgent, LLMCoordinator
from ..models import (
    ACTION_LABELS,
    Coord,
    DisasterObservation,
    DisasterState,
    Drone,
    DroneActions,
    Event,
    HOTSPOTS,
    EVENT_TYPES,
    SEVERITY_CONFIG,
    build_observation,
    build_team_metrics,
    compute_baseline_reward,
    compute_episode_bonus,
    compute_grid_coverage,
    compute_level5_reward,
    compute_reward,
    event_type_priority,
    get_fov_cells,
    manhattan_distance,
    normalize_actions,
    normalize_targets,
    severity_label_from_score,
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
    COORDINATOR_RECENT_MEMORY = 25
    MAX_SPREAD_EVENTS_PER_STEP = 12

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
        coordinator: Optional[CoordinatorAgent] = None,
    ) -> None:
        super().__init__()
        if level not in {3, 4, 5, 6}:
            raise ValueError(f"level must be 3, 4, 5, or 6; got {level}.")

        self.level = level
        self.grid_size = grid_size
        self.episode_length = episode_length
        self.agent_ids = [f"drone_{index + 1}" for index in range(n_drones)]
        self.coordinator = coordinator or (LLMCoordinator() if level == 6 else None)
        self.action_space: Dict[str, Any] = (
            {
                "type": "multi_agent_target_assignment",
                "agents": self.agent_ids,
                "target_shape": [2],
                "description": "Coordinator-assigned target coordinates for each drone.",
            }
            if self.level == 6
            else {
                "type": "multi_agent_discrete",
                "agents": self.agent_ids,
                "n": 5,
                "actions": ACTION_LABELS,
            }
        )
        self.observation_space: Dict[str, Any] = {
            "type": "per_agent_partial_observation",
            "coordination": "centralized_coordinator" if self.level == 6 else "decentralized",
            "communication": "none",
            "reward_mode": self.reward_mode,
            "agents": self.agent_ids,
            "position": {"shape": [2], "dtype": "int"},
            "assigned_target": {"shape": [2], "dtype": "int"},
            "visible_cells": {"shape": ["variable", 2], "dtype": "int"},
            "visible_events": {
                "shape": ["variable"],
                "fields": [
                    "id",
                    "location",
                    "severity",
                    "type",
                    "type_priority",
                    "severity_score",
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
            "coordinator": {
                "fields": [
                    "timestep",
                    "drone_positions",
                    "visible_active_events",
                    "known_detected_events",
                    "team_frontier_cells",
                    "recently_observed_cells",
                ],
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
        self._recently_observed_cells: list[Coord] = []
        self.last_assigned_targets: Dict[str, Coord] = {}
        self.last_coordinator_observation: Dict[str, Any] = {}
        self.reset(seed=seed)

    def get_metadata(self) -> EnvironmentMetadata:
        descriptions = {
            3: "Level 3 decentralized RL environment with shared global reward and no communication.",
            4: "Level 4 decentralized coordination environment with implicit FOV overlap and coverage reward shaping.",
            5: "Level 5 long-horizon disaster surveillance with urgency, prioritization, delayed rewards, and hotspot-biased event spawning.",
            6: "Level 6 coordinator-driven disaster surveillance with centralized target assignment and drone execution toward goals.",
        }
        return EnvironmentMetadata(
            name="disaster-surveillance-grid",
            description=descriptions[self.level],
            version="0.5.0",
        )

    @property
    def reward_mode(self) -> str:
        if self.level == 3:
            return "shared_global_baseline"
        if self.level == 4:
            return "shared_global_overlap_and_coverage_shaping"
        if self.level == 5:
            return "shared_global_long_horizon_priority_shaping"
        return "shared_global_coordinator_priority_shaping"

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
                    "type": event.type,
                    "type_priority": event_type_priority(event.type),
                    "severity_score": event.severity_score,
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
                    "current_target": drone.current_target,
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
        self._recently_observed_cells = []
        self.last_assigned_targets = {}
        self.last_coordinator_observation = {}
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
            "coordination_mode": "centralized_coordinator" if self.level == 6 else "decentralized_no_communication",
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
            "total_growth_penalty": 0.0,
            "unique_cells_visited": 0,
            "grid_coverage": 0.0,
            "grid_coverage_percent": 0.0,
            "events_spawned_by_severity": dict(zero_by_severity),
            "events_detected_by_severity": dict(zero_by_severity),
            "missed_by_severity": dict(zero_by_severity),
            "pending_by_severity": dict(zero_by_severity),
            "detected_on_time_by_severity": dict(zero_by_severity),
            "avg_latency_by_severity": {severity: None for severity in SEVERITY_CONFIG},
            "on_time_detection_rate": 0.0,
            "high_priority_miss_rate": 0.0,
            "episode_rescue_score": 0.0,
            "reward_breakdown": {
                "time_penalty": 0.0,
                "detection_reward": 0.0,
                "miss_penalty": 0.0,
                "growth_penalty": 0.0,
                "overlap_penalty": 0.0,
                "coverage_reward": 0.0,
                "episode_bonus": 0.0,
            },
            "last_assigned_targets": {},
            "target_assignment_count": 0,
            "target_progress_sum": 0.0,
            "movement_distance_sum": 0.0,
            "path_efficiency": 0.0,
            "coordination_quality": 0.0,
            "coordinator_backend": type(self.coordinator).__name__ if self.coordinator is not None else None,
            "coordinator_model_name": getattr(self.coordinator, "model_name", None),
            "coordinator_decision_source": None,
            "coordinator_fallback_count": 0,
            "coordinator_fallback_reason": None,
            "last_llm_latency_ms": None,
            "last_llm_raw_response": None,
            "last_llm_debug": None,
            "_latencies_by_severity": latency_lists,
        }
        self._mark_initial_cells_visited()
        self._sync_coverage_metrics()
        self._sync_priority_metrics()
        self._last_reward = 0.0
        return self._build_current_observation(reward=0.0, done=False)

    def step(
        self,
        action: DroneActions | Mapping[str, Any] | Sequence[Any] | None,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DisasterObservation:
        del timeout_s, kwargs

        if self.timestep >= self.episode_length:
            return self._build_current_observation(reward=0.0, done=True)

        processed_timestep = self.timestep
        self._spawn_step_event()

        if self.level == 6:
            coordinator_observation = self.build_coordinator_observation()
            assigned_targets = self._resolve_coordinator_targets(action, coordinator_observation)
            self._apply_coordinator_targets(assigned_targets)
        else:
            coordinator_observation = None
            actions = normalize_actions(action, self.agent_ids)
            for drone_id, drone in self.drones.items():
                drone.move(actions[drone_id], self.grid_size)

        self._update_events()
        fovs = self._compute_fovs()
        self._remember_observed_cells(fovs)
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

        observation = self._build_current_observation(
            reward=reward,
            done=done,
            coordinator_observation=coordinator_observation,
        )
        observation.metadata.update(reward_info)
        observation.metadata["processed_timestep"] = processed_timestep
        observation.metadata["next_timestep"] = self.timestep
        observation.metadata["last_step_detected"] = len(detected_events)
        observation.metadata["last_step_missed"] = len(missed_events)
        observation.metadata["coordination_mode"] = self.metrics["coordination_mode"]
        observation.metadata["reward_mode"] = self.reward_mode
        observation.metadata["assigned_targets"] = dict(self.last_assigned_targets)
        return observation

    def render_ascii(self) -> str:
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for x, y in self.visited_cells:
            grid[y][x] = "v"
        for event in self.events:
            if event.is_active(self.timestep) and not event.detected:
                x, y = event.location
                grid[y][x] = event.severity[0]
        for drone_id, target in self.last_assigned_targets.items():
            x, y = target
            if grid[y][x] == ".":
                grid[y][x] = "T"
        for index, drone in enumerate(self.drones.values(), start=1):
            x, y = drone.position
            grid[y][x] = str(index)
        return "\n".join(" ".join(row) for row in grid)

    def build_coordinator_observation(self) -> Dict[str, Any]:
        fovs = self._compute_fovs()
        visible_cells = set().union(*fovs.values()) if fovs else set()
        team_frontier_cells = sorted(cell for cell in visible_cells if cell not in self.visited_cells)
        visible_active_events = [
            {
                "id": event.id,
                "location": event.location,
                "severity": event.severity,
                "type": event.type,
                "type_priority": event_type_priority(event.type),
                "severity_score": event.severity_score,
                "time_remaining": event.end_time - self.timestep,
                "deadline_remaining": event.deadline_step - self.timestep,
            }
            for event in self.events
            if event.is_active(self.timestep) and event.location in visible_cells and not event.detected
        ]
        known_detected_events = sorted(
            {
                (
                    entry["id"],
                    entry["severity"],
                    tuple(entry["location"]),
                    int(entry["detected_at"]),
                )
                for drone in self.drones.values()
                for entry in drone.detected_event_history
            }
        )

        observation = {
            "timestep": self.timestep,
            "grid_size": self.grid_size,
            "drone_positions": {drone_id: drone.position for drone_id, drone in self.drones.items()},
            "visible_active_events": visible_active_events,
            "known_detected_events": [
                {
                    "id": event_id,
                    "severity": severity,
                    "location": location,
                    "detected_at": detected_at,
                }
                for event_id, severity, location, detected_at in known_detected_events
            ],
            "team_frontier_cells": team_frontier_cells,
            "recently_observed_cells": list(self._recently_observed_cells[-self.COORDINATOR_RECENT_MEMORY :]),
            "recent_team_coverage_ratio": self.metrics.get("grid_coverage", 0.0),
        }
        self.last_coordinator_observation = observation
        return observation

    def _compute_fovs(self) -> Dict[str, set[Coord]]:
        return {
            drone_id: get_fov_cells(drone, self.grid_size, self.fov_radius)
            for drone_id, drone in self.drones.items()
        }

    def _mark_initial_cells_visited(self) -> None:
        initial_fovs = self._compute_fovs()
        initial_visible_cells = set().union(*initial_fovs.values()) if initial_fovs else set()
        self.visited_cells.update(initial_visible_cells)
        self._remember_observed_cells(initial_fovs)

    def _remember_observed_cells(self, fovs: Mapping[str, set[Coord]]) -> None:
        visible_cells = sorted(set().union(*fovs.values()) if fovs else set())
        for cell in visible_cells:
            self._recently_observed_cells.append(cell)
        if len(self._recently_observed_cells) > self.COORDINATOR_RECENT_MEMORY:
            self._recently_observed_cells = self._recently_observed_cells[-self.COORDINATOR_RECENT_MEMORY :]

    def _spawn_step_event(self) -> None:
        drift = float(np.sin(self.timestep / 10.0))
        adjusted_p_spawn = float(np.clip(self.p_spawn * (1.0 + 0.5 * drift), 0.0, 1.0))
        new_event = spawn_event(
            self.rng,
            self.timestep,
            self.next_event_id,
            self.grid_size,
            adjusted_p_spawn,
        )
        if new_event is None:
            return

        self._register_new_event(new_event)

    def _register_new_event(self, event: Event) -> None:
        self.events.append(event)
        self.next_event_id = max(self.next_event_id, event.id + 1)
        self.metrics["total_events_spawned"] += 1
        self.metrics["events_spawned_by_severity"][event.severity] += 1
        self._sync_priority_metrics()

    def _valid_cell(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _event_exists_at(self, location: Coord, event_type: str) -> bool:
        return any(
            event.location == location and event.type == event_type and not event.detected and event.is_active(self.timestep)
            for event in self.events
        )

    def _spawn_dynamic_event(self, *, location: Coord, event_type: str, template: Event) -> Event:
        if event_type == "road_blockage":
            blockage_level = float(np.clip(max(template.blockage_level, 0.4) * self.rng.uniform(0.9, 1.15), 0.3, 2.0))
            severity_score = float(np.clip(max(template.severity_score, 0.5) * self.rng.uniform(0.85, 1.0), 0.4, 2.0))
            return Event(
                id=self.next_event_id,
                location=location,
                start_time=self.timestep,
                duration=max(3, int(template.duration)),
                severity=severity_label_from_score(severity_score),
                type="road_blockage",
                severity_score=severity_score,
                blockage_level=blockage_level,
            )

        if event_type == "injured_civilian":
            medical_urgency = float(np.clip(max(template.medical_urgency, 0.8) * self.rng.uniform(0.9, 1.1), 0.7, 2.5))
            severity_score = float(np.clip(max(template.severity_score, 1.4) * self.rng.uniform(0.9, 1.05), 1.2, 3.2))
            return Event(
                id=self.next_event_id,
                location=location,
                start_time=self.timestep,
                duration=max(3, int(template.duration)),
                severity=severity_label_from_score(severity_score),
                type="injured_civilian",
                severity_score=severity_score,
                medical_urgency=medical_urgency,
            )

        if event_type == "flood_zone":
            water_level = float(np.clip(max(template.water_level, 1.2) * self.rng.uniform(0.95, 1.1), 1.0, 4.0))
            spread_pressure = float(np.clip(max(template.spread_pressure, 0.6) * self.rng.uniform(0.9, 1.15), 0.5, 2.0))
            severity_score = float(np.clip(water_level, 1.0, 4.0))
            return Event(
                id=self.next_event_id,
                location=location,
                start_time=self.timestep,
                duration=max(4, int(template.duration)),
                severity=severity_label_from_score(severity_score),
                type="flood_zone",
                severity_score=severity_score,
                water_level=water_level,
                spread_pressure=spread_pressure,
            )

        if event_type == "riot":
            crowd_pressure = float(np.clip(template.crowd_pressure * self.rng.uniform(0.8, 1.2), 0.3, 2.0))
            severity_score = float(np.clip(template.severity_score * self.rng.uniform(0.88, 1.04), 1.6, 4.5))
            return Event(
                id=self.next_event_id,
                location=location,
                start_time=self.timestep,
                duration=max(4, int(template.duration)),
                severity=severity_label_from_score(severity_score),
                type="riot",
                severity_score=severity_score,
                crowd_pressure=crowd_pressure,
            )

        if event_type == "fire":
            fuel = float(np.clip(max(template.fuel, 0.8) * self.rng.uniform(0.8, 1.1), 0.5, 3.0))
            intensity = float(np.clip(max(template.intensity, 0.5) * self.rng.uniform(0.75, 0.98), 0.4, 4.5))
            severity_score = float(intensity)
            return Event(
                id=self.next_event_id,
                location=location,
                start_time=self.timestep,
                duration=max(3, int(template.duration)),
                severity=severity_label_from_score(severity_score),
                type="fire",
                severity_score=severity_score,
                fuel=fuel,
                intensity=intensity,
            )

        if event_type == "structural_collapse":
            structural_instability = float(
                np.clip(max(template.structural_instability, 0.6) * self.rng.uniform(0.85, 1.15), 0.4, 2.4)
            )
            debris_risk = float(np.clip(max(template.debris_risk, 0.5) * self.rng.uniform(0.8, 1.2), 0.3, 2.0))
            severity_score = float(np.clip(template.severity_score * self.rng.uniform(0.75, 0.95), 1.5, 4.5))
            return Event(
                id=self.next_event_id,
                location=location,
                start_time=self.timestep,
                duration=max(4, int(template.duration)),
                severity=severity_label_from_score(severity_score),
                type="structural_collapse",
                severity_score=severity_score,
                structural_instability=structural_instability,
                debris_risk=debris_risk,
            )

        gas_pressure = float(np.clip(max(template.gas_pressure, 1.2) * self.rng.uniform(0.9, 1.2), 0.8, 3.0))
        toxicity = float(np.clip(max(template.toxicity, 1.0) * self.rng.uniform(0.9, 1.15), 0.7, 2.8))
        severity_score = float(
            np.clip(max(template.severity_score, 2.4) * self.rng.uniform(0.85, 1.05), 2.0, 4.5)
        )
        return Event(
            id=self.next_event_id,
            location=location,
            start_time=self.timestep,
            duration=max(4, int(template.duration)),
            severity=severity_label_from_score(severity_score),
            type="gas_leak",
            severity_score=severity_score,
            gas_pressure=gas_pressure,
            toxicity=toxicity,
        )

    def _queue_spread_event(self, *, location: Coord, event_type: str, template: Event, new_events: list[Event]) -> None:
        if len(new_events) >= self.MAX_SPREAD_EVENTS_PER_STEP:
            return
        if self._event_exists_at(location, event_type):
            return
        if any(event.location == location and event.type == event_type for event in new_events):
            return
        new_events.append(self._spawn_dynamic_event(location=location, event_type=event_type, template=template))

    def _update_events(self) -> None:
        new_events: list[Event] = []
        for event in list(self.events):
            if not event.is_active(self.timestep) or event.detected:
                continue
            if event.type == "riot":
                self._update_riot(event, new_events)
            elif event.type == "road_blockage":
                self._update_road_blockage(event)
            elif event.type == "injured_civilian":
                self._update_injured_civilian(event)
            elif event.type == "flood_zone":
                self._update_flood_zone(event, new_events)
            elif event.type == "fire":
                self._update_fire(event, new_events)
            elif event.type == "structural_collapse":
                self._update_structural_collapse(event, new_events)
            elif event.type == "gas_leak":
                self._update_gas_leak(event, new_events)
        for event in new_events:
            self._register_new_event(event)

    def _update_road_blockage(self, event: Event) -> None:
        event.blockage_level *= 1.01
        event.severity_score = float(min(1.6, event.severity_score + 0.03 * max(event.blockage_level, 0.5)))
        event.severity = severity_label_from_score(event.severity_score)
        if not event.detected:
            event.end_time = event.end_time + 1

    def _update_injured_civilian(self, event: Event) -> None:
        event.medical_urgency += 0.08
        event.severity_score = float(min(3.2, event.severity_score + 0.12 * max(event.medical_urgency, 0.8)))
        event.severity = severity_label_from_score(event.severity_score)
        if not event.detected:
            event.end_time = event.end_time + 1

    def _update_flood_zone(self, event: Event, new_events: list[Event]) -> None:
        event.water_level += 0.12 * max(event.spread_pressure, 0.5)
        event.spread_pressure *= 1.02
        event.severity_score = float(min(4.0, event.water_level))
        event.severity = severity_label_from_score(event.severity_score)
        if not event.detected:
            event.end_time = event.end_time + 1

        x, y = event.location
        spread_probability = min(0.12, 0.04 * float(event.spread_pressure))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not self._valid_cell(nx, ny):
                continue
            if self.rng.random() < spread_probability:
                self._queue_spread_event(location=(nx, ny), event_type="flood_zone", template=event, new_events=new_events)

    def _update_riot(self, event: Event, new_events: list[Event]) -> None:
        event.severity_score += 0.14 * float(event.crowd_pressure)
        event.crowd_pressure *= 1.015
        event.severity = severity_label_from_score(event.severity_score)
        if not event.detected:
            event.end_time = event.end_time + 1

        x, y = event.location
        spread_probability = 0.03 if event.severity == "MEDIUM" else 0.07 if event.severity == "HIGH" else 0.0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not self._valid_cell(nx, ny):
                continue
            if self.rng.random() < spread_probability:
                self._queue_spread_event(location=(nx, ny), event_type="riot", template=event, new_events=new_events)

    def _update_fire(self, event: Event, new_events: list[Event]) -> None:
        event.intensity += 0.4 * float(event.fuel)
        event.fuel *= 0.97
        event.severity_score = float(event.intensity)
        event.severity = severity_label_from_score(event.severity_score)

        x, y = event.location
        spread_probability = min(0.35, 0.13 * float(event.intensity))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not self._valid_cell(nx, ny):
                continue
            if self.rng.random() < spread_probability:
                self._queue_spread_event(location=(nx, ny), event_type="fire", template=event, new_events=new_events)

    def _update_structural_collapse(self, event: Event, new_events: list[Event]) -> None:
        event.structural_instability += 0.04 * max(event.debris_risk, 0.5)
        event.debris_risk *= 1.01
        event.severity_score = float(min(2.4, event.severity_score + 0.06 * event.structural_instability))
        event.severity = severity_label_from_score(event.severity_score)
        if not event.detected:
            event.end_time = event.end_time + 1

        if event.severity != "HIGH":
            return

        x, y = event.location
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not self._valid_cell(nx, ny):
                continue
            if self.rng.random() < 0.02:
                self._queue_spread_event(
                    location=(nx, ny),
                    event_type="structural_collapse",
                    template=event,
                    new_events=new_events,
                )

    def _update_gas_leak(self, event: Event, new_events: list[Event]) -> None:
        event.gas_pressure += 0.16
        event.toxicity *= 1.05
        event.severity_score = float(min(4.5, event.gas_pressure + 0.9 * event.toxicity))
        event.severity = severity_label_from_score(event.severity_score)
        if not event.detected:
            event.end_time = event.end_time + 1

        x, y = event.location
        gas_spread_probability = min(0.2, 0.065 * float(event.gas_pressure + event.toxicity))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not self._valid_cell(nx, ny):
                continue
            if self.rng.random() < gas_spread_probability:
                self._queue_spread_event(location=(nx, ny), event_type="gas_leak", template=event, new_events=new_events)
            if event.severity == "HIGH" and self.rng.random() < 0.04:
                self._queue_spread_event(location=(nx, ny), event_type="fire", template=event, new_events=new_events)

    def _resolve_coordinator_targets(
        self,
        action: DroneActions | Mapping[str, Any] | Sequence[Any] | None,
        coordinator_observation: Mapping[str, Any],
    ) -> Dict[str, Coord]:
        if action is None:
            decision = self.coordinator.decide(coordinator_observation) if hasattr(self.coordinator, "decide") else None
            targets = decision.targets if decision is not None else self.coordinator.act(coordinator_observation)
            decision_metadata = decision.metadata if decision is not None else getattr(self.coordinator, "last_metadata", {})
        elif isinstance(action, DroneActions):
            if action.targets is not None:
                targets = normalize_targets(action, self.agent_ids, self.grid_size)
                decision_metadata = {
                    "decision_source": "external_targets",
                    "model_name": getattr(self.coordinator, "model_name", None),
                }
            elif action.actions is not None:
                raise ValueError("Level 6 does not accept direct per-drone movement actions.")
            else:
                decision = self.coordinator.decide(coordinator_observation) if hasattr(self.coordinator, "decide") else None
                targets = decision.targets if decision is not None else self.coordinator.act(coordinator_observation)
                decision_metadata = decision.metadata if decision is not None else getattr(self.coordinator, "last_metadata", {})
        elif isinstance(action, Mapping):
            sample_value = next(iter(action.values()), None)
            if sample_value is None:
                decision = self.coordinator.decide(coordinator_observation) if hasattr(self.coordinator, "decide") else None
                targets = decision.targets if decision is not None else self.coordinator.act(coordinator_observation)
                decision_metadata = decision.metadata if decision is not None else getattr(self.coordinator, "last_metadata", {})
            elif isinstance(sample_value, (tuple, list)):
                targets = normalize_targets(action, self.agent_ids, self.grid_size)
                decision_metadata = {
                    "decision_source": "external_targets",
                    "model_name": getattr(self.coordinator, "model_name", None),
                }
            else:
                raise ValueError("Level 6 expects coordinator targets, not direct movement actions.")
        else:
            targets = normalize_targets(action, self.agent_ids, self.grid_size)
            decision_metadata = {
                "decision_source": "external_targets",
                "model_name": getattr(self.coordinator, "model_name", None),
            }

        if not isinstance(targets, dict):
            targets = dict(targets)

        normalized_targets = normalize_targets(targets, self.agent_ids, self.grid_size)
        self.last_assigned_targets = normalized_targets
        self.metrics["last_assigned_targets"] = dict(normalized_targets)
        self.metrics["target_assignment_count"] += 1
        unique_targets = len(set(normalized_targets.values()))
        self.metrics["coordination_quality"] = unique_targets / float(len(self.agent_ids))
        self.metrics["coordinator_decision_source"] = decision_metadata.get("decision_source")
        self.metrics["coordinator_model_name"] = decision_metadata.get("model_name")
        self.metrics["coordinator_fallback_reason"] = decision_metadata.get("fallback_reason")
        self.metrics["last_llm_latency_ms"] = decision_metadata.get("llm_latency_ms")
        self.metrics["last_llm_raw_response"] = decision_metadata.get("llm_raw_response")
        self.metrics["last_llm_debug"] = decision_metadata.get("llm_debug")
        if decision_metadata.get("decision_source") == "heuristic_fallback":
            self.metrics["coordinator_fallback_count"] += 1
        return normalized_targets

    def _apply_coordinator_targets(self, assigned_targets: Mapping[str, Coord]) -> None:
        for drone_id, target in assigned_targets.items():
            drone = self.drones[drone_id]
            distance_before = manhattan_distance(drone.position, target)
            movement_distance = 0.0 if distance_before == 0 else 1.0
            drone.move_toward_target(target, self.grid_size)
            distance_after = manhattan_distance(drone.position, target)
            progress = max(0, distance_before - distance_after)
            self.metrics["target_progress_sum"] += float(progress)
            self.metrics["movement_distance_sum"] += movement_distance
        movement_total = self.metrics["movement_distance_sum"]
        self.metrics["path_efficiency"] = (
            self.metrics["target_progress_sum"] / movement_total if movement_total else 0.0
        )

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
            active_undetected_events=[event for event in self.events if event.is_active(self.timestep) and not event.detected],
        )

    def _apply_reward_metrics(self, reward_info: Mapping[str, Any]) -> None:
        self.visited_cells.update(reward_info["new_cells"])
        self.metrics["total_overlap_penalty"] += reward_info["overlap_penalty"]
        self.metrics["total_coverage_reward"] += reward_info["coverage_reward"]
        self.metrics["total_detection_reward"] += reward_info["detection_reward"]
        self.metrics["total_miss_penalty"] += reward_info["miss_penalty"]
        self.metrics["total_growth_penalty"] += reward_info.get("growth_penalty", 0.0)
        for key, value in reward_info["reward_breakdown"].items():
            if key == "episode_bonus":
                continue
            self.metrics["reward_breakdown"].setdefault(key, 0.0)
            self.metrics["reward_breakdown"][key] += float(value)
        self._sync_coverage_metrics()
        self._sync_priority_metrics()

    def _sync_coverage_metrics(self) -> None:
        self.metrics["unique_cells_visited"] = len(self.visited_cells)
        self.metrics["grid_coverage"] = compute_grid_coverage(self.visited_cells, self.grid_size)
        self.metrics["grid_coverage_percent"] = 100.0 * self.metrics["grid_coverage"]

    def _sync_priority_metrics(self) -> None:
        pending_by_severity = {severity: 0 for severity in SEVERITY_CONFIG}
        for event in self.events:
            if not event.detected:
                pending_by_severity[event.severity] += 1
        self.metrics["pending_by_severity"] = pending_by_severity

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

    def _build_current_observation(
        self,
        reward: float,
        done: bool,
        coordinator_observation: Optional[Dict[str, Any]] = None,
    ) -> DisasterObservation:
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
            coordinator_observation=coordinator_observation or self.last_coordinator_observation,
            assigned_targets=self.last_assigned_targets,
        )


def _print_severity_metrics(metrics: Mapping[str, Any]) -> None:
    print("Severity metrics:")
    for severity in SEVERITY_CONFIG:
        print(
            "  {severity}: spawned={spawned} detected={detected} on_time={on_time} missed={missed} pending={pending} avg_latency={latency}".format(
                severity=severity,
                spawned=metrics["events_spawned_by_severity"][severity],
                detected=metrics["events_detected_by_severity"][severity],
                on_time=metrics["detected_on_time_by_severity"][severity],
                missed=metrics["missed_by_severity"][severity],
                pending=metrics["pending_by_severity"][severity],
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
    episode_length: Optional[int] = None,
) -> Dict[str, Any]:
    env = DisasterSurveillanceEnvironment(
        seed=seed,
        level=level,
        episode_length=episode_length or DisasterSurveillanceEnvironment.EPISODE_LENGTH,
    )
    observation = env.reset(seed=seed)
    if verbose:
        print("Initial observation:")
        print(observation.model_dump())

    while not observation.done:
        action: DroneActions | Dict[str, Any]
        if level == 6:
            coordinator_observation = env.build_coordinator_observation()
            action = None
            if render:
                print(f"\nCoordinator observation at t={env.timestep}: {coordinator_observation}")
                print(f"Drone positions before move: {coordinator_observation['drone_positions']}")
        else:
            action = DroneActions(actions={agent_id: int(env.rng.integers(0, 5)) for agent_id in env.agent_ids})

        observation = env.step(action)
        if render:
            print(f"t={observation.metadata['processed_timestep']} reward={observation.reward}")
            print(f"Assigned targets: {env.last_assigned_targets}")
            positions_after_move = {drone_id: agent_obs["position"] for drone_id, agent_obs in observation.agents.items()}
            print(f"Drone positions after move: {positions_after_move}")
            print(env.render_ascii())

    public_metrics = {key: value for key, value in env.metrics.items() if not key.startswith("_")}
    if verbose:
        print("\nFinal metrics:")
        for key, value in public_metrics.items():
            print(f"{key}: {value}")
        _print_severity_metrics(public_metrics)
        _print_reward_breakdown(public_metrics)
        if level == 6:
            print("Coordinator summary:")
            print(f"  coordinator_backend: {public_metrics['coordinator_backend']}")
            print(f"  coordinator_model_name: {public_metrics['coordinator_model_name']}")
            print(f"  coordinator_decision_source: {public_metrics['coordinator_decision_source']}")
            print(f"  coordinator_fallback_count: {public_metrics['coordinator_fallback_count']}")
            print(f"  coordinator_fallback_reason: {public_metrics['coordinator_fallback_reason']}")
            print(f"  last_llm_debug: {public_metrics['last_llm_debug']}")
            print(f"  last_assigned_targets: {public_metrics['last_assigned_targets']}")
            print(f"  path_efficiency: {public_metrics['path_efficiency']:.2f}")
            print(f"  coordination_quality: {public_metrics['coordination_quality']:.2f}")
        print(f"\nTotal reward: {public_metrics['total_reward']}")
        print(f"Coverage %: {public_metrics['grid_coverage_percent']:.1f}")
    return public_metrics


def run_random_episodes(
    *,
    episodes: int = 1,
    seed: int = 7,
    level: int = 4,
    render: bool = False,
    episode_length: Optional[int] = None,
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
            episode_length=episode_length,
        )
        all_metrics.append(metrics)
        if episodes > 1:
            print(
                "episode={episode} seed={seed} level={level} total_reward={reward:.1f} "
                "detected={detected} missed={missed} high_miss_rate={high_miss:.2f} "
                "coverage={coverage:.1f}% path_efficiency={path_efficiency:.2f}".format(
                    episode=episode_index + 1,
                    seed=episode_seed,
                    level=level,
                    reward=metrics["total_reward"],
                    detected=metrics["events_detected"],
                    missed=metrics["events_missed"],
                    high_miss=metrics["high_priority_miss_rate"],
                    coverage=metrics["grid_coverage_percent"],
                    path_efficiency=metrics.get("path_efficiency", 0.0),
                )
            )

    if episodes > 1:
        mean_reward = float(np.mean([metrics["total_reward"] for metrics in all_metrics]))
        mean_coverage = float(np.mean([metrics["grid_coverage_percent"] for metrics in all_metrics]))
        mean_high_miss = float(np.mean([metrics["high_priority_miss_rate"] for metrics in all_metrics]))
        mean_path_efficiency = float(np.mean([metrics.get("path_efficiency", 0.0) for metrics in all_metrics]))
        print("\nAggregate metrics:")
        print(f"episodes: {episodes}")
        print(f"level: {level}")
        print(f"mean_total_reward: {mean_reward:.2f}")
        print(f"mean_coverage_percent: {mean_coverage:.2f}")
        print(f"mean_high_priority_miss_rate: {mean_high_miss:.2f}")
        print(f"mean_path_efficiency: {mean_path_efficiency:.2f}")

    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random rollouts for the disaster surveillance environment.")
    parser.add_argument(
        "--level",
        type=int,
        choices=[3, 4, 5, 6],
        default=4,
        help="Environment level: 3 baseline, 4 shaped coordination, 5 urgency, or 6 coordinator control.",
    )
    parser.add_argument("--episodes", "-k", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--episode-length", type=int, default=None, help="Override episode length for debugging.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed. Episode i uses seed + i.")
    parser.add_argument("--render", action="store_true", help="Render ASCII grid per step. Only enabled for a single episode.")
    args = parser.parse_args()
    run_random_episodes(
        episodes=args.episodes,
        seed=args.seed,
        level=args.level,
        render=args.render,
        episode_length=args.episode_length,
    )


if __name__ == "__main__":
    main()
