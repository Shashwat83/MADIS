from __future__ import annotations

import numpy as np

from disaster_surveillance_env.coordinator import HeuristicCoordinator, LLMCoordinator
from disaster_surveillance_env.models import (
    DroneActions,
    Event,
    compute_episode_bonus,
    compute_event_reward,
    compute_miss_penalty,
    move_toward_target,
    sample_event_location_with_hotspots,
)
from disaster_surveillance_env.server.disaster_surveillance_environment import (
    DisasterSurveillanceEnvironment,
)


def test_high_severity_reward_dominates_low_severity_reward() -> None:
    low_event = Event(id=1, location=(0, 0), start_time=0, duration=8, severity="LOW")
    high_event = Event(id=2, location=(0, 0), start_time=0, duration=8, severity="HIGH")

    low_reward = compute_event_reward(low_event, detection_time=1)
    high_reward = compute_event_reward(high_event, detection_time=1)

    assert high_reward > low_reward
    assert high_reward >= 40.0


def test_miss_penalty_is_severity_weighted() -> None:
    low_penalty = compute_miss_penalty(Event(id=1, location=(0, 0), start_time=0, duration=5, severity="LOW"))
    medium_penalty = compute_miss_penalty(Event(id=2, location=(0, 0), start_time=0, duration=5, severity="MEDIUM"))
    high_penalty = compute_miss_penalty(Event(id=3, location=(0, 0), start_time=0, duration=5, severity="HIGH"))

    assert low_penalty == -10.0
    assert medium_penalty == -25.0
    assert high_penalty == -60.0


def test_hotspot_sampling_biases_toward_hotspots() -> None:
    rng = np.random.default_rng(123)
    hotspot_hits = 0
    total_samples = 500
    hotspot_cells = {
        (x, y)
        for cx, cy in [(2, 2), (7, 7)]
        for x in range(10)
        for y in range(10)
        if (x - cx) ** 2 + (y - cy) ** 2 <= 4
    }

    for _ in range(total_samples):
        location = sample_event_location_with_hotspots(rng, 10)
        if location in hotspot_cells:
            hotspot_hits += 1

    assert hotspot_hits / total_samples > 0.5


def test_level5_observation_includes_urgency_fields() -> None:
    env = DisasterSurveillanceEnvironment(level=5, seed=0, p_spawn=0.0)
    obs = env.reset(seed=0)

    event = Event(id=10, location=env.drones["drone_1"].position, start_time=0, duration=6, severity="HIGH")
    env.events = [event]

    obs = env._build_current_observation(reward=0.0, done=False)
    visible_event = obs.agents["drone_1"]["visible_events"][0]

    assert visible_event["severity"] == "HIGH"
    assert "time_remaining" in visible_event
    assert "deadline_remaining" in visible_event


def test_episode_bonus_rewards_on_time_high_priority_behavior() -> None:
    bonus, breakdown = compute_episode_bonus(
        {
            "events_spawned_by_severity": {"LOW": 0, "MEDIUM": 0, "HIGH": 2},
            "detected_on_time_by_severity": {"LOW": 0, "MEDIUM": 0, "HIGH": 2},
            "missed_by_severity": {"LOW": 0, "MEDIUM": 0, "HIGH": 0},
            "mean_detection_latency": 1.5,
        },
        latency_threshold=3.0,
    )

    assert bonus == 70.0
    assert breakdown["all_high_detected_on_time_bonus"] == 50.0
    assert breakdown["low_latency_team_bonus"] == 20.0


def test_level5_step_applies_episode_bonus_and_metrics() -> None:
    env = DisasterSurveillanceEnvironment(level=5, seed=1, episode_length=1, p_spawn=0.0)
    env.reset(seed=1)
    env.metrics["events_spawned_by_severity"]["HIGH"] = 1
    env.metrics["detected_on_time_by_severity"]["HIGH"] = 1
    env.metrics["detection_latencies"] = [1]
    env.metrics["mean_detection_latency"] = 1.0

    obs = env.step({agent_id: 4 for agent_id in env.agent_ids})

    assert obs.done is True
    assert obs.metadata["episode_bonus"] == 70.0
    assert env.metrics["total_episode_bonus"] == 70.0
    assert env.metrics["reward_breakdown"]["episode_bonus"] == 70.0


def test_move_toward_target_moves_one_step() -> None:
    next_position, action = move_toward_target((2, 2), (5, 2))
    assert next_position == (3, 2)
    assert action == 3


def test_level6_heuristic_coordinator_assigns_targets() -> None:
    coordinator = HeuristicCoordinator()
    targets = coordinator.act(
        {
            "grid_size": 10,
            "drone_positions": {"drone_1": (0, 0), "drone_2": (9, 0), "drone_3": (5, 9)},
            "visible_active_events": [],
            "known_detected_events": [],
            "team_frontier_cells": [(1, 1), (8, 1), (5, 8)],
            "recently_observed_cells": [],
        }
    )
    assert set(targets) == {"drone_1", "drone_2", "drone_3"}
    assert len(set(targets.values())) == 3


def test_level6_step_moves_drones_toward_targets_and_logs_assignment() -> None:
    env = DisasterSurveillanceEnvironment(level=6, seed=2, p_spawn=0.0)
    env.reset(seed=2)
    start_positions = {drone_id: drone.position for drone_id, drone in env.drones.items()}
    targets = {drone_id: (9, 9) for drone_id in env.agent_ids}

    obs = env.step(DroneActions(targets=targets))

    for drone_id, start in start_positions.items():
        end = obs.agents[drone_id]["position"]
        assert abs(end[0] - start[0]) + abs(end[1] - start[1]) <= 1
    assert obs.metadata["assigned_targets"] == targets
    assert env.metrics["target_assignment_count"] == 1


def test_level6_rejects_direct_drone_actions() -> None:
    env = DisasterSurveillanceEnvironment(level=6, seed=3, p_spawn=0.0)
    env.reset(seed=3)

    try:
        env.step(DroneActions(actions={agent_id: 4 for agent_id in env.agent_ids}))
    except ValueError as exc:
        assert "does not accept direct per-drone movement actions" in str(exc)
    else:
        raise AssertionError("Level 6 should reject direct movement actions.")


def test_level9_step_supports_coordinator_reports_and_openenv_flow() -> None:
    env = DisasterSurveillanceEnvironment(level=9, seed=4, p_spawn=0.0, coordinator=HeuristicCoordinator())
    env.reset(seed=4)
    env.reported_events = [
        {
            "id": "report_1",
            "location": (1, 1),
            "severity": "HIGH",
            "type": "riot",
            "type_priority": 6,
            "severity_score": 2.8,
            "credibility": 0.55,
            "reported_at": 0,
            "expires_at": 6,
            "source": "scripted_adversary",
        }
    ]

    observation = env.step(None)

    assert observation.coordinator is not None
    assert "reported_events" in observation.coordinator
    assert observation.metadata["coordination_mode"] == "centralized_coordinator"


def test_llm_coordinator_parses_json_targets() -> None:
    class FakeBackend:
        def generate(self, prompt: str) -> str:
            assert "Return JSON only" in prompt
            return '{"drone_1": [1, 2], "drone_2": [3, 4], "drone_3": [5, 6]}'

    coordinator = LLMCoordinator(backend=FakeBackend(), model_name="fake-small")
    decision = coordinator.decide(
        {
            "timestep": 0,
            "grid_size": 10,
            "drone_positions": {"drone_1": (0, 0), "drone_2": (1, 1), "drone_3": (2, 2)},
            "visible_active_events": [],
            "known_detected_events": [],
            "team_frontier_cells": [],
            "recently_observed_cells": [],
            "recent_team_coverage_ratio": 0.1,
        }
    )

    assert decision.targets == {"drone_1": (1, 2), "drone_2": (3, 4), "drone_3": (5, 6)}
    assert decision.metadata["decision_source"] == "llm"
    assert decision.metadata["model_name"] == "fake-small"


def test_llm_coordinator_falls_back_to_heuristic_on_backend_failure() -> None:
    class FailingBackend:
        def generate(self, prompt: str) -> str:
            raise RuntimeError("backend unavailable")

    coordinator = LLMCoordinator(backend=FailingBackend(), model_name="fake-small")
    decision = coordinator.decide(
        {
            "timestep": 0,
            "grid_size": 10,
            "drone_positions": {"drone_1": (0, 0), "drone_2": (9, 0), "drone_3": (5, 9)},
            "visible_active_events": [],
            "known_detected_events": [],
            "team_frontier_cells": [(1, 1), (8, 1), (5, 8)],
            "recently_observed_cells": [],
            "recent_team_coverage_ratio": 0.0,
        }
    )

    assert set(decision.targets) == {"drone_1", "drone_2", "drone_3"}
    assert decision.metadata["decision_source"] == "heuristic_fallback"
    assert "backend unavailable" in decision.metadata["fallback_reason"]
