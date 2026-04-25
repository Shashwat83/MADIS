from __future__ import annotations

import numpy as np

from disaster_surveillance_env.models import (
    Event,
    compute_episode_bonus,
    compute_event_reward,
    compute_miss_penalty,
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
