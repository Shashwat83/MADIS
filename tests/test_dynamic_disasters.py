from __future__ import annotations

from typing import Any, Optional

import numpy as np

from disaster_surveillance_env.models import Event, compute_level5_reward
from disaster_surveillance_env.server.disaster_surveillance_environment import DisasterSurveillanceEnvironment


class _FixedRng:
    """Deterministic RNG helper for forcing spread branches in tests."""

    def __init__(self, *, random_value: float = 0.0, uniform_value: float = 1.0) -> None:
        self._random_value = float(random_value)
        self._uniform_value = float(uniform_value)

    def random(self) -> float:
        return self._random_value

    def uniform(self, low: float, high: float) -> float:
        del low, high
        return self._uniform_value

    def integers(self, low: int, high: Optional[int] = None) -> int:
        if high is None:
            high = low
            low = 0
        return int(low)

    def choice(self, values: Any, p: Optional[Any] = None) -> Any:
        del p
        if isinstance(values, int):
            return 0
        return values[0]


def _seeded_env(*, seed: int, level: int = 6) -> DisasterSurveillanceEnvironment:
    env = DisasterSurveillanceEnvironment(level=level, seed=seed, p_spawn=0.0)
    env.reset(seed=seed)
    return env


def test_riot_escalates_and_can_spread_with_forced_rng() -> None:
    env = _seeded_env(seed=101, level=6)
    env.rng = _FixedRng(random_value=0.0, uniform_value=1.0)  # force spread

    riot = Event(
        id=env.next_event_id,
        location=(5, 5),
        start_time=env.timestep,
        duration=8,
        severity="HIGH",
        type="riot",
        severity_score=3.2,
        crowd_pressure=1.5,
    )
    env._register_new_event(riot)
    before = riot.severity_score

    env._update_events()

    assert riot.severity_score > before
    assert any(event.type == "riot" for event in env.events if event.id != riot.id)


def test_fire_intensity_increases_and_can_spread_with_forced_rng() -> None:
    env = _seeded_env(seed=102, level=6)
    env.rng = _FixedRng(random_value=0.0, uniform_value=0.9)  # force spread + stable params

    fire = Event(
        id=env.next_event_id,
        location=(4, 4),
        start_time=env.timestep,
        duration=6,
        severity="MEDIUM",
        type="fire",
        severity_score=2.2,
        fuel=1.4,
        intensity=0.8,
    )
    env._register_new_event(fire)
    before_intensity = fire.intensity

    env._update_events()

    assert fire.intensity > before_intensity
    assert any(event.type == "fire" for event in env.events if event.id != fire.id)


def test_gas_leak_high_severity_can_spawn_fire_neighbors_with_forced_rng() -> None:
    env = _seeded_env(seed=103, level=6)
    env.rng = _FixedRng(random_value=0.0, uniform_value=1.0)  # force both spread and fire conversion

    leak = Event(
        id=env.next_event_id,
        location=(3, 3),
        start_time=env.timestep,
        duration=10,
        severity="HIGH",
        type="gas_leak",
        severity_score=3.4,
        gas_pressure=3.0,
        toxicity=2.0,
    )
    env._register_new_event(leak)

    env._update_events()

    assert any(event.type == "fire" for event in env.events if event.id != leak.id)


def test_flood_spread_is_bounded_per_step_and_water_level_increases() -> None:
    env = _seeded_env(seed=104, level=6)
    env.rng = _FixedRng(random_value=0.0, uniform_value=1.0)  # force spread

    flood = Event(
        id=env.next_event_id,
        location=(6, 6),
        start_time=env.timestep,
        duration=12,
        severity="MEDIUM",
        type="flood_zone",
        severity_score=2.2,
        water_level=1.6,
        spread_pressure=2.0,
    )
    env._register_new_event(flood)
    before_water = flood.water_level
    before_count = len(env.events)

    env._update_events()

    assert flood.water_level > before_water
    assert len(env.events) - before_count <= env.MAX_SPREAD_EVENTS_PER_STEP
    assert any(event.type == "flood_zone" for event in env.events if event.id != flood.id)


def test_reward_growth_penalty_scales_with_undetected_event_severity_score() -> None:
    active_undetected = [
        Event(id=1, location=(0, 0), start_time=0, duration=10, severity="LOW", severity_score=1.5),
        Event(id=2, location=(1, 1), start_time=0, duration=10, severity="HIGH", severity_score=3.0),
    ]
    reward, info = compute_level5_reward(
        detected_events=[],
        missed_events=[],
        fovs={},
        visited_cells=set(),
        reward_per_new_cell=0.0,
        active_undetected_events=active_undetected,
    )
    del reward
    expected = -0.1 * (1.5 + 3.0)
    assert float(info["growth_penalty"]) == expected


def test_level9_scripted_adversary_adds_reported_event_with_forced_rng() -> None:
    env = _seeded_env(seed=105, level=9)
    env.rng = _FixedRng(random_value=0.0, uniform_value=0.5)

    env._update_adversarial_reports()

    assert len(env.reported_events) == 1
    report = env.reported_events[0]
    assert report["source"] == "scripted_adversary"
    assert report["type"] in {"riot", "fire", "gas_leak", "flood_zone"}
    assert env.metrics["false_reports_issued"] == 1
    assert env.metrics["false_reports_active"] == 1


def test_level9_false_report_penalty_uses_fov_radius_match() -> None:
    env = _seeded_env(seed=106, level=9)
    env.reported_events = [
        {
            "id": "report_1",
            "location": (4, 4),
            "severity": "HIGH",
            "type": "gas_leak",
            "type_priority": 7,
            "severity_score": 3.2,
            "credibility": 0.7,
            "reported_at": 0,
            "expires_at": 5,
            "source": "scripted_adversary",
        }
    ]
    env.last_assigned_targets = {"drone_1": (6, 4), "drone_2": (1, 1), "drone_3": (2, 2)}

    penalty, info = env._compute_false_report_penalty()

    assert penalty < 0.0
    assert info["false_report_targets"] == 1
    assert info["matched_false_reports"][0]["location"] == (6, 4)


def test_level9_false_report_expiry_removes_stale_reports() -> None:
    env = _seeded_env(seed=107, level=9)
    env.reported_events = [
        {
            "id": "report_1",
            "location": (3, 3),
            "severity": "MEDIUM",
            "type": "fire",
            "type_priority": 5,
            "severity_score": 2.4,
            "credibility": 0.5,
            "reported_at": 0,
            "expires_at": 1,
            "source": "scripted_adversary",
        }
    ]
    env.timestep = 1

    env._update_adversarial_reports()

    assert env.reported_events == []
    assert env.metrics["false_reports_active"] == 0


def test_level9_false_report_rejection_rate_tracks_uninvestigated_reports() -> None:
    env = _seeded_env(seed=108, level=9)
    env.metrics["false_reports_issued"] = 2
    env.metrics["_investigated_false_report_ids"] = {"report_1"}

    env._sync_false_report_metrics()

    assert env.metrics["false_report_investigations"] == 1
    assert env.metrics["false_report_rejection_rate"] == 0.5
