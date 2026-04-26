from __future__ import annotations

import json

from disaster_surveillance_env.grpo.rewards import json_valid_reward, parse_or_fallback_completion
from disaster_surveillance_env.grpo.state import restore_environment, snapshot_environment
from disaster_surveillance_env.server.disaster_surveillance_environment import DisasterSurveillanceEnvironment


def test_snapshot_and_restore_preserve_core_env_fields() -> None:
    env = DisasterSurveillanceEnvironment(level=6, seed=11, p_spawn=0.0)
    observation = env.reset(seed=11)
    env.step({"drone_1": (1, 1), "drone_2": (8, 1), "drone_3": (5, 8)})

    snapshot = snapshot_environment(env)
    restored = restore_environment(snapshot)

    assert restored.timestep == env.timestep
    assert restored.next_event_id == env.next_event_id
    assert restored.visited_cells == env.visited_cells
    assert restored.last_assigned_targets == env.last_assigned_targets
    assert restored.metrics["level"] == env.metrics["level"]


def test_parse_or_fallback_completion_falls_back_to_stay_targets_on_invalid_json() -> None:
    env = DisasterSurveillanceEnvironment(level=6, seed=12, p_spawn=0.0)
    env.reset(seed=12)
    snapshot_json = json.dumps(snapshot_environment(env))

    parsed = parse_or_fallback_completion("not json", snapshot_json)

    assert parsed.parse_success is False
    assert set(parsed.targets) == {"drone_1", "drone_2", "drone_3"}


def test_json_valid_reward_marks_valid_and_invalid_completions() -> None:
    env = DisasterSurveillanceEnvironment(level=6, seed=13, p_spawn=0.0)
    env.reset(seed=13)
    snapshot_json = json.dumps(snapshot_environment(env))

    rewards = json_valid_reward(
        ['{"drone_1":[1,1],"drone_2":[8,1],"drone_3":[5,8]}', "oops"],
        [snapshot_json, snapshot_json],
    )

    assert rewards == [1.0, -1.0]
