from __future__ import annotations

import json

from disaster_surveillance_env.sft.dataset import rollout_sft_examples
from disaster_surveillance_env.sft.parsing import ACTION_OUTPUT_MODE, TARGET_OUTPUT_MODE
from disaster_surveillance_env.sft.policies import HeuristicTeacherPolicy, OracleTeacherPolicy


def test_rollout_sft_examples_generates_heuristic_target_records() -> None:
    examples = rollout_sft_examples(
        episodes=1,
        teacher_policy=HeuristicTeacherPolicy(),
        dataset_type="heuristic",
        output_mode=TARGET_OUTPUT_MODE,
        seed=5,
        episode_length=3,
        p_spawn=0.0,
    )

    assert len(examples) == 3
    payload = json.loads(examples[0].response)
    assert set(payload) == {"drone_1", "drone_2", "drone_3"}
    assert examples[0].metadata["visibility_mode"] == "partial"


def test_rollout_sft_examples_marks_oracle_records_and_action_output() -> None:
    examples = rollout_sft_examples(
        episodes=1,
        teacher_policy=OracleTeacherPolicy(),
        dataset_type="oracle",
        output_mode=ACTION_OUTPUT_MODE,
        seed=7,
        episode_length=2,
        p_spawn=0.0,
    )

    assert len(examples) == 2
    payload = json.loads(examples[0].response)
    assert set(payload) == {"drone_1", "drone_2", "drone_3"}
    assert examples[0].metadata["visibility_mode"] == "oracle"
