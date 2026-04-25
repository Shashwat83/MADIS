from __future__ import annotations

import json

from disaster_surveillance_env.sft.training import (
    build_run_manifest,
    truncate_prompt_completion,
    validate_sft_record,
)


def test_validate_sft_record_accepts_minimal_valid_record() -> None:
    validate_sft_record({"prompt": "state", "response": '{"drone_1":[1,1]}'})


def test_truncate_prompt_completion_masks_prompt_and_preserves_response() -> None:
    packed = truncate_prompt_completion(
        [1, 2, 3, 4],
        [5, 6],
        max_seq_length=5,
        eos_token_id=99,
    )

    assert packed["input_ids"] == [3, 4, 5, 6, 99]
    assert packed["labels"] == [-100, -100, 5, 6, 99]
    assert packed["prompt_tokens"] == 2
    assert packed["response_tokens"] == 3


def test_build_run_manifest_contains_grpo_reuse_contract(tmp_path) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text(json.dumps({"prompt": "p", "response": "r"}) + "\n", encoding="utf-8")

    manifest = build_run_manifest(
        model_name="Qwen/Qwen3-1.7B",
        output_mode="targets",
        max_seq_length=1024,
        train_jsonl=train_jsonl,
        eval_jsonl=None,
        train_records=1,
        eval_records=0,
        training_args={"learning_rate": 2e-4},
    )

    assert manifest["artifact_type"] == "peft_lora"
    assert manifest["grpo_reuse"]["recommended_start_model_type"] == "peft_adapter"
    assert manifest["dataset_manifest"]["train_records"] == 1
