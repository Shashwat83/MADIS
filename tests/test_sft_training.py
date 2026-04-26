from __future__ import annotations

import json
from types import SimpleNamespace

from disaster_surveillance_env.sft.training import (
    build_sft_results,
    build_run_manifest,
    ensure_run_paths,
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


def test_build_sft_results_emits_grpo_ready_metadata(tmp_path) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    eval_jsonl = tmp_path / "eval.jsonl"
    train_jsonl.write_text(
        json.dumps(
            {
                "prompt": "p",
                "response": "r",
                "metadata": {"dataset_type": "heuristic", "visibility_mode": "partial"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    eval_jsonl.write_text(json.dumps({"prompt": "p2", "response": "r2"}) + "\n", encoding="utf-8")

    manifest = build_run_manifest(
        model_name="Qwen/Qwen3-1.7B",
        output_mode="targets",
        max_seq_length=1024,
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        train_records=1,
        eval_records=1,
        training_args={"per_device_train_batch_size": 1, "learning_rate": 2e-4},
    )
    run_paths = ensure_run_paths(tmp_path / "run")
    trainer_state = SimpleNamespace(
        global_step=12,
        train_loss=0.25,
        best_metric=0.2,
        best_model_checkpoint="checkpoint-10",
        train_runtime=12.3,
        train_samples_per_second=1.5,
        train_steps_per_second=0.5,
        total_flos=123.0,
        log_history=[
            {"step": 5, "epoch": 0.5, "loss": 1.0, "grad_norm": 0.1, "learning_rate": 1e-4},
            {"step": 10, "epoch": 1.0, "eval_loss": 0.2},
        ],
    )

    checkpoint_dir = run_paths.checkpoints_dir / "checkpoint-12"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results = build_sft_results(
        run_name="demo-run",
        run_paths=run_paths,
        manifest=manifest,
        trainer_state=trainer_state,
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
    )

    assert results["run_name"] == "demo-run"
    assert results["dataset_summary"]["dataset_type"] == "heuristic"
    assert results["dataset_summary"]["visibility_mode"] == "partial"
    assert results["training_results"]["eval_ran_during_training"] is True
    assert results["artifacts"]["checkpoint_path"].endswith("checkpoint-12")
