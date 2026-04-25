from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence


DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(slots=True)
class RunPaths:
    root: Path
    adapter_dir: Path
    tokenizer_dir: Path
    metadata_dir: Path
    checkpoints_dir: Path


def iter_jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL record: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object per line.")
            yield record


def validate_sft_record(record: Mapping[str, Any]) -> None:
    if "prompt" not in record or "response" not in record:
        raise ValueError("Each SFT record must contain 'prompt' and 'response'.")
    if not isinstance(record["prompt"], str) or not isinstance(record["response"], str):
        raise ValueError("'prompt' and 'response' must both be strings.")
    metadata = record.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError("'metadata' must be a mapping when present.")


def count_valid_records(path: Path) -> int:
    count = 0
    for record in iter_jsonl_records(path):
        validate_sft_record(record)
        count += 1
    return count


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_run_paths(output_dir: Path) -> RunPaths:
    paths = RunPaths(
        root=output_dir,
        adapter_dir=output_dir / "adapter",
        tokenizer_dir=output_dir / "tokenizer",
        metadata_dir=output_dir / "metadata",
        checkpoints_dir=output_dir / "checkpoints",
    )
    for path in (paths.root, paths.adapter_dir, paths.tokenizer_dir, paths.metadata_dir, paths.checkpoints_dir):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def build_dataset_manifest(
    *,
    train_jsonl: Path,
    eval_jsonl: Optional[Path],
    train_records: int,
    eval_records: int,
) -> Dict[str, Any]:
    manifest = {
        "train_jsonl": str(train_jsonl.resolve()),
        "train_records": train_records,
        "train_sha256": file_sha256(train_jsonl),
    }
    if eval_jsonl is not None:
        manifest.update(
            {
                "eval_jsonl": str(eval_jsonl.resolve()),
                "eval_records": eval_records,
                "eval_sha256": file_sha256(eval_jsonl),
            }
        )
    return manifest


def build_run_manifest(
    *,
    model_name: str,
    output_mode: str,
    max_seq_length: int,
    train_jsonl: Path,
    eval_jsonl: Optional[Path],
    train_records: int,
    eval_records: int,
    training_args: Mapping[str, Any],
    adapter_format: str = "peft_lora",
) -> Dict[str, Any]:
    return {
        "artifact_type": adapter_format,
        "base_model_name": model_name,
        "output_mode": output_mode,
        "max_seq_length": max_seq_length,
        "dataset_manifest": build_dataset_manifest(
            train_jsonl=train_jsonl,
            eval_jsonl=eval_jsonl,
            train_records=train_records,
            eval_records=eval_records,
        ),
        "training_args": dict(training_args),
        "grpo_reuse": {
            "recommended_start_model_type": "peft_adapter",
            "load_with_base_model": model_name,
            "adapter_subdir": "adapter",
            "tokenizer_subdir": "tokenizer",
            "prompt_parser_contract": output_mode,
        },
    }


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def truncate_prompt_completion(
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    *,
    max_seq_length: int,
    eos_token_id: Optional[int],
) -> Dict[str, list[int]]:
    response_tail = list(response_ids)
    if eos_token_id is not None and (not response_tail or response_tail[-1] != eos_token_id):
        response_tail = response_tail + [int(eos_token_id)]

    if len(response_tail) >= max_seq_length:
        trimmed_response = response_tail[-max_seq_length:]
        return {
            "input_ids": trimmed_response,
            "labels": trimmed_response.copy(),
            "attention_mask": [1] * len(trimmed_response),
            "prompt_tokens": 0,
            "response_tokens": len(trimmed_response),
        }

    available_prompt_tokens = max_seq_length - len(response_tail)
    trimmed_prompt = list(prompt_ids)[-available_prompt_tokens:]
    input_ids = trimmed_prompt + response_tail
    labels = ([-100] * len(trimmed_prompt)) + response_tail.copy()
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "prompt_tokens": len(trimmed_prompt),
        "response_tokens": len(response_tail),
    }

