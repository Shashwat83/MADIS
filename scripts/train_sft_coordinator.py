from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disaster_surveillance_env.coordinator import get_configured_model_name
from disaster_surveillance_env.sft.training import (
    DEFAULT_LORA_TARGET_MODULES,
    build_run_manifest,
    count_valid_records,
    ensure_run_paths,
    save_json,
    truncate_prompt_completion,
)

def _import_training_stack() -> dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional local deps
        raise RuntimeError(
            "Local SFT training requires optional dependencies. Install them with "
            "`pip install -e '.[sft]'` before running training."
        ) from exc

    return {
        "torch": torch,
        "load_dataset": load_dataset,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def _select_torch_dtype(torch_module: Any, *, bf16: bool, fp16: bool) -> Any:
    if bf16:
        return torch_module.bfloat16
    if fp16:
        return torch_module.float16
    return torch_module.float32


def _build_tokenize_function(tokenizer: Any, *, max_seq_length: int):
    eos_token_id = tokenizer.eos_token_id

    def tokenize_record(example: dict[str, Any]) -> dict[str, Any]:
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(example["response"], add_special_tokens=False)["input_ids"]
        packed = truncate_prompt_completion(
            prompt_ids,
            response_ids,
            max_seq_length=max_seq_length,
            eos_token_id=eos_token_id,
        )
        packed["sequence_length"] = len(packed["input_ids"])
        return packed

    return tokenize_record


def _save_run_artifacts(
    *,
    paths: Any,
    tokenizer: Any,
    trainer: Any,
    manifest: dict[str, Any],
) -> None:
    trainer.save_model(str(paths.adapter_dir))
    tokenizer.save_pretrained(str(paths.tokenizer_dir))
    save_json(paths.metadata_dir / "run_manifest.json", manifest)
    save_json(
        paths.metadata_dir / "grpo_reuse.json",
        {
            **manifest["grpo_reuse"],
            "adapter_path": str(paths.adapter_dir.resolve()),
            "tokenizer_path": str(paths.tokenizer_dir.resolve()),
        },
    )
    save_json(paths.metadata_dir / "trainer_state_summary.json", trainer.state.log_history[-20:] if trainer.state.log_history else {"log_history": []})


def run_training(args: argparse.Namespace) -> None:
    train_records = count_valid_records(args.train_jsonl)
    eval_records = count_valid_records(args.eval_jsonl) if args.eval_jsonl is not None else 0

    print(f"Validated train records: {train_records}")
    if args.eval_jsonl is not None:
        print(f"Validated eval records: {eval_records}")

    run_paths = ensure_run_paths(args.output_dir)

    training_args_manifest = {
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "max_steps": args.max_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "use_4bit": args.use_4bit,
        "gradient_checkpointing": args.gradient_checkpointing,
        "seed": args.seed,
    }

    manifest = build_run_manifest(
        model_name=args.model_name,
        output_mode=args.output_mode,
        max_seq_length=args.max_seq_length,
        train_jsonl=args.train_jsonl,
        eval_jsonl=args.eval_jsonl,
        train_records=train_records,
        eval_records=eval_records,
        training_args=training_args_manifest,
    )
    save_json(run_paths.metadata_dir / "run_manifest.json", manifest)

    if args.dry_run:
        print(f"Dry run complete for model {args.model_name}")
        return

    stack = _import_training_stack()
    torch = stack["torch"]
    load_dataset = stack["load_dataset"]
    LoraConfig = stack["LoraConfig"]
    get_peft_model = stack["get_peft_model"]
    prepare_model_for_kbit_training = stack["prepare_model_for_kbit_training"]
    AutoModelForCausalLM = stack["AutoModelForCausalLM"]
    AutoTokenizer = stack["AutoTokenizer"]
    BitsAndBytesConfig = stack["BitsAndBytesConfig"]
    DataCollatorForSeq2Seq = stack["DataCollatorForSeq2Seq"]
    Trainer = stack["Trainer"]
    TrainingArguments = stack["TrainingArguments"]

    train_dataset = load_dataset("json", data_files=str(args.train_jsonl), split="train")
    eval_dataset = (
        load_dataset("json", data_files=str(args.eval_jsonl), split="train")
        if args.eval_jsonl is not None
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    torch_dtype = _select_torch_dtype(torch, bf16=args.bf16, fp16=args.fp16)
    quantization_config: Optional[Any] = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_target_modules = [module.strip() for module in args.lora_target_modules.split(",") if module.strip()]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules,
    )
    model = get_peft_model(model, peft_config)

    tokenize_record = _build_tokenize_function(tokenizer, max_seq_length=args.max_seq_length)
    remove_columns = train_dataset.column_names
    train_dataset = train_dataset.map(tokenize_record, remove_columns=remove_columns)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(tokenize_record, remove_columns=eval_dataset.column_names)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    training_arguments = TrainingArguments(
        output_dir=str(run_paths.checkpoints_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        load_best_model_at_end=bool(eval_dataset is not None and args.load_best_model_at_end),
        metric_for_best_model="eval_loss" if eval_dataset is not None and args.load_best_model_at_end else None,
        greater_is_better=False if eval_dataset is not None and args.load_best_model_at_end else None,
        group_by_length=True,
        length_column_name="sequence_length",
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    _save_run_artifacts(paths=run_paths, tokenizer=tokenizer, trainer=trainer, manifest=manifest)
    print(f"Saved adapter to {run_paths.adapter_dir}")
    print(f"Saved tokenizer to {run_paths.tokenizer_dir}")
    print(f"Saved run metadata to {run_paths.metadata_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local LoRA-based SFT coordinator model.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default=get_configured_model_name())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-mode", choices=["targets", "actions"], default="targets")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default=",".join(DEFAULT_LORA_TARGET_MODULES),
        help="Comma-separated LoRA target modules.",
    )
    parser.add_argument("--use-4bit", action="store_true", help="Enable QLoRA-style 4-bit loading when supported.")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--load-best-model-at-end", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset shape without launching training.")
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
