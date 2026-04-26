from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disaster_surveillance_env.grpo.dataset import export_grpo_prompt_jsonl, iter_grpo_prompt_examples
from disaster_surveillance_env.grpo.eval import run_adapter_eval, run_backend_eval
from disaster_surveillance_env.grpo.local_peft_backend import LoadedPeftCoordinatorBackend
from disaster_surveillance_env.grpo.rewards import environment_step_reward, json_valid_reward, unique_target_reward
from disaster_surveillance_env.hf_hub_auth import from_pretrained_token_kwargs
from disaster_surveillance_env.sft.training import save_json
from scripts.analyze_baseline import save_line_plot


SERIOUS_RECOMMENDED_CONFIG = {
    "training_episodes": 1200,
    "rollouts_per_update": 64,
    "grpo_group_size": 4,
    "prompts_per_update": 16,
    "episode_length": 50,
    "log_every_episodes": 5,
    "small_eval_every_episodes": 25,
    "small_eval_episodes": 25,
    "medium_eval_every_episodes": 100,
    "medium_eval_episodes": 100,
    "full_eval_every_episodes": 200,
    "full_eval_episodes": 400,
}

PILOT_DEFAULT_CONFIG = {
    "training_episodes": 400,
    "rollouts_per_update": 4,
    "grpo_group_size": 4,
    "prompts_per_update": 1,
    "episode_length": 50,
    "log_every_episodes": 5,
    "small_eval_every_episodes": 25,
    "small_eval_episodes": 25,
    "medium_eval_every_episodes": 100,
    "medium_eval_episodes": 100,
    "full_eval_every_episodes": 200,
    "full_eval_episodes": 400,
}


def _import_grpo_stack() -> Dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "GRPO training requires optional dependencies. Install them with "
            "`pip install -e '.[grpo]'` in Colab or locally."
        ) from exc
    return {
        "torch": torch,
        "load_dataset": load_dataset,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "TrainerCallback": TrainerCallback,
        "GRPOConfig": GRPOConfig,
        "GRPOTrainer": GRPOTrainer,
    }


def _ensure_prompt_dataset(path: Path, *, episodes: int, seed: int, episode_length: int) -> Path:
    if path.exists():
        return path
    written = export_grpo_prompt_jsonl(
        path,
        iter_grpo_prompt_examples(
            episodes=episodes,
            seed=seed,
            episode_length=episode_length,
        ),
    )
    print(f"Generated {written} prompt-only GRPO examples at {path}")
    return path


def _append_csv_row(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name) for name in fieldnames})


def _filter_supported_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs supported by the installed runtime signature."""
    supported = inspect.signature(callable_obj).parameters
    return {key: value for key, value in kwargs.items() if key in supported}


def _save_training_plots(csv_path: Path, output_dir: Path) -> None:
    if not csv_path.exists():
        return
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    if not rows:
        return

    steps = [int(float(row["step"])) for row in rows]
    reward_values = [float(row["reward"]) for row in rows if row.get("reward")]
    reward_steps = [int(float(row["step"])) for row in rows if row.get("reward")]
    loss_values = [float(row["loss"]) for row in rows if row.get("loss")]
    loss_steps = [int(float(row["step"])) for row in rows if row.get("loss")]
    completion_lengths = [float(row["completion_mean_length"]) for row in rows if row.get("completion_mean_length")]
    completion_steps = [int(float(row["step"])) for row in rows if row.get("completion_mean_length")]

    plot_dir = output_dir / "training" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if reward_values:
        save_line_plot(plot_dir / "grpo_reward.svg", reward_steps, {"reward": reward_values}, "GRPO Reward vs Step", "Reward", smooth=False)
    if loss_values:
        save_line_plot(plot_dir / "grpo_loss.svg", loss_steps, {"loss": loss_values}, "GRPO Loss vs Step", "Loss", smooth=False)
    if completion_lengths:
        save_line_plot(
            plot_dir / "completion_length.svg",
            completion_steps,
            {"completion_mean_length": completion_lengths},
            "Completion Mean Length vs Step",
            "Tokens",
            smooth=False,
        )


def _consume_eval_trigger(episodes_seen: int, next_trigger: int, every: int) -> tuple[Optional[int], int]:
    if every <= 0 or episodes_seen < next_trigger:
        return None, next_trigger
    triggered = next_trigger
    while episodes_seen >= next_trigger:
        next_trigger += every
    return triggered, next_trigger


def _summarize_eval(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    episode_count = len(metrics)
    return {
        "episodes": episode_count,
        "mean_total_reward": sum(float(row["total_reward"]) for row in metrics) / float(episode_count or 1),
        "mean_grid_coverage_percent": sum(float(row["grid_coverage_percent"]) for row in metrics) / float(episode_count or 1),
        "mean_high_priority_miss_rate": sum(float(row["high_priority_miss_rate"]) for row in metrics) / float(episode_count or 1),
        "mean_on_time_detection_rate": sum(float(row["on_time_detection_rate"]) for row in metrics) / float(episode_count or 1),
        "mean_path_efficiency": sum(float(row.get("derived_path_efficiency", 0.0)) for row in metrics) / float(episode_count or 1),
    }


def _make_callback(
    *,
    torch_module: Any,
    tokenizer: Any,
    model_name: str,
    eval_output_dir: Path,
    effective_prompt_batch: int,
    log_every_episodes: int,
    plot_every_steps: int,
    small_eval_every_episodes: int,
    small_eval_episodes: int,
    medium_eval_every_episodes: int,
    medium_eval_episodes: int,
    full_eval_every_episodes: int,
    full_eval_episodes: int,
    eval_seed: int,
    episode_length: int,
    trainer_callback_base: type,
):
    class MADISGRPOCallback(trainer_callback_base):
        def __init__(self) -> None:
            self.next_log_episode = log_every_episodes
            self.next_small_eval = small_eval_every_episodes
            self.next_medium_eval = medium_eval_every_episodes
            self.next_full_eval = full_eval_every_episodes
            self.training_csv = eval_output_dir / "training" / "grpo_training_metrics.csv"
            self.eval_summary_csv = eval_output_dir / "training" / "grpo_eval_summary.csv"

        def on_log(self, args, state, control, logs=None, model=None, processing_class=None, **kwargs):
            if not state.is_local_process_zero or logs is None:
                return
            step = int(state.global_step)
            episodes_seen = step * effective_prompt_batch
            row = {
                "step": step,
                "episodes_seen": episodes_seen,
                "reward": logs.get("reward"),
                "reward_std": logs.get("reward_std"),
                "loss": logs.get("loss"),
                "kl": logs.get("kl"),
                "completion_mean_length": logs.get("completions/mean_length"),
                "completion_clipped_ratio": logs.get("completions/clipped_ratio"),
                "json_valid_mean": logs.get("reward/json_valid_reward/mean"),
                "unique_target_mean": logs.get("reward/unique_target_reward/mean"),
                "env_step_reward_mean": logs.get("reward/environment_step_reward/mean"),
            }
            _append_csv_row(
                self.training_csv,
                list(row.keys()),
                row,
            )
            should_refresh_plots = bool(plot_every_steps) and step % max(1, plot_every_steps) == 0

            while episodes_seen >= self.next_log_episode:
                print(
                    "episodes_seen={episodes} step={step} reward={reward} loss={loss} json_valid={json_valid}".format(
                        episodes=self.next_log_episode,
                        step=step,
                        reward=row.get("reward"),
                        loss=row.get("loss"),
                        json_valid=row.get("json_valid_mean"),
                    )
                )
                self.next_log_episode += log_every_episodes

            backend = LoadedPeftCoordinatorBackend(model=model, tokenizer=processing_class or tokenizer, torch_module=torch_module)
            triggered, self.next_small_eval = _consume_eval_trigger(episodes_seen, self.next_small_eval, small_eval_every_episodes)
            if triggered is not None:
                should_refresh_plots = True
                self._run_eval(
                    eval_type="small",
                    eval_episodes=small_eval_episodes,
                    trigger_episodes=triggered,
                    backend=backend,
                    step=step,
                )
            triggered, self.next_medium_eval = _consume_eval_trigger(episodes_seen, self.next_medium_eval, medium_eval_every_episodes)
            if triggered is not None:
                should_refresh_plots = True
                self._run_eval(
                    eval_type="medium",
                    eval_episodes=medium_eval_episodes,
                    trigger_episodes=triggered,
                    backend=backend,
                    step=step,
                )
            triggered, self.next_full_eval = _consume_eval_trigger(episodes_seen, self.next_full_eval, full_eval_every_episodes)
            if triggered is not None:
                should_refresh_plots = True
                self._run_eval(
                    eval_type="full",
                    eval_episodes=full_eval_episodes,
                    trigger_episodes=triggered,
                    backend=backend,
                    step=step,
                )
            if should_refresh_plots:
                _save_training_plots(self.training_csv, eval_output_dir)

        def _run_eval(self, *, eval_type: str, eval_episodes: int, trigger_episodes: int, backend: object, step: int) -> None:
            output_dir = eval_output_dir / f"{eval_type}_eval_ep{trigger_episodes:04d}"
            metrics = run_backend_eval(
                backend=backend,
                model_name=model_name,
                episodes=eval_episodes,
                seed=eval_seed + trigger_episodes,
                output_dir=output_dir,
                episode_length=episode_length,
            )
            summary = _summarize_eval(metrics)
            row = {
                "eval_type": eval_type,
                "trigger_episodes": trigger_episodes,
                "trainer_step": step,
                **summary,
            }
            _append_csv_row(self.eval_summary_csv, list(row.keys()), row)
            print(
                "[eval:{kind}] episodes_seen={episodes} mean_reward={reward:.3f} high_miss={miss:.3f} coverage={coverage:.2f}".format(
                    kind=eval_type,
                    episodes=trigger_episodes,
                    reward=summary["mean_total_reward"],
                    miss=summary["mean_high_priority_miss_rate"],
                    coverage=summary["mean_grid_coverage_percent"],
                )
            )

    return MADISGRPOCallback


def _load_sft_artifact_info(sft_run_dir: Path) -> Dict[str, Any]:
    grpo_reuse = json.loads((sft_run_dir / "metadata" / "grpo_reuse.json").read_text(encoding="utf-8"))
    sft_results_path = sft_run_dir / "metadata" / "sft_results.json"
    if sft_results_path.exists():
        sft_results = json.loads(sft_results_path.read_text(encoding="utf-8"))
    else:
        sft_results = {
            "status": "missing",
            "warning": "metadata/sft_results.json was not found. GRPO can continue using grpo_reuse metadata only.",
        }
    return {"grpo_reuse": grpo_reuse, "sft_results": sft_results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MADIS coordinator with TRL GRPO starting from the SFT adapter.")
    parser.add_argument("--sft-run-dir", type=Path, default=ROOT / "outputs" / "sft" / "qwen3-1.7b-coordinator-sft")
    parser.add_argument("--train-prompts-jsonl", type=Path, default=None)
    parser.add_argument("--eval-prompts-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "grpo" / "qwen3_1_7b_level8")
    parser.add_argument("--eval-output-dir", type=Path, default=ROOT / "outputs" / "evals" / "qwen3_1_7b_level8_400")
    parser.add_argument("--config-preset", choices=["pilot", "serious"], default="pilot")
    parser.add_argument("--training-episodes", type=int, default=None)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--rollouts-per-update", type=int, default=None)
    parser.add_argument("--grpo-group-size", type=int, default=None)
    parser.add_argument("--prompts-per-update", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--logging-episodes", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=10, help="TRL logging interval in trainer steps.")
    parser.add_argument(
        "--plot-every-steps",
        type=int,
        default=25,
        help="Regenerate training SVG plots every N trainer steps. Set to 0 to disable periodic plot refresh.",
    )
    parser.add_argument("--small-eval-every-episodes", type=int, default=None)
    parser.add_argument("--small-eval-episodes", type=int, default=None)
    parser.add_argument("--medium-eval-every-episodes", type=int, default=None)
    parser.add_argument("--medium-eval-episodes", type=int, default=None)
    parser.add_argument("--full-eval-every-episodes", type=int, default=None)
    parser.add_argument("--full-eval-episodes", type=int, default=None)
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Checkpoint save interval in trainer steps. If omitted, it is set to --save-every-pct of max_steps.",
    )
    parser.add_argument(
        "--save-every-pct",
        type=int,
        default=10,
        help="When --save-steps is omitted, save a checkpoint every N percent of training progress.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use-vllm", action="store_true", help="Enable vLLM generation if your TRL install supports it.")
    parser.add_argument("--log-completions", action="store_true", help="Enable TRL completion logging (can be slow).")
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help='Optional attention backend to request from transformers, e.g. "flash_attention_2" or "sdpa".',
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    preset = dict(PILOT_DEFAULT_CONFIG if args.config_preset == "pilot" else SERIOUS_RECOMMENDED_CONFIG)
    training_episodes = args.training_episodes or preset["training_episodes"]
    episode_length = args.episode_length or preset["episode_length"]
    group_size = args.grpo_group_size or preset["grpo_group_size"]
    prompts_per_update = args.prompts_per_update or preset["prompts_per_update"]
    rollouts_per_update = args.rollouts_per_update or preset["rollouts_per_update"]
    if prompts_per_update * group_size != rollouts_per_update:
        raise ValueError("prompts_per_update * grpo_group_size must equal rollouts_per_update.")

    log_every_episodes = args.logging_episodes or preset["log_every_episodes"]
    small_eval_every = args.small_eval_every_episodes or preset["small_eval_every_episodes"]
    small_eval_episodes = args.small_eval_episodes or preset["small_eval_episodes"]
    medium_eval_every = args.medium_eval_every_episodes or preset["medium_eval_every_episodes"]
    medium_eval_episodes = args.medium_eval_episodes or preset["medium_eval_episodes"]
    full_eval_every = args.full_eval_every_episodes or preset["full_eval_every_episodes"]
    full_eval_episodes = args.full_eval_episodes or preset["full_eval_episodes"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.eval_output_dir.mkdir(parents=True, exist_ok=True)
    train_prompts_jsonl = args.train_prompts_jsonl or (args.output_dir / "data" / "train_prompts.jsonl")
    _ensure_prompt_dataset(train_prompts_jsonl, episodes=training_episodes, seed=args.seed, episode_length=episode_length)
    if args.eval_prompts_jsonl is not None:
        _ensure_prompt_dataset(
            args.eval_prompts_jsonl,
            episodes=full_eval_episodes,
            seed=args.eval_seed,
            episode_length=episode_length,
        )

    sft_info = _load_sft_artifact_info(args.sft_run_dir)
    grpo_reuse = sft_info["grpo_reuse"]
    base_model_name = grpo_reuse["load_with_base_model"]
    adapter_path = grpo_reuse["adapter_path"]
    tokenizer_path = grpo_reuse["tokenizer_path"]

    save_json(
        args.output_dir / "run_config.json",
        {
            "preset": args.config_preset,
            "pilot_defaults": PILOT_DEFAULT_CONFIG,
            "serious_recommended_config": SERIOUS_RECOMMENDED_CONFIG,
            "active_config": {
                "training_episodes": training_episodes,
                "rollouts_per_update": rollouts_per_update,
                "grpo_group_size": group_size,
                "prompts_per_update": prompts_per_update,
                "episode_length": episode_length,
                "log_every_episodes": log_every_episodes,
                "small_eval_every_episodes": small_eval_every,
                "small_eval_episodes": small_eval_episodes,
                "medium_eval_every_episodes": medium_eval_every,
                "medium_eval_episodes": medium_eval_episodes,
                "full_eval_every_episodes": full_eval_every,
                "full_eval_episodes": full_eval_episodes,
            },
            "sft_artifact": sft_info,
        },
    )

    if args.dry_run:
        print("GRPO dry run configuration validated.")
        print(f"Train prompt dataset: {train_prompts_jsonl}")
        if args.eval_prompts_jsonl is not None:
            print(f"Eval prompt dataset: {args.eval_prompts_jsonl}")
        print(f"Eval output dir: {args.eval_output_dir}")
        return

    stack = _import_grpo_stack()
    torch = stack["torch"]
    load_dataset = stack["load_dataset"]
    PeftModel = stack["PeftModel"]
    AutoModelForCausalLM = stack["AutoModelForCausalLM"]
    AutoTokenizer = stack["AutoTokenizer"]
    BitsAndBytesConfig = stack["BitsAndBytesConfig"]
    TrainerCallback = stack["TrainerCallback"]
    GRPOConfig = stack["GRPOConfig"]
    GRPOTrainer = stack["GRPOTrainer"]

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=args.trust_remote_code,
        **from_pretrained_token_kwargs(),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    model_kwargs.update(from_pretrained_token_kwargs())
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.use_4bit:
        compute_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_kwargs["dtype"] = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    try:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    except TypeError:
        dtype = model_kwargs.pop("dtype", None)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_dataset = load_dataset("json", data_files=str(train_prompts_jsonl), split="train")

    effective_prompt_batch = prompts_per_update
    max_steps = max(1, training_episodes // max(1, effective_prompt_batch))
    if args.save_steps is None:
        pct = max(1, min(100, int(args.save_every_pct)))
        save_steps = max(1, int(round(max_steps * (pct / 100.0))))
    else:
        save_steps = max(1, int(args.save_steps))
    callback_cls = _make_callback(
        torch_module=torch,
        tokenizer=tokenizer,
        model_name=base_model_name,
        eval_output_dir=args.eval_output_dir,
        effective_prompt_batch=effective_prompt_batch,
        log_every_episodes=log_every_episodes,
        plot_every_steps=args.plot_every_steps,
        small_eval_every_episodes=small_eval_every,
        small_eval_episodes=small_eval_episodes,
        medium_eval_every_episodes=medium_eval_every,
        medium_eval_episodes=medium_eval_episodes,
        full_eval_every_episodes=full_eval_every,
        full_eval_episodes=full_eval_episodes,
        eval_seed=args.eval_seed,
        episode_length=episode_length,
        trainer_callback_base=TrainerCallback,
    )

    grpo_config_kwargs = _filter_supported_kwargs(
        GRPOConfig.__init__,
        {
            "output_dir": str(args.output_dir / "checkpoints"),
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "epsilon": args.epsilon,
            "num_generations": group_size,
            "per_device_train_batch_size": prompts_per_update,
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "logging_steps": args.logging_steps,
            "save_steps": save_steps,
            "max_steps": max_steps,
            "bf16": args.bf16,
            "fp16": args.fp16,
            "gradient_checkpointing": args.gradient_checkpointing,
            "report_to": [],
            "seed": args.seed,
            "log_completions": args.log_completions,
            "use_vllm": args.use_vllm,
        },
    )
    grpo_args = GRPOConfig(**grpo_config_kwargs)

    trainer_kwargs = _filter_supported_kwargs(
        GRPOTrainer.__init__,
        {
            "model": model,
            "processing_class": tokenizer,
            "tokenizer": tokenizer,
            "reward_funcs": [json_valid_reward, unique_target_reward, environment_step_reward],
            "train_dataset": train_dataset,
            "args": grpo_args,
            "callbacks": [callback_cls()],
        },
    )
    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(args.output_dir / "adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "tokenizer"))

    final_eval_metrics = run_adapter_eval(
        adapter_path=str(args.output_dir / "adapter"),
        tokenizer_path=str(args.output_dir / "tokenizer"),
        base_model_name=base_model_name,
        episodes=full_eval_episodes,
        seed=args.eval_seed,
        output_dir=args.eval_output_dir,
        episode_length=episode_length,
    )
    final_summary = _summarize_eval(final_eval_metrics)
    save_json(
        args.output_dir / "final_summary.json",
        {
            "final_eval": final_summary,
            "adapter_path": str((args.output_dir / "adapter").resolve()),
            "tokenizer_path": str((args.output_dir / "tokenizer").resolve()),
            "eval_output_dir": str(args.eval_output_dir.resolve()),
        },
    )
    print(f"Saved GRPO adapter to {args.output_dir / 'adapter'}")
    print(f"Saved final Level 8 evaluation outputs to {args.eval_output_dir}")


if __name__ == "__main__":
    main()
