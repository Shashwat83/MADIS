from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import inspect
import time
from html import escape
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disaster_surveillance_env.coordinator import (
    HeuristicCoordinator,
    build_coordinator_prompt,
    extract_observation_from_prompt,
    parse_coordinator_targets,
)
from disaster_surveillance_env.models import DroneActions, manhattan_distance
from disaster_surveillance_env.server.disaster_surveillance_environment import DisasterSurveillanceEnvironment


def require_training_imports() -> Dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
        from trl import GRPOConfig, GRPOTrainer
    except Exception as exc:
        raise RuntimeError(
            "Missing training dependencies. In Colab run: "
            "pip install -U transformers trl peft accelerate bitsandbytes datasets"
        ) from exc

    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "Trainer": Trainer,
        "TrainerCallback": TrainerCallback,
        "TrainingArguments": TrainingArguments,
        "GRPOConfig": GRPOConfig,
        "GRPOTrainer": GRPOTrainer,
    }


class EtaCallback:
    def __init__(self, total_steps: int, label: str, print_every: int = 5) -> None:
        self.total_steps = max(1, total_steps)
        self.label = label
        self.print_every = max(1, print_every)
        self.started_at = time.perf_counter()

    def maybe_print(self, step: int, extra: str = "") -> None:
        if step != 1 and step % self.print_every != 0 and step < self.total_steps:
            return
        elapsed = time.perf_counter() - self.started_at
        avg = elapsed / float(max(1, step))
        eta = max(0, self.total_steps - step) * avg
        suffix = f" {extra}" if extra else ""
        print(
            f"[{self.label}] step={step}/{self.total_steps} elapsed={elapsed:.1f}s eta={eta:.1f}s{suffix}",
            flush=True,
        )


def make_trainer_callback(base_callback_cls: Any, total_steps: int, label: str, print_every: int) -> Any:
    class ProgressCallback(base_callback_cls):
        def __init__(self) -> None:
            self.eta = EtaCallback(total_steps=total_steps, label=label, print_every=print_every)

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            self.eta.maybe_print(int(state.global_step))
            return control

    return ProgressCallback()


def coordinator_targets_to_json(targets: Mapping[str, Sequence[int]]) -> str:
    return json.dumps(
        {drone_id: [int(coord[0]), int(coord[1])] for drone_id, coord in targets.items()},
        separators=(",", ":"),
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _nice_bounds(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if low == high:
        padding = abs(low) * 0.1 or 1.0
        return low - padding, high + padding
    padding = (high - low) * 0.08
    return low - padding, high + padding


def _save_metric_plot(path: Path, rows: Sequence[Mapping[str, Any]], metric: str, title: str) -> None:
    points = [
        (int(row["step"]), float(row[metric]))
        for row in rows
        if row.get("step") is not None and _is_number(row.get(metric))
    ]
    if not points:
        return

    width, height = 900, 460
    left, right, top, bottom = 72, 28, 54, 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = _nice_bounds(y_values)
    x_span = max(1, x_max - x_min)
    y_span = max(1e-9, y_max - y_min)

    def sx(step: float) -> float:
        return left + ((step - x_min) / x_span) * plot_w

    def sy(value: float) -> float:
        return top + (1.0 - ((value - y_min) / y_span)) * plot_h

    polyline = " ".join(f"{sx(step):.2f},{sy(value):.2f}" for step, value in points)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2:.2f}" y="28" font-family="Arial, sans-serif" font-size="20" '
        f'font-weight="700" text-anchor="middle" fill="#111827">{escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f9fafb" stroke="#d1d5db" />',
    ]
    for tick in range(6):
        ratio = tick / 5
        y = top + ratio * plot_h
        value = y_max - ratio * y_span
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb" />')
        elements.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" font-family="Arial, sans-serif" font-size="11" '
            f'text-anchor="end" fill="#111827">{value:.4g}</text>'
        )
    for tick in range(6):
        ratio = tick / 5
        x = left + ratio * plot_w
        value = round(x_min + ratio * x_span)
        elements.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#f3f4f6" />')
        elements.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 22}" font-family="Arial, sans-serif" font-size="11" '
            f'text-anchor="middle" fill="#111827">{value}</text>'
        )

    elements.append(
        f'<polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="2.4" '
        'stroke-linejoin="round" stroke-linecap="round" />'
    )
    elements.append(
        f'<text x="{width / 2:.2f}" y="{height - 18}" font-family="Arial, sans-serif" font-size="13" '
        'text-anchor="middle" fill="#111827">Training Step</text>'
    )
    elements.append(
        f'<text x="18" y="{height / 2:.2f}" font-family="Arial, sans-serif" font-size="13" '
        f'text-anchor="middle" fill="#111827">{escape(metric)}</text>'
    )
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements))


def save_training_diagnostics(trainer: Any, output_dir: Path, phase: str) -> None:
    log_rows = [
        dict(row)
        for row in getattr(trainer.state, "log_history", [])
        if isinstance(row, Mapping) and row.get("step") is not None
    ]
    if not log_rows:
        print(f"[{phase}] no trainer log history available for loss/metric plots", flush=True)
        return

    numeric_keys = sorted(
        {
            key
            for row in log_rows
            for key, value in row.items()
            if key != "step" and _is_number(value)
        }
    )
    fieldnames = ["step", *numeric_keys]
    diagnostics_dir = output_dir / "training_diagnostics" / phase
    _write_csv(diagnostics_dir / "trainer_log.csv", log_rows, fieldnames)

    preferred_metrics = [
        "loss",
        "reward",
        "rewards",
        "mean_reward",
        "kl",
        "grad_norm",
        "learning_rate",
    ]
    plotted = 0
    for metric in preferred_metrics:
        if metric not in numeric_keys:
            continue
        _save_metric_plot(
            diagnostics_dir / f"{metric}.svg",
            log_rows,
            metric,
            f"{phase.upper()} {metric} vs Training Step",
        )
        plotted += 1

    print(
        f"[{phase}] saved training diagnostics to {diagnostics_dir} "
        f"(csv=trainer_log.csv, plots={plotted})",
        flush=True,
    )


def generate_prompt_dataset(
    *,
    num_prompts: int,
    seed: int,
    episode_length: int,
    rollout_steps_per_episode: int,
    print_every: int,
) -> List[Dict[str, str]]:
    heuristic = HeuristicCoordinator()
    rows: List[Dict[str, str]] = []
    episode = 0
    started_at = time.perf_counter()
    while len(rows) < num_prompts:
        episode += 1
        env = DisasterSurveillanceEnvironment(
            level=6,
            seed=seed + episode - 1,
            episode_length=episode_length,
            coordinator=heuristic,
        )
        observation = env.reset(seed=seed + episode - 1)
        for _ in range(rollout_steps_per_episode):
            if observation.done or len(rows) >= num_prompts:
                break
            coordinator_observation = env.build_coordinator_observation()
            decision = heuristic.decide(coordinator_observation)
            prompt = build_coordinator_prompt(coordinator_observation)
            completion = coordinator_targets_to_json(decision.targets)
            rows.append({"prompt": prompt, "completion": completion, "text": f"{prompt}{completion}"})
            observation = env.step(DroneActions(targets=decision.targets))

        if episode == 1 or episode % print_every == 0 or len(rows) >= num_prompts:
            elapsed = time.perf_counter() - started_at
            avg = elapsed / float(max(1, len(rows)))
            remaining = max(0, num_prompts - len(rows)) * avg
            print(
                f"[dataset] episode={episode} prompts={len(rows)}/{num_prompts} "
                f"elapsed={elapsed:.1f}s eta={remaining:.1f}s",
                flush=True,
            )
    return rows[:num_prompts]


def target_reward(prompt: str, completion: str) -> float:
    observation = extract_observation_from_prompt(prompt)
    try:
        targets = parse_coordinator_targets(completion, observation)
    except Exception:
        return -3.0

    reward = 2.0
    unique_targets = len(set(targets.values()))
    reward += unique_targets / float(max(1, len(targets)))
    if unique_targets < len(targets):
        reward -= 1.0 * (len(targets) - unique_targets)

    grid_size = int(observation.get("grid_size", 10))
    for target in targets.values():
        if not (0 <= target[0] < grid_size and 0 <= target[1] < grid_size):
            reward -= 2.0

    frontier = {tuple(cell) for cell in observation.get("team_frontier_cells", [])}
    if frontier:
        frontier_hits = sum(1 for target in set(targets.values()) if target in frontier)
        reward += 0.3 * frontier_hits

    severity_weight = {"HIGH": 6.0, "MEDIUM": 3.0, "LOW": 1.0}
    drone_positions = {drone_id: tuple(pos) for drone_id, pos in observation["drone_positions"].items()}
    for event in observation.get("visible_active_events", []):
        event_location = tuple(event["location"])
        weight = severity_weight.get(str(event.get("severity")), 1.0)
        nearest_target_distance = min(manhattan_distance(target, event_location) for target in targets.values())
        nearest_drone_distance = min(manhattan_distance(position, event_location) for position in drone_positions.values())
        reward += weight / float(1 + nearest_target_distance)
        if nearest_target_distance <= nearest_drone_distance:
            reward += 0.5 * weight

    return float(reward)


def build_reward_func() -> Any:
    def reward_func(prompts: Sequence[str], completions: Sequence[Any], **kwargs: Any) -> List[float]:
        rewards: List[float] = []
        for prompt, completion in zip(prompts, completions):
            if isinstance(completion, list) and completion and isinstance(completion[0], Mapping):
                text = str(completion[0].get("content", ""))
            else:
                text = str(completion)
            rewards.append(target_reward(prompt, text))
        return rewards

    return reward_func


def load_qwen_for_training(model_name: str, imports: Mapping[str, Any]) -> tuple[Any, Any]:
    torch = imports["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for efficient GRPO training. Select a Colab GPU runtime first.")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)

    tokenizer = imports["AutoTokenizer"].from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = imports["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = imports["AutoModelForCausalLM"].from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model = imports["prepare_model_for_kbit_training"](model)
    lora_config = imports["LoraConfig"](
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = imports["get_peft_model"](model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def run_sft_warmup(
    *,
    model: Any,
    tokenizer: Any,
    dataset: Any,
    imports: Mapping[str, Any],
    output_dir: Path,
    max_steps: int,
    print_every: int,
) -> Any:
    if max_steps <= 0:
        print("[sft] skipped", flush=True)
        return model

    def tokenize(batch: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=1152)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    args = imports["TrainingArguments"](
        output_dir=str(output_dir / "sft_warmup"),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=print_every,
        save_steps=max_steps,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = imports["Trainer"](
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=imports["DataCollatorForLanguageModeling"](tokenizer=tokenizer, mlm=False),
        callbacks=[make_trainer_callback(imports["TrainerCallback"], max_steps, "sft", print_every)],
    )
    trainer.train()
    trainer.save_model(str(output_dir / "sft_warmup"))
    save_training_diagnostics(trainer, output_dir, "sft")
    return model


def run_grpo(
    *,
    model: Any,
    tokenizer: Any,
    dataset: Any,
    imports: Mapping[str, Any],
    output_dir: Path,
    max_steps: int,
    num_generations: int,
    print_every: int,
) -> None:
    if max_steps <= 0:
        print("[grpo] skipped", flush=True)
        return

    per_device_batch_size = max(2, num_generations)
    config_kwargs = {
        "output_dir": str(output_dir / "grpo"),
        "max_steps": max_steps,
        "learning_rate": 5e-6,
        "per_device_train_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": 2,
        "num_generations": num_generations,
        "max_prompt_length": 1024,
        "max_completion_length": 128,
        "temperature": 0.7,
        "top_p": 0.9,
        "beta": 0.02,
        "fp16": True,
        "logging_steps": print_every,
        "save_steps": max_steps,
        "report_to": [],
    }

    accepted_args = set(inspect.signature(imports["GRPOConfig"].__init__).parameters)
    filtered_kwargs = {
        key: value for key, value in config_kwargs.items()
        if key in accepted_args
    }
    dropped_args = sorted(set(config_kwargs) - set(filtered_kwargs))
    if dropped_args:
        print(f"[grpo] dropped unsupported GRPOConfig args: {dropped_args}", flush=True)

    args = imports["GRPOConfig"](**filtered_kwargs)


    trainer = imports["GRPOTrainer"](
        model=model,
        args=args,
        train_dataset=dataset.remove_columns(["completion", "text"]),
        reward_funcs=build_reward_func(),
        processing_class=tokenizer,
        callbacks=[make_trainer_callback(imports["TrainerCallback"], max_steps, "grpo", print_every)],
    )
    trainer.train()
    trainer.save_model(str(output_dir / "grpo_lora"))
    save_training_diagnostics(trainer, output_dir, "grpo")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Level 6 coordinator LLM with optional SFT warmup and GRPO.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="Local Hugging Face model id.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "qwen3_grpo_coordinator")
    parser.add_argument("--num-prompts", type=int, default=1024, help="Prompt states cached before training.")
    parser.add_argument("--episode-length", type=int, default=10, help="Short horizon used for prompt collection.")
    parser.add_argument("--rollout-steps-per-episode", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft-steps", type=int, default=100, help="Set to 0 to skip SFT warmup.")
    parser.add_argument("--grpo-steps", type=int, default=300)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--print-every", type=int, default=5)
    args = parser.parse_args()

    started_at = time.perf_counter()
    try:
        imports = require_training_imports()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[setup] model={args.model} output_dir={args.output_dir} "
            f"num_prompts={args.num_prompts} sft_steps={args.sft_steps} grpo_steps={args.grpo_steps}",
            flush=True,
        )
        rows = generate_prompt_dataset(
            num_prompts=args.num_prompts,
            seed=args.seed,
            episode_length=args.episode_length,
            rollout_steps_per_episode=args.rollout_steps_per_episode,
            print_every=args.print_every,
        )
        dataset = imports["Dataset"].from_list(rows)
        dataset.save_to_disk(str(args.output_dir / "prompt_cache"))
        print(f"[dataset] saved prompt cache to {args.output_dir / 'prompt_cache'}", flush=True)

        model, tokenizer = load_qwen_for_training(args.model, imports)
        model = run_sft_warmup(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            imports=imports,
            output_dir=args.output_dir,
            max_steps=args.sft_steps,
            print_every=args.print_every,
        )
        run_grpo(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            imports=imports,
            output_dir=args.output_dir,
            max_steps=args.grpo_steps,
            num_generations=args.num_generations,
            print_every=args.print_every,
        )
        elapsed = time.perf_counter() - started_at
        print(f"[done] training completed in {elapsed:.1f}s. Adapter saved under {args.output_dir}", flush=True)
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        print(f"[ERROR] training failed after {elapsed:.1f}s: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
