from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disaster_surveillance_env.coordinator import LLMCoordinator
from disaster_surveillance_env.grpo.local_peft_backend import LoadedPeftCoordinatorBackend
from disaster_surveillance_env.hf_hub_auth import from_pretrained_token_kwargs
from disaster_surveillance_env.server.disaster_surveillance_environment import DisasterSurveillanceEnvironment
from scripts.analyze_baseline import save_line_plot, save_outputs


def _import_stack() -> Dict[str, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Comparison requires optional dependencies. Install them with "
            "`pip install -e '.[grpo]'` (or at least transformers+torch+peft)."
        ) from exc
    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


def _load_grpo_reuse(sft_run_dir: Path) -> Dict[str, Any]:
    return json.loads((sft_run_dir / "metadata" / "grpo_reuse.json").read_text(encoding="utf-8"))


def _summarize(metrics: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    count = len(metrics)
    return {
        "episodes": count,
        "mean_total_reward": sum(float(row["total_reward"]) for row in metrics) / float(count or 1),
        "mean_high_priority_miss_rate": sum(float(row["high_priority_miss_rate"]) for row in metrics) / float(count or 1),
        "mean_on_time_detection_rate": sum(float(row["on_time_detection_rate"]) for row in metrics) / float(count or 1),
        "mean_grid_coverage_percent": sum(float(row["grid_coverage_percent"]) for row in metrics) / float(count or 1),
        "mean_path_efficiency": sum(float(row.get("derived_path_efficiency", 0.0)) for row in metrics) / float(count or 1),
    }


def _run_eval(
    *,
    backend: object,
    model_name: str,
    policy_type: str,
    episodes: int,
    seed: int,
    output_dir: Path,
    level: int = 6,
    episode_length: Optional[int] = None,
) -> List[Dict[str, Any]]:
    coordinator = LLMCoordinator(backend=backend, model_name=model_name)
    results: List[Dict[str, Any]] = []

    for index in range(episodes):
        env_kwargs: Dict[str, Any] = {"seed": seed + index, "level": level, "coordinator": coordinator}
        if episode_length is not None:
            env_kwargs["episode_length"] = episode_length
        env = DisasterSurveillanceEnvironment(**env_kwargs)
        observation = env.reset(seed=seed + index)
        while not observation.done:
            observation = env.step(None)
        metrics = {key: value for key, value in env.metrics.items() if not key.startswith("_")}
        metrics["episode"] = index + 1
        metrics["seed"] = seed + index
        metrics["model_name"] = model_name
        metrics["policy_type"] = policy_type
        metrics["derived_path_efficiency"] = float(metrics.get("path_efficiency", 0.0))
        target_assignments = int(metrics.get("target_assignment_count", 0))
        fallback_count = int(metrics.get("coordinator_fallback_count", 0))
        metrics["derived_fallback_rate"] = fallback_count / float(target_assignments or 1)
        results.append(metrics)

    save_outputs(results, output_dir)
    return results


def _plot_comparison(output_dir: Path, *, per_variant: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    episodes = list(range(1, 1 + max((len(rows) for rows in per_variant.values()), default=0)))
    if not episodes:
        return

    def series_for(key: str) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for name, rows in per_variant.items():
            out[name] = [float(row.get(key, 0.0)) for row in rows]
        return out

    save_line_plot(plot_dir / "total_reward.svg", episodes, series_for("total_reward"), "Total Reward vs Episode", "Reward", smooth=False)
    save_line_plot(
        plot_dir / "high_priority_miss_rate.svg",
        episodes,
        series_for("high_priority_miss_rate"),
        "High-Priority Miss Rate vs Episode",
        "Miss Rate",
        smooth=False,
    )
    save_line_plot(
        plot_dir / "on_time_detection_rate.svg",
        episodes,
        series_for("on_time_detection_rate"),
        "On-Time Detection Rate vs Episode",
        "Rate",
        smooth=False,
    )
    save_line_plot(
        plot_dir / "grid_coverage_percent.svg",
        episodes,
        series_for("grid_coverage_percent"),
        "Coverage % vs Episode",
        "Coverage %",
        smooth=False,
    )
    save_line_plot(
        plot_dir / "path_efficiency.svg",
        episodes,
        series_for("derived_path_efficiency"),
        "Path Efficiency vs Episode",
        "Path Efficiency",
        smooth=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base vs SFT vs SFT+GRPO coordinator policies on fixed seeds.")
    parser.add_argument("--sft-run-dir", type=Path, default=ROOT / "outputs" / "sft" / "qwen3-1.7b-coordinator-sft")
    parser.add_argument("--grpo-run-dir", type=Path, default=ROOT / "outputs" / "grpo" / "qwen3_1_7b_level8")
    parser.add_argument("--base-model-name", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--level", type=int, default=6)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "comparisons" / "base_vs_sft_vs_grpo")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", type=str, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    grpo_reuse = _load_grpo_reuse(args.sft_run_dir)
    base_model_name = args.base_model_name or str(grpo_reuse["load_with_base_model"])
    sft_adapter_path = Path(grpo_reuse["adapter_path"])
    sft_tokenizer_path = Path(grpo_reuse["tokenizer_path"])

    grpo_adapter_path = args.grpo_run_dir / "adapter"
    grpo_tokenizer_path = args.grpo_run_dir / "tokenizer"

    stack = _import_stack()
    torch = stack["torch"]
    PeftModel = stack["PeftModel"]
    AutoModelForCausalLM = stack["AutoModelForCausalLM"]
    AutoTokenizer = stack["AutoTokenizer"]
    BitsAndBytesConfig = stack["BitsAndBytesConfig"]

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": args.trust_remote_code,
        **from_pretrained_token_kwargs(),
    }
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        model_kwargs["dtype"] = dtype

    tokenizer_base = AutoTokenizer.from_pretrained(
        str(sft_tokenizer_path),
        trust_remote_code=args.trust_remote_code,
        **from_pretrained_token_kwargs(),
    )
    if tokenizer_base.pad_token is None:
        tokenizer_base.pad_token = tokenizer_base.eos_token

    tokenizer_grpo = AutoTokenizer.from_pretrained(
        str(grpo_tokenizer_path) if grpo_tokenizer_path.exists() else str(sft_tokenizer_path),
        trust_remote_code=args.trust_remote_code,
        **from_pretrained_token_kwargs(),
    )
    if tokenizer_grpo.pad_token is None:
        tokenizer_grpo.pad_token = tokenizer_grpo.eos_token

    try:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    except TypeError:
        fixed = dict(model_kwargs)
        maybe = fixed.pop("dtype", None)
        if maybe is not None:
            fixed["torch_dtype"] = maybe
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **fixed)
    base_model.eval()

    base_backend = LoadedPeftCoordinatorBackend(model=base_model, tokenizer=tokenizer_base, torch_module=torch)
    sft_model = PeftModel.from_pretrained(base_model, str(sft_adapter_path))
    sft_model.eval()
    sft_backend = LoadedPeftCoordinatorBackend(model=sft_model, tokenizer=tokenizer_base, torch_module=torch)

    if not grpo_adapter_path.exists():
        raise FileNotFoundError(f"GRPO adapter not found at {grpo_adapter_path}")
    grpo_model = PeftModel.from_pretrained(base_model, str(grpo_adapter_path))
    grpo_model.eval()
    grpo_backend = LoadedPeftCoordinatorBackend(model=grpo_model, tokenizer=tokenizer_grpo, torch_module=torch)

    base_out = args.output_dir / "base"
    sft_out = args.output_dir / "sft"
    grpo_out = args.output_dir / "grpo"

    base_metrics = _run_eval(
        backend=base_backend,
        model_name=base_model_name,
        policy_type="base_model",
        episodes=args.episodes,
        seed=args.seed,
        output_dir=base_out,
        level=args.level,
        episode_length=args.episode_length,
    )
    sft_metrics = _run_eval(
        backend=sft_backend,
        model_name=base_model_name,
        policy_type="sft_adapter",
        episodes=args.episodes,
        seed=args.seed,
        output_dir=sft_out,
        level=args.level,
        episode_length=args.episode_length,
    )
    grpo_metrics = _run_eval(
        backend=grpo_backend,
        model_name=base_model_name,
        policy_type="sft_plus_grpo_adapter",
        episodes=args.episodes,
        seed=args.seed,
        output_dir=grpo_out,
        level=args.level,
        episode_length=args.episode_length,
    )

    _plot_comparison(args.output_dir, per_variant={"base": base_metrics, "sft": sft_metrics, "grpo": grpo_metrics})

    per_episode_rewards = {
        name: [float(row["total_reward"]) for row in rows]
        for name, rows in {"base": base_metrics, "sft": sft_metrics, "grpo": grpo_metrics}.items()
    }
    summary = {
        "config": {
            "base_model_name": base_model_name,
            "sft_run_dir": str(args.sft_run_dir.resolve()),
            "grpo_run_dir": str(args.grpo_run_dir.resolve()),
            "episodes": args.episodes,
            "seed": args.seed,
            "level": args.level,
            "episode_length": args.episode_length,
        },
        "base": _summarize(base_metrics),
        "sft": _summarize(sft_metrics),
        "grpo": _summarize(grpo_metrics),
        "per_episode_rewards": per_episode_rewards,
        "output_dir": str(args.output_dir.resolve()),
    }
    (args.output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote comparison summary to {args.output_dir / 'comparison_summary.json'}")


if __name__ == "__main__":
    main()

