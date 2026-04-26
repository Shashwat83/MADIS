from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..coordinator import LLMCoordinator
from ..server.disaster_surveillance_environment import DisasterSurveillanceEnvironment
from .local_peft_backend import LoadedPeftCoordinatorBackend, LocalPeftCoordinatorBackend
from scripts.analyze_baseline import save_outputs


def run_adapter_eval(
    *,
    adapter_path: str,
    tokenizer_path: str,
    base_model_name: str,
    episodes: int,
    seed: int,
    output_dir: Path,
    level: int = 6,
    episode_length: Optional[int] = None,
) -> List[Dict[str, Any]]:
    backend = LocalPeftCoordinatorBackend(
        base_model_name=base_model_name,
        adapter_path=adapter_path,
        tokenizer_path=tokenizer_path,
    )
    coordinator = LLMCoordinator(backend=backend, model_name=base_model_name)
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
        metrics["model_name"] = base_model_name
        metrics["policy_type"] = "grpo_adapter"
        metrics["derived_path_efficiency"] = float(metrics.get("path_efficiency", 0.0))
        target_assignments = int(metrics.get("target_assignment_count", 0))
        fallback_count = int(metrics.get("coordinator_fallback_count", 0))
        metrics["derived_fallback_rate"] = fallback_count / float(target_assignments or 1)
        results.append(metrics)

    save_outputs(results, output_dir)
    return results


def run_backend_eval(
    *,
    backend: object,
    model_name: str,
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
        metrics["policy_type"] = "grpo_adapter"
        metrics["derived_path_efficiency"] = float(metrics.get("path_efficiency", 0.0))
        target_assignments = int(metrics.get("target_assignment_count", 0))
        fallback_count = int(metrics.get("coordinator_fallback_count", 0))
        metrics["derived_fallback_rate"] = fallback_count / float(target_assignments or 1)
        results.append(metrics)

    save_outputs(results, output_dir)
    return results
