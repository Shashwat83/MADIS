from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence

import numpy as np

from ..models import HOTSPOTS
from ..server.disaster_surveillance_environment import DisasterSurveillanceEnvironment
from ..sft.policies import HeuristicTeacherPolicy
from ..sft.prompting import TARGET_OUTPUT_MODE, build_coordinator_prompt
from .state import snapshot_environment


@dataclass(slots=True)
class GRPOPromptExample:
    prompt: str
    snapshot_json: str
    episode_index: int
    sampled_timestep: int
    seed: int


def iter_grpo_prompt_examples(
    *,
    episodes: int,
    seed: int = 42,
    level: int = 6,
    episode_length: int | None = None,
) -> Iterator[GRPOPromptExample]:
    if episodes < 1:
        raise ValueError("episodes must be >= 1")

    teacher = HeuristicTeacherPolicy()
    for episode_index in range(episodes):
        env_kwargs: Dict[str, Any] = {"seed": seed + episode_index, "level": level}
        if episode_length is not None:
            env_kwargs["episode_length"] = episode_length
        env = DisasterSurveillanceEnvironment(**env_kwargs)
        observation = env.reset(seed=seed + episode_index)
        chosen_timestep = int(env.rng.integers(0, env.episode_length))

        while not observation.done and env.timestep < chosen_timestep:
            coordinator_observation = env.build_coordinator_observation()
            targets = teacher.decide_targets(coordinator_observation, env=env)
            observation = env.step({"drone_1": targets["drone_1"], "drone_2": targets["drone_2"], "drone_3": targets["drone_3"]})

        coordinator_observation = env.build_coordinator_observation()
        prompt = build_coordinator_prompt(
            coordinator_observation,
            episode_length=env.episode_length,
            hotspots=HOTSPOTS,
            output_mode=TARGET_OUTPUT_MODE,
            include_hidden_state=False,
        )
        snapshot_json = json.dumps(
            snapshot_environment(env),
            separators=(",", ":"),
        )
        yield GRPOPromptExample(
            prompt=prompt,
            snapshot_json=snapshot_json,
            episode_index=episode_index,
            sampled_timestep=env.timestep,
            seed=seed + episode_index,
        )


def export_grpo_prompt_jsonl(path: str | Path, examples: Iterable[GRPOPromptExample]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(
                json.dumps(
                    {
                        "prompt": example.prompt,
                        "snapshot_json": example.snapshot_json,
                        "episode_index": example.episode_index,
                        "sampled_timestep": example.sampled_timestep,
                        "seed": example.seed,
                    },
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
            count += 1
    return count

