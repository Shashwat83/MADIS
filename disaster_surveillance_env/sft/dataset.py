from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, Optional

from ..models import HOTSPOTS, DroneActions, normalize_targets
from ..server.disaster_surveillance_environment import DisasterSurveillanceEnvironment
from .parsing import ACTION_OUTPUT_MODE, TARGET_OUTPUT_MODE
from .policies import TeacherPolicy
from .prompting import build_coordinator_prompt


DatasetType = Literal["action_format", "heuristic", "oracle"]
VisibilityMode = Literal["partial", "oracle"]


@dataclass(slots=True)
class SFTExample:
    prompt: str
    response: str
    metadata: Dict[str, Any]


def rollout_sft_examples(
    *,
    episodes: int,
    teacher_policy: TeacherPolicy,
    dataset_type: DatasetType,
    output_mode: str = TARGET_OUTPUT_MODE,
    seed: int = 42,
    level: int = 6,
    episode_length: Optional[int] = None,
    p_spawn: Optional[float] = None,
) -> List[SFTExample]:
    return list(
        iter_sft_examples(
            episodes=episodes,
            teacher_policy=teacher_policy,
            dataset_type=dataset_type,
            output_mode=output_mode,
            seed=seed,
            level=level,
            episode_length=episode_length,
            p_spawn=p_spawn,
        )
    )


def iter_sft_examples(
    *,
    episodes: int,
    teacher_policy: TeacherPolicy,
    dataset_type: DatasetType,
    output_mode: str = TARGET_OUTPUT_MODE,
    seed: int = 42,
    level: int = 6,
    episode_length: Optional[int] = None,
    p_spawn: Optional[float] = None,
) -> Iterator[SFTExample]:
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    if level != 6:
        raise ValueError("SFT rollout generation currently expects the Level 6 coordinator environment.")

    visibility_mode: VisibilityMode = "oracle" if dataset_type == "oracle" else "partial"

    for episode_index in range(episodes):
        env_kwargs: Dict[str, Any] = {"seed": seed + episode_index, "level": level}
        if episode_length is not None:
            env_kwargs["episode_length"] = episode_length
        if p_spawn is not None:
            env_kwargs["p_spawn"] = p_spawn
        env = DisasterSurveillanceEnvironment(**env_kwargs)
        observation = env.reset(seed=seed + episode_index)

        bounded_steps = 0
        while not observation.done and bounded_steps < env.episode_length:
            coordinator_observation = env.build_coordinator_observation()
            prompt = build_coordinator_prompt(
                coordinator_observation,
                episode_length=env.episode_length,
                hotspots=HOTSPOTS,
                output_mode=output_mode,
                include_hidden_state=(visibility_mode == "oracle"),
            )
            targets = normalize_targets(
                teacher_policy.decide_targets(coordinator_observation, env=env),
                env.agent_ids,
                env.grid_size,
            )
            response_payload: Mapping[str, Any]
            if output_mode == ACTION_OUTPUT_MODE:
                response_payload = teacher_policy.decide_actions(coordinator_observation, env=env)
            else:
                response_payload = targets

            yield SFTExample(
                prompt=prompt,
                response=json.dumps(response_payload, separators=(",", ":")),
                metadata={
                    "dataset_type": dataset_type,
                    "visibility_mode": visibility_mode,
                    "output_mode": output_mode,
                    "episode_index": episode_index,
                    "seed": seed + episode_index,
                    "timestep": env.timestep,
                    "episode_length": env.episode_length,
                    "grid_size": env.grid_size,
                    "drone_ids": list(env.agent_ids),
                },
            )

            observation = env.step(DroneActions(targets=targets))
            bounded_steps += 1

        if bounded_steps >= env.episode_length and not observation.done:
            raise RuntimeError("SFT rollout exceeded the episode-length safety bound.")

def export_jsonl(path: str | Path, examples: Iterable[SFTExample]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count
