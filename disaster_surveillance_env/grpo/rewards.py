from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from ..models import DroneActions, normalize_targets
from ..sft.parsing import parse_target_json
from .state import restore_environment


@dataclass(slots=True)
class ParsedCompletion:
    parse_success: bool
    targets: Dict[str, tuple[int, int]]
    invalid_reason: str | None
    unique_target_ratio: float


def parse_or_fallback_completion(
    completion: str,
    snapshot_json: str | None = None,
    *,
    snapshot: Mapping[str, Any] | None = None,
) -> ParsedCompletion:
    if snapshot is None:
        if snapshot_json is None:
            raise ValueError("Either snapshot_json or snapshot must be provided.")
        snapshot = json.loads(snapshot_json)
    drone_ids = [drone["id"] for drone in snapshot["drones"]]
    grid_size = int(snapshot.get("grid_size", 10))
    default_targets = {
        drone["id"]: tuple(drone["position"])
        for drone in snapshot["drones"]
    }
    try:
        targets = parse_target_json(completion, drone_ids=drone_ids, grid_size=grid_size)
        parse_success = True
        invalid_reason = None
    except Exception as exc:
        targets = default_targets
        parse_success = False
        invalid_reason = f"{type(exc).__name__}: {exc}"
    normalized = normalize_targets(targets, drone_ids, grid_size)
    unique_target_ratio = len(set(normalized.values())) / float(len(drone_ids))
    return ParsedCompletion(
        parse_success=parse_success,
        targets=normalized,
        invalid_reason=invalid_reason,
        unique_target_ratio=unique_target_ratio,
    )


def json_valid_reward(completions: Sequence[str], snapshot_json: Sequence[str], **kwargs: Any) -> List[float]:
    rewards: List[float] = []
    for completion, snapshot_text in zip(completions, snapshot_json):
        snapshot = json.loads(snapshot_text)
        rewards.append(1.0 if parse_or_fallback_completion(completion, snapshot=snapshot).parse_success else -1.0)
    return rewards


def unique_target_reward(completions: Sequence[str], snapshot_json: Sequence[str], **kwargs: Any) -> List[float]:
    ratios: List[float] = []
    for completion, snapshot_text in zip(completions, snapshot_json):
        snapshot = json.loads(snapshot_text)
        ratios.append(parse_or_fallback_completion(completion, snapshot=snapshot).unique_target_ratio)
    return ratios


def environment_step_reward(
    completions: Sequence[str],
    snapshot_json: Sequence[str],
    **kwargs: Any,
) -> List[float]:
    rewards: List[float] = []
    for completion, snapshot_text in zip(completions, snapshot_json):
        snapshot = json.loads(snapshot_text)
        parsed = parse_or_fallback_completion(completion, snapshot=snapshot)
        env = restore_environment(
            snapshot,
        )
        observation = env.step(DroneActions(targets=parsed.targets))
        reward = float(observation.reward)
        if not parsed.parse_success:
            reward -= 2.0
        rewards.append(reward)
    return rewards
