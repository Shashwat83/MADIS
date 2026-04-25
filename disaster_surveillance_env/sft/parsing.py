from __future__ import annotations

import json
import re
from typing import Any, Dict, Mapping, Sequence

from ..models import ACTION_LABELS, Coord


ACTION_OUTPUT_MODE = "actions"
TARGET_OUTPUT_MODE = "targets"
ACTION_NAME_TO_ID = {label: action_id for action_id, label in ACTION_LABELS.items()}
VALID_ACTION_NAMES = frozenset(ACTION_NAME_TO_ID)


def _extract_json_object(raw_text: str) -> Mapping[str, Any]:
    cleaned_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE).strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        candidate = fenced_match.group(1)
    else:
        start = cleaned_text.find("{")
        end = cleaned_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Response did not contain a JSON object.")
        candidate = cleaned_text[start : end + 1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON payload must be an object.")
    return parsed


def sanitize_action_mapping(
    parsed: Mapping[str, Any],
    drone_ids: Sequence[str],
) -> Dict[str, str]:
    sanitized: Dict[str, str] = {}
    for drone_id in drone_ids:
        raw_value = parsed.get(drone_id, "STAY")
        action_name = str(raw_value).strip().upper()
        sanitized[drone_id] = action_name if action_name in VALID_ACTION_NAMES else "STAY"
    return sanitized


def parse_action_json(raw_text: str, drone_ids: Sequence[str]) -> Dict[str, str]:
    parsed = _extract_json_object(raw_text)
    return sanitize_action_mapping(parsed, drone_ids)


def parse_target_json(raw_text: str, drone_ids: Sequence[str], grid_size: int) -> Dict[str, Coord]:
    parsed = _extract_json_object(raw_text)
    targets: Dict[str, Coord] = {}
    for drone_id in drone_ids:
        value = parsed.get(drone_id)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Invalid target format for {drone_id}: {value}.")
        x = max(0, min(grid_size - 1, int(value[0])))
        y = max(0, min(grid_size - 1, int(value[1])))
        targets[drone_id] = (x, y)
    return targets
