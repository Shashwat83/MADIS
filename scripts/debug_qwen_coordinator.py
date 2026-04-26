from __future__ import annotations

import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disaster_surveillance_env.coordinator import DEFAULT_REMOTE_LLM, LLMCoordinator, get_configured_model_name


def main() -> None:
    print("Hosted coordinator environment check:")
    print(f"  DEFAULT_REMOTE_LLM: {DEFAULT_REMOTE_LLM}")
    print(f"  HF_COORDINATOR_MODEL: {os.environ.get('HF_COORDINATOR_MODEL')}")
    print(f"  MODEL_NAME: {os.environ.get('MODEL_NAME')}")
    print(f"  configured_model: {get_configured_model_name()}")
    print(f"  API_BASE_URL: {os.environ.get('API_BASE_URL')}")
    print(f"  HF_TOKEN_loaded: {bool(os.environ.get('HF_TOKEN'))}")

    observation = {
        "timestep": 0,
        "grid_size": 10,
        "drone_positions": {
            "drone_1": (0, 7),
            "drone_2": (6, 4),
            "drone_3": (4, 8),
        },
        "visible_active_events": [
            {
                "id": 1,
                "location": (7, 7),
                "severity": "HIGH",
                "time_remaining": 8,
                "deadline_remaining": 3,
            }
        ],
        "team_frontier_cells": [(1, 1), (8, 1), (5, 8), (7, 7), (2, 2)],
        "recent_team_coverage_ratio": 0.34,
        "known_detected_events": [],
    }
    coordinator = LLMCoordinator()
    decision = coordinator.decide(observation)
    print("targets:", decision.targets)
    print("metadata:")
    for key, value in decision.metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
