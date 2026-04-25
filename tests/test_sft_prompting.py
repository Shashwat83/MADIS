from __future__ import annotations

from disaster_surveillance_env.models import HOTSPOTS
from disaster_surveillance_env.sft.prompting import ACTION_OUTPUT_MODE, TARGET_OUTPUT_MODE, build_coordinator_prompt


def test_build_coordinator_prompt_uses_public_observation_fields_only() -> None:
    prompt = build_coordinator_prompt(
        {
            "timestep": 4,
            "grid_size": 10,
            "drone_positions": {"drone_1": (0, 0), "drone_2": (1, 1), "drone_3": (2, 2)},
            "visible_active_events": [{"id": 7, "severity": "HIGH", "location": (4, 4), "time_remaining": 3, "deadline_remaining": 1}],
            "known_detected_events": [],
            "team_frontier_cells": [(1, 2), (2, 3)],
            "recently_observed_cells": [(0, 0), (0, 1)],
            "recent_team_coverage_ratio": 0.42,
            "hidden_active_events": [{"id": 999}],
        },
        episode_length=50,
        hotspots=HOTSPOTS,
        output_mode=TARGET_OUTPUT_MODE,
    )

    assert "hidden_active_events" not in prompt
    assert "Grid size: 10x10" in prompt
    assert '"drone_1":[x,y]' in prompt


def test_build_coordinator_prompt_supports_action_output_mode() -> None:
    prompt = build_coordinator_prompt(
        {
            "timestep": 0,
            "grid_size": 10,
            "drone_positions": {"drone_1": (0, 0), "drone_2": (1, 1), "drone_3": (2, 2)},
            "visible_active_events": [],
            "known_detected_events": [],
            "team_frontier_cells": [],
            "recently_observed_cells": [],
            "recent_team_coverage_ratio": 0.0,
        },
        output_mode=ACTION_OUTPUT_MODE,
    )

    assert '[UP, DOWN, LEFT, RIGHT, STAY]' in prompt
    assert '"drone_1":"UP"' in prompt
