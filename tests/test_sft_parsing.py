from __future__ import annotations

from disaster_surveillance_env.sft.parsing import parse_action_json, parse_target_json


def test_parse_action_json_replaces_invalid_actions_with_stay() -> None:
    parsed = parse_action_json(
        '{"drone_1":"UP","drone_2":"invalid","drone_3":"LEFT"}',
        ["drone_1", "drone_2", "drone_3"],
    )

    assert parsed == {"drone_1": "UP", "drone_2": "STAY", "drone_3": "LEFT"}


def test_parse_target_json_clips_out_of_bounds_coordinates() -> None:
    parsed = parse_target_json(
        '{"drone_1":[-4,12],"drone_2":[3,4],"drone_3":[9,9]}',
        ["drone_1", "drone_2", "drone_3"],
        grid_size=10,
    )

    assert parsed["drone_1"] == (0, 9)
    assert parsed["drone_2"] == (3, 4)
