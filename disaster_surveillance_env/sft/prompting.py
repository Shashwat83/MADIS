from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


ACTION_OUTPUT_MODE = "actions"
TARGET_OUTPUT_MODE = "targets"


def _summarize_hotspots(hotspots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "center": tuple(hotspot["center"]),
            "radius": int(hotspot["radius"]),
        }
        for hotspot in hotspots
    ]


def build_coordinator_prompt(
    observation: Mapping[str, Any],
    *,
    episode_length: int | None = None,
    hotspots: Sequence[Mapping[str, Any]] = (),
    output_mode: str = TARGET_OUTPUT_MODE,
    include_hidden_state: bool = False,
) -> str:
    """Build a stable text prompt from the coordinator observation.

    By default this uses only the public coordinator observation. Oracle pipelines
    can attach extra privileged fields manually and set include_hidden_state=True
    in metadata outside the prompt.
    """

    if output_mode not in {ACTION_OUTPUT_MODE, TARGET_OUTPUT_MODE}:
        raise ValueError(f"Unsupported output mode '{output_mode}'.")

    timestep = int(observation["timestep"])
    grid_size = int(observation["grid_size"])
    header = [
        "You are a disaster-response drone coordinator.",
        f"Grid size: {grid_size}x{grid_size}",
        (
            f"Time step: {timestep}/{episode_length}"
            if episode_length is not None
            else f"Time step: {timestep}"
        ),
        "",
        "Drone positions:",
    ]

    for drone_id, position in observation["drone_positions"].items():
        header.append(f"- {drone_id}: {tuple(position)}")

    visible_events = list(observation.get("visible_active_events", []))
    header.append("")
    header.append("Visible active events:")
    if visible_events:
        for event in visible_events:
            header.append(
                "- id={id} severity={severity} location={location} time_remaining={time_remaining} deadline_remaining={deadline_remaining}".format(
                    id=event["id"],
                    severity=event["severity"],
                    location=tuple(event["location"]),
                    time_remaining=event["time_remaining"],
                    deadline_remaining=event["deadline_remaining"],
                )
            )
    else:
        header.append("- none")

    known_detected = list(observation.get("known_detected_events", []))
    header.append("")
    header.append("Known detected events:")
    if known_detected:
        for event in known_detected:
            header.append(
                "- id={id} severity={severity} location={location} detected_at={detected_at}".format(
                    id=event["id"],
                    severity=event["severity"],
                    location=tuple(event["location"]),
                    detected_at=event["detected_at"],
                )
            )
    else:
        header.append("- none")

    header.append("")
    header.append(f"Team frontier cells: {json.dumps(list(observation.get('team_frontier_cells', []))[:20])}")
    header.append(
        f"Recently observed cells: {json.dumps(list(observation.get('recently_observed_cells', []))[:25])}"
    )
    header.append(
        "Recent team coverage ratio: {:.2f}".format(float(observation.get("recent_team_coverage_ratio", 0.0)))
    )

    if hotspots:
        header.append("")
        header.append(f"Hotspots: {json.dumps(_summarize_hotspots(hotspots), separators=(',', ':'))}")

    header.append("")
    if output_mode == ACTION_OUTPUT_MODE:
        header.extend(
            [
                "Choose one action for each drone from [UP, DOWN, LEFT, RIGHT, STAY].",
                "Return JSON only.",
                "Return exactly one valid JSON object and nothing else.",
                'JSON format: {"drone_1":"UP","drone_2":"STAY","drone_3":"LEFT"}',
            ]
        )
    else:
        header.extend(
            [
                "Assign one target grid cell to each drone.",
                "Prioritize HIGH severity events, then MEDIUM, then LOW.",
                "Avoid assigning the same target to multiple drones unless necessary.",
                "Return JSON only.",
                "Return exactly one valid JSON object and nothing else.",
                'JSON format: {"drone_1":[x,y],"drone_2":[x,y],"drone_3":[x,y]}',
            ]
        )

    if include_hidden_state:
        header.append("Note: this is an oracle supervision example.")

    return "\n".join(header)
