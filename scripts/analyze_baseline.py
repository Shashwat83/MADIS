from __future__ import annotations

import argparse
import csv
from html import escape
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disaster_surveillance_env.coordinator import get_configured_model_name
from disaster_surveillance_env.models import DroneActions
from disaster_surveillance_env.server.disaster_surveillance_environment import DisasterSurveillanceEnvironment


REWARD_BREAKDOWN_KEYS = (
    "time_penalty",
    "detection_reward",
    "miss_penalty",
    "overlap_penalty",
    "coverage_reward",
    "episode_bonus",
)


def run_analysis_episode(seed: int, level: int = 6, episode_length: int | None = None) -> Dict[str, Any]:
    env_kwargs: Dict[str, Any] = {"seed": seed, "level": level}
    if episode_length is not None:
        env_kwargs["episode_length"] = episode_length
    env = DisasterSurveillanceEnvironment(**env_kwargs)
    observation = env.reset(seed=seed)

    while not observation.done:
        if level == 6:
            observation = env.step(None)
        else:
            actions = {
                agent_id: int(env.rng.integers(0, 5))
                for agent_id in env.agent_ids
            }
            observation = env.step(DroneActions(actions=actions))

    metrics = {key: value for key, value in env.metrics.items() if not key.startswith("_")}
    total_steps = sum(drone.steps_taken for drone in env.drones.values())
    useful_cells = float(metrics.get("unique_cells_visited", 0))
    metrics["derived_path_efficiency"] = (
        float(metrics.get("path_efficiency", 0.0))
        if level == 6
        else useful_cells / float(total_steps or 1)
    )

    target_assignments = int(metrics.get("target_assignment_count", 0))
    fallback_count = int(metrics.get("coordinator_fallback_count", 0))
    metrics["derived_fallback_rate"] = fallback_count / float(target_assignments or 1)
    metrics["model_name"] = metrics.get("coordinator_model_name") or get_configured_model_name()
    metrics["policy_type"] = "qwen_llm_coordinator" if level == 6 else "baseline_random"
    return metrics


def run_episodes(episodes: int, seed: int, level: int, episode_length: int | None = None) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    started_at = time.perf_counter()
    for index in range(episodes):
        episode = index + 1
        episode_started_at = time.perf_counter()
        try:
            metrics = run_analysis_episode(seed=seed + index, level=level, episode_length=episode_length)
        except Exception as exc:
            elapsed = time.perf_counter() - started_at
            print(
                f"[ERROR] analysis failed at episode={episode}/{episodes} "
                f"after {elapsed:.1f}s: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

        metrics["episode"] = episode
        metrics["seed"] = seed + index
        results.append(metrics)
        if episode == 1 or episode % 5 == 0 or episode == episodes:
            elapsed = time.perf_counter() - started_at
            avg_episode_time = elapsed / float(episode)
            remaining = max(0, episodes - episode) * avg_episode_time
            episode_time = time.perf_counter() - episode_started_at
            print(
                "episode={episode}/{episodes} episode_time={episode_time:.1f}s elapsed={elapsed:.1f}s eta={eta:.1f}s "
                "reward={reward:.1f} coverage={coverage:.1f}% detected={detected} missed={missed} "
                "high_miss={high_miss:.2f} fallback_rate={fallback:.2f}".format(
                    episode=episode,
                    episodes=episodes,
                    episode_time=episode_time,
                    elapsed=elapsed,
                    eta=remaining,
                    reward=metrics["total_reward"],
                    coverage=metrics["grid_coverage_percent"],
                    detected=metrics["events_detected"],
                    missed=metrics["events_missed"],
                    high_miss=metrics["high_priority_miss_rate"],
                    fallback=metrics["derived_fallback_rate"],
                )
            )
    return results


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def moving_average(values: Sequence[float], window: int = 25) -> List[float]:
    if len(values) < window:
        return [float(value) for value in values]
    smoothed: List[float] = []
    for index in range(len(values)):
        if index < window - 1:
            smoothed.append(float(values[index]))
            continue
        chunk = values[index - window + 1 : index + 1]
        smoothed.append(sum(float(value) for value in chunk) / float(window))
    return smoothed


def _nice_bounds(values: Iterable[float]) -> tuple[float, float]:
    data = [float(value) for value in values]
    if not data:
        return 0.0, 1.0
    low = min(data)
    high = max(data)
    if low == high:
        padding = abs(low) * 0.1 or 1.0
        return low - padding, high + padding
    padding = (high - low) * 0.08
    return low - padding, high + padding


def _polyline(points: Sequence[tuple[float, float]], color: str, width: float, opacity: float = 1.0) -> str:
    point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline points="{point_text}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" opacity="{opacity}" stroke-linejoin="round" stroke-linecap="round" />'
    )


def _svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="#111827">{escape(text)}</text>'
    )


def episode_axis_ticks(episodes: Sequence[int], max_ticks: int = 12) -> List[int]:
    ticks = list(dict.fromkeys(int(episode) for episode in episodes))
    if len(ticks) <= max_ticks:
        return ticks

    step = max(1, round(len(ticks) / float(max_ticks - 1)))
    sampled = ticks[::step]
    if ticks[-1] not in sampled:
        sampled.append(ticks[-1])
    return sampled


def save_line_plot(
    path: Path,
    episodes: Sequence[int],
    series: Mapping[str, Sequence[float]],
    title: str,
    ylabel: str,
    smooth: bool = True,
) -> None:
    width, height = 1000, 520
    left, right, top, bottom = 76, 32, 58, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#ea580c"]
    all_values: List[float] = []
    for values in series.values():
        all_values.extend(float(value) for value in values)
        if smooth:
            all_values.extend(moving_average(values))
    y_min, y_max = _nice_bounds(all_values)
    x_min, x_max = min(episodes), max(episodes)
    x_span = max(1, x_max - x_min)
    y_span = y_max - y_min

    def sx(episode: float) -> float:
        return left + ((episode - x_min) / x_span) * plot_w

    def sy(value: float) -> float:
        return top + (1.0 - ((value - y_min) / y_span)) * plot_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        _svg_text(width / 2, 28, title, size=20, anchor="middle", weight="700"),
        _svg_text(width / 2, height - 18, "Episode", size=13, anchor="middle"),
        _svg_text(18, height / 2, ylabel, size=13, anchor="middle"),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f9fafb" stroke="#d1d5db" />',
    ]
    for tick in range(6):
        ratio = tick / 5
        y = top + ratio * plot_h
        value = y_max - ratio * y_span
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb" />')
        elements.append(_svg_text(left - 10, y + 4, f"{value:.2f}", size=11, anchor="end"))
    for value in episode_axis_ticks(episodes):
        x = sx(float(value))
        elements.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#f3f4f6" />')
        elements.append(_svg_text(x, top + plot_h + 22, str(value), size=11, anchor="middle"))

    legend_x = left + 12
    legend_y = top + 18
    for index, (label, values) in enumerate(series.items()):
        color = colors[index % len(colors)]
        points = [(sx(float(ep)), sy(float(value))) for ep, value in zip(episodes, values)]
        elements.append(_polyline(points, color=color, width=1.2, opacity=0.35 if smooth else 0.9))
        legend_label = label
        if smooth:
            smooth_values = moving_average(values)
            smooth_points = [(sx(float(ep)), sy(float(value))) for ep, value in zip(episodes, smooth_values)]
            elements.append(_polyline(smooth_points, color=color, width=2.4))
            legend_label = f"{label} + 25-ep MA"
        y = legend_y + index * 20
        elements.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 26}" y2="{y}" stroke="{color}" stroke-width="3" />')
        elements.append(_svg_text(legend_x + 34, y + 4, legend_label, size=12))

    elements.append("</svg>")
    path.write_text("\n".join(elements))


def save_reward_breakdown_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    width, height = 1200, 620
    left, right, top, bottom = 76, 180, 58, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    episodes = [int(row["episode"]) for row in rows]
    colors = {
        "time_penalty": "#6b7280",
        "detection_reward": "#16a34a",
        "miss_penalty": "#dc2626",
        "overlap_penalty": "#7c3aed",
        "coverage_reward": "#0284c7",
        "episode_bonus": "#f59e0b",
    }
    positive_totals = []
    negative_totals = []
    for row in rows:
        values = [float(row[key]) for key in REWARD_BREAKDOWN_KEYS]
        positive_totals.append(sum(value for value in values if value > 0))
        negative_totals.append(sum(value for value in values if value < 0))
    y_min, y_max = _nice_bounds([*positive_totals, *negative_totals, 0.0])
    y_span = y_max - y_min

    def sy(value: float) -> float:
        return top + (1.0 - ((value - y_min) / y_span)) * plot_h

    bar_gap = 1.0
    bar_w = max(1.0, plot_w / max(1, len(rows)) - bar_gap)
    zero_y = sy(0.0)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        _svg_text(width / 2, 28, "Reward Breakdown Stacked By Episode", size=20, anchor="middle", weight="700"),
        _svg_text((width - right + width) / 2, top, "Components", size=13, anchor="middle", weight="700"),
        _svg_text(width / 2, height - 18, "Episode", size=13, anchor="middle"),
        _svg_text(18, height / 2, "Reward Component Sum", size=13, anchor="middle"),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f9fafb" stroke="#d1d5db" />',
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + plot_w}" y2="{zero_y:.2f}" stroke="#111827" stroke-width="1" />',
    ]
    for tick in range(6):
        ratio = tick / 5
        y = top + ratio * plot_h
        value = y_max - ratio * y_span
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb" />')
        elements.append(_svg_text(left - 10, y + 4, f"{value:.1f}", size=11, anchor="end"))

    for row_index, row in enumerate(rows):
        x = left + row_index * (plot_w / max(1, len(rows))) + bar_gap / 2
        pos_base = 0.0
        neg_base = 0.0
        for key in REWARD_BREAKDOWN_KEYS:
            value = float(row[key])
            if value == 0.0:
                continue
            if value > 0:
                y_top = sy(pos_base + value)
                y_bottom = sy(pos_base)
                pos_base += value
            else:
                y_top = sy(neg_base)
                y_bottom = sy(neg_base + value)
                neg_base += value
            rect_y = min(y_top, y_bottom)
            rect_h = abs(y_bottom - y_top)
            elements.append(
                f'<rect x="{x:.2f}" y="{rect_y:.2f}" width="{bar_w:.2f}" height="{rect_h:.2f}" '
                f'fill="{colors[key]}" opacity="0.9" />'
            )
    x_min, x_max = min(episodes), max(episodes)
    x_span = max(1, x_max - x_min)
    for value in episode_axis_ticks(episodes):
        x = left + ((value - x_min) / x_span) * plot_w
        elements.append(_svg_text(x, top + plot_h + 22, str(value), size=11, anchor="middle"))

    for key in REWARD_BREAKDOWN_KEYS:
        legend_index = REWARD_BREAKDOWN_KEYS.index(key)
        y = top + 28 + legend_index * 24
        x = width - right + 24
        elements.append(f'<rect x="{x}" y="{y - 11}" width="14" height="14" fill="{colors[key]}" />')
        elements.append(_svg_text(x + 22, y, key, size=12))

    elements.append("</svg>")
    path.write_text("\n".join(elements))


def build_cache_rows(metrics: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    source_codes = {None: 0, "none": 0, "llm": 1, "heuristic": 2, "heuristic_fallback": 3, "external_targets": 4}
    rows: Dict[str, List[Dict[str, Any]]] = {
        "total_reward": [],
        "grid_coverage_percent": [],
        "events_detected_missed": [],
        "high_priority_miss_rate": [],
        "on_time_detection_rate": [],
        "reward_breakdown": [],
        "path_efficiency": [],
        "coordinator_fallback": [],
        "all_metrics": [],
    }

    for item in metrics:
        episode = int(item["episode"])
        rows["total_reward"].append({"episode": episode, "total_reward": item["total_reward"]})
        rows["grid_coverage_percent"].append(
            {"episode": episode, "grid_coverage_percent": item["grid_coverage_percent"]}
        )
        rows["events_detected_missed"].append(
            {
                "episode": episode,
                "events_detected": item["events_detected"],
                "events_missed": item["events_missed"],
            }
        )
        rows["high_priority_miss_rate"].append(
            {"episode": episode, "high_priority_miss_rate": item["high_priority_miss_rate"]}
        )
        rows["on_time_detection_rate"].append(
            {"episode": episode, "on_time_detection_rate": item["on_time_detection_rate"]}
        )
        breakdown = item.get("reward_breakdown", {})
        rows["reward_breakdown"].append(
            {
                "episode": episode,
                **{key: float(breakdown.get(key, 0.0)) for key in REWARD_BREAKDOWN_KEYS},
            }
        )
        rows["path_efficiency"].append(
            {"episode": episode, "path_efficiency": item["derived_path_efficiency"]}
        )
        source = item.get("coordinator_decision_source") or "none"
        rows["coordinator_fallback"].append(
            {
                "episode": episode,
                "coordinator_decision_source": source,
                "decision_source_code": source_codes.get(source, -1),
                "coordinator_fallback_count": item.get("coordinator_fallback_count", 0),
                "target_assignment_count": item.get("target_assignment_count", 0),
                "fallback_rate": item["derived_fallback_rate"],
                "last_llm_diagnosis": (item.get("last_llm_debug") or {}).get("diagnosis"),
            }
        )
        rows["all_metrics"].append(
            {
                "episode": episode,
                "seed": item["seed"],
                "model_name": item["model_name"],
                "policy_type": item["policy_type"],
                "level": item["level"],
                "total_reward": item["total_reward"],
                "grid_coverage_percent": item["grid_coverage_percent"],
                "events_detected": item["events_detected"],
                "events_missed": item["events_missed"],
                "high_priority_miss_rate": item["high_priority_miss_rate"],
                "on_time_detection_rate": item["on_time_detection_rate"],
                "path_efficiency": item["derived_path_efficiency"],
                "fallback_rate": item["derived_fallback_rate"],
                "coordinator_decision_source": source,
                "coordinator_fallback_count": item.get("coordinator_fallback_count", 0),
                "target_assignment_count": item.get("target_assignment_count", 0),
                "last_llm_diagnosis": (item.get("last_llm_debug") or {}).get("diagnosis"),
            }
        )
    return rows


def save_outputs(metrics: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    csv_dir = output_dir / "csv"
    plot_dir = output_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = build_cache_rows(metrics)
    write_csv(csv_dir / "total_reward.csv", rows["total_reward"], ["episode", "total_reward"])
    write_csv(
        csv_dir / "grid_coverage_percent.csv",
        rows["grid_coverage_percent"],
        ["episode", "grid_coverage_percent"],
    )
    write_csv(
        csv_dir / "events_detected_missed.csv",
        rows["events_detected_missed"],
        ["episode", "events_detected", "events_missed"],
    )
    write_csv(
        csv_dir / "high_priority_miss_rate.csv",
        rows["high_priority_miss_rate"],
        ["episode", "high_priority_miss_rate"],
    )
    write_csv(
        csv_dir / "on_time_detection_rate.csv",
        rows["on_time_detection_rate"],
        ["episode", "on_time_detection_rate"],
    )
    write_csv(csv_dir / "reward_breakdown.csv", rows["reward_breakdown"], ["episode", *REWARD_BREAKDOWN_KEYS])
    write_csv(csv_dir / "path_efficiency.csv", rows["path_efficiency"], ["episode", "path_efficiency"])
    write_csv(
        csv_dir / "coordinator_fallback.csv",
        rows["coordinator_fallback"],
        [
            "episode",
            "coordinator_decision_source",
            "decision_source_code",
            "coordinator_fallback_count",
            "target_assignment_count",
            "fallback_rate",
            "last_llm_diagnosis",
        ],
    )
    write_csv(
        csv_dir / "all_metrics.csv",
        rows["all_metrics"],
        [
            "episode",
            "seed",
            "model_name",
            "policy_type",
            "level",
            "total_reward",
            "grid_coverage_percent",
            "events_detected",
            "events_missed",
            "high_priority_miss_rate",
            "on_time_detection_rate",
            "path_efficiency",
            "fallback_rate",
            "coordinator_decision_source",
            "coordinator_fallback_count",
            "target_assignment_count",
            "last_llm_diagnosis",
        ],
    )

    episodes = [row["episode"] for row in rows["total_reward"]]
    save_line_plot(
        plot_dir / "total_reward.svg",
        episodes,
        {"total_reward": [row["total_reward"] for row in rows["total_reward"]]},
        "Total Reward vs Episode",
        "Total Reward",
    )
    save_line_plot(
        plot_dir / "grid_coverage_percent.svg",
        episodes,
        {"grid_coverage_percent": [row["grid_coverage_percent"] for row in rows["grid_coverage_percent"]]},
        "Grid Coverage Percent vs Episode",
        "Grid Coverage (%)",
    )
    save_line_plot(
        plot_dir / "events_detected_missed.svg",
        episodes,
        {
            "events_detected": [row["events_detected"] for row in rows["events_detected_missed"]],
            "events_missed": [row["events_missed"] for row in rows["events_detected_missed"]],
        },
        "Events Detected And Missed vs Episode",
        "Event Count",
    )
    save_line_plot(
        plot_dir / "high_priority_miss_rate.svg",
        episodes,
        {"high_priority_miss_rate": [row["high_priority_miss_rate"] for row in rows["high_priority_miss_rate"]]},
        "High Priority Miss Rate vs Episode",
        "Miss Rate",
    )
    save_line_plot(
        plot_dir / "on_time_detection_rate.svg",
        episodes,
        {"on_time_detection_rate": [row["on_time_detection_rate"] for row in rows["on_time_detection_rate"]]},
        "On-Time Detection Rate vs Episode",
        "On-Time Rate",
    )
    save_reward_breakdown_plot(plot_dir / "reward_breakdown_stacked.svg", rows["reward_breakdown"])
    save_line_plot(
        plot_dir / "path_efficiency.svg",
        episodes,
        {"path_efficiency": [row["path_efficiency"] for row in rows["path_efficiency"]]},
        "Path Efficiency vs Episode",
        "Unique FOV Cells / Drone Step",
    )
    save_line_plot(
        plot_dir / "coordinator_fallback_rate.svg",
        episodes,
        {"fallback_rate": [row["fallback_rate"] for row in rows["coordinator_fallback"]]},
        "Coordinator Fallback Rate vs Episode",
        "Fallback Rate",
        smooth=False,
    )


def default_output_dir(level: int, episodes: int) -> Path:
    if level == 6:
        return ROOT / "outputs" / "evals" / f"qwen3_1_7b_level6_{episodes}"
    return ROOT / "outputs" / "evals" / f"baseline_level{level}_{episodes}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rollout analysis and cache plot datapoints.")
    parser.add_argument("--episodes", "-k", type=int, default=4, help="Number of episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--level", type=int, default=6, choices=[3, 4, 5, 6], help="Environment level to evaluate.")
    parser.add_argument("--episode-length", type=int, default=None, help="Optional shorter episode length for debugging.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV caches and plots.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or default_output_dir(args.level, args.episodes)

    print(f"Configured model: {get_configured_model_name()}")
    print(f"Running Level {args.level} for {args.episodes} episodes.")
    metrics = run_episodes(
        episodes=args.episodes,
        seed=args.seed,
        level=args.level,
        episode_length=args.episode_length,
    )
    save_outputs(metrics, output_dir)
    print(f"\nSaved CSV caches to: {output_dir / 'csv'}")
    print(f"Saved plots to: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
