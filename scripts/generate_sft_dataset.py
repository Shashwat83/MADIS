from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disaster_surveillance_env.sft.dataset import export_jsonl, iter_sft_examples
from disaster_surveillance_env.sft.parsing import ACTION_OUTPUT_MODE, TARGET_OUTPUT_MODE
from disaster_surveillance_env.sft.policies import (
    ActionFormatTeacherPolicy,
    HeuristicTeacherPolicy,
    OracleTeacherPolicy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JSONL datasets for coordinator SFT.")
    parser.add_argument("--dataset-type", choices=["action_format", "heuristic", "oracle"], required=True)
    parser.add_argument("--output-mode", choices=[ACTION_OUTPUT_MODE, TARGET_OUTPUT_MODE], default=TARGET_OUTPUT_MODE)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.dataset_type == "action_format":
        teacher = ActionFormatTeacherPolicy()
    elif args.dataset_type == "heuristic":
        teacher = HeuristicTeacherPolicy()
    else:
        teacher = OracleTeacherPolicy()

    examples = iter_sft_examples(
        episodes=args.episodes,
        teacher_policy=teacher,
        dataset_type=args.dataset_type,
        output_mode=args.output_mode,
        seed=args.seed,
        episode_length=args.episode_length,
    )
    written = export_jsonl(args.output, examples)
    print(f"Wrote {written} examples to {args.output}")


if __name__ == "__main__":
    main()
