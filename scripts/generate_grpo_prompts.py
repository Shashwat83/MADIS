from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disaster_surveillance_env.grpo.dataset import export_grpo_prompt_jsonl, iter_grpo_prompt_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prompt-only GRPO datasets from MADIS environment states.")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    written = export_grpo_prompt_jsonl(
        args.output,
        iter_grpo_prompt_examples(
            episodes=args.episodes,
            seed=args.seed,
            episode_length=args.episode_length,
        ),
    )
    print(f"Wrote {written} GRPO prompt examples to {args.output}")


if __name__ == "__main__":
    main()
