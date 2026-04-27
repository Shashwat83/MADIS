---
title: MADIS OpenEnv
emoji: 🚁
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# MADIS
#
Disaster Surveillance OpenEnv

## Walkthrough video

Project overview and demo: `https://youtu.be/VK2XK-aKVHU?si=9HLUOIuAXnI1kJ_d`

This repository now follows the standard OpenEnv environment layout instead of keeping
the implementation in a single file.

## Environment levels

This repo implements a **multi-agent** drone surveillance environment with multiple “levels” of increasing difficulty/realism:

- **Level 3**: baseline decentralized multi-agent RL interface with a shared global reward.
- **Level 4**: adds overlap and coverage shaping.
- **Level 5**: adds urgency/deadlines, severity-aware rewards/penalties, delayed bonuses, hotspot-biased spawning, and a growth penalty for undetected active disasters.
- **Level 6**: introduces a **central coordinator policy** that assigns target coordinates to 3 drones (LLM coordinator with heuristic fallback).
- **Level 9**: adversarial setting where a scripted adversary injects **false-positive incident reports** with credibility; the coordinator must prioritize confirmed visible events and avoid wasting steps chasing decoys (false-report penalty is applied when drones are assigned to false reports).

Core mechanics:
- **3 drones** move on a 10x10 grid with partial observability (FOV).
- **Dynamic disasters** can escalate/spread (e.g. riot/fire/gas leak/flood zone behaviors).
- Episode metrics include total reward, high-priority miss rate, on-time detection rate, coverage, and path efficiency.

## Structure

- `disaster_surveillance_env/`
  - `models.py`: shared action, observation, state, and simulation helpers
  - `client.py`: typed OpenEnv client
  - `server/`: server-side environment implementation and FastAPI app
- `scripts/run_random_episode.py`: local smoke-test runner
- `openenv.yaml`: OpenEnv manifest
- `pyproject.toml`: package metadata and dependencies
- `outputs/`: runtime logs/evals directory

## Local Usage

Install the project:

```bash
pip install -e .
```

Run the server locally:

```bash
python -m disaster_surveillance_env.server.app --port 8000
```

Run the local random rollout:

```bash
python3 scripts/run_random_episode.py --level 6 --episodes 1
```

Try the adversarial variant:

```bash
python3 scripts/run_random_episode.py --level 9 --episodes 1 --render
```

Level 6/9 can use a coordinator model (default `Qwen/Qwen3-1.7B`) via Hugging Face router when available. If inference is unavailable locally, the env falls back to the heuristic coordinator and logs that fallback in metrics.

## Training + evaluation utilities

- **SFT**: `scripts/train_sft_coordinator.py`
- **GRPO**: `scripts/train_grpo_coordinator.py`
- **Compare base vs SFT vs SFT+GRPO** (fixed seeds, summary + plots): `scripts/compare_coordinator_variants.py`

## Hugging Face authentication (Colab + local)

Many training and inference paths download models from the Hugging Face Hub. For best performance, set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) so requests are authenticated and not rate-limited.

### Colab

Add a secret named `HF_TOKEN` in the Colab UI (Settings → Secrets), then run:

```python
import os

try:
    from google.colab import userdata  # type: ignore
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    pass
```

If you see messages about vault timeouts, it usually means the secret is not available to the runtime from the Colab UI.

### GPU dtype tip (T4)

On T4-class GPUs, prefer `--fp16` (and optionally `--use-4bit`) for SFT/GRPO training. `--bf16` often provides no benefit on T4.

### GRPO can look “stuck”

`scripts/train_grpo_coordinator.py` runs periodic evaluations that execute full environment rollouts. For quick iterations, keep `--config-preset pilot` and reduce evaluation frequency/size with:

- `--small-eval-every-episodes`, `--small-eval-episodes`
- `--medium-eval-every-episodes`, `--medium-eval-episodes`
- `--full-eval-every-episodes`, `--full-eval-episodes`

Run the Level 3 baseline:

```bash
python3 scripts/run_random_episode.py --level 3
```

Run `k` episodes:

```bash
python3 scripts/run_random_episode.py --level 6 --episodes 10
```

Run the 400-episode baseline analysis and cache CSV/SVG plot data:

```bash
python3 scripts/analyze_baseline.py --episodes 400
```

Run tests:

```bash
pytest
```
