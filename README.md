# Disaster Surveillance OpenEnv

This repository now follows the standard OpenEnv environment layout instead of keeping
the implementation in a single file.

## Environment Level

Current setup is `Level 6 (Coordinator-Driven Planning)`:

- 3 independent RL drone agents
- an inference-only coordinator LLM assigns high-level target coordinates to drones
- drones execute one step toward their assigned targets each timestep
- reward still includes event detection, severity-aware miss penalties, timestep, FOV overlap, team coverage, and delayed episode-end bonuses
- Level 3 baseline mode is also available and keeps only the basic shared reward
- Level 4 remains available with overlap and coverage shaping
- Level 5 adds severity, deadlines, hotspot-biased spawning, delayed rewards, and urgency-sensitive prioritization
- Level 6 adds a centralized coordinator interface and an inference-only small-model LLM coordinator with heuristic fallback
- team-level FOV coverage is tracked without double counting overlapping visible cells
- no communication between agents
- coordination now comes from explicit target assignment, while the reward structure still pushes urgent, high-value event handling

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

Level 6 uses the coordinator model `Qwen/Qwen3-1.7B` by default through Hugging Face inference when available. If inference is unavailable locally, the env falls back to the heuristic coordinator and logs that fallback in metrics.

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

Train the local Qwen coordinator with optional SFT warmup and GRPO on Colab:

```bash
pip install -U "transformers>=4.45.0" "trl>=0.14.0" peft accelerate bitsandbytes datasets
pip install -e ".[llm]"
python3 scripts/train_grpo_coordinator.py \
  --model Qwen/Qwen3-1.7B \
  --output-dir outputs/qwen3_grpo_coordinator \
  --num-prompts 1024 \
  --episode-length 10 \
  --sft-steps 100 \
  --grpo-steps 300 \
  --num-generations 2
```

Evaluate the trained LoRA coordinator locally:

```bash
export USE_LOCAL_QWEN=true
export HF_COORDINATOR_MODEL=Qwen/Qwen3-1.7B
export LOCAL_QWEN_ADAPTER_PATH=outputs/qwen3_grpo_coordinator/grpo_lora
python3 scripts/analyze_baseline.py --level 6 --episodes 400
```

Run tests:

```bash
pytest
```
