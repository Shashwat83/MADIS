# Disaster Surveillance OpenEnv

This repository now follows the standard OpenEnv environment layout instead of keeping
the implementation in a single file.

## Environment Level

Current setup is `Level 5 (Long-Horizon Complexity)`:

- 3 independent RL drone agents
- each drone receives local observations and a shared global reward
- reward includes event detection, severity-aware miss penalties, timestep, FOV overlap, team coverage, and delayed episode-end bonuses
- Level 3 baseline mode is also available and keeps only the basic shared reward
- Level 4 remains available with overlap and coverage shaping
- Level 5 adds severity, deadlines, hotspot-biased spawning, delayed rewards, and urgency-sensitive prioritization
- team-level FOV coverage is tracked without double counting overlapping visible cells
- no communication between agents
- no centralized controller
- coordination emerges because drones are penalized for overlapping FOVs, rewarded for covering new cells, and strongly incentivized to rescue urgent high-severity events on time

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
python3 scripts/run_random_episode.py --level 5 --episodes 1
```

Run the Level 3 baseline:

```bash
python3 scripts/run_random_episode.py --level 3
```

Run `k` episodes:

```bash
python3 scripts/run_random_episode.py --level 5 --episodes 10
```

Run tests:

```bash
pytest
```
