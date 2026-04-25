# Disaster Surveillance OpenEnv

This repository now follows the standard OpenEnv environment layout instead of keeping
the implementation in a single file.

## Environment Level

Current setup is `Level 3 (Decentralized RL Begins)`:

- 3 independent RL drone agents
- each drone learns where to move and how to explore
- shared global reward
- no communication between agents
- per-drone observations only expose local information plus shared team metrics

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
python3 scripts/run_random_episode.py
```
