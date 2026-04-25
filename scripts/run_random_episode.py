from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disaster_surveillance_env.server.disaster_surveillance_environment import run_random_episode


if __name__ == "__main__":
    run_random_episode(seed=42, render=False)
