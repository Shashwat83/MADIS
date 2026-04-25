from __future__ import annotations

import argparse

import uvicorn
from openenv.core.env_server import create_app

from ..models import DisasterObservation, DroneActions
from .disaster_surveillance_environment import DisasterSurveillanceEnvironment


app = create_app(
    DisasterSurveillanceEnvironment,
    DroneActions,
    DisasterObservation,
    env_name="disaster_surveillance_env",
    max_concurrent_envs=32,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
