from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import uvicorn
from openenv.core.env_server import create_app
from starlette.responses import HTMLResponse

from ..models import DisasterObservation, DroneActions
from .disaster_surveillance_environment import DisasterSurveillanceEnvironment


app = create_app(
    DisasterSurveillanceEnvironment,
    DroneActions,
    DisasterObservation,
    env_name="disaster_surveillance_env",
    max_concurrent_envs=32,
)

README_PATH = Path(__file__).resolve().parents[2] / "README.md"


@app.get("/", include_in_schema=False)
def home() -> HTMLResponse:
    if README_PATH.exists():
        readme = README_PATH.read_text(encoding="utf-8", errors="replace")
    else:
        readme = "README.md not found."
    return HTMLResponse(
        "<html><head><title>MADIS OpenEnv</title></head>"
        "<body style='margin:24px;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;'>"
        "<h2>MADIS OpenEnv</h2>"
        "<pre style='white-space:pre-wrap;word-wrap:break-word;background:#f8fafc;padding:16px;border-radius:10px;border:1px solid #e5e7eb;'>"
        + escape(readme)
        + "</pre></body></html>"
    )


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
