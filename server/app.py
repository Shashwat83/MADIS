from __future__ import annotations

import uvicorn

from disaster_surveillance_env.server.app import app as _app

app = _app


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)


__all__ = ["app", "main"]


if __name__ == '__main__':
    main()
