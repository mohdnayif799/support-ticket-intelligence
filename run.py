#!/usr/bin/env python3
"""Single-command entrypoint.

    python run.py

Equivalent to `uvicorn app.api.main:app`. Starts the REST API and serves the UI
at http://127.0.0.1:8000 from the same process.
"""

from __future__ import annotations

import sys

import uvicorn

from app.config import get_settings


def main() -> int:
    settings = get_settings()
    if not settings.csv_path.exists():
        print(f"Dataset not found at {settings.csv_path}", file=sys.stderr)
        print("Place support_tickets.csv there or set CSV_PATH.", file=sys.stderr)
        return 1

    print(f"UI and API on http://{settings.api_host}:{settings.api_port}")
    print(f"Interactive API docs at http://{settings.api_host}:{settings.api_port}/docs")
    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
