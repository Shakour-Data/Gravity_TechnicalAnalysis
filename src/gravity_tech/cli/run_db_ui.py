"""
Spin up the Gravity Technical Analysis API (DB UI included) with a single command.

Usage (from repo root):
    python -m gravity_tech.cli.run_db_ui --port 8000 --reload
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn


def _ensure_src_on_path() -> None:
    """Ensure `src` is on sys.path so `gravity_tech.main` imports cleanly."""
    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Gravity Technical Analysis API (includes DB UI at /api/v1/db/ui)."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for local development"
    )
    parser.add_argument(
        "--log-level", default="info", choices=["critical", "error", "warning", "info", "debug", "trace"]
    )
    args = parser.parse_args()

    _ensure_src_on_path()

    uvicorn.run(
        "gravity_tech.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
