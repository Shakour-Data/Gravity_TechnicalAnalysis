#!/usr/bin/env python3
"""
Aggregate tool performance stats into materialized table.

Usage:
    python -m scripts.maintenance.aggregate_tool_performance --hours 24

This script loads tool_performance_history, aggregates by (tool, regime,
 time frame) over the requested window, and upserts into tool_performance_stats.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root and src are on path
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from gravity_tech.database.database_manager import DatabaseManager  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate tool performance stats")
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Lookback window in hours (default: 24)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manager = DatabaseManager(auto_setup=True)
    count = manager.aggregate_tool_performance_stats(period_hours=args.hours, now=datetime.utcnow())
    manager.close()
    print(f"Aggregated {count} tool performance buckets over last {args.hours}h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
