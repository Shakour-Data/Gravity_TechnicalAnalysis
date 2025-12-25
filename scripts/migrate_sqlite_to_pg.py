"""
Thin wrapper to reuse the existing TSE SQLite -> Postgres migration helper.

This script matches the interface expected by `scripts/etl/run_stack_pipeline.py`
(`--tse-sqlite` and `--pg-dsn`) while delegating the actual work to
`scripts/migrate_tse_sqlite_to_pg.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from migrate_tse_sqlite_to_pg import main as migrate_tse_to_pg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate TSE SQLite data into Postgres.")
    parser.add_argument(
        "--tse-sqlite",
        default="services/data_ingestion/data/tse_data.db",
        help="Path to the source TSE SQLite DB.",
    )
    parser.add_argument(
        "--pg-dsn",
        required=True,
        help="Postgres DSN, e.g. postgresql://gravity:gravity_db_pass@localhost:5545/tech_analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    migrate_tse_to_pg(Path(args.tse_sqlite), args.pg_dsn)


if __name__ == "__main__":
    main()
