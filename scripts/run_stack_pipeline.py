#!/usr/bin/env python3
"""
Unified pipeline runner for the Gravity stack.

Steps performed (unless explicitly skipped):
  1. Run the TSE ingestion CLI (services/data_ingestion) to refresh SQLite caches.
  2. Ensure the PostgreSQL schemas/tables exist.
  3. Migrate the refreshed SQLite data into the shared Postgres database.
  4. Execute the historical analysis batch and persist scores.

Usage examples:
  python scripts/run_stack_pipeline.py --mode init --pg-dsn postgresql://gravity:gravity@localhost:5544/tech_analysis
  python scripts/run_stack_pipeline.py --mode daily --pg-dsn postgresql://... --analysis-limit 80 --lookback-days 365
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import psycopg

from _paths import ANALYSIS_SRC, REPO_ROOT

ROOT = REPO_ROOT
INGESTION_DIR = ROOT / "services" / "data_ingestion"
INGESTION_DB = INGESTION_DIR / "data" / "tse_data.db"
SQL_SCHEMA_FILE = ROOT / "scripts" / "postgres_schema.sql"
PYTHON_BIN = os.environ.get("PYTHON", sys.executable)


def run_cmd(args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    where = f" (cwd={cwd})" if cwd else ""
    print(f"[pipeline] ➤ Running: {' '.join(args)}{where}")
    subprocess.run(args, check=True, cwd=cwd, env=env)


def ensure_ingestion(mode: str) -> None:
    if mode == "init":
        steps = [
            ["create-db"],
            ["create-indices-tables"],
            ["load-initial"],
            ["load-all-prices"],
        ]
    else:
        steps = [["load-all-prices"]]

    for step in steps:
        run_cmd([PYTHON_BIN, "main.py", *step], cwd=INGESTION_DIR)


def ensure_postgres_schema(pg_dsn: str) -> None:
    if not SQL_SCHEMA_FILE.exists():
        raise FileNotFoundError(f"Schema file not found: {SQL_SCHEMA_FILE}")
    print("[pipeline] ➤ Ensuring postgres schema objects exist ...")
    sql_text = SQL_SCHEMA_FILE.read_text(encoding="utf-8")
    statements: list[str] = []
    buff: list[str] = []
    for raw_line in sql_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("--"):
            continue
        buff.append(raw_line)
        if line.endswith(";"):
            statements.append("\n".join(buff))
            buff = []
    if buff:
        statements.append("\n".join(buff))

    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
        conn.commit()


def migrate_sqlite(pg_dsn: str) -> None:
    if not INGESTION_DB.exists():
        raise FileNotFoundError(
            dedent(
                f"""
                TSE SQLite database not found at {INGESTION_DB}.
                Run the ingestion CLI first (services/data_ingestion/main.py).
                """
            ).strip()
        )
    run_cmd(
        [
            PYTHON_BIN,
            "scripts/migrate_sqlite_to_pg.py",
            "--tse-sqlite",
            str(INGESTION_DB),
            "--pg-dsn",
            pg_dsn,
        ],
        cwd=ROOT,
    )


def run_analysis(pg_dsn: str, limit: int, lookback_days: int) -> None:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    path_entries = [str(ROOT), str(ANALYSIS_SRC)]
    if existing:
        path_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(path_entries)
    run_cmd(
        [
            PYTHON_BIN,
            "scripts/run_full_batch_analysis.py",
            "--pg-dsn",
            pg_dsn,
            "--limit",
            str(limit),
            "--lookback-days",
            str(lookback_days),
        ],
        cwd=ROOT,
        env=env,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ingestion + migration + analysis in one shot.")
    parser.add_argument("--pg-dsn", required=True, help="Postgres DSN, e.g. postgresql://user:pass@host:5432/db")
    parser.add_argument("--mode", choices=["init", "daily"], default="daily", help="Init = full bootstrap, daily = incremental.")
    parser.add_argument("--analysis-limit", type=int, default=80, help="Max symbols to process during analysis.")
    parser.add_argument("--lookback-days", type=int, default=365, help="Lookback horizon for candles.")
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip calling the ingestion CLI.")
    parser.add_argument("--skip-migration", action="store_true", help="Skip copying SQLite output into Postgres.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip the historical analysis batch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_ingestion:
        ensure_ingestion(args.mode)
    else:
        print("[pipeline] ⚠ Skipping ingestion as requested.")

    ensure_postgres_schema(args.pg_dsn)

    if not args.skip_migration:
        migrate_sqlite(args.pg_dsn)
    else:
        print("[pipeline] ⚠ Skipping SQLite -> Postgres migration as requested.")

    if not args.skip_analysis:
        run_analysis(args.pg_dsn, args.analysis_limit, args.lookback_days)
    else:
        print("[pipeline] ⚠ Skipping analytical batch as requested.")


if __name__ == "__main__":
    main()
