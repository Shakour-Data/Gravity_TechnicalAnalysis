"""
Clone tse_input tables from a source Postgres database into a target Postgres database.

This implementation streams each table via PostgreSQL COPY in binary mode to avoid
loading large tables (e.g., price_data) entirely into memory. Destination tables are
truncated by default before loading.

Usage example:
  python scripts/clone_tse_to_target.py ^
    --src-dsn postgresql://gravity:gravity_db_pass@127.0.0.1:5544/gravity_tse ^
    --dst-dsn postgresql://gravity:gravity_db_pass@127.0.0.1:5544/tech_analysis
"""

from __future__ import annotations

import argparse
import os
import tempfile
from typing import Iterable

import psycopg2


def copy_table_via_copy(conn_s, conn_d, tbl: str, truncate: bool = True) -> int:
    """
    Stream-copy a single table using COPY BINARY.

    Returns:
        int: number of rows in the destination after copy.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Export from source
        with conn_s.cursor() as cur_s, open(tmp_path, "wb") as f_out:
            cur_s.copy_expert(f"COPY tse_input.{tbl} TO STDOUT WITH BINARY", f_out)

        # Import into destination
        with conn_d.cursor() as cur_d, open(tmp_path, "rb") as f_in:
            if truncate:
                cur_d.execute(f"TRUNCATE tse_input.{tbl} RESTART IDENTITY CASCADE")
            cur_d.copy_expert(f"COPY tse_input.{tbl} FROM STDIN WITH BINARY", f_in)
            cur_d.execute(f"SELECT COUNT(*) FROM tse_input.{tbl}")
            rows = cur_d.fetchone()[0]
            conn_d.commit()

        print(f"{tbl}: copied {rows} rows")
        return int(rows)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone tse_input tables from source to target Postgres.")
    parser.add_argument("--src-dsn", required=True, help="Source DSN (has tse_input data)")
    parser.add_argument("--dst-dsn", required=True, help="Destination DSN")
    parser.add_argument(
        "--tables",
        help="Comma-separated table names to copy (default: all tse_input tables)",
        default=None,
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Do not truncate destination tables before copy (default: truncate).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    conn_s = psycopg2.connect(args.src_dsn)
    conn_d = psycopg2.connect(args.dst_dsn)

    default_tables: Iterable[str] = (
        "markets",
        "panels",
        "sectors",
        "companies",
        "indices_info",
        "last_updates",
        "price_data",
        "market_indices",
        "sector_indices",
    )
    tables = [t.strip() for t in args.tables.split(",")] if args.tables else default_tables

    for t in tables:
        copy_table_via_copy(conn_s, conn_d, t, truncate=not args.no_truncate)

    conn_s.close()
    conn_d.close()
    print("? Clone completed")


if __name__ == "__main__":
    main()
