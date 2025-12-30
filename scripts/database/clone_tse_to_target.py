"""
Clone data from a source Postgres (tse_input tables) to a target Postgres.

Usage example:
  python scripts/clone_tse_to_target.py \
    --src-dsn postgresql://gravity:gravity_db_pass@127.0.0.1:5544/gravity_tse \
    --dst-dsn postgresql://gravity:gravity_db_pass@127.0.0.1:5544/tech_analysis
"""

from __future__ import annotations

import argparse

import psycopg2


def copy_table(cur_s, cur_d, tbl: str):
    cur_s.execute(f"SELECT * FROM tse_input.{tbl}")
    rows = cur_s.fetchall()
    cols = [desc[0] for desc in cur_s.description]
    placeholders = ", ".join(["%s"] * len(cols))
    cur_d.execute(f"TRUNCATE tse_input.{tbl} RESTART IDENTITY CASCADE")
    if rows:
        cur_d.executemany(
            f"INSERT INTO tse_input.{tbl} ({', '.join(cols)}) VALUES ({placeholders})", rows
        )
    print(f"{tbl}: {len(rows)} rows copied")


def main():
    parser = argparse.ArgumentParser(
        description="Clone tse_input tables from source to target Postgres."
    )
    parser.add_argument("--src-dsn", required=True, help="Source DSN (has tse_input data)")
    parser.add_argument("--dst-dsn", required=True, help="Destination DSN")
    args = parser.parse_args()

    conn_s = psycopg2.connect(args.src_dsn)
    conn_d = psycopg2.connect(args.dst_dsn)
    cur_s = conn_s.cursor()
    cur_d = conn_d.cursor()

    tables = [
        "markets",
        "panels",
        "sectors",
        "companies",
        "indices_info",
        "last_updates",
        "price_data",
        "market_indices",
        "sector_indices",
    ]
    for t in tables:
        copy_table(cur_s, cur_d, t)
    conn_d.commit()
    cur_s.close()
    cur_d.close()
    conn_s.close()
    conn_d.close()
    print("✅ Clone completed")


if __name__ == "__main__":
    main()
