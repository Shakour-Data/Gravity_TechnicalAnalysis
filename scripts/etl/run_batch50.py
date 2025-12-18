#!/usr/bin/env python3
"""
DEPRECATED: use scripts/etl/run_batch50_full_ingest.py instead.

Batch updater for TSE data + analysis (50 symbols per run).

What it does per invocation:
1) Picks the next N symbols (default 50) ordered by last_date ascending from the
   source SQLite DB (tse_data.db).
2) Fetches fresh OHLCV for just those symbols using temp_gravity_tse DataFetcher.
3) Runs the full analysis pipeline for the same symbols into the target DB.

Run this script repeatedly until no symbols remain with stale last_date.

Example:
  python scripts/etl/run_batch50.py \
    --source-db temp_gravity_tse/data/tse_data.db \
    --target-db data/TechAnalysis.db \
    --batch-size 50 \
    --limit 0
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List
try:
    import psycopg2  # type: ignore
except Exception:  # pragma: no cover
    psycopg2 = None

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMP_TSE_DIR = REPO_ROOT / "temp_gravity_tse"
TEMP_TSE_SRC = TEMP_TSE_DIR / "src"


def pick_symbols(db_path: Path, batch_size: int, min_candles: int) -> List[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT lu.symbol
            FROM last_updates lu
            JOIN (
                SELECT ticker, COUNT(*) AS cnt
                FROM price_data
                GROUP BY ticker
            ) p ON lu.symbol = p.ticker
            WHERE p.cnt >= ?
            ORDER BY lu.last_date ASC, lu.symbol ASC
            LIMIT ?
            """,
            (min_candles, batch_size),
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def fetch_prices_for_symbols(
    db_path: Path,
    symbols: Iterable[str],
    fetch_indices: bool = True,
    fetch_usd: bool = True,
) -> None:
    """Run DataFetcher for only the provided symbols."""
    # We need the parent of "src" on sys.path to import "src.*"
    sys.path.insert(0, str(TEMP_TSE_DIR))
    import src.config as cfg  # type: ignore
    from src.database import init_price_data  # type: ignore
    from src.fetcher import DataFetcher  # type: ignore

    symbols_set = set(symbols)
    if not symbols_set:
        print("No symbols to fetch.")
        return

    # Point config/DB to the provided path
    cfg.DB_FILE = str(db_path)
    init_price_data.DB_FILE = str(db_path)

    orig_load_json = DataFetcher.load_json

    def load_json_override(filepath):
        data = orig_load_json(filepath)
        # Filter companies list to the batch symbols
        if Path(filepath).resolve() == Path(cfg.COMPANIES_FILE).resolve():
            filtered = [c for c in data if c.get("Ticker") in symbols_set]
            print(f"[fetch] Filtered companies from {len(data)} to {len(filtered)}")
            return filtered
        return data

    DataFetcher.load_json = staticmethod(load_json_override)  # type: ignore[assignment]
    print(f"[fetch] Fetching prices for {len(symbols_set)} symbols...")
    DataFetcher.fetch_all_prices_to_json()
    if fetch_usd:
        # USD fetch is embedded at end of fetch_all_prices_to_json; kept for clarity
        pass
    if fetch_indices:
        print("[fetch] Fetching market indices (TSETMC)...")
        DataFetcher.fetch_all_market_indices_to_json()
        print("[fetch] Fetching sector indices (TSETMC)...")
        DataFetcher.fetch_all_sector_indices_to_json()
    print("[fetch] Done.")


def run_analysis_pipeline(symbols: Iterable[str], source_db: Path, target_db: str, limit: int) -> None:
    """Call the existing full pipeline for the given symbols."""
    sym_list = list(symbols)
    if not sym_list:
        print("No symbols for analysis.")
        return

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "etl" / "run_full_pipeline.py"),
        "--source-db",
        str(source_db),
        "--target-db",
        target_db,
        "--symbols",
        ",".join(sym_list),
        "--limit",
        str(limit),
    ]
    print(f"[analysis] Running pipeline for {len(sym_list)} symbols...")
    subprocess.run(cmd, check=True)
    print("[analysis] Done.")


def ensure_target_ready(target_db: str) -> None:
    """Preflight for target DB (especially Postgres) to fail fast with a clear message."""
    if target_db.lower().startswith("postgres"):
        if psycopg2 is None:
            raise SystemExit("psycopg2 not installed; cannot connect to Postgres target.")
        try:
            conn = psycopg2.connect(target_db)
            conn.close()
            print("[preflight] Postgres connection successful.")
        except Exception as exc:
            raise SystemExit(
                f"Cannot connect to Postgres target ({target_db}). Ensure container/service is up and DSN is correct. Error: {exc}"
            )
    else:
        # SQLite: ensure parent dir exists
        Path(target_db).parent.mkdir(parents=True, exist_ok=True)


def verify_source_data(db_path: Path, symbols: Iterable[str]) -> None:
    """Basic sanity check that fetched symbols have rows and updated dates."""
    sym_list = list(symbols)
    conn = sqlite3.connect(db_path)
    try:
        placeholders = ",".join("?" for _ in sym_list)
        if not placeholders:
            return
        cur = conn.cursor()
        cur.execute(
            f"SELECT ticker, COUNT(*) AS c, MIN(date), MAX(date) FROM price_data "
            f"WHERE ticker IN ({placeholders}) GROUP BY ticker",
            sym_list,
        )
        stats = {row[0]: (row[1], row[2], row[3]) for row in cur.fetchall()}

        cur.execute(
            f"SELECT symbol, last_date FROM last_updates WHERE symbol IN ({placeholders})",
            sym_list,
        )
        last_updates = {row[0]: row[1] for row in cur.fetchall()}

        missing_rows = [s for s in sym_list if s not in stats or stats[s][0] == 0]
        if missing_rows:
            print(f"[verify] WARNING: no rows in price_data for {len(missing_rows)} symbols: {missing_rows[:5]}...")

        for s in sym_list:
            if s in stats:
                c, dmin, dmax = stats[s]
                lu = last_updates.get(s)
                print(f"[verify] {s}: rows={c}, min_date={dmin}, max_date={dmax}, last_update={lu}")
            else:
                print(f"[verify] {s}: MISSING in price_data/last_updates")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch fetch + analysis for N symbols.")
    parser.add_argument(
        "--source-db",
        default=str(TEMP_TSE_DIR / "data" / "tse_data.db"),
        help="Path to source TSE SQLite DB (input).",
    )
    default_target = (
        os.getenv("ANALYSIS_TARGET_DB")
        or os.getenv("DATABASE_URL")
        or "postgresql://postgres:Bedaan4D@127.0.0.1:5432/bedaan4d_db"
    )
    parser.add_argument(
        "--target-db",
        default=default_target,
        help="Path/DSN to target analysis DB (output). Default prefers Postgres (Docker DSN).",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Number of symbols per batch.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max candles per symbol for analysis (0 = all history).",
    )
    parser.add_argument(
        "--min-candles",
        type=int,
        default=400,
        help="Skip symbols with fewer candles than this (prevents pipeline failures).",
    )
    parser.add_argument(
        "--no-indices",
        action="store_true",
        help="Do not fetch market/sector indices during batch fetch (default: fetch).",
    )
    parser.add_argument(
        "--no-usd",
        action="store_true",
        help="Do not fetch USD during batch fetch (default: fetch).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep running batches until no symbols remain (according to filters).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Optional cap when --loop is set (0 = unlimited).",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip source data verification after each batch.",
    )
    args = parser.parse_args()

    source_db = Path(args.source_db).resolve()
    target_db_input = args.target_db
    target_db = target_db_input if str(target_db_input).lower().startswith("postgres") else str(Path(target_db_input).resolve())

    ensure_target_ready(target_db)

    iteration = 0
    while True:
        if args.loop and args.max_iterations and iteration >= args.max_iterations:
            print(f"Reached max iterations ({args.max_iterations}); stopping.")
            break

        symbols = pick_symbols(source_db, args.batch_size, args.min_candles)
        if not symbols:
            if iteration == 0:
                print("No symbols found to process (maybe already up-to-date or below min-candles).")
            else:
                print("All eligible symbols processed; stopping.")
            break

        iteration += 1
        print(f"\n=== Batch {iteration} ===")
        print(f"Selected {len(symbols)} symbols (oldest first, min_candles={args.min_candles}).")

        batch_start = time.perf_counter()
        fetch_prices_for_symbols(
            source_db,
            symbols,
            fetch_indices=not args.no_indices,
            fetch_usd=not args.no_usd,
        )
        run_analysis_pipeline(symbols, source_db, target_db, args.limit)
        elapsed = time.perf_counter() - batch_start

        if not args.skip_verify:
            verify_source_data(source_db, symbols)

        print(f"Batch {iteration} completed in {elapsed:.2f} seconds.")

        if not args.loop:
            break

    print("Done.")


if __name__ == "__main__":
    main()
