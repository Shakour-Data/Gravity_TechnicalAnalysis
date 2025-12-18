#!/usr/bin/env python3
"""
Batch validation helper: verifies that all requested symbols exist in target tables,
reports row counts per table, time ranges, and coverage vs source candles.

Usage:
python scripts/etl/validate_batch.py \
  --target-db "postgresql://gravity:gravity_db_pass@127.0.0.1:5545/tech_analysis" \
  --source-db "E:\\Shakour\\MyProjects\\GravityTseHisPrice\\data\\tse_data.db" \
  --symbols-file batch1_symbols.txt \
  --limit 300
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import psycopg2

TABLES = [
    "analysis_results",
    "historical_scores",
    "historical_indicator_scores",
    "tool_performance_history",
    "backtest_runs",
    "pattern_detection_results",
    "ml_weights_history",
]


def load_symbols(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fmt_list(items: Iterable[str], limit: int = 5) -> str:
    lst = list(items)
    if not lst:
        return "-"
    if len(lst) <= limit:
        return ", ".join(lst)
    return ", ".join(lst[:limit]) + f" ... (+{len(lst) - limit})"


def pg_connect(dsn: str):
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    return conn


def coverage_vs_source(src_db: Path, symbols: List[str], limit: int) -> List[Tuple[str, int]]:
    """Return candle counts per symbol (up to limit) from source DB."""
    conn = sqlite3.connect(src_db)
    cur = conn.cursor()
    result = []
    for sym in symbols:
        cur.execute(
            "SELECT COUNT(*) FROM (SELECT date FROM price_data WHERE ticker=? ORDER BY date DESC LIMIT ?)",
            (sym, limit),
        )
        result.append((sym, cur.fetchone()[0]))
    cur.close()
    conn.close()
    return result


def check_tables(conn, symbols: List[str]) -> str:
    lines = []
    cur = conn.cursor()
    sym_tuple = tuple(symbols)

    missing: dict[str, Set[str]] = {}
    for tbl in TABLES:
        cur.execute(f"SELECT COUNT(*), COUNT(DISTINCT symbol) FROM {tbl} WHERE symbol IN %s", (sym_tuple,))
        rows, distinct_syms = cur.fetchone()
        lines.append(f"{tbl:30s} rows={rows:8d} symbols={distinct_syms:5d}")
        cur.execute(f"SELECT DISTINCT symbol FROM {tbl} WHERE symbol IN %s", (sym_tuple,))
        have = {r[0] for r in cur.fetchall()}
        miss = set(symbols) - have
        if miss:
            missing[tbl] = miss

    # time ranges
    lines.append("")
    cur.execute("SELECT MIN(analysis_date), MAX(analysis_date) FROM analysis_results WHERE symbol IN %s", (sym_tuple,))
    lines.append(f"analysis_results range: {cur.fetchone()}")
    cur.execute("SELECT MIN(ts), MAX(ts) FROM historical_scores WHERE symbol IN %s", (sym_tuple,))
    lines.append(f"historical_scores range: {cur.fetchone()}")
    cur.execute("SELECT MIN(ts), MAX(ts) FROM ml_weights_history WHERE symbol IN %s", (sym_tuple,))
    lines.append(f"ml_weights_history range: {cur.fetchone()}")

    if missing:
        lines.append("")
        lines.append("Missing symbols per table:")
        for tbl, miss in sorted(missing.items()):
            lines.append(f"  {tbl}: {fmt_list(sorted(miss), limit=10)}")
    else:
        lines.append("")
        lines.append("All symbols present in all tables.")

    cur.close()
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate that batch symbols are fully stored in target DB.")
    parser.add_argument("--target-db", required=True, help="Postgres DSN for target.")
    parser.add_argument("--source-db", required=True, help="Source SQLite DB (for candle coverage).")
    parser.add_argument("--symbols-file", required=True, help="File with symbols, one per line.")
    parser.add_argument("--limit", type=int, default=300, help="Max candles to compare with source.")
    args = parser.parse_args()

    symbols = load_symbols(Path(args.symbols_file))
    if not symbols:
        raise SystemExit("No symbols loaded from symbols-file.")

    report_lines = []
    report_lines.append(f"Generated at: {datetime.utcnow().isoformat()}Z")
    report_lines.append(f"Target: {args.target_db}")
    report_lines.append(f"Symbols: {len(symbols)}")
    report_lines.append("")

    # Postgres coverage
    conn = pg_connect(args.target_db)
    report_lines.append(check_tables(conn, symbols))
    conn.close()

    # Candle coverage vs source
    src_counts = coverage_vs_source(Path(args.source_db), symbols, args.limit)
    avg_candles = sum(c for _, c in src_counts) / len(src_counts) if src_counts else 0
    min_candles = min((c for _, c in src_counts), default=0)
    report_lines.append("")
    report_lines.append(f"Source candle coverage (limit={args.limit}): avg={avg_candles:.1f}, min={min_candles}")

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
