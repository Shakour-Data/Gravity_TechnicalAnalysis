#!/usr/bin/env python3
"""
Generate a per-batch verification report to confirm all tables are filled.

Usage:
python scripts/etl/report_batch.py \
  --target-db "postgresql://gravity:gravity@127.0.0.1:5544/tech_analysis" \
  --symbols "SYM1,SYM2,..." \
  --outfile batch1_report.txt

You can also provide --symbols-file with one symbol per line.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg2

TABLES_WITH_SYMBOL = [
    "analysis_results",
    "historical_scores",
    "historical_indicator_scores",
    "tool_performance_history",
    "backtest_runs",
    "pattern_detection_results",
]


def load_symbols(args: argparse.Namespace) -> list[str]:
    symbols: list[str] = []
    if args.symbols:
        symbols.extend([s.strip() for s in args.symbols.split(",") if s.strip()])
    if args.symbols_file:
        for line in Path(args.symbols_file).read_text(encoding="utf-8").splitlines():
            sym = line.strip()
            if sym:
                symbols.append(sym)
    symbols = sorted(set(symbols))
    return symbols


def format_list(items: Iterable[str], limit: int = 10) -> str:
    lst = list(items)
    if not lst:
        return "-"
    if len(lst) <= limit:
        return ", ".join(lst)
    return ", ".join(lst[:limit]) + f" ... (+{len(lst) - limit})"


def report(target_db: str, symbols: list[str], outfile: Path | None) -> None:
    if not symbols:
        raise SystemExit("No symbols provided. Use --symbols or --symbols-file.")

    conn = psycopg2.connect(target_db)
    cur = conn.cursor()

    lines: list[str] = []
    lines.append(f"Report generated at {datetime.now(UTC).isoformat()}")
    lines.append(f"Target DB: {target_db}")
    lines.append(f"Batch size: {len(symbols)}")
    lines.append(f"Symbols: {format_list(symbols, limit=20)}")
    lines.append("")

    sym_tuple = tuple(symbols)
    missing: dict[str, set[str]] = defaultdict(set)

    for tbl in TABLES_WITH_SYMBOL:
        cur.execute(
            f"SELECT COUNT(*) , COUNT(DISTINCT symbol) FROM {tbl} WHERE symbol IN %s",
            (sym_tuple,),
        )
        rows, distinct_syms = cur.fetchone()
        lines.append(f"{tbl:30s} rows={rows:8d} symbols={distinct_syms:5d}")

        cur.execute(
            f"SELECT DISTINCT symbol FROM {tbl} WHERE symbol IN %s",
            (sym_tuple,),
        )
        present = {row[0] for row in cur.fetchall()}
        missing_tbl = set(symbols) - present
        if missing_tbl:
            missing[tbl].update(missing_tbl)

    # 90-day coverage for analysis_results
    lines.append("")
    lines.append("analysis_results (last 90d):")
    try:
        cur.execute(
            "SELECT symbol, COUNT(*), MIN(ts), MAX(ts) "
            "FROM analysis_results WHERE symbol IN %s AND ts >= %s GROUP BY symbol",
            (sym_tuple, datetime.now(UTC) - timedelta(days=90)),
        )
        rows90 = cur.fetchall()
        for sym, cnt, ts_min, ts_max in rows90:
            lines.append(f"  {sym}: rows_90d={cnt:6d} min_ts={ts_min} max_ts={ts_max}")
    except Exception:
        # Fallback for schemas without ts; try analysis_date/created_at
        try:
            cur.execute(
                "SELECT symbol, COUNT(*), MIN(analysis_date), MAX(analysis_date) "
                "FROM analysis_results WHERE symbol IN %s AND analysis_date >= %s GROUP BY symbol",
                (sym_tuple, datetime.now(UTC).date() - timedelta(days=90)),
            )
            rows90 = cur.fetchall()
            for sym, cnt, ts_min, ts_max in rows90:
                lines.append(f"  {sym}: rows_90d={cnt:6d} min_ts={ts_min} max_ts={ts_max}")
        except Exception:
            lines.append("  (timestamp column not available for 90d check)")

    if missing:
        lines.append("")
        lines.append("Missing symbols by table:")
        for tbl, syms in sorted(missing.items()):
            lines.append(f"  {tbl}: {format_list(sorted(syms), limit=20)}")
    else:
        lines.append("")
        lines.append("All symbols present in all tables.")

    out_text = "\n".join(lines)
    if outfile:
        outfile.write_text(out_text, encoding="utf-8")
        print(f"Saved report to {outfile}")
    else:
        print(out_text)

    cur.close()
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that all batch symbols exist in all analysis tables."
    )
    parser.add_argument("--target-db", required=True, help="Postgres DSN.")
    parser.add_argument("--symbols", help="Comma-separated symbols.")
    parser.add_argument("--symbols-file", help="Path to a file containing symbols (one per line).")
    parser.add_argument("--outfile", help="Optional path to save the report text.")
    args = parser.parse_args()

    symbols = load_symbols(args)
    outfile = Path(args.outfile).resolve() if args.outfile else None
    report(args.target_db, symbols, outfile)


if __name__ == "__main__":
    main()
