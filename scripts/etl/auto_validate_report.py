#!/usr/bin/env python3
"""
Run automatic validation checks for a batch of symbols and emit a Markdown report.

Checks performed:
- Row counts and distinct symbols per table.
- Time ranges per table.
- Duplicate keys (based on natural unique keys) per table.
- Trend-score sign agreement vs بازده روز بعد (from source SQLite price_data).
- Basic sanity: volatility_score منفی؟

Usage:
python scripts/etl/auto_validate_report.py \
  --target-db "postgresql://gravity:gravity_db_pass@127.0.0.1:5545/tech_analysis" \
  --source-db "E:\\Shakour\\MyProjects\\GravityTseHisPrice\\data\\tse_data.db" \
  --symbols-file batch1_symbols.txt \
  --outfile docs/validation_report_batch1.md \
  --limit 500
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor


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


def load_prices(src_db: Path, symbols: Iterable[str], limit: int) -> Dict[str, List[Tuple[date, float]]]:
    conn = sqlite3.connect(src_db)
    cur = conn.cursor()
    result: Dict[str, List[Tuple[date, float]]] = {}
    limit_val = limit if limit and limit > 0 else -1
    for sym in symbols:
        rows = cur.execute(
            """
            SELECT date, adj_close
            FROM price_data
            WHERE ticker=?
            ORDER BY date DESC
            LIMIT ?
            """,
            (sym, limit_val),
        ).fetchall()
        parsed_rev = []
        for d, c in rows:
            try:
                parsed_rev.append((datetime.fromisoformat(d).date(), float(c)))
            except Exception:
                continue
        result[sym] = list(reversed(parsed_rev))
    cur.close()
    conn.close()
    return result


def trend_agreement(pg_conn, symbols: List[str], prices: Dict[str, List[Tuple[date, float]]], limit: int) -> Tuple[int, int]:
    """Compare sign(trend_score) vs sign(next-day return) from source prices."""
    cur = pg_conn.cursor(cursor_factory=RealDictCursor)
    lim_clause = "" if limit <= 0 else f"LIMIT {limit}"
    cur.execute(
        f"""
        SELECT symbol, ts, trend_score
        FROM historical_scores
        WHERE symbol = ANY(%s)
        ORDER BY ts DESC
        {lim_clause}
        """,
        (symbols,),
    )
    total = 0
    agree = 0
    for row in cur.fetchall():
        sym = row["symbol"]
        ts = row["ts"]
        trend = float(row["trend_score"] or 0.0)
        day = ts.date()
        price_list = prices.get(sym, [])
        # find index for current day
        idx = next((i for i, (d, _c) in enumerate(price_list) if d == day), -1)
        if idx == -1 or idx + 1 >= len(price_list):
            continue
        c0 = price_list[idx][1]
        c1 = price_list[idx + 1][1]
        if c0 == 0:
            continue
        ret = (c1 - c0) / c0
        if ret > 0 and trend > 0:
            agree += 1
        elif ret < 0 and trend < 0:
            agree += 1
        total += 1
    cur.close()
    return agree, total


def counts_and_ranges(pg_conn, symbols: List[str]) -> List[str]:
    cur = pg_conn.cursor()
    lines: List[str] = []
    sym_tuple = tuple(symbols)
    for tbl in TABLES:
        cur.execute(f"SELECT COUNT(*), COUNT(DISTINCT symbol) FROM {tbl} WHERE symbol IN %s", (sym_tuple,))
        rows, distinct_syms = cur.fetchone()
        lines.append(f"- {tbl}: rows={rows}, symbols={distinct_syms}")
    lines.append("")
    cur.execute("SELECT MIN(analysis_date), MAX(analysis_date) FROM analysis_results WHERE symbol IN %s", (sym_tuple,))
    lines.append(f"- analysis_results range: {cur.fetchone()}")
    cur.execute("SELECT MIN(ts), MAX(ts) FROM historical_scores WHERE symbol IN %s", (sym_tuple,))
    lines.append(f"- historical_scores range: {cur.fetchone()}")
    cur.execute("SELECT MIN(ts), MAX(ts) FROM ml_weights_history WHERE symbol IN %s", (sym_tuple,))
    lines.append(f"- ml_weights_history range: {cur.fetchone()}")
    cur.close()
    return lines


def duplicate_checks(pg_conn, symbols: List[str]) -> List[str]:
    cur = pg_conn.cursor()
    sym_tuple = tuple(symbols)
    lines: List[str] = []
    dup_queries = {
        "historical_scores": "symbol, ts, timeframe",
        # align with unique index (coalesce + ::text)
        "historical_indicator_scores": "symbol, ts, timeframe, indicator_name, coalesce(indicator_params::text,'__NULL__')",
        "ml_weights_history": "symbol, ts, model_name, timeframe",
        "tool_performance_history": "symbol, timeframe, prediction_timestamp, tool_name",
        "backtest_runs": "symbol, interval, period_start, period_end, model_version",
        "pattern_detection_results": "symbol, timeframe, timestamp, pattern_type, pattern_name",
    }
    for tbl, key in dup_queries.items():
        cur.execute(
            f"SELECT COUNT(*) FROM (SELECT {key} FROM {tbl} WHERE symbol IN %s GROUP BY {key} HAVING COUNT(*)>1) d",
            (sym_tuple,),
        )
        dups = cur.fetchone()[0]
        lines.append(f"- {tbl} duplicates (by {key}): {dups}")
    cur.close()
    return lines


def negative_vol_checks(pg_conn, symbols: List[str]) -> int:
    cur = pg_conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM historical_scores WHERE symbol = ANY(%s) AND volatility_score < 0",
        (symbols,),
    )
    c = cur.fetchone()[0]
    cur.close()
    return c


def coverage_stats(prices: Dict[str, List[Tuple[date, float]]], limit: int) -> Tuple[float, int]:
    counts = []
    for rows in prices.values():
        if limit > 0:
            counts.append(min(len(rows), limit))
        else:
            counts.append(len(rows))
    avg = sum(counts) / len(counts) if counts else 0
    mn = min(counts) if counts else 0
    return avg, mn


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto validate batch and generate Markdown report.")
    parser.add_argument("--target-db", required=True, help="Postgres DSN for target.")
    parser.add_argument("--source-db", required=True, help="SQLite DB path for price_data.")
    parser.add_argument("--symbols-file", required=True, help="File with symbols, one per line.")
    parser.add_argument("--outfile", required=True, help="Path to write Markdown report.")
    parser.add_argument("--limit", type=int, default=500, help="Max candles per symbol to consider (0 = all).")
    args = parser.parse_args()

    symbols = load_symbols(Path(args.symbols_file))
    if not symbols:
        raise SystemExit("No symbols loaded.")

    # load data
    prices = load_prices(Path(args.source_db), symbols, args.limit)
    avg_c, min_c = coverage_stats(prices, args.limit)

    pg_conn = psycopg2.connect(args.target_db)
    pg_conn.autocommit = True

    report: List[str] = []
    report.append(f"# گزارش اعتبارسنجی بچ ({len(symbols)} نماد)")
    report.append(f"- زمان اجرا: {datetime.now(timezone.utc).isoformat()}")
    report.append(f"- DSN: {args.target_db}")
    report.append(f"- سورس: {args.source_db}")
    report.append(f"- limit کندل: {args.limit if args.limit>0 else 'همه'}")
    report.append("")

    report.append("## شمارش و پوشش جداول")
    report.extend(counts_and_ranges(pg_conn, symbols))
    report.append("")

    report.append("## تکرار داده (duplicates)")
    report.extend(duplicate_checks(pg_conn, symbols))
    report.append("")

    report.append("## منفی بودن نوسان (volatility_score < 0)")
    neg_vol = negative_vol_checks(pg_conn, symbols)
    report.append(f"- historical_scores: {neg_vol} ردیف منفی")
    report.append("")

    report.append("## تطابق جهت trend_score با بازده روز بعد")
    agree, total = trend_agreement(pg_conn, symbols, prices, args.limit)
    rate = (agree / total * 100.0) if total else 0.0
    report.append(f"- total مقایسه: {total}")
    report.append(f"- توافق جهت: {agree} ({rate:.2f}%)")
    report.append("")

    report.append("## پوشش کندل سورس")
    report.append(f"- میانگین کندل (تا limit): {avg_c:.1f}")
    report.append(f"- حداقل کندل (تا limit): {min_c}")

    Path(args.outfile).write_text("\n".join(report), encoding="utf-8")
    print(f"Report saved to {args.outfile}")


if __name__ == "__main__":
    main()
