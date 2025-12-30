#!/usr/bin/env python3
"""
Recompute basic metrics (trend/momentum/volatility) from source price_data and compare
against historical_scores in Postgres. Emits a Markdown report.

Usage:
python scripts/etl/recompute_validation.py \
  --target-db "postgresql://gravity:gravity_db_pass@127.0.0.1:5545/tech_analysis" \
  --source-db "E:\\Shakour\\MyProjects\\GravityTseHisPrice\\data\\tse_data.db" \
  --symbols-file batch1_symbols.txt \
  --window 30 \
  --outfile docs/reports/recompute_report.md

Notes:
- This uses a rolling window of closes ending at each ts to recompute:
    trend = (close_t / close_{t-window+1}) - 1
    vol   = stddev of daily returns in that window
- Compares to historical_scores (trend_score, volatility_score) at the same ts.
- Reports mean absolute error (MAE) and count of mismatches above tolerance.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor


def load_symbols(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_prices(src_db: Path, symbols: Iterable[str]) -> dict[str, list[tuple[str, float]]]:
    conn = sqlite3.connect(src_db)
    cur = conn.cursor()
    data: dict[str, list[tuple[str, float]]] = {}
    for sym in symbols:
        rows = cur.execute(
            """
            SELECT date, adj_close
            FROM price_data
            WHERE ticker=?
            ORDER BY date ASC
            """,
            (sym,),
        ).fetchall()
        data[sym] = [(r[0], float(r[1])) for r in rows]
    cur.close()
    conn.close()
    return data


def rolling_metrics(closes: list[tuple[str, float]], window: int) -> dict[str, tuple[float, float]]:
    """
    Return mapping from date string -> (trend, vol) using rolling window ending at that date.
    trend: (close / close_{t-window+1}) - 1
    vol: stddev of daily returns in window
    """
    result: dict[str, tuple[float, float]] = {}
    prices = [c for _, c in closes]
    dates = [d for d, _ in closes]
    rets: list[float] = []
    for i in range(1, len(prices)):
        if prices[i - 1] != 0:
            rets.append((prices[i] - prices[i - 1]) / prices[i - 1])
        else:
            rets.append(0.0)
        if i + 1 >= window:
            start = i + 1 - window
            window_prices = prices[start : i + 1]
            window_rets = rets[start : i + 1]
            if window_prices[0] == 0:
                trend = 0.0
            else:
                trend = window_prices[-1] / window_prices[0] - 1.0
            if window_rets:
                mean_r = sum(window_rets) / len(window_rets)
                var = sum((r - mean_r) ** 2 for r in window_rets) / len(window_rets)
                vol = math.sqrt(var)
            else:
                vol = 0.0
            result[dates[i]] = (trend, vol)
    return result


def fetch_scores(pg_conn, symbols: list[str]) -> dict[str, dict[str, tuple[float, float]]]:
    """Return mapping symbol -> date(str) -> (trend_score, volatility_score)."""
    cur = pg_conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        """
        SELECT symbol, ts, trend_score, volatility_score
        FROM historical_scores
        WHERE symbol = ANY(%s)
        ORDER BY ts ASC
        """,
        (symbols,),
    )
    out: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for row in cur.fetchall():
        d = row["ts"].date().isoformat()
        out[row["symbol"]][d] = (
            float(row["trend_score"] or 0.0),
            float(row["volatility_score"] or 0.0),
        )
    cur.close()
    return out


def compare(
    prices: dict[str, list[tuple[str, float]]],
    db_scores: dict[str, dict[str, tuple[float, float]]],
    window: int,
):
    stats = {
        "trend_mae": 0.0,
        "vol_mae": 0.0,
        "count": 0,
        "trend_miss": 0,
        "vol_miss": 0,
    }
    mismatches: list[str] = []
    tolerance_trend = 0.02  # 2%
    tolerance_vol = 0.02
    for sym, rows in prices.items():
        if sym not in db_scores:
            continue
        roll = rolling_metrics(rows, window)
        for d, (trend_calc, vol_calc) in roll.items():
            if d not in db_scores[sym]:
                continue
            trend_db, vol_db = db_scores[sym][d]
            stats["trend_mae"] += abs(trend_calc - trend_db)
            stats["vol_mae"] += abs(vol_calc - vol_db)
            stats["count"] += 1
            if abs(trend_calc - trend_db) > tolerance_trend:
                stats["trend_miss"] += 1
                mismatches.append(f"{sym} {d} trend calc={trend_calc:.4f} db={trend_db:.4f}")
            if abs(vol_calc - vol_db) > tolerance_vol:
                stats["vol_miss"] += 1
                mismatches.append(f"{sym} {d} vol calc={vol_calc:.4f} db={vol_db:.4f}")
    if stats["count"]:
        stats["trend_mae"] /= stats["count"]
        stats["vol_mae"] /= stats["count"]
    return stats, mismatches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute metrics from source and compare with DB."
    )
    parser.add_argument("--target-db", required=True, help="Postgres DSN.")
    parser.add_argument("--source-db", required=True, help="SQLite source DB.")
    parser.add_argument("--symbols-file", required=True, help="File with symbols (one per line).")
    parser.add_argument("--window", type=int, default=30, help="Rolling window size (days).")
    parser.add_argument("--outfile", required=True, help="Markdown report path.")
    args = parser.parse_args()

    symbols = load_symbols(Path(args.symbols_file))
    prices = load_prices(Path(args.source_db), symbols)

    pg_conn = psycopg2.connect(args.target_db)
    db_scores = fetch_scores(pg_conn, symbols)
    pg_conn.close()

    stats, mismatches = compare(prices, db_scores, args.window)

    lines = []
    lines.append(f"# گزارش بازمحاسبه ({len(symbols)} نماد)")
    lines.append(f"- زمان اجرا: {datetime.utcnow().isoformat()}Z")
    lines.append(f"- DSN: {args.target_db}")
    lines.append(f"- window: {args.window}")
    lines.append("")
    lines.append("## خلاصه خطاها (MAE)")
    lines.append(f"- trend MAE: {stats['trend_mae']:.4f}")
    lines.append(f"- volatility MAE: {stats['vol_mae']:.4f}")
    lines.append(f"- تعداد مقایسه: {stats['count']}")
    lines.append(f"- trend اختلاف > 0.02: {stats['trend_miss']}")
    lines.append(f"- vol اختلاف > 0.02: {stats['vol_miss']}")
    lines.append("")
    lines.append("## نمونه اختلاف‌ها (تا 20 مورد)")
    for m in mismatches[:20]:
        lines.append(f"- {m}")

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved to {args.outfile}")


if __name__ == "__main__":
    main()
