#!/usr/bin/env python3
"""
Count symbols with complete OHLCV data in the SQLite source DB.
A symbol is considered 'complete' if it has at least N candles and no missing/null/NaN values in OHLCV columns.

Usage:
  python scripts/count_complete_symbols.py --source-db data/TechAnalysis.db --min-candles 120
"""

import argparse
import math
import sqlite3


def is_valid_row(row):
    # row: (open, high, low, close, volume)
    return all(
        r is not None and not (isinstance(r, float) and (math.isnan(r) or math.isinf(r)))
        for r in row
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-db", required=True)
    p.add_argument("--min-candles", type=int, default=120)
    args = p.parse_args()

    conn = sqlite3.connect(args.source_db)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT ticker FROM price_data")
    symbols = [r[0] for r in cur.fetchall()]
    complete = []
    for sym in symbols:
        cur.execute(
            "SELECT open, high, low, close, volume FROM price_data WHERE ticker=? ORDER BY date ASC",
            (sym,),
        )
        rows = cur.fetchall()
        if len(rows) < args.min_candles:
            continue
        if all(is_valid_row(row) and row[1] >= row[2] and row[4] >= 0 for row in rows):
            complete.append(sym)
    print(f"Total symbols: {len(symbols)}")
    print(f"Symbols with >= {args.min_candles} valid candles: {len(complete)}")
    print("Sample symbols:", complete[:10])


if __name__ == "__main__":
    main()
