"""
Mirror tse_input price/index data into analytics market_data_cache.

Usage:
  PYTHONIOENCODING=utf-8 python scripts/mirror_price_to_cache.py \
    --pg-dsn postgresql://gravity:gravity_db_pass@localhost:5544/gravity_tse

Notes:
  - Processes price_data year-by-year to reduce transaction size.
  - Mirrors market_indices (as IDX_*) and sector_indices (as SEC_*).
"""

from __future__ import annotations

import argparse
from datetime import date

import psycopg2


def mirror_prices(cur, start_year: int, end_year: int):
    for year in range(start_year, end_year + 1):
        d1, d2 = date(year, 1, 1), date(year, 12, 31)
        print(f"Mirroring price_data for {year} ...")
        cur.execute(
            """
            INSERT INTO tech_analysis.market_data_cache(symbol,timeframe,ts,open,high,low,close,volume)
            SELECT symbol,'1d', trading_date::timestamp, adj_open, adj_high, adj_low, adj_close, adj_volume
            FROM tse_input.price_data
            WHERE trading_date BETWEEN %s AND %s
            ON CONFLICT (symbol,timeframe,ts) DO UPDATE
            SET open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close, volume=EXCLUDED.volume
            """,
            (d1, d2),
        )


def mirror_indices(cur):
    print("Mirroring market_indices ...")
    cur.execute(
        """
        INSERT INTO tech_analysis.market_data_cache(symbol,timeframe,ts,open,high,low,close,volume)
        SELECT 'IDX_'||index_code,'1d', trading_date::timestamp, open, high, low, close, 0
        FROM tse_input.market_indices
        ON CONFLICT (symbol,timeframe,ts) DO UPDATE
        SET open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close, volume=EXCLUDED.volume
        """
    )
    print("Mirroring sector_indices ...")
    cur.execute(
        """
        INSERT INTO tech_analysis.market_data_cache(symbol,timeframe,ts,open,high,low,close,volume)
        SELECT 'SEC_'||sector_code::text,'1d', trading_date::timestamp, open, high, low, close, 0
        FROM tse_input.sector_indices
        ON CONFLICT (symbol,timeframe,ts) DO UPDATE
        SET open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close, volume=EXCLUDED.volume
        """
    )


def main():
    parser = argparse.ArgumentParser(description="Mirror tse_input.* into analytics.market_data_cache.")
    parser.add_argument("--pg-dsn", required=True, help="Postgres DSN, e.g. postgresql://user:pass@host:port/db")
    args = parser.parse_args()

    conn = psycopg2.connect(args.pg_dsn)
    cur = conn.cursor()

    cur.execute("SELECT MIN(trading_date), MAX(trading_date) FROM tse_input.price_data")
    min_max = cur.fetchone()
    if not min_max or not min_max[0] or not min_max[1]:
        print("No price_data found; aborting.")
        cur.close()
        conn.close()
        return
    start_year, end_year = min_max[0].year, min_max[1].year

    mirror_prices(cur, start_year, end_year)
    mirror_indices(cur)

    conn.commit()
    cur.close()
    conn.close()
    print("✅ market_data_cache mirrored.")


if __name__ == "__main__":
    main()
