"""
Utility script to migrate SQLite inputs into PostgreSQL.

What it does:
  1) Copy TSE input data (companies, sectors, panels, markets, indices, price_data)
     from GravityTseHisPrice SQLite into Postgres schema `tse_input`.
  2) Populate `tech_analysis.symbols` from companies.ticker.
  3) Mirror daily prices into `tech_analysis.market_data_cache` with timeframe='1d'
     so the analysis stack can run immediately.

Run (example):
  set PG_DSN=postgresql://gravity:gravity@localhost:5432/tech_analysis
  python scripts/migrate_sqlite_to_pg.py ^
      --tse-sqlite "E:\\Shakour\\MyProjects\\GravityTseHisPrice\\data\\tse_data.db" ^
      --tech-sqlite "data/TechAnalysis.db"
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import psycopg2
from psycopg2.extras import execute_values


def chunked(iterable: Iterable, size: int) -> Iterator[list]:
    chunk: list = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def open_sqlite(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def copy_simple_table(
    pg_conn,
    sqlite_conn,
    sqlite_query: str,
    target_sql: str,
    fmt_row,
    batch_size: int = 1000,
):
    cur_src = sqlite_conn.cursor()
    cur_src.execute(sqlite_query)
    rows = (fmt_row(row) for row in cur_src)
    with pg_conn.cursor() as cur:
        for batch in chunked(rows, batch_size):
            execute_values(cur, target_sql, batch, page_size=batch_size)
    pg_conn.commit()


def parse_date(value: str | None):
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def migrate_tse(sqlite_path: Path, pg_conn):
    if not sqlite_path.exists():
        raise FileNotFoundError(f"TSE SQLite not found: {sqlite_path}")

    src = open_sqlite(sqlite_path)

    # markets
    copy_simple_table(
        pg_conn,
        src,
        "SELECT market_id, market_name FROM markets",
        """
        INSERT INTO tse_input.markets (market_id, market_name)
        VALUES %s
        ON CONFLICT (market_id) DO UPDATE SET market_name = EXCLUDED.market_name
        """,
        lambda r: (r["market_id"], r["market_name"]),
    )

    # panels
    copy_simple_table(
        pg_conn,
        src,
        "SELECT panel_id, panel_name FROM panels",
        """
        INSERT INTO tse_input.panels (panel_id, panel_name)
        VALUES %s
        ON CONFLICT (panel_id) DO UPDATE SET panel_name = EXCLUDED.panel_name
        """,
        lambda r: (r["panel_id"], r["panel_name"]),
    )

    # sectors
    copy_simple_table(
        pg_conn,
        src,
        "SELECT sector_id, sector_name, sector_name_en, us_sector FROM sectors",
        """
        INSERT INTO tse_input.sectors (sector_id, sector_name, sector_name_en, us_sector)
        VALUES %s
        ON CONFLICT (sector_id) DO UPDATE SET
            sector_name = EXCLUDED.sector_name,
            sector_name_en = EXCLUDED.sector_name_en,
            us_sector = EXCLUDED.us_sector
        """,
        lambda r: (r["sector_id"], r["sector_name"], r["sector_name_en"], r["us_sector"]),
    )

    # companies
    copy_simple_table(
        pg_conn,
        src,
        "SELECT company_id, ticker, name, sector_id, panel_id, market_id FROM companies",
        """
        INSERT INTO tse_input.companies (company_id, ticker, name, sector_id, panel_id, market_id)
        VALUES %s
        ON CONFLICT (company_id) DO UPDATE SET
            ticker = EXCLUDED.ticker,
            name = EXCLUDED.name,
            sector_id = EXCLUDED.sector_id,
            panel_id = EXCLUDED.panel_id,
            market_id = EXCLUDED.market_id
        """,
        lambda r: (r["company_id"], r["ticker"], r["name"], r["sector_id"], r["panel_id"], r["market_id"]),
    )

    # indices_info
    copy_simple_table(
        pg_conn,
        src,
        "SELECT index_code, index_name_fa, index_name_en, index_type FROM indices_info",
        """
        INSERT INTO tse_input.indices_info (index_code, index_name_fa, index_name_en, index_type)
        VALUES %s
        ON CONFLICT (index_code) DO UPDATE SET
            index_name_fa = EXCLUDED.index_name_fa,
            index_name_en = EXCLUDED.index_name_en,
            index_type = EXCLUDED.index_type
        """,
        lambda r: (r["index_code"], r["index_name_fa"], r["index_name_en"], r["index_type"]),
    )

    # price_data
    copy_simple_table(
        pg_conn,
        src,
        "SELECT ticker, date, adj_open, adj_high, adj_low, adj_close, adj_final, adj_volume FROM price_data",
        """
        INSERT INTO tse_input.price_data (
            symbol, trading_date, adj_open, adj_high, adj_low, adj_close, adj_final, adj_volume
        ) VALUES %s
        ON CONFLICT (symbol, trading_date) DO UPDATE SET
            adj_open = EXCLUDED.adj_open,
            adj_high = EXCLUDED.adj_high,
            adj_low = EXCLUDED.adj_low,
            adj_close = EXCLUDED.adj_close,
            adj_final = EXCLUDED.adj_final,
            adj_volume = EXCLUDED.adj_volume
        """,
        lambda r: (
            r["ticker"],
            parse_date(r["date"]),
            r["adj_open"],
            r["adj_high"],
            r["adj_low"],
            r["adj_close"],
            r["adj_final"],
            r["adj_volume"],
        ),
        batch_size=5000,
    )

    # market_indices
    copy_simple_table(
        pg_conn,
        src,
        "SELECT index_code, date, open, high, low, close FROM market_indices",
        """
        INSERT INTO tse_input.market_indices (index_code, trading_date, open, high, low, close)
        VALUES %s
        ON CONFLICT (index_code, trading_date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close
        """,
        lambda r: (
            r["index_code"],
            parse_date(r["date"]),
            r["open"],
            r["high"],
            r["low"],
            r["close"],
        ),
        batch_size=2000,
    )

    # sector_indices
    copy_simple_table(
        pg_conn,
        src,
        "SELECT sector_code, date, open, high, low, close FROM sector_indices",
        """
        INSERT INTO tse_input.sector_indices (sector_code, trading_date, open, high, low, close)
        VALUES %s
        ON CONFLICT (sector_code, trading_date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close
        """,
        lambda r: (
            r["sector_code"],
            parse_date(r["date"]),
            r["open"],
            r["high"],
            r["low"],
            r["close"],
        ),
        batch_size=2000,
    )

    # last_updates
    copy_simple_table(
        pg_conn,
        src,
        "SELECT symbol, last_date FROM last_updates",
        """
        INSERT INTO tse_input.last_updates (symbol, last_date)
        VALUES %s
        ON CONFLICT (symbol) DO UPDATE SET last_date = EXCLUDED.last_date
        """,
        lambda r: (r["symbol"], parse_date(r["last_date"])),
    )

    src.close()


def populate_symbols_from_companies(pg_conn):
    with pg_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tech_analysis.symbols (symbol, company_id, sector_id, panel_id, market_id, name)
            SELECT ticker, company_id, sector_id, panel_id, market_id, name
            FROM tse_input.companies
            ON CONFLICT (symbol) DO UPDATE SET
                company_id = EXCLUDED.company_id,
                sector_id = EXCLUDED.sector_id,
                panel_id = EXCLUDED.panel_id,
                market_id = EXCLUDED.market_id,
                name = EXCLUDED.name
            """
        )
    pg_conn.commit()


def populate_symbols_for_indices(pg_conn):
    """Create symbol aliases for market/sector indices (IDX_*, SEC_*) to align with analysis tables."""
    with pg_conn.cursor() as cur:
        # Market indices
        cur.execute(
            """
            INSERT INTO tech_analysis.symbols (symbol, name)
            SELECT CONCAT('IDX_', index_code) AS symbol, index_name_fa
            FROM tse_input.indices_info
            WHERE index_type = 'market'
            ON CONFLICT (symbol) DO NOTHING
            """
        )
        # Sector indices
        cur.execute(
            """
            INSERT INTO tech_analysis.symbols (symbol, name)
            SELECT CONCAT('SEC_', sector_id)::text AS symbol, sector_name
            FROM tse_input.sectors
            ON CONFLICT (symbol) DO NOTHING
            """
        )
    pg_conn.commit()


def mirror_price_to_market_cache(pg_conn, batch_size: int = 5000):
    """Copy daily price data into tech_analysis.market_data_cache with timeframe='1d'."""
    with pg_conn.cursor() as cur:
        cur.execute("SET search_path TO tse_input, public")
        cur.execute(
            "SELECT symbol, trading_date, adj_open, adj_high, adj_low, adj_close, adj_volume FROM price_data"
        )
        rows = (
            (
                r[0],
                "1d",
                datetime.combine(r[1], datetime.min.time()),
                r[2],
                r[3],
                r[4],
                r[5],
                r[6],
            )
            for r in cur.fetchall()
        )

    # Use a new cursor for insert to avoid fetched data buffer issues
    with pg_conn.cursor() as cur_ins:
        for batch in chunked(rows, batch_size):
            execute_values(
                cur_ins,
                """
                INSERT INTO tech_analysis.market_data_cache
                    (symbol, timeframe, ts, open, high, low, close, volume)
                VALUES %s
                ON CONFLICT (symbol, timeframe, ts) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
                """,
                batch,
                page_size=batch_size,
            )
    pg_conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Migrate SQLite data to PostgreSQL.")
    parser.add_argument("--tse-sqlite", default=r"E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db")
    parser.add_argument("--tech-sqlite", default="data/TechAnalysis.db")
    parser.add_argument("--pg-dsn", default=None, help="Postgres DSN, e.g. postgresql://user:pass@host:5432/db")
    args = parser.parse_args()

    pg_dsn = args.pg_dsn or Path(".env")
    if args.pg_dsn:
        dsn = args.pg_dsn
    else:
        # Fallback to DATABASE_URL env if available
        import os

        dsn = os.getenv("PG_DSN") or os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("PG DSN not provided. Use --pg-dsn or set PG_DSN / DATABASE_URL.")

    print(f"Connecting to Postgres: {dsn}")
    pg_conn = psycopg2.connect(dsn)

    try:
        print("Migrating TSE data ...")
        migrate_tse(Path(args.tse_sqlite), pg_conn)
        print("Populating symbols ...")
        populate_symbols_from_companies(pg_conn)
        populate_symbols_for_indices(pg_conn)
        print("Mirroring price_data into market_data_cache ...")
        mirror_price_to_market_cache(pg_conn)
        print("Done.")
    finally:
        pg_conn.close()


if __name__ == "__main__":
    main()
