"""
Migration helper: copy TSE SQLite data into Postgres (tse_input schema).

Usage:
    PYTHONPATH=apps/analysis_api/src python scripts/migrate_tse_sqlite_to_pg.py \
        --sqlite services/data_ingestion/data/tse_data.db \
        --pg postgresql://gravity:gravity_db_pass@localhost:5545/tech_analysis

Notes:
- Creates schema tse_input and tables compatible with TSEDatabaseConnector.
- Copies: sectors, markets, panels, companies, price_data, last_updates,
          indices_info, market_indices, sector_indices, usd_prices.
- Assumes daily timeframe data; copies all rows from SQLite.
"""

import argparse
import sqlite3
import sys
from collections.abc import Iterable
from pathlib import Path

import psycopg2
import psycopg2.extras

TABLE_DEFS: tuple[str, str] = (
    (
        "sectors",
        """
        CREATE TABLE IF NOT EXISTS tse_input.sectors (
            sector_id INTEGER PRIMARY KEY,
            sector_name TEXT UNIQUE,
            sector_name_en TEXT,
            us_sector TEXT
        )
        """,
    ),
    (
        "markets",
        """
        CREATE TABLE IF NOT EXISTS tse_input.markets (
            market_id INTEGER PRIMARY KEY,
            market_name TEXT UNIQUE
        )
        """,
    ),
    (
        "panels",
        """
        CREATE TABLE IF NOT EXISTS tse_input.panels (
            panel_id INTEGER PRIMARY KEY,
            panel_name TEXT UNIQUE
        )
        """,
    ),
    (
        "companies",
        """
        CREATE TABLE IF NOT EXISTS tse_input.companies (
            company_id TEXT PRIMARY KEY,
            ticker TEXT UNIQUE,
            name TEXT,
            sector_id INTEGER,
            panel_id INTEGER,
            market_id INTEGER,
            FOREIGN KEY(sector_id) REFERENCES tse_input.sectors(sector_id),
            FOREIGN KEY(panel_id) REFERENCES tse_input.panels(panel_id),
            FOREIGN KEY(market_id) REFERENCES tse_input.markets(market_id)
        )
        """,
    ),
    (
        "price_data",
        """
        CREATE TABLE IF NOT EXISTS tse_input.price_data (
            id SERIAL PRIMARY KEY,
            trading_date DATE,
            j_date TEXT,
            adj_open REAL,
            adj_high REAL,
            adj_low REAL,
            adj_close REAL,
            adj_final REAL,
            adj_volume REAL,
            symbol TEXT,
            ticker TEXT,
            company_id TEXT,
            FOREIGN KEY(company_id) REFERENCES tse_input.companies(company_id),
            UNIQUE(ticker, trading_date)
        )
        """,
    ),
    (
        "last_updates",
        """
        CREATE TABLE IF NOT EXISTS tse_input.last_updates (
            symbol TEXT PRIMARY KEY,
            last_date TEXT
        )
        """,
    ),
    (
        "indices_info",
        """
        CREATE TABLE IF NOT EXISTS tse_input.indices_info (
            index_code TEXT PRIMARY KEY,
            index_name_fa TEXT NOT NULL,
            index_name_en TEXT,
            index_type TEXT NOT NULL CHECK(index_type IN ('market', 'sector'))
        )
        """,
    ),
    (
        "market_indices",
        """
        CREATE TABLE IF NOT EXISTS tse_input.market_indices (
            id SERIAL PRIMARY KEY,
            index_code TEXT NOT NULL,
            j_date TEXT NOT NULL,
            trading_date DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            FOREIGN KEY(index_code) REFERENCES tse_input.indices_info(index_code),
            UNIQUE(index_code, trading_date)
        )
        """,
    ),
    (
        "sector_indices",
        """
        CREATE TABLE IF NOT EXISTS tse_input.sector_indices (
            id SERIAL PRIMARY KEY,
            sector_code INTEGER NOT NULL,
            j_date TEXT NOT NULL,
            trading_date DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            FOREIGN KEY(sector_code) REFERENCES tse_input.sectors(sector_id),
            UNIQUE(sector_code, trading_date)
        )
        """,
    ),
    (
        "usd_prices",
        """
        CREATE TABLE IF NOT EXISTS tse_input.usd_prices (
            id SERIAL PRIMARY KEY,
            trading_date DATE,
            j_date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            final REAL,
            UNIQUE(trading_date)
        )
        """,
    ),
)


def copy_rows(
    sqlite_conn: sqlite3.Connection,
    pg_conn: psycopg2.extensions.connection,
    table: str,
    query: str,
    columns: Iterable[str],
    target_table: str | None = None,
    chunk_size: int = 5000,
) -> int:
    target = target_table or f"tse_input.{table}"
    total = 0
    with sqlite_conn as sconn, pg_conn.cursor() as cur:
        sconn.row_factory = sqlite3.Row
        src = sconn.execute(query)
        while True:
            batch = src.fetchmany(chunk_size)
            if not batch:
                break
            psycopg2.extras.execute_batch(
                cur,
                f"INSERT INTO {target} ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))}) "
                f"ON CONFLICT DO NOTHING",
                [tuple(r[c] for c in columns) for r in batch],
                page_size=chunk_size,
            )
            total += len(batch)
            pg_conn.commit()
    return total


def main(sqlite_path: Path, pg_dsn: str) -> None:
    if not sqlite_path.exists():
        print(f"SQLite path not found: {sqlite_path}", file=sys.stderr)
        sys.exit(1)

    sconn = sqlite3.connect(sqlite_path)
    pconn = psycopg2.connect(pg_dsn)
    pconn.autocommit = False

    with pconn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS tse_input;")
        for _, ddl in TABLE_DEFS:
            cur.execute(ddl)
    pconn.commit()

    total = {}
    total["sectors"] = copy_rows(
        sconn,
        pconn,
        "sectors",
        "SELECT sector_id, sector_name, sector_name_en, us_sector FROM sectors",
        ["sector_id", "sector_name", "sector_name_en", "us_sector"],
        "tse_input.sectors",
    )
    total["markets"] = copy_rows(
        sconn,
        pconn,
        "markets",
        "SELECT market_id, market_name FROM markets",
        ["market_id", "market_name"],
        "tse_input.markets",
    )
    total["panels"] = copy_rows(
        sconn,
        pconn,
        "panels",
        "SELECT panel_id, panel_name FROM panels",
        ["panel_id", "panel_name"],
        "tse_input.panels",
    )
    total["companies"] = copy_rows(
        sconn,
        pconn,
        "companies",
        "SELECT company_id, ticker, name, sector_id, panel_id, market_id FROM companies",
        ["company_id", "ticker", "name", "sector_id", "panel_id", "market_id"],
        "tse_input.companies",
    )
    total["price_data"] = copy_rows(
        sconn,
        pconn,
        "price_data",
        """
        SELECT date as trading_date, j_date, adj_open, adj_high, adj_low, adj_close,
               adj_final, adj_volume, ticker, company_id, ticker as symbol
        FROM price_data
        """,
        [
            "trading_date",
            "j_date",
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
            "adj_final",
            "adj_volume",
            "ticker",
            "company_id",
            "symbol",
        ],
        "tse_input.price_data",
    )
    total["last_updates"] = copy_rows(
        sconn,
        pconn,
        "last_updates",
        "SELECT symbol, last_date FROM last_updates",
        ["symbol", "last_date"],
        "tse_input.last_updates",
    )
    total["indices_info"] = copy_rows(
        sconn,
        pconn,
        "indices_info",
        "SELECT index_code, index_name_fa, index_name_en, index_type FROM indices_info",
        ["index_code", "index_name_fa", "index_name_en", "index_type"],
        "tse_input.indices_info",
    )
    total["market_indices"] = copy_rows(
        sconn,
        pconn,
        "market_indices",
        "SELECT index_code, j_date, date as trading_date, open, high, low, close FROM market_indices",
        ["index_code", "j_date", "trading_date", "open", "high", "low", "close"],
        "tse_input.market_indices",
    )
    total["sector_indices"] = copy_rows(
        sconn,
        pconn,
        "sector_indices",
        "SELECT sector_code, j_date, date as trading_date, open, high, low, close FROM sector_indices",
        ["sector_code", "j_date", "trading_date", "open", "high", "low", "close"],
        "tse_input.sector_indices",
    )
    total["usd_prices"] = copy_rows(
        sconn,
        pconn,
        "usd_prices",
        "SELECT date as trading_date, j_date, open, high, low, close, final FROM usd_prices",
        ["trading_date", "j_date", "open", "high", "low", "close", "final"],
        "tse_input.usd_prices",
    )

    pconn.close()
    sconn.close()
    for k, v in total.items():
        print(f"{k}: {v} rows copied")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", default="services/data_ingestion/data/tse_data.db")
    parser.add_argument(
        "--pg", default="postgresql://gravity:gravity_db_pass@localhost:5545/tech_analysis"
    )
    args = parser.parse_args()
    main(Path(args.sqlite), args.pg)
