"""
Sync core.dim_symbols from tse_input.* sources.

Usage:
  PYTHONIOENCODING=utf-8 python scripts/sync_dim_symbols.py --pg-dsn postgresql://gravity:gravity_db_pass@localhost:5544/gravity_tse
"""

from __future__ import annotations

import argparse

import psycopg2


def main():
    parser = argparse.ArgumentParser(description="Sync core.dim_symbols from tse_input tables.")
    parser.add_argument("--pg-dsn", required=True, help="Postgres DSN, e.g. postgresql://user:pass@host:port/db")
    args = parser.parse_args()

    conn = psycopg2.connect(args.pg_dsn)
    cur = conn.cursor()

    cur.execute("CREATE SCHEMA IF NOT EXISTS core;")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS core.dim_symbols (
          id BIGSERIAL PRIMARY KEY,
          symbol TEXT UNIQUE NOT NULL,
          company_id TEXT,
          sector_id INTEGER,
          panel_id INTEGER,
          market_id INTEGER,
          name TEXT,
          kind TEXT CHECK (kind IN ('equity','market_index','sector_index','fx','other'))
        );
        """
    )

    cur.execute(
        """
        INSERT INTO core.dim_symbols (symbol, company_id, sector_id, panel_id, market_id, name, kind)
        SELECT ticker, company_id, sector_id, panel_id, market_id, name, 'equity'
        FROM tse_input.companies
        ON CONFLICT (symbol) DO UPDATE SET
          company_id=EXCLUDED.company_id,
          sector_id=EXCLUDED.sector_id,
          panel_id=EXCLUDED.panel_id,
          market_id=EXCLUDED.market_id,
          name=EXCLUDED.name,
          kind='equity';
        """
    )

    cur.execute(
        """
        INSERT INTO core.dim_symbols (symbol, name, kind)
        SELECT 'IDX_'||index_code, index_name_fa, 'market_index'
        FROM tse_input.indices_info WHERE index_type='market'
        ON CONFLICT (symbol) DO NOTHING;
        """
    )

    cur.execute(
        """
        INSERT INTO core.dim_symbols (symbol, name, kind)
        SELECT 'SEC_'||sector_id::text, sector_name, 'sector_index'
        FROM tse_input.sectors
        ON CONFLICT (symbol) DO NOTHING;
        """
    )

    conn.commit()
    cur.close()
    conn.close()
    print("✅ dim_symbols synced.")


if __name__ == "__main__":
    main()
