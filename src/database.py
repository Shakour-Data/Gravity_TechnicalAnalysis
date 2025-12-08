import logging
import sqlite3
from typing import Any

import pandas as pd

from src.config import TSE_DB_FILE

logger = logging.getLogger(__name__)

class TSEDatabaseConnector:
    """
    Connector for the External TSE Database (Input Source).
    This class handles reading market data from the pre-built SQLite database.
    """
    def __init__(self, db_file: str):
        self.db_file = db_file

    def get_connection(self) -> sqlite3.Connection:
        """Creates and returns a database connection with foreign keys enabled."""
        conn = sqlite3.connect(self.db_file)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # Note: The database is assumed to be pre-built.
    # The create_tables and insert methods are kept for reference or initialization if needed,
    # but the primary role of this class in this project is to READ data.

    def create_tables(self):
        """Creates all necessary tables if they do not exist."""
        conn = self.get_connection()
        cursor = conn.cursor()

        # 1. sectors
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sectors (
            sector_id INTEGER PRIMARY KEY,
            sector_name TEXT UNIQUE,
            sector_name_en TEXT,
            us_sector TEXT
        )
        """)

        # 2. markets
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS markets (
            market_id INTEGER PRIMARY KEY,
            market_name TEXT UNIQUE
        )
        """)

        # 3. panels
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS panels (
            panel_id INTEGER PRIMARY KEY,
            panel_name TEXT UNIQUE
        )
        """)

        # 4. companies
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            company_id TEXT PRIMARY KEY,
            ticker TEXT UNIQUE,
            name TEXT,
            sector_id INTEGER,
            panel_id INTEGER,
            market_id INTEGER,
            FOREIGN KEY(sector_id) REFERENCES sectors(sector_id),
            FOREIGN KEY(panel_id) REFERENCES panels(panel_id),
            FOREIGN KEY(market_id) REFERENCES markets(market_id)
        )
        """)

        # 5. price_data
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,         -- YYYY-MM-DD (Gregorian)
            j_date TEXT,       -- YYYY-MM-DD (Jalali)
            adj_open REAL,
            adj_high REAL,
            adj_low REAL,
            adj_close REAL,
            adj_final REAL,
            adj_volume REAL,
            ticker TEXT,
            company_id TEXT,
            FOREIGN KEY(company_id) REFERENCES companies(company_id),
            UNIQUE(ticker, date)
        )
        """)

        # 6. last_updates
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS last_updates (
            symbol TEXT PRIMARY KEY,
            last_date TEXT
        )
        """)

        # 7. indices_info
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS indices_info (
            index_code TEXT PRIMARY KEY,
            index_name_fa TEXT NOT NULL,
            index_name_en TEXT,
            index_type TEXT NOT NULL CHECK(index_type IN ('market', 'sector'))
        )
        """)

        # 8. market_indices
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_indices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            index_code TEXT NOT NULL,
            j_date TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            FOREIGN KEY(index_code) REFERENCES indices_info(index_code),
            UNIQUE(index_code, date)
        )
        """)

        # 9. sector_indices
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sector_indices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sector_code TEXT NOT NULL,
            j_date TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            FOREIGN KEY(sector_code) REFERENCES sectors(sector_id),
            UNIQUE(sector_code, date)
        )
        """)

        # 10. usd_prices
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS usd_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            j_date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            final REAL,
            UNIQUE(date)
        )
        """)

        # Create Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_ticker_date ON price_data(ticker, date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_date ON price_data(date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_index_code_date ON market_indices(index_code, date);")

        conn.commit()
        conn.close()

    def insert_indices_info(self):
        """Inserts default indices info."""
        default_indices = [
            ('CWI', 'شاخص کل', 'Overall Index', 'market'),
            ('EWI', 'شاخص هم‌وزن', 'Equal Weighted Index', 'market'),
            # Add more default indices as needed
        ]
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT OR IGNORE INTO indices_info (index_code, index_name_fa, index_name_en, index_type)
            VALUES (?, ?, ?, ?)
        """, default_indices)
        conn.commit()
        conn.close()

    def insert_price_data(self, records: list[dict[str, Any]]):
        """
        Inserts price data in bulk.
        Expected keys in records: 'Date', 'J-Date', 'Adj Open', 'Adj High', 'Adj Low',
        'Adj Close', 'Adj Final', 'Adj Volume', 'Ticker', 'CompanyID'
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        data_to_insert = []
        for r in records:
            if r.get('Ticker') == 'USD':
                continue # Skip USD, handled in usd_prices

            data_to_insert.append((
                r.get('Date'),
                r.get('J-Date'),
                r.get('Adj Open'),
                r.get('Adj High'),
                r.get('Adj Low'),
                r.get('Adj Close'),
                r.get('Adj Final'),
                r.get('Adj Volume'),
                r.get('Ticker'),
                r.get('CompanyID')
            ))

        cursor.executemany("""
            INSERT OR REPLACE INTO price_data (
                date, j_date, adj_open, adj_high, adj_low, adj_close,
                adj_final, adj_volume, ticker, company_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data_to_insert)

        conn.commit()
        conn.close()

    def insert_market_indices(self, index_code: str, index_name_fa: str, df: pd.DataFrame):
        """
        Inserts market indices from a DataFrame.
        DataFrame should have columns for Open, High, Low, Close.
        Index or a column should contain Date/J-Date.
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Ensure index info exists
        cursor.execute("INSERT OR IGNORE INTO indices_info (index_code, index_name_fa, index_type) VALUES (?, ?, 'market')", (index_code, index_name_fa))

        data_to_insert = []
        for idx, row in df.iterrows():
            # Assuming df has 'Date' and 'J-Date' columns, or they are in the index
            # Adjust based on actual DataFrame structure passed
            date_val = row.get('Date') if 'Date' in row else (idx if isinstance(idx, str) else str(idx)) # Fallback
            j_date_val = row.get('J-Date', '')

            data_to_insert.append((
                index_code,
                j_date_val,
                str(date_val),
                row.get('Open'),
                row.get('High'),
                row.get('Low'),
                row.get('Close')
            ))

        cursor.executemany("""
            INSERT OR REPLACE INTO market_indices (
                index_code, j_date, date, open, high, low, close
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, data_to_insert)

        conn.commit()
        conn.close()

    def insert_last_updates(self, updates: dict[str, str]):
        """
        Updates the last_updates table.
        updates: dict of {symbol: last_date}
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        data = list(updates.items())

        cursor.executemany("""
            INSERT OR REPLACE INTO last_updates (symbol, last_date)
            VALUES (?, ?)
        """, data)

        conn.commit()
        conn.close()

    def fetch_price_data(self, ticker: str, start_date: str | None = None, end_date: str | None = None) -> list[dict[str, Any]]:
        """
        Fetches price data for a given ticker.
        Returns a list of dictionaries compatible with Candle schema.
        """
        from datetime import datetime

        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT date, adj_open, adj_high, adj_low, adj_close, adj_volume FROM price_data WHERE ticker = ?"
        params = [ticker]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date ASC"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        candles = []
        for row in rows:
            try:
                dt = datetime.strptime(row["date"], "%Y-%m-%d")
            except ValueError:
                # Handle potential different date formats or errors
                continue

            candles.append({
                "timestamp": dt,
                "open": row["adj_open"],
                "high": row["adj_high"],
                "low": row["adj_low"],
                "close": row["adj_close"],
                "volume": row["adj_volume"]
            })

        return candles

    def fetch_market_index(self, index_code: str, start_date: str | None = None, end_date: str | None = None) -> list[dict[str, Any]]:
        """
        Fetches market index data (e.g., CWI).
        """
        from datetime import datetime

        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT date, open, high, low, close FROM market_indices WHERE index_code = ?"
        params = [index_code]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date ASC"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        candles = []
        for row in rows:
            try:
                dt = datetime.strptime(row["date"], "%Y-%m-%d")
            except ValueError:
                continue

            candles.append({
                "timestamp": dt,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": 0  # Indices often don't have volume in the same way, or it's not in this table
            })

        return candles

# Global instance for accessing the External TSE Database
tse_data_source = TSEDatabaseConnector(TSE_DB_FILE)
# Alias for backward compatibility if needed, but prefer tse_data_source
init_price_data = tse_data_source
