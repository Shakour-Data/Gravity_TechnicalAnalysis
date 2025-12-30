"""
Database Initialization Script
===============================
This script initializes the TSE database with the correct schema.
It should be run once at the beginning or when you need to reset the database.

Usage: python fix_db.py
"""

import os
import sqlite3
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DB_FILE


def initialize_database():
    """Initialize the database with all required tables and structure."""

    # Backup existing database if it exists
    if os.path.exists(DB_FILE):
        backup_file = DB_FILE.replace(".db", "_backup.db")
        print(f"⚠️  Database exists. Creating backup at: {backup_file}")
        import shutil

        shutil.copy2(DB_FILE, backup_file)

    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    print("\n" + "=" * 80)
    print("INITIALIZING TSE DATABASE")
    print("=" * 80)

    # 1. Create indices_info table
    print("\n1. Creating indices_info table...")
    cursor.execute("DROP TABLE IF EXISTS market_indices")
    cursor.execute("DROP TABLE IF EXISTS indices_info")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS indices_info (
            index_code TEXT PRIMARY KEY,
            index_name_fa TEXT NOT NULL,
            index_name_en TEXT,
            index_type TEXT NOT NULL CHECK(index_type IN ('market', 'sector'))
        )
    """)
    print("   ✓ indices_info table created")

    # 2. Create market_indices table (OHLC only)
    print("\n2. Creating market_indices table...")
    cursor.execute("""
        CREATE TABLE market_indices (
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
    print("   ✓ market_indices table created")

    # 3. Create sectors table
    print("\n3. Creating sectors table...")
    cursor.execute("DROP TABLE IF EXISTS sector_indices")
    cursor.execute("DROP TABLE IF EXISTS companies")
    cursor.execute("DROP TABLE IF EXISTS price_data")
    cursor.execute("DROP TABLE IF EXISTS sectors")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sectors (
            sector_id INTEGER PRIMARY KEY,
            sector_name TEXT UNIQUE,
            sector_name_en TEXT,
            us_sector TEXT
        )
    """)
    print("   ✓ sectors table created")

    # 4. Create sector_indices table (OHLC only)
    print("\n4. Creating sector_indices table...")
    cursor.execute("""
        CREATE TABLE sector_indices (
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
    print("   ✓ sector_indices table created")

    # 5. Create markets table
    print("\n5. Creating markets table...")
    cursor.execute("DROP TABLE IF EXISTS markets")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS markets (
            market_id INTEGER PRIMARY KEY,
            market_name TEXT UNIQUE
        )
    """)
    print("   ✓ markets table created")

    # 6. Create panels table
    print("\n6. Creating panels table...")
    cursor.execute("DROP TABLE IF EXISTS panels")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS panels (
            panel_id INTEGER PRIMARY KEY,
            panel_name TEXT UNIQUE
        )
    """)
    print("   ✓ panels table created")

    # 7. Create companies table (without subsector_id and other empty columns)
    print("\n7. Creating companies table...")
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
    print("   ✓ companies table created")

    # 8. Create price_data table (without sector_id)
    print("\n8. Creating price_data table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            j_date TEXT,
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
    print("   ✓ price_data table created")

    # 9. Create usd_prices table
    print("\n9. Creating usd_prices table...")
    cursor.execute("DROP TABLE IF EXISTS usd_prices")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usd_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE,
            j_date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            final REAL
        )
    """)
    print("   ✓ usd_prices table created")

    # 10. Create last_updates table
    print("\n10. Creating last_updates table...")
    cursor.execute("DROP TABLE IF EXISTS last_updates")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS last_updates (
            symbol TEXT PRIMARY KEY,
            last_date TEXT
        )
    """)
    print("   ✓ last_updates table created")

    # 11. Create indexes for better performance
    print("\n11. Creating indexes for better performance...")
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_price_data_ticker ON price_data(ticker)",
        "CREATE INDEX IF NOT EXISTS idx_price_data_date ON price_data(date)",
        "CREATE INDEX IF NOT EXISTS idx_price_data_company ON price_data(company_id)",
        "CREATE INDEX IF NOT EXISTS idx_market_indices_code ON market_indices(index_code)",
        "CREATE INDEX IF NOT EXISTS idx_market_indices_date ON market_indices(date)",
        "CREATE INDEX IF NOT EXISTS idx_sector_indices_code ON sector_indices(sector_code)",
        "CREATE INDEX IF NOT EXISTS idx_sector_indices_date ON sector_indices(date)",
        "CREATE INDEX IF NOT EXISTS idx_usd_prices_date ON usd_prices(date)",
    ]

    for idx_sql in indexes:
        cursor.execute(idx_sql)
    print(f"   ✓ Created {len(indexes)} indexes")

    conn.commit()
    conn.close()

    print("\n" + "=" * 80)
    print("✅ DATABASE INITIALIZED SUCCESSFULLY!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Load basic data: python main.py load-basic-data")
    print("  2. Load market indices: python main.py load-market-indices")
    print("  3. Load sector indices: python main.py load-sector-indices")
    print("  4. Fetch price data: python main.py fetch")


if __name__ == "__main__":
    initialize_database()
