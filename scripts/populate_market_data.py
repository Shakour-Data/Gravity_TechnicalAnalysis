"""
Populate market_data_cache from TSE database.

This script loads market data from the external TSE database and populates
the market_data_cache table in the internal TechAnalysis.db database.
"""

import importlib

from _paths import REPO_ROOT, extend_sys_path

extend_sys_path()

# Now import after directory change
config = importlib.import_module('config')
database = importlib.import_module('database')
gravity_tech_db = importlib.import_module('gravity_tech.database.database_manager')

TSE_DB_FILE = config.TSE_DB_FILE
TSEDatabaseConnector = database.TSEDatabaseConnector
DatabaseManager = gravity_tech_db.DatabaseManager
DatabaseType = gravity_tech_db.DatabaseType


def populate_market_data_cache():
    """Populate market_data_cache from TSE database."""
    print("🔄 Populating market_data_cache from TSE database...")

    # Connect to TSE database
    tse_conn = TSEDatabaseConnector(TSE_DB_FILE)

    # Connect to internal database
    db_manager = DatabaseManager(
        db_type=DatabaseType.SQLITE,
        sqlite_path=str(REPO_ROOT / "data" / "TechAnalysis.db")
    )

    # Get list of symbols with sufficient data
    symbols = tse_conn.list_symbols(limit=100, min_rows=200)
    print(f"📊 Found {len(symbols)} symbols with sufficient data")

    total_rows = 0

    for symbol in symbols:
        print(f"📈 Processing {symbol}...")

        # Fetch price data from TSE
        price_data = tse_conn.fetch_price_data(symbol)

        if not price_data:
            print(f"⚠️  No data for {symbol}, skipping")
            continue

        # Convert to format expected by upsert_market_data
        rows = []
        for candle in price_data:
            rows.append({
                "symbol": symbol,
                "timeframe": "1d",
                "timestamp": candle["timestamp"].isoformat(),
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
                "volume": candle["volume"]
            })

        # Upsert to internal database
        count = db_manager.upsert_market_data(rows)
        total_rows += count
        print(f"✅ Inserted {count} rows for {symbol}")

    print(f"🎉 Total rows inserted: {total_rows}")
    print("✅ market_data_cache population complete!")


if __name__ == "__main__":
    populate_market_data_cache()
