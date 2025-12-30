"""
List all symbols/indices in the TSE database and count the number of available candles for each.

Usage:
  python scripts/list_symbol_counts.py
"""

from database import TSEDatabaseConnector

RAW_DB_URL = "postgresql://gravity:gravity_db_pass@localhost:5544/gravity_tse"


def main():
    db = TSEDatabaseConnector(RAW_DB_URL)
    print("Symbols with candle counts (price_data):")
    symbols = db.list_symbols(limit=1000, min_rows=1)
    for sym in symbols:
        candles = db.fetch_price_data(sym)
        print(f"{sym}: {len(candles)} candles")

    print("\nMarket indices with candle counts:")
    indices = db.list_market_indices(limit=100)
    for idx in indices:
        candles = db.fetch_market_index(idx)
        print(f"{idx}: {len(candles)} candles")

    print("\nSector indices with candle counts:")
    sectors = db.list_sector_indices(limit=100)
    for sec in sectors:
        candles = db.fetch_sector_index(sec)
        print(f"{sec}: {len(candles)} candles")


if __name__ == "__main__":
    main()
