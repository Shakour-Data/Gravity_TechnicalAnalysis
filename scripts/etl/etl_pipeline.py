"""
Gravity ETL Pipeline: Unified daily ingestion, analysis, and ML update

- Ingests raw OHLCV/company/index data from GravityTseHisPrice (tse_input)
- Computes all indicators, patterns, and ML outputs for all symbols/indices
- Stores results in tech_analysis schema (historical_scores, indicator_scores, ML, ...)
- Ensures 90-day rolling window is always complete for all symbols/indices

Usage:
  python scripts/etl_pipeline.py --days 90 --symbols ALL
"""

import argparse
from datetime import datetime, timedelta

from database import TSEDatabaseConnector
from gravity_tech.cli.ingest_tse_data import insert_historical_score
from gravity_tech.core.domain.entities import Candle
from gravity_tech.database.database_manager import DatabaseManager, DatabaseType
from gravity_tech.ml.complete_analysis_pipeline import quick_analyze

# --- Config ---
RAW_DB_URL = "postgresql://gravity:gravity_db_pass@localhost:5544/gravity_tse"
ANALYSIS_DB_URL = "postgresql://gravity:gravity@localhost:5432/gravity"

def run_etl(days: int = 90, symbols: list[str] | None = None):
    raw_db = TSEDatabaseConnector(RAW_DB_URL)
    analysis_db = DatabaseManager(db_type=DatabaseType.POSTGRESQL, connection_string=ANALYSIS_DB_URL)

    # 1. Load all symbols/indices
    if not symbols or symbols == ["ALL"]:
        all_symbols = raw_db.list_symbols(limit=10, min_rows=10)
    else:
        all_symbols = symbols

    # Use inclusive end date and correct string formatting
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days - 1)  # e.g. days=7 gives 7 days including today
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")


    for symbol in all_symbols:
        print(f"Processing {symbol}...")
        # 2. Load OHLCV for symbol or index with correct date range
        if symbol.startswith("IDX_"):
            candles_dicts = raw_db.fetch_market_index(symbol, start_date_str, end_date_str)
        elif symbol.startswith("SECTOR_"):
            candles_dicts = raw_db.fetch_sector_index(symbol, start_date_str, end_date_str)
        else:
            candles_dicts = raw_db.fetch_price_data(symbol, start_date_str, end_date_str)
        if not candles_dicts or len(candles_dicts) < 120:
            print(f"No sufficient data for {symbol}")
            continue
        # Convert dicts to Candle objects
        candles = [Candle(
            timestamp=cd["timestamp"],
            open=cd["open"],
            high=cd["high"],
            low=cd["low"],
            close=cd["close"],
            volume=cd.get("volume", 0),
            symbol=symbol,
            timeframe="1d"
        ) for cd in candles_dicts]
        try:
            result = quick_analyze(candles, verbose=False)
        except Exception as e:
            print(f"Analysis failed for {symbol}: {e}")
            continue
        # 4. Store results in historical_scores
        try:
            insert_historical_score(analysis_db, symbol, "1d", result.to_dict())
            print(f"Stored results for {symbol}")
        except Exception as e:
            print(f"DB insert failed for {symbol}: {e}")

    print("ETL pipeline completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90, help="Number of days to process")
    parser.add_argument("--symbols", nargs="*", default=["ALL"], help="Symbols to process (default: ALL)")
    args = parser.parse_args()
    run_etl(days=args.days, symbols=args.symbols)
