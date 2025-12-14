"""
Generate sample TSE data for testing and training.

This script generates synthetic TSE market data and inserts it into the database.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd
from gravity_tech.config.settings import settings
from sqlalchemy import create_engine, text


def generate_tse_sample_data(num_symbols=10, num_days=365):
    """Generate sample TSE data."""
    symbols = [f"IRO{i+1:06d}" for i in range(num_symbols)]  # TSE symbol format

    data = []
    base_date = datetime.now() - timedelta(days=num_days)

    for symbol in symbols:
        # Generate price series with some trend and volatility
        base_price = np.random.uniform(1000, 50000)  # Iranian Rial prices
        prices = [base_price]

        for _ in range(1, num_days):
            # Random walk with slight upward trend
            change = np.random.normal(0.001, 0.02)  # Mean return 0.1%, std 2%
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 100))  # Floor at 100

        # Generate OHLCV data
        for i, close in enumerate(prices):
            date = base_date + timedelta(days=i)

            # Generate OHLC around close price
            volatility = 0.02  # 2% daily volatility
            high = close * (1 + abs(np.random.normal(0, volatility)))
            low = close * (1 - abs(np.random.normal(0, volatility)))
            open_price = close * (1 + np.random.normal(0, volatility/2))

            # Ensure OHLC relationships
            high = max(high, open_price, close)
            low = min(low, open_price, close)

            volume = np.random.randint(10000, 1000000)  # Trading volume

            data.append({
                'symbol': symbol,
                'timestamp': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': int(volume)
            })

    return pd.DataFrame(data)


def insert_sample_data_to_db(df):
    """Insert sample data into PostgreSQL database."""
    engine = create_engine(settings.database_url)

    try:
        # Clear existing data
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM companies"))
            conn.execute(text("DELETE FROM price_data"))
            conn.execute(text("DELETE FROM panels"))
            conn.execute(text("DELETE FROM markets"))
            conn.execute(text("DELETE FROM sectors"))
            conn.commit()
        print("✅ Cleared existing data from all tables")

        # Insert new data
        df.to_sql('price_data', engine, if_exists='append', index=False)
        print(f"✅ Inserted {len(df)} records into price_data")

        # Insert sample companies data
        companies_data = []
        sectors_data = []
        markets_data = []
        panels_data = []

        unique_symbols = df['symbol'].unique()

        for i, symbol in enumerate(unique_symbols):
            companies_data.append({
                'company_id': symbol,
                'ticker': symbol,
                'name': f'شرکت نمونه {i+1}',
                'sector_id': (i % 5) + 1,
                'panel_id': (i % 3) + 1,
                'market_id': 1
            })

        # Insert sectors
        for i in range(1, 6):
            sectors_data.append({
                'sector_id': i,
                'sector_name': f'بخش {i}',
                'sector_name_en': f'Sector {i}',
                'us_sector': f'US Sector {i}'
            })

        # Insert markets
        markets_data.append({
            'market_id': 1,
            'market_name': 'بورس تهران'
        })

        # Insert panels
        for i in range(1, 4):
            panels_data.append({
                'panel_id': i,
                'panel_name': f'تابلو {i}'
            })

        # Insert data
        pd.DataFrame(sectors_data).to_sql('sectors', engine, if_exists='append', index=False)
        pd.DataFrame(markets_data).to_sql('markets', engine, if_exists='append', index=False)
        pd.DataFrame(panels_data).to_sql('panels', engine, if_exists='append', index=False)
        pd.DataFrame(companies_data).to_sql('companies', engine, if_exists='append', index=False)

        print("✅ Inserted reference data (companies, sectors, markets, panels)")

    except Exception as e:
        print(f"❌ Error inserting data: {e}")
        raise


if __name__ == "__main__":
    print("Generating sample TSE data...")

    # Generate sample data
    df = generate_tse_sample_data(num_symbols=20, num_days=200)
    print(f"Generated {len(df)} records for {len(df['symbol'].unique())} symbols")

    # Insert to database
    insert_sample_data_to_db(df)

    print("✅ Sample TSE data generation completed!")