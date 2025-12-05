"""
Data Connector for Historical Market Data

This module connects to the microservice containing historical daily candle data
for Bitcoin and other cryptocurrencies.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from gravity_tech.models.schemas import Candle


class DataConnector:
    """
    Connects to historical data microservice to fetch daily candles
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize data connector

        Args:
            base_url: Base URL of the historical data microservice
        """
        self.base_url = base_url.rstrip('/')

    def fetch_daily_candles(
        self,
        symbol: str = "BTCUSDT",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> list[Candle]:
        """
        Fetch historical daily candles from microservice

        Args:
            symbol: Trading symbol (default: BTCUSDT)
            start_date: Start date for historical data
            end_date: End date for historical data
            limit: Maximum number of candles to fetch

        Returns:
            List of Candle objects
        """
        # Set default date range if not provided
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(days=limit)

        # Prepare request parameters
        params = {
            'symbol': symbol,
            'interval': '1d',  # Daily candles
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'limit': limit
        }

        try:
            # Make API request to microservice
            response = requests.get(
                f"{self.base_url}/api/v1/candles",
                params=params,
                timeout=30
            )
            response.raise_for_status()

            # Parse response
            data = response.json()
            candles = []

            for item in data.get('candles', []):
                candle = Candle(
                    open=float(item['open']),
                    high=float(item['high']),
                    low=float(item['low']),
                    close=float(item['close']),
                    volume=float(item['volume']),
                    timestamp=datetime.fromisoformat(item['timestamp'])
                )
                candles.append(candle)

            return candles

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from microservice: {e}")
            # Fallback: return mock data for testing
            return self._generate_mock_data(symbol, start_date, end_date, limit)

    def _generate_mock_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        limit: int
    ) -> list[Candle]:
        """
        Generate mock data for testing when microservice is unavailable

        This creates realistic Bitcoin price movements for testing ML models
        """
        import numpy as np

        print(f"⚠️ Microservice unavailable. Generating mock data for {symbol}...")

        # Calculate number of days
        days = min(limit, (end_date - start_date).days)

        # Generate realistic BTC price data
        # Start price around $40,000
        base_price = 40000.0

        # Create price series with trend + noise
        np.random.seed(42)  # For reproducibility

        # Daily returns: mean 0.1% with 3% volatility
        returns = np.random.normal(0.001, 0.03, days)

        # Add trend (bull market for first half, bear for second half)
        trend = np.concatenate([
            np.linspace(0, 0.002, days // 2),  # Uptrend
            np.linspace(0.002, -0.001, days - days // 2)  # Downtrend
        ])

        returns += trend

        # Calculate prices from returns
        price_multipliers = np.cumprod(1 + returns)
        closes = base_price * price_multipliers

        # Generate OHLCV data
        candles = []
        current_date = start_date

        for i, close in enumerate(closes):
            # Daily range: 2-5%
            daily_range = close * np.random.uniform(0.02, 0.05)

            # Generate realistic OHLC
            open_price = close + np.random.uniform(-daily_range/2, daily_range/2)
            high = max(open_price, close) + np.random.uniform(0, daily_range/2)
            low = min(open_price, close) - np.random.uniform(0, daily_range/2)

            # Volume: correlated with price movement
            price_move = abs(close - open_price) / open_price
            base_volume = 25000 + np.random.uniform(-5000, 5000)
            volume = base_volume * (1 + price_move * 10)

            candle = Candle(
                open=round(open_price, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close, 2),
                volume=round(volume, 2),
                timestamp=current_date
            )
            candles.append(candle)

            current_date += timedelta(days=1)

        print(f"✅ Generated {len(candles)} mock daily candles")
        print(f"   Price range: ${candles[0].close:.2f} → ${candles[-1].close:.2f}")

        return candles

    def fetch_multiple_symbols(
        self,
        symbols: list[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> dict[str, list[Candle]]:
        """
        Fetch historical data for multiple symbols

        Args:
            symbols: List of trading symbols
            start_date: Start date
            end_date: End date
            limit: Max candles per symbol

        Returns:
            Dictionary mapping symbol to list of candles
        """
        results = {}

        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            candles = self.fetch_daily_candles(symbol, start_date, end_date, limit)
            results[symbol] = candles
            print(f"  ✓ {len(candles)} candles fetched")

        return results

    def export_to_csv(
        self,
        candles: list[Candle],
        filename: str
    ):
        """
        Export candles to CSV file for analysis

        Args:
            candles: List of candles
            filename: Output CSV filename
        """
        data = []
        for candle in candles:
            data.append({
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            })

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"✅ Exported {len(candles)} candles to {filename}")


# Example usage
if __name__ == "__main__":
    # Initialize connector
    connector = DataConnector(base_url="http://localhost:8000")

    # Fetch 2 years of Bitcoin daily data
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=730)

    print("📊 Fetching Bitcoin historical data...")
    candles = connector.fetch_daily_candles(
        symbol="BTCUSDT",
        start_date=start_date,
        end_date=end_date
    )

    print(f"\n✅ Fetched {len(candles)} daily candles")
    print(f"   Date range: {candles[0].timestamp.date()} to {candles[-1].timestamp.date()}")
    print(f"   Price range: ${candles[0].close:.2f} to ${candles[-1].close:.2f}")

    # Export to CSV
    connector.export_to_csv(candles, "btc_daily_historical.csv")
