"""
Sample Data Generation Utilities

Author: Gravity Tech Team
Date: 2024
Version: 1.0
License: MIT

Helper functions for generating sample data for testing and demo purposes.
"""

from datetime import datetime, timedelta

from gravity_tech.models.schemas import Candle


def generate_sample_candles(
    num_candles: int = 100,
    base_price: float = 40000,
    trend: str = "sideways"
) -> list[Candle]:
    """
    Generate sample candles for testing.

    Args:
        num_candles: Number of candles required
        base_price: Base price
        trend: Trend type ("uptrend", "downtrend", "sideways")

    Returns:
        List of sample candles

    Example:
        >>> candles = generate_sample_candles(100, 40000, "uptrend")
        >>> print(len(candles))
        100
    """
    candles = []
    current_price = base_price
    base_time = datetime.now() - timedelta(hours=num_candles)

    # Set trend direction
    if trend == "uptrend":
        trend_multiplier = 1
    elif trend == "downtrend":
        trend_multiplier = -1
    else:  # sideways
        trend_multiplier = 0

    for i in range(num_candles):
        # Simulate price movement
        trend_movement = trend_multiplier * (i * 50)
        noise = ((i % 20) - 10) * 100

        open_price = current_price + trend_movement
        close_price = open_price + noise
        high_price = max(open_price, close_price) + 200
        low_price = min(open_price, close_price) - 200

        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=float(open_price),
            high=float(high_price),
            low=float(low_price),
            close=float(close_price),
            volume=float(1000 + (i * 50))
        ))

        # Update current price for next candle
        current_price = close_price

    return candles


def generate_volatile_candles(num_candles: int = 100) -> list[Candle]:
    """
    Generate sample candles with high volatility.

    Args:
        num_candles: Number of candles required

    Returns:
        List of volatile candles
    """
    import random

    candles = []
    base_price = 40000
    base_time = datetime.now() - timedelta(hours=num_candles)

    for i in range(num_candles):
        volatility = random.uniform(500, 2000)
        open_price = base_price + random.uniform(-volatility, volatility)
        close_price = open_price + random.uniform(-volatility, volatility)
        high_price = max(open_price, close_price) + random.uniform(0, volatility)
        low_price = min(open_price, close_price) - random.uniform(0, volatility)

        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=float(open_price),
            high=float(high_price),
            low=float(low_price),
            close=float(close_price),
            volume=float(random.uniform(500, 5000))
        ))

        base_price = close_price

    return candles
