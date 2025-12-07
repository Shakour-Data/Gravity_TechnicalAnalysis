"""
General Utility Functions

This module provides general utility functions for data processing,
conversions, validation, and calculations.

Author: Gravity Tech Team
Date: December 5, 2025
Version: 1.0.0
License: MIT
"""

import re
from datetime import datetime, timedelta
from typing import Any

# Timeframe and Conversion Utilities
# ═══════════════════════════════════════════════════════════════════

def convert_timeframe_to_seconds(timeframe: str) -> int:
    """
    Convert timeframe string to seconds.

    Args:
        timeframe: Timeframe string (e.g., "1m", "1h", "1d")

    Returns:
        Timeframe in seconds

    Examples:
        >>> convert_timeframe_to_seconds("1m")
        60
        >>> convert_timeframe_to_seconds("1h")
        3600
    """
    timeframe = timeframe.lower().strip()

    # Extract number and unit
    match = re.match(r'^(\d+)([mhdw])$', timeframe)
    if not match:
        raise ValueError(f"Invalid timeframe format: {timeframe}")

    number = int(match.group(1))
    unit = match.group(2)

    multipliers = {
        'm': 60,        # minutes
        'h': 3600,      # hours
        'd': 86400,     # days
        'w': 604800     # weeks
    }

    return number * multipliers[unit]


def convert_price_units(price: float, from_unit: str = "USD", to_unit: str = "USD") -> float:
    """
    Convert price between different units/currencies.

    Args:
        price: Price value
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted price

    Note: This is a placeholder - real implementation would use exchange rates
    """
    if from_unit == to_unit:
        return price

    # Placeholder conversion rates (would be fetched from API in real implementation)
    rates = {
        "USD": 1.0,
        "EUR": 0.85,
        "GBP": 0.73,
        "JPY": 110.0,
        "IRR": 42000.0  # Iranian Rial
    }

    if from_unit not in rates or to_unit not in rates:
        raise ValueError(f"Unsupported currency conversion: {from_unit} to {to_unit}")

    # Convert to USD first, then to target
    usd_price = price / rates[from_unit]
    return usd_price * rates[to_unit]


# ═══════════════════════════════════════════════════════════════════
# Validation Helpers
# ═══════════════════════════════════════════════════════════════════

def validate_symbol(symbol: str) -> bool:
    """
    Validate trading symbol format.

    Args:
        symbol: Trading symbol

    Returns:
        True if valid
    """
    if not symbol or not isinstance(symbol, str):
        return False

    # Basic validation: alphanumeric, underscores, dots allowed
    pattern = r'^[A-Z0-9_.]{1,20}$'
    return bool(re.match(pattern, symbol.upper()))


def validate_timeframe(timeframe: str) -> bool:
    """
    Validate timeframe format.

    Args:
        timeframe: Timeframe string

    Returns:
        True if valid
    """
    try:
        convert_timeframe_to_seconds(timeframe)
        return True
    except ValueError:
        return False


def validate_price_range(price: float, min_price: float = 0.0001, max_price: float = 1000000.0) -> bool:
    """
    Validate price is within acceptable range.

    Args:
        price: Price value
        min_price: Minimum acceptable price
        max_price: Maximum acceptable price

    Returns:
        True if valid
    """
    return isinstance(price, int | float) and min_price <= price <= max_price


def validate_volume(volume: float) -> bool:
    """
    Validate volume value.

    Args:
        volume: Volume value

    Returns:
        True if valid
    """
    return isinstance(volume, int | float) and volume >= 0


def validate_candle_data(candle: Any) -> bool:
    """
    Validate candle data structure.

    Args:
        candle: Candle object or dict

    Returns:
        True if valid
    """
    required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    if hasattr(candle, '__dict__'):
        # Object with attributes
        return all(hasattr(candle, field) for field in required_fields)
    elif isinstance(candle, dict):
        # Dictionary
        return all(field in candle for field in required_fields)
    else:
        return False


# ═══════════════════════════════════════════════════════════════════
# Data Aggregation
# ═══════════════════════════════════════════════════════════════════

def aggregate_ohlc(candles: list, timeframe: str) -> list:
    """
    Aggregate OHLC data to higher timeframe.

    Args:
        candles: List of candle data
        timeframe: Target timeframe

    Returns:
        Aggregated candles
    """
    if not candles:
        return []

    # Placeholder implementation
    # Real implementation would group candles by timeframe and aggregate OHLCV
    return candles[:1]  # Return first candle as placeholder


def aggregate_multiple_timeframes(candles: list, timeframes: list[str]) -> dict:
    """
    Aggregate data to multiple timeframes.

    Args:
        candles: Base candle data
        timeframes: List of target timeframes

    Returns:
        Dict with timeframe keys and aggregated data
    """
    result = {}
    for tf in timeframes:
        result[tf] = aggregate_ohlc(candles, tf)
    return result


# ═══════════════════════════════════════════════════════════════════
# Statistical Helpers
# ═══════════════════════════════════════════════════════════════════

def calculate_sma(data: list[float], period: int) -> list[float]:
    """
    Calculate Simple Moving Average.

    Args:
        data: Price data
        period: SMA period

    Returns:
        SMA values
    """
    if len(data) < period:
        return []

    sma_values = []
    for i in range(period - 1, len(data)):
        sma = sum(data[i - period + 1:i + 1]) / period
        sma_values.append(sma)

    return sma_values


def calculate_std_dev(data: list[float], period: int) -> list[float]:
    """
    Calculate Standard Deviation.

    Args:
        data: Price data
        period: Period for calculation

    Returns:
        Standard deviation values
    """
    if len(data) < period:
        return []

    std_values = []
    for i in range(period - 1, len(data)):
        window = data[i - period + 1:i + 1]
        mean = sum(window) / period
        variance = sum((x - mean) ** 2 for x in window) / period
        std_values.append(variance ** 0.5)

    return std_values


def calculate_returns(data: list[float]) -> list[float]:
    """
    Calculate price returns.

    Args:
        data: Price data

    Returns:
        Return values
    """
    if len(data) < 2:
        return []

    returns = []
    for i in range(1, len(data)):
        ret = (data[i] - data[i-1]) / data[i-1] if data[i-1] != 0 else 0
        returns.append(ret)

    return returns


def calculate_correlation(data1: list[float], data2: list[float]) -> float:
    """
    Calculate correlation between two datasets.

    Args:
        data1: First dataset
        data2: Second dataset

    Returns:
        Correlation coefficient
    """
    if len(data1) != len(data2) or len(data1) < 2:
        return 0.0

    n = len(data1)
    sum_x = sum(data1)
    sum_y = sum(data2)
    sum_xy = sum(x * y for x, y in zip(data1, data2, strict=True))
    sum_x2 = sum(x ** 2 for x in data1)
    sum_y2 = sum(y ** 2 for y in data2)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5

    return numerator / denominator if denominator != 0 else 0.0


def normalize_data(data: list[float]) -> list[float]:
    """
    Normalize data to 0-1 range.

    Args:
        data: Input data

    Returns:
        Normalized data
    """
    if not data:
        return []

    min_val = min(data)
    max_val = max(data)

    if max_val == min_val:
        return [0.5] * len(data)  # All same values

    return [(x - min_val) / (max_val - min_val) for x in data]


# ═══════════════════════════════════════════════════════════════════
# Cache Helpers
# ═══════════════════════════════════════════════════════════════════

def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate cache key from arguments.

    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Cache key string
    """
    import hashlib

    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))

    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def cache_expiration_check(cache_time: datetime, ttl_seconds: int) -> bool:
    """
    Check if cache entry has expired.

    Args:
        cache_time: When item was cached
        ttl_seconds: Time to live in seconds

    Returns:
        True if expired
    """
    return (datetime.now() - cache_time).total_seconds() > ttl_seconds


# ═══════════════════════════════════════════════════════════════════
# DateTime Helpers
# ═══════════════════════════════════════════════════════════════════

def get_market_hours(market: str = "US") -> dict:
    """
    Get market trading hours.

    Args:
        market: Market identifier

    Returns:
        Dict with open/close times
    """
    # Placeholder - real implementation would have market-specific hours
    markets = {
        "US": {"open": "09:30", "close": "16:00", "timezone": "America/New_York"},
        "EU": {"open": "09:00", "close": "17:30", "timezone": "Europe/London"},
        "ASIA": {"open": "09:00", "close": "15:00", "timezone": "Asia/Tokyo"}
    }

    return markets.get(market.upper(), markets["US"])


def is_market_open(market: str = "US") -> bool:
    """
    Check if market is currently open.

    Args:
        market: Market identifier

    Returns:
        True if market is open
    """
    # Placeholder - real implementation would check current time vs market hours
    return True  # Assume always open for testing


def get_next_trading_day(current_date: datetime, market: str = "US") -> datetime:
    """
    Get next trading day.

    Args:
        current_date: Current date
        market: Market identifier

    Returns:
        Next trading day
    """
    # Skip weekends
    next_day = current_date + timedelta(days=1)
    while next_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
        next_day += timedelta(days=1)

    return next_day


# ═══════════════════════════════════════════════════════════════════
# Error Handling
# ═══════════════════════════════════════════════════════════════════

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero

    Returns:
        Division result or default
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except (ZeroDivisionError, TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float.

    Args:
        value: Value to convert
        default: Default value

    Returns:
        Float value or default
    """
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def fill_missing_values(data: list, method: str = "forward") -> list:
    """
    Fill missing values in data.

    Args:
        data: Data with potential None values
        method: Fill method ("forward", "backward", "zero", "mean")

    Returns:
        Data with missing values filled
    """
    if not data:
        return data

    filled_data = data.copy()

    if method == "forward":
        last_valid = None
        for i, val in enumerate(filled_data):
            if val is not None:
                last_valid = val
            elif last_valid is not None:
                filled_data[i] = last_valid

    elif method == "backward":
        next_valid = None
        for i in range(len(filled_data) - 1, -1, -1):
            if filled_data[i] is not None:
                next_valid = filled_data[i]
            elif next_valid is not None:
                filled_data[i] = next_valid

    elif method == "zero":
        filled_data = [val if val is not None else 0 for val in filled_data]

    elif method == "mean":
        valid_values = [val for val in filled_data if val is not None]
        if valid_values:
            mean_val = sum(valid_values) / len(valid_values)
            filled_data = [val if val is not None else mean_val for val in filled_data]

    return filled_data