"""
Tests for data transformers
"""

import pytest
import math
from gravity_pipeline.transformers import DataCleaner


@pytest.fixture
def cleaner():
    """Create data cleaner instance"""
    return DataCleaner(
        remove_outliers=True,
        fill_missing_with="previous",
    )


@pytest.mark.asyncio
async def test_clean_valid_candles(cleaner):
    """Test cleaning valid OHLCV data"""
    candles = [
        {
            "timestamp": "2024-01-01",
            "open": "100",
            "high": "110",
            "low": "90",
            "close": "105",
            "volume": "1000",
        },
        {
            "timestamp": "2024-01-02",
            "open": "105",
            "high": "115",
            "low": "95",
            "close": "110",
            "volume": "1200",
        }
    ]
    
    cleaned = await cleaner.transform(candles)
    
    assert len(cleaned) == 2
    assert cleaned[0]["open"] == 100.0
    assert cleaned[0]["close"] == 105.0
    assert isinstance(cleaned[0]["volume"], float)


@pytest.mark.asyncio
async def test_skip_invalid_ohlc(cleaner):
    """Test skipping invalid OHLC data"""
    cleaner.fill_missing_with = "skip"
    
    candles = [
        {
            "timestamp": "2024-01-01",
            "open": "100",
            "high": "90",  # high < low!
            "low": "110",
            "close": "105",
            "volume": "1000",
        }
    ]
    
    cleaned = await cleaner.transform(candles)
    
    # Should skip invalid candle
    assert len(cleaned) == 0


@pytest.mark.asyncio
async def test_remove_outliers(cleaner):
    """Test outlier removal"""
    cleaner.fill_missing_with = "skip"
    
    candles = [
        {
            "timestamp": "2024-01-01",
            "open": "100",
            "high": "250",  # 150% change - outlier!
            "low": "90",
            "close": "240",
            "volume": "1000",
        }
    ]
    
    cleaned = await cleaner.transform(candles)
    
    # Should remove outlier
    assert len(cleaned) == 0


@pytest.mark.asyncio
async def test_handle_missing_values_skip(cleaner):
    """Test skipping candles with missing values"""
    cleaner.fill_missing_with = "skip"
    
    candles = [
        {
            "timestamp": "2024-01-01",
            "open": "100",
            "high": "110",
            # missing: low
            "close": "105",
            "volume": "1000",
        }
    ]
    
    cleaned = await cleaner.transform(candles)
    
    # Should skip due to missing value
    assert len(cleaned) == 0


@pytest.mark.asyncio
async def test_fill_missing_zero(cleaner):
    """Test filling missing values with zeros"""
    cleaner.fill_missing_with = "zero"
    
    candles = [
        {
            "timestamp": "2024-01-01",
            "open": "100",
            "high": "110",
            # missing: low
            "close": "105",
            "volume": "1000",
        }
    ]
    
    cleaned = await cleaner.transform(candles)
    
    # Should fill with zero
    assert len(cleaned) == 1
    assert cleaned[0]["low"] == 0.0


@pytest.mark.asyncio
async def test_convert_string_to_float(cleaner):
    """Test converting string prices to float"""
    candles = [
        {
            "timestamp": "2024-01-01",
            "open": "100.5",
            "high": "110.75",
            "low": "90.25",
            "close": "105.5",
            "volume": "1000.0",
        }
    ]
    
    cleaned = await cleaner.transform(candles)
    
    assert cleaned[0]["open"] == 100.5
    assert cleaned[0]["high"] == 110.75
    assert isinstance(cleaned[0]["volume"], float)


@pytest.mark.asyncio
async def test_reject_nan_inf(cleaner):
    """Test rejecting NaN and Inf values"""
    cleaner.fill_missing_with = "skip"
    
    candles = [
        {
            "timestamp": "2024-01-01",
            "open": "100",
            "high": float('inf'),  # Inf!
            "low": "90",
            "close": "105",
            "volume": "1000",
        }
    ]
    
    cleaned = await cleaner.transform(candles)
    
    # Should skip due to Inf
    assert len(cleaned) == 0


@pytest.mark.asyncio
async def test_stats(cleaner):
    """Test cleaner statistics"""
    candles = [
        {
            "timestamp": "2024-01-01",
            "open": "100",
            "high": "110",
            "low": "90",
            "close": "105",
            "volume": "1000",
        }
    ]
    
    await cleaner.transform(candles)
    
    stats = cleaner.get_stats()
    assert stats["processed"] >= 1
