"""
Tests for data validators
"""

import pytest
from gravity_pipeline.validators import DataQualityValidator


@pytest.fixture
def validator():
    """Create data quality validator"""
    return DataQualityValidator(
        check_ohlc=True,
        check_volume=True,
        check_timestamps=True,
        check_nan_inf=True,
    )


@pytest.mark.asyncio
async def test_validate_good_candles(validator):
    """Test validating good data"""
    candles = [
        {
            "timestamp": "2024-01-01",
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": 1000.0,
        },
        {
            "timestamp": "2024-01-02",
            "open": 105.0,
            "high": 115.0,
            "low": 95.0,
            "close": 110.0,
            "volume": 1200.0,
        }
    ]
    
    valid, invalid = await validator.validate(candles)
    
    assert valid == 2
    assert invalid == 0


@pytest.mark.asyncio
async def test_reject_invalid_ohlc(validator):
    """Test rejecting invalid OHLC"""
    candles = [
        {
            "timestamp": "2024-01-01",
            "open": 100.0,
            "high": 90.0,  # high < low!
            "low": 110.0,
            "close": 105.0,
            "volume": 1000.0,
        }
    ]
    
    valid, invalid = await validator.validate(candles)
    
    assert valid == 0
    assert invalid == 1


@pytest.mark.asyncio
async def test_reject_negative_volume(validator):
    """Test rejecting negative volume"""
    candles = [
        {
            "timestamp": "2024-01-01",
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": -100.0,  # Negative!
        }
    ]
    
    valid, invalid = await validator.validate(candles)
    
    assert valid == 0
    assert invalid == 1


@pytest.mark.asyncio
async def test_reject_nan_values(validator):
    """Test rejecting NaN values"""
    candles = [
        {
            "timestamp": "2024-01-01",
            "open": float('nan'),  # NaN!
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": 1000.0,
        }
    ]
    
    valid, invalid = await validator.validate(candles)
    
    assert valid == 0
    assert invalid == 1


@pytest.mark.asyncio
async def test_reject_inf_values(validator):
    """Test rejecting Inf values"""
    candles = [
        {
            "timestamp": "2024-01-01",
            "open": 100.0,
            "high": float('inf'),  # Inf!
            "low": 90.0,
            "close": 105.0,
            "volume": 1000.0,
        }
    ]
    
    valid, invalid = await validator.validate(candles)
    
    assert valid == 0
    assert invalid == 1


@pytest.mark.asyncio
async def test_detect_duplicates(validator):
    """Test detecting duplicate timestamps"""
    validator.check_duplicates = True
    
    candles = [
        {
            "timestamp": "2024-01-01",
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": 1000.0,
        },
        {
            "timestamp": "2024-01-01",  # Duplicate!
            "open": 101.0,
            "high": 111.0,
            "low": 91.0,
            "close": 106.0,
            "volume": 1100.0,
        }
    ]
    
    valid, invalid = await validator.validate(candles)
    
    # Should detect duplicate
    assert invalid >= 1


@pytest.mark.asyncio
async def test_partial_validation(validator):
    """Test with some valid and some invalid candles"""
    candles = [
        {  # Valid
            "timestamp": "2024-01-01",
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": 1000.0,
        },
        {  # Invalid - high < low
            "timestamp": "2024-01-02",
            "open": 100.0,
            "high": 90.0,
            "low": 110.0,
            "close": 105.0,
            "volume": 1000.0,
        },
        {  # Valid
            "timestamp": "2024-01-03",
            "open": 105.0,
            "high": 115.0,
            "low": 95.0,
            "close": 110.0,
            "volume": 1200.0,
        }
    ]
    
    valid, invalid = await validator.validate(candles)
    
    assert valid == 2
    assert invalid == 1


@pytest.mark.asyncio
async def test_stats(validator):
    """Test validator statistics"""
    candles = [
        {
            "timestamp": "2024-01-01",
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": 1000.0,
        }
    ]
    
    await validator.validate(candles)
    
    stats = validator.get_stats()
    assert stats["checked"] >= 1
    assert stats["invalid"] == 0
