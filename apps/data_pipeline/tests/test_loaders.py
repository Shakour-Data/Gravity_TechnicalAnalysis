"""
Tests for data loaders
"""

import os
import sqlite3

import pytest
from gravity_pipeline.loaders import SQLiteLoader


@pytest.fixture
def temp_db():
    """Create temporary SQLite database"""
    db_path = "./test_gravity.db"
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def sqlite_loader(temp_db):
    """Create SQLite loader instance"""
    return SQLiteLoader(db_path=temp_db)


@pytest.mark.asyncio
async def test_validate_connection(sqlite_loader):
    """Test SQLite connection validation"""
    result = await sqlite_loader.validate_connection()
    assert result is True


@pytest.mark.asyncio
async def test_load_valid_candles(sqlite_loader):
    """Test loading valid candles"""
    candles = [
        {
            "symbol": "BTCUSDT",
            "timestamp": "2024-01-01",
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": 1000.0,
        },
        {
            "symbol": "BTCUSDT",
            "timestamp": "2024-01-02",
            "open": 105.0,
            "high": 115.0,
            "low": 95.0,
            "close": 110.0,
            "volume": 1200.0,
        },
    ]

    loaded = await sqlite_loader.load(candles)

    assert loaded == 2
    assert sqlite_loader.loaded_count == 2


@pytest.mark.asyncio
async def test_table_creation(sqlite_loader, temp_db):
    """Test automatic table creation"""
    assert sqlite_loader.auto_create_table is True

    candles = [
        {
            "symbol": "TEST",
            "timestamp": "2024-01-01",
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": 1000.0,
        }
    ]

    await sqlite_loader.load(candles)

    # Verify table was created
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{sqlite_loader.table_name}'"
    )
    result = cursor.fetchone()
    conn.close()

    assert result is not None


@pytest.mark.asyncio
async def test_batch_loading(sqlite_loader):
    """Test batch loading"""
    sqlite_loader.batch_size = 10

    # Create 25 candles (should load in 3 batches)
    candles = [
        {
            "symbol": f"SYM{i}",
            "timestamp": f"2024-01-{i:02d}",
            "open": float(100 + i),
            "high": float(110 + i),
            "low": float(90 + i),
            "close": float(105 + i),
            "volume": float(1000 + i * 10),
        }
        for i in range(1, 26)
    ]

    loaded = await sqlite_loader.load(candles)

    assert loaded == 25


@pytest.mark.asyncio
async def test_duplicate_handling(sqlite_loader):
    """Test handling of duplicate candles"""

    # Load first set
    candles1 = [
        {
            "symbol": "BTCUSDT",
            "timestamp": "2024-01-01",
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": 1000.0,
        }
    ]

    loaded1 = await sqlite_loader.load(candles1)
    assert loaded1 == 1

    # Load again with same symbol/timestamp but different price
    candles2 = [
        {
            "symbol": "BTCUSDT",
            "timestamp": "2024-01-01",
            "open": 101.0,
            "high": 111.0,
            "low": 91.0,
            "close": 106.0,
            "volume": 1100.0,
        }
    ]

    loaded2 = await sqlite_loader.load(candles2)
    assert loaded2 == 1  # Should replace, not insert duplicate


@pytest.mark.asyncio
async def test_stats(sqlite_loader):
    """Test loader statistics"""
    candles = [
        {
            "symbol": "TEST",
            "timestamp": "2024-01-01",
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": 1000.0,
        }
    ]

    await sqlite_loader.load(candles)

    stats = sqlite_loader.get_stats()
    assert stats["loaded"] == 1
    assert stats["errors"] == 0
