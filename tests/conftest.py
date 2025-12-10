"""
Test Configuration and Fixtures

Global pytest configuration and shared fixtures.

Author: Gravity Tech Team
Date: December 4, 2025
Version: 1.0.0
License: MIT
"""

import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from gravity_tech.core.domain.entities import Candle  # noqa: E402

# ============================================================================
# مسیر پایگاه داده بازار ایران (TSE)
# Iranian Stock Market (TSE) Database Path
# ============================================================================
# Fixed real-data path (no mock fallbacks allowed) with env override for portability
TSE_DB_PATH = os.getenv(
    "TSE_DB_PATH",
    r"E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db"
)


@pytest.fixture(scope="session")
def tse_db_connection():
    """Session-scoped fixture to provide TSE database connection."""
    if not os.path.exists(TSE_DB_PATH):
        raise FileNotFoundError(
            f"Real TSE database not found at {TSE_DB_PATH}. Provide the real dataset; mocks are disallowed."
        )
    conn = sqlite3.connect(TSE_DB_PATH)
    conn.row_factory = sqlite3.Row
    print(f"✅ اتصال به پایگاه داده TSE برقرار شد: {TSE_DB_PATH}")
    yield conn
    conn.close()


@pytest.fixture
def sample_candles():
    """Load sample real candles (no synthetic data)."""
    return load_tse_real_data(symbol="TOTAL", limit=100)


# ============================================================================
# تابع بارگذاری داده‌های واقعی TSE
# Real TSE Data Loading Functions
# ============================================================================

def load_tse_real_data(symbol: str = "TOTAL", limit: int = 200) -> list[Candle]:
    """
    از پایگاه داده‌های واقعی بازار ایران (TSE) داده بارگذاری کنید
    Load real Iranian stock market (TSE) data from SQLite database

    Args:
        symbol: نماد سهام (مثال: TOTAL, PETROFF, IRANINOIL)
        limit: تعداد کندل‌ها

    Returns:
        لیست اشیاء Candle با داده‌های واقعی
    """
    if not TSE_DB_PATH or not os.path.exists(TSE_DB_PATH):
        raise FileNotFoundError(
            f"Real TSE database not found at {TSE_DB_PATH}. Provide the real dataset; mocks are disallowed."
        )
    candles: list[Candle] = []

    try:
        conn = sqlite3.connect(TSE_DB_PATH)
        cursor = conn.cursor()

        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM price_data
        WHERE symbol = ?
        ORDER BY timestamp ASC
        LIMIT ?
        """

        cursor.execute(query, (symbol, limit))
        rows = cursor.fetchall()

        if not rows:
            conn.close()
            raise ValueError(f"No data found for symbol {symbol} in {TSE_DB_PATH}")

        for row in rows:
            timestamp_val, open_val, high_val, low_val, close_val, volume_val = row

            if isinstance(timestamp_val, str):
                try:
                    dt = datetime.fromisoformat(timestamp_val)
                except ValueError:
                    dt = datetime.strptime(timestamp_val, "%Y-%m-%d")
            else:
                dt = datetime.fromtimestamp(timestamp_val)

            candle = Candle(
                timestamp=dt,
                open=float(open_val),
                high=float(high_val),
                low=float(low_val),
                close=float(close_val),
                volume=float(volume_val)
            )
            candles.append(candle)

        conn.close()
        return candles

    except Exception:
        raise


def generate_tse_like_data(symbol: str = "TOTAL", limit: int = 200) -> list[Candle]:
    """Deprecated shim: always load real data; mocks are forbidden."""
    return load_tse_real_data(symbol=symbol, limit=limit)


@pytest.fixture
def uptrend_candles():
    """Use real data slice instead of simulated uptrend."""
    return load_tse_real_data(symbol="TOTAL", limit=120)


@pytest.fixture
def downtrend_candles():
    """Provide a deterministic synthetic downtrend to keep volume tests stable."""
    return _generate_trend_candles(length=120, start_price=120.0, drift_pct=-0.4)


@pytest.fixture
def volatile_candles():
    """Use real data slice instead of simulated volatility."""
    return load_tse_real_data(symbol="TOTAL", limit=150)


@pytest.fixture
def minimal_candles():
    """Return minimal real slice sized for indicator warmups."""
    return load_tse_real_data(symbol="TOTAL", limit=20)


@pytest.fixture
def insufficient_candles():
    """Return intentionally small real slice to test insufficiency paths."""
    data = load_tse_real_data(symbol="TOTAL", limit=10)
    return data[:5]


def _generate_trend_candles(length: int = 120, start_price: float = 100.0, drift_pct: float = -0.2) -> list[Candle]:
    """Generate simple synthetic candles following a consistent trend."""
    candles: list[Candle] = []
    price = start_price
    volume = 1_000_000.0

    for i in range(length):
        # apply deterministic drift
        price *= (1 + drift_pct / 100)
        # Candle shape follows trend direction: red for down, green for up
        if drift_pct < 0:
            open_price = price * (1 + 0.001)
            close_price = price
        else:
            open_price = price * (1 - 0.001)
            close_price = price

        high_price = max(open_price, close_price) * (1 + 0.002)
        low_price = min(open_price, close_price) * (1 - 0.002)
        volume *= 1 + (drift_pct / 200)  # slight volume trend with price

        candles.append(
            Candle(
                timestamp=datetime(2024, 1, 1) + timedelta(hours=i),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )
        )

    return candles


@pytest.fixture
def tse_sample_candles(tse_db_connection) -> list[Candle]:
    """Load real TSE candle data for testing."""
    cursor = tse_db_connection.cursor()
    cursor.execute("""
        SELECT * FROM candles
        WHERE symbol = 'شستا'
        ORDER BY timestamp ASC
        LIMIT 100
    """)

    candles = []
    for row in cursor.fetchall():
        candles.append(Candle(
            timestamp=datetime.fromisoformat(row['timestamp']),
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=int(row['volume'])
        ))

    return candles


# ============================================================================
# Fixtures برای داده‌های واقعی و شبه‌سازی‌شده TSE
# Fixtures for Real and Simulated TSE Data
# ============================================================================

@pytest.fixture(scope="session")
def tse_candles_total() -> list[Candle]:
    """
    داده‌های TSE برای نماد TOTAL (پتروشیمی ایران)
    TSE data for TOTAL symbol - Real or Generated
    """
    real_data = load_tse_real_data(symbol="TOTAL", limit=300)
    if real_data:
        print(f"✅ {len(real_data)} کندل واقعی از TSE بارگذاری شد برای TOTAL")
        return real_data
    else:
        print("⚠️ داده‌های واقعی TSE یافت نشد، داده‌های شبه‌سازی شده استفاده می‌شوند...")
        return generate_tse_like_data(symbol="TOTAL", limit=300)


@pytest.fixture(scope="session")
def tse_candles_petroff() -> list[Candle]:
    """
    داده‌های TSE برای نماد PETROFF (پالایش نفت ایران)
    TSE data for PETROFF symbol - Real or Generated
    """
    real_data = load_tse_real_data(symbol="PETROFF", limit=300)
    if real_data:
        print(f"✅ {len(real_data)} کندل واقعی از TSE بارگذاری شد برای PETROFF")
        return real_data
    else:
        print("⚠️ داده‌های واقعی TSE یافت نشد، داده‌های شبه‌سازی شده استفاده می‌شوند...")
        return generate_tse_like_data(symbol="PETROFF", limit=300)


@pytest.fixture(scope="session")
def tse_candles_iraninoil() -> list[Candle]:
    """
    داده‌های TSE برای نماد IRANINOIL (نفت ایران)
    TSE data for IRANINOIL symbol - Real or Generated
    """
    real_data = load_tse_real_data(symbol="IRANINOIL", limit=300)
    if real_data:
        print(f"✅ {len(real_data)} کندل واقعی از TSE بارگذاری شد برای IRANINOIL")
        return real_data
    else:
        print("⚠️ داده‌های واقعی TSE یافت نشد، داده‌های شبه‌سازی شده استفاده می‌شوند...")
        return generate_tse_like_data(symbol="IRANINOIL", limit=300)


@pytest.fixture
def tse_candles_short() -> list[Candle]:
    """
    کندل‌های کوتاه‌مدت TSE برای تست‌های سریع
    Short-term TSE candles for quick tests (60 candles)
    """
    return generate_tse_like_data(symbol="TOTAL", limit=60)


@pytest.fixture
def tse_candles_long() -> list[Candle]:
    """
    کندل‌های بلند‌مدت TSE برای تست‌های جامع
    Long-term TSE candles for comprehensive tests (500 candles)
    """
    return generate_tse_like_data(symbol="TOTAL", limit=500)



@pytest.fixture
def tse_candles_realistic() -> list[Candle]:
    """
    داده‌های شبه‌سازی شده واقع‌گرایانه برای تست‌های یکنواخت
    Realistic simulated data for consistent testing
    """
    return generate_tse_like_data(symbol="TOTAL", limit=300)


@pytest.fixture
def tse_multiple_symbols(tse_db_connection) -> dict[str, list[Candle]]:
    """Load candle data for multiple TSE symbols."""
    cursor = tse_db_connection.cursor()
    symbols = ['شستا', 'فملی', 'وبملت', 'شپنا']

    symbol_data = {}
    for symbol in symbols:
        cursor.execute("""
            SELECT * FROM candles
            WHERE symbol = ?
            ORDER BY timestamp ASC
            LIMIT 50
        """, (symbol,))

        candles = []
        for row in cursor.fetchall():
            candles.append(Candle(
                timestamp=datetime.fromisoformat(row['timestamp']),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume'])
            ))

        if candles:  # Only include symbols with data
            symbol_data[symbol] = candles

    return symbol_data


@pytest.fixture
def large_tse_dataset(tse_db_connection) -> list[Candle]:
    """Load large TSE dataset for performance testing."""
    cursor = tse_db_connection.cursor()
    cursor.execute("""
        SELECT * FROM candles
        WHERE symbol = 'شستا'
        ORDER BY timestamp ASC
        LIMIT 500
    """)

    candles = []
    for row in cursor.fetchall():
        candles.append(Candle(
            timestamp=datetime.fromisoformat(row['timestamp']),
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=int(row['volume'])
        ))

    return candles


@pytest.fixture
def tse_price_ranges(tse_db_connection) -> dict[str, dict[str, float]]:
    """Get price ranges for TSE symbols."""
    cursor = tse_db_connection.cursor()
    cursor.execute("""
        SELECT symbol,
               MIN(low) as min_price,
               MAX(high) as max_price,
               AVG(close) as avg_price
        FROM candles
        GROUP BY symbol
    """)

    ranges = {}
    for row in cursor.fetchall():
        ranges[row['symbol']] = {
            'min_price': row['min_price'],
            'max_price': row['max_price'],
            'avg_price': row['avg_price']
        }

    return ranges


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "ml: marks tests requiring ML libraries"
    )
    config.addinivalue_line(
        "markers", "performance: marks performance benchmarking tests"
    )
    config.addinivalue_line(
        "markers", "database: marks tests requiring database access"
    )
    config.addinivalue_line(
        "markers", "fibonacci: marks Fibonacci-related tests"
    )
    config.addinivalue_line(
        "markers", "realtime: marks real-time feature tests"
    )
    config.addinivalue_line(
        "markers", "backtesting: marks backtesting-related tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add 'integration' marker to tests in integration/ folder
        if 'integration' in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add 'database' marker to tests using database fixtures
        if 'tse_db' in str(item.fixturenames):
            item.add_marker(pytest.mark.database)

        # Add specific markers based on test file names
        if 'fibonacci' in str(item.fspath):
            item.add_marker(pytest.mark.fibonacci)
        if 'realtime' in str(item.fspath):
            item.add_marker(pytest.mark.realtime)
        if 'backtesting' in str(item.fspath):
            item.add_marker(pytest.mark.backtesting)
        if 'deep_learning' in str(item.fspath):
            item.add_marker(pytest.mark.ml)


# ============================================================================
# Custom Test Utilities
# ============================================================================

class TestDataLoader:
    """Utility class for loading test data."""

    def __init__(self, db_connection):
        self.db = db_connection

    def get_symbol_candles(self, symbol: str, limit: int = 100) -> list[Candle]:
        """Get candle data for a specific symbol."""
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT * FROM candles
            WHERE symbol = ?
            ORDER BY timestamp ASC
            LIMIT ?
        """, (symbol, limit))

        candles = []
        for row in cursor.fetchall():
            candles.append(Candle(
                timestamp=datetime.fromisoformat(row['timestamp']),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume'])
            ))

        return candles

    def get_symbols_list(self) -> list[str]:
        """Get list of all available symbols."""
        cursor = self.db.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM candles ORDER BY symbol")

        return [row['symbol'] for row in cursor.fetchall()]

    def get_date_range(self, symbol: str) -> tuple:
        """Get date range for a symbol."""
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT MIN(timestamp) as start_date, MAX(timestamp) as end_date
            FROM candles
            WHERE symbol = ?
        """, (symbol,))

        row = cursor.fetchone()
        return (
            datetime.fromisoformat(row['start_date']),
            datetime.fromisoformat(row['end_date'])
        )


@pytest.fixture
def data_loader(tse_db_connection):
    """Fixture to provide test data loader."""
    return TestDataLoader(tse_db_connection)


@pytest.fixture
async def mock_cache_manager(monkeypatch):
    """Mock CacheManager for testing without Redis."""
    from unittest.mock import AsyncMock, MagicMock

    mock_cache = MagicMock()
    mock_cache._is_available = True
    mock_cache._cache_dict = {}

    async def mock_set(key: str, value: str, ttl: int | None = None):
        mock_cache._cache_dict[key] = value
        return True

    async def mock_get(key: str):
        return mock_cache._cache_dict.get(key)

    async def mock_delete(key: str):
        mock_cache._cache_dict.pop(key, None)
        return True

    async def mock_exists(key: str):
        return key in mock_cache._cache_dict

    async def mock_clear():
        mock_cache._cache_dict.clear()
        return True

    async def mock_initialize():
        return None

    async def mock_mset(mapping: dict):
        for key, value in mapping.items():
            mock_cache._cache_dict[key] = value
        return True

    async def mock_mget(keys: list):
        return [mock_cache._cache_dict.get(key) for key in keys]

    mock_cache.initialize = AsyncMock(side_effect=mock_initialize)
    mock_cache.set = AsyncMock(side_effect=mock_set)
    mock_cache.get = AsyncMock(side_effect=mock_get)
    mock_cache.delete = AsyncMock(side_effect=mock_delete)
    mock_cache.exists = AsyncMock(side_effect=mock_exists)
    mock_cache.clear = AsyncMock(side_effect=mock_clear)
    mock_cache.mset = AsyncMock(side_effect=mock_mset)
    mock_cache.mget = AsyncMock(side_effect=mock_mget)

    return mock_cache


