"""
Test Configuration and Fixtures

Global pytest configuration and shared fixtures.

Author: Gravity Tech Team
Date: December 4, 2025
Version: 1.0.0
License: MIT
"""

import sys
from pathlib import Path
import pytest
import sqlite3
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from gravity_tech.core.domain.entities import Candle

# ============================================================================
# مسیر پایگاه داده بازار ایران (TSE)
# Iranian Stock Market (TSE) Database Path
# ============================================================================
TSE_DB_PATH = r"E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db"


@pytest.fixture(scope="session")
def tse_db_connection():
    """Session-scoped fixture to provide TSE database connection."""
    # ابتدا سعی کنید از پایگاه داده واقعی بازار ایران استفاده کنید
    if os.path.exists(TSE_DB_PATH):
        try:
            conn = sqlite3.connect(TSE_DB_PATH)
            conn.row_factory = sqlite3.Row
            print(f"✅ اتصال به پایگاه داده TSE برقرار شد: {TSE_DB_PATH}")
            yield conn
            conn.close()
            return
        except Exception as e:
            print(f"⚠️ خطا در اتصال به TSE: {e}")
    
    # اگر پایگاه داده واقعی وجود نداشت، یک پایگاه داده موقت بسازید
    print(f"⚠️ پایگاه داده واقعی TSE یافت نشد: {TSE_DB_PATH}")
    print("📦 یک پایگاه داده موقت ایجاد شود...")
    
    db_path = project_root / "data" / "tse_data.db"
    os.makedirs(db_path.parent, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    # جدول ایجاد کنید اگر وجود نداشت
    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL
        )
    """)
    conn.commit()
    
    yield conn
    conn.close()


@pytest.fixture
def sample_candles():
    """Create sample candle data for testing"""
    candles = []
    base_price = 40000
    base_time = datetime.now() - timedelta(days=100)

    for i in range(100):
        open_price = base_price + (i * 10)
        close_price = open_price + ((i % 10) - 5) * 50
        high_price = max(open_price, close_price) + 100
        low_price = min(open_price, close_price) - 100

        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=1000 + (i * 10)
        ))

    return candles


# ============================================================================
# تابع بارگذاری داده‌های واقعی TSE
# Real TSE Data Loading Functions
# ============================================================================

def load_tse_real_data(symbol: str = "TOTAL", limit: int = 200) -> List[Candle]:
    """
    از پایگاه داده‌های واقعی بازار ایران (TSE) داده بارگذاری کنید
    Load real Iranian stock market (TSE) data from SQLite database
    
    Args:
        symbol: نماد سهام (مثال: TOTAL, PETROFF, IRANINOIL)
        limit: تعداد کندل‌ها
        
    Returns:
        لیست اشیاء Candle با داده‌های واقعی
    """
    candles = []
    
    if not os.path.exists(TSE_DB_PATH):
        return None
    
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
            return None
        
        for row in rows:
            timestamp_val, open_val, high_val, low_val, close_val, volume_val = row
            
            if isinstance(timestamp_val, str):
                try:
                    dt = datetime.fromisoformat(timestamp_val)
                except:
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
        
    except Exception as e:
        print(f"❌ خطا در بارگذاری داده‌های TSE: {str(e)}")
        return None


def generate_tse_like_data(symbol: str = "TOTAL", limit: int = 200) -> List[Candle]:
    """
    داده‌های شبه‌سازی شده واقع‌گرایانه برای بازار ایران
    Generate realistic simulated data similar to TSE market patterns
    
    Args:
        symbol: نماد سهام
        limit: تعداد کندل‌ها
        
    Returns:
        لیست اشیاء Candle با داده‌های شبه‌سازی شده
    """
    candles = []
    base_time = datetime(2024, 1, 1)
    
    # داده‌های ایرانی معمولاً در رنج خاصی هستند
    base_price = 10000  # قیمت پایه تریال
    volatility = 0.02  # نوسان 2%
    
    np.random.seed(42)
    price = base_price
    
    for i in range(limit):
        # شبیه‌سازی قیمت با تصادفی‌بودن و ترند
        trend = i * 0.5  # ترند آهسته
        random_change = np.random.normal(0, price * volatility)
        
        price = price + trend + random_change
        
        # باز، بالا، پایین بر اساس بسته‌شدن
        open_price = price + np.random.normal(0, price * volatility)
        close_price = price
        high_price = max(open_price, close_price) + abs(np.random.normal(0, price * volatility))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, price * volatility))
        
        # حجم معمولی برای بازار ایران
        volume = np.random.uniform(100000, 5000000)
        
        candle = Candle(
            timestamp=base_time + timedelta(days=i),
            open=float(max(open_price, 1)),
            high=float(max(high_price, 1)),
            low=float(max(low_price, 1)),
            close=float(max(close_price, 1)),
            volume=float(volume)
        )
        candles.append(candle)
    
    return candles


@pytest.fixture
def uptrend_candles():
    """Create uptrend candle data"""
    candles = []
    base_price = 40000
    base_time = datetime.now() - timedelta(days=100)

    for i in range(100):
        open_price = base_price + (i * 100)  # Clear uptrend
        close_price = open_price + 80
        high_price = close_price + 50
        low_price = open_price - 30

        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=1000 + (i * 10)
        ))

    return candles


@pytest.fixture
def downtrend_candles():
    """Create downtrend candle data"""
    candles = []
    base_price = 50000
    base_time = datetime.now() - timedelta(days=100)

    for i in range(100):
        open_price = base_price - (i * 100)  # Clear downtrend
        close_price = open_price - 80
        high_price = open_price + 30
        low_price = close_price - 50

        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=1000 + (i * 10)
        ))

    return candles


@pytest.fixture
def volatile_candles():
    """Create volatile candle data"""
    np.random.seed(42)
    candles = []
    base_price = 40000
    base_time = datetime.now() - timedelta(days=100)

    for i in range(100):
        volatility = np.random.uniform(-500, 500)
        open_price = base_price + volatility
        close_price = open_price + np.random.uniform(-300, 300)
        high_price = max(open_price, close_price) + np.random.uniform(100, 500)
        low_price = min(open_price, close_price) - np.random.uniform(100, 500)

        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=1000 + np.random.randint(0, 500)
        ))

    return candles


@pytest.fixture
def minimal_candles():
    """Create minimal candle data (14 candles for RSI, etc.)"""
    candles = []
    base_time = datetime(2024, 1, 1)

    for i in range(14):
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=100.0 + i,
            high=102.0 + i,
            low=99.0 + i,
            close=101.0 + i,
            volume=1000000
        ))

    return candles


@pytest.fixture
def insufficient_candles():
    """Create insufficient candle data (too few for most indicators)"""
    candles = []
    base_time = datetime(2024, 1, 1)

    for i in range(5):
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1000000
        ))

    return candles


@pytest.fixture
def tse_sample_candles(tse_db_connection) -> List[Candle]:
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
def tse_candles_total() -> List[Candle]:
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
def tse_candles_petroff() -> List[Candle]:
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
def tse_candles_iraninoil() -> List[Candle]:
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
def tse_candles_short() -> List[Candle]:
    """
    کندل‌های کوتاه‌مدت TSE برای تست‌های سریع
    Short-term TSE candles for quick tests (60 candles)
    """
    return generate_tse_like_data(symbol="TOTAL", limit=60)


@pytest.fixture
def tse_candles_long() -> List[Candle]:
    """
    کندل‌های بلند‌مدت TSE برای تست‌های جامع
    Long-term TSE candles for comprehensive tests (500 candles)
    """
    return generate_tse_like_data(symbol="TOTAL", limit=500)


@pytest.fixture
def tse_multiple_symbols() -> Dict[str, List[Candle]]:
    """
    داده‌های TSE برای نمادهای مختلف
    TSE data for multiple symbols
    """
    symbols = ["TOTAL", "PETROFF", "IRANINOIL"]
    result = {}
    
    for symbol in symbols:
        real_data = load_tse_real_data(symbol=symbol, limit=200)
        if real_data:
            result[symbol] = real_data
        else:
            result[symbol] = generate_tse_like_data(symbol=symbol, limit=200)
    
    return result


@pytest.fixture
def tse_candles_realistic() -> List[Candle]:
    """
    داده‌های شبه‌سازی شده واقع‌گرایانه برای تست‌های یکنواخت
    Realistic simulated data for consistent testing
    """
    return generate_tse_like_data(symbol="TOTAL", limit=300)


@pytest.fixture
def tse_multiple_symbols(tse_db_connection) -> Dict[str, List[Candle]]:
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
def large_tse_dataset(tse_db_connection) -> List[Candle]:
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
def tse_price_ranges(tse_db_connection) -> Dict[str, Dict[str, float]]:
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

    def get_symbol_candles(self, symbol: str, limit: int = 100) -> List[Candle]:
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

    def get_symbols_list(self) -> List[str]:
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

