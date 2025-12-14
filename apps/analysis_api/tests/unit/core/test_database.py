"""
Unit tests for src/database.py

Tests database connection and data retrieval functionality.
"""

import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

from src.database import TSEDatabaseConnector


class TestTSEDatabaseConnector:
    """Test TSEDatabaseConnector functionality."""

    def test_init(self):
        """Test TSEDatabaseConnector initialization."""
        connector = TSEDatabaseConnector("test.db")
        assert connector.db_file == "test.db"

    def test_get_connection(self):
        """Test database connection creation."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            connector = TSEDatabaseConnector(tmp.name)

            conn = connector.get_connection()
            assert isinstance(conn, sqlite3.Connection)

            # Check that foreign keys are enabled
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys")
            result = cursor.fetchone()
            assert result[0] == 1  # Foreign keys enabled

            conn.close()

    @patch('sqlite3.connect')
    def test_get_connection_with_foreign_keys(self, mock_connect):
        """Test that foreign keys are enabled on connection."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        connector = TSEDatabaseConnector("test.db")
        conn = connector.get_connection()

        # Verify foreign keys pragma was executed
        mock_conn.execute.assert_called_with("PRAGMA foreign_keys = ON")
        assert conn == mock_conn

    def test_list_symbols(self):
        """Test listing symbols from database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            # Create a test database with price_data table
            conn = sqlite3.connect(tmp.name)
            conn.execute("""
                CREATE TABLE price_data (
                    ticker TEXT,
                    date TEXT,
                    adj_open REAL,
                    adj_high REAL,
                    adj_low REAL,
                    adj_close REAL,
                    adj_volume INTEGER
                )
            """)

            # Insert test data for multiple tickers
            prices = [
                ('ABC', '2023-01-01', 100.0, 105.0, 95.0, 102.0, 1000),
                ('ABC', '2023-01-02', 102.0, 108.0, 100.0, 106.0, 1200),
                ('ABC', '2023-01-03', 106.0, 112.0, 104.0, 110.0, 1300),
                ('DEF', '2023-01-01', 200.0, 205.0, 195.0, 202.0, 2000),
                ('DEF', '2023-01-02', 202.0, 208.0, 200.0, 206.0, 2200),
                ('DEF', '2023-01-03', 206.0, 212.0, 204.0, 210.0, 2300),
            ]
            conn.executemany("""
                INSERT INTO price_data (ticker, date, adj_open, adj_high, adj_low, adj_close, adj_volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, prices)
            conn.commit()
            conn.close()

            connector = TSEDatabaseConnector(tmp.name)
            result = connector.list_symbols(limit=2, min_rows=2)

            assert isinstance(result, list)
            assert len(result) <= 2
            assert all(isinstance(symbol, str) for symbol in result)

    def test_list_market_indices(self):
        """Test listing market indices."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            # Create test database
            conn = sqlite3.connect(tmp.name)
            conn.execute("""
                CREATE TABLE market_indices (
                    index_code TEXT PRIMARY KEY,
                    index_name_fa TEXT
                )
            """)

            indices = [('INDEX1', 'شاخص اول'), ('INDEX2', 'شاخص دوم')]
            conn.executemany("INSERT INTO market_indices (index_code, index_name_fa) VALUES (?, ?)", indices)
            conn.commit()
            conn.close()

            connector = TSEDatabaseConnector(tmp.name)
            result = connector.list_market_indices(limit=5)

            assert isinstance(result, list)
            assert len(result) <= 5

    def test_list_sector_indices(self):
        """Test listing sector indices."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            # Create test database
            conn = sqlite3.connect(tmp.name)
            conn.execute("""
                CREATE TABLE sector_indices (
                    sector_code TEXT PRIMARY KEY,
                    sector_name TEXT
                )
            """)

            sectors = [('SEC1', 'بخش اول'), ('SEC2', 'بخش دوم')]
            conn.executemany("INSERT INTO sector_indices (sector_code, sector_name) VALUES (?, ?)", sectors)
            conn.commit()
            conn.close()

            connector = TSEDatabaseConnector(tmp.name)
            result = connector.list_sector_indices(limit=10)

            assert isinstance(result, list)
            assert len(result) <= 10

    def test_fetch_price_data(self):
        """Test fetching price data for a symbol."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            # Create test database
            conn = sqlite3.connect(tmp.name)
            conn.execute("""
                CREATE TABLE price_data (
                    ticker TEXT,
                    date TEXT,
                    adj_open REAL,
                    adj_high REAL,
                    adj_low REAL,
                    adj_close REAL,
                    adj_volume INTEGER
                )
            """)

            # Insert test data
            prices = [
                ('ABC', '2023-01-01', 100.0, 105.0, 95.0, 102.0, 1000),
                ('ABC', '2023-01-02', 102.0, 108.0, 100.0, 106.0, 1200),
            ]
            conn.executemany("""
                INSERT INTO price_data (ticker, date, adj_open, adj_high, adj_low, adj_close, adj_volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, prices)
            conn.commit()
            conn.close()

            connector = TSEDatabaseConnector(tmp.name)
            result = connector.fetch_price_data('ABC')

            assert isinstance(result, list)
            assert len(result) == 2
            assert all(isinstance(record, dict) for record in result)
            assert 'timestamp' in result[0]
            assert 'open' in result[0]
            assert 'close' in result[0]

    def test_fetch_price_data_with_date_range(self):
        """Test fetching price data with date range."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            # Create test database
            conn = sqlite3.connect(tmp.name)
            conn.execute("""
                CREATE TABLE price_data (
                    ticker TEXT,
                    date TEXT,
                    adj_open REAL,
                    adj_high REAL,
                    adj_low REAL,
                    adj_close REAL,
                    adj_volume INTEGER
                )
            """)

            # Insert test data
            prices = [
                ('ABC', '2023-01-01', 100.0, 105.0, 95.0, 102.0, 1000),
                ('ABC', '2023-01-02', 102.0, 108.0, 100.0, 106.0, 1200),
                ('ABC', '2023-01-03', 106.0, 112.0, 104.0, 110.0, 1300),
            ]
            conn.executemany("""
                INSERT INTO price_data (ticker, date, adj_open, adj_high, adj_low, adj_close, adj_volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, prices)
            conn.commit()
            conn.close()

            connector = TSEDatabaseConnector(tmp.name)
            result = connector.fetch_price_data('ABC', start_date='2023-01-02', end_date='2023-01-02')

            assert len(result) == 1
            assert result[0]['timestamp'].strftime('%Y-%m-%d') == '2023-01-02'

    def test_fetch_market_index(self):
        """Test fetching market index data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            # Create test database
            conn = sqlite3.connect(tmp.name)
            conn.execute("""
                CREATE TABLE market_indices (
                    index_code TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL
                )
            """)

            data = [
                ('INDEX1', '2023-01-01', 1000.0, 1010.0, 990.0, 1005.0),
                ('INDEX1', '2023-01-02', 1005.0, 1020.0, 1000.0, 1015.0),
            ]
            conn.executemany("""
                INSERT INTO market_indices (index_code, date, open, high, low, close)
                VALUES (?, ?, ?, ?, ?, ?)
            """, data)
            conn.commit()
            conn.close()

            connector = TSEDatabaseConnector(tmp.name)
            result = connector.fetch_market_index('INDEX1')

            assert isinstance(result, list)
            assert len(result) == 2
            assert all(isinstance(record, dict) for record in result)

    def test_fetch_sector_index(self):
        """Test fetching sector index data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            # Create test database
            conn = sqlite3.connect(tmp.name)
            conn.execute("""
                CREATE TABLE sector_indices (
                    sector_code TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL
                )
            """)

            data = [
                ('SEC1', '2023-01-01', 500.0, 510.0, 490.0, 505.0),
                ('SEC1', '2023-01-02', 505.0, 520.0, 500.0, 515.0),
            ]
            conn.executemany("""
                INSERT INTO sector_indices (sector_code, date, open, high, low, close)
                VALUES (?, ?, ?, ?, ?, ?)
            """, data)
            conn.commit()
            conn.close()

            connector = TSEDatabaseConnector(tmp.name)
            result = connector.fetch_sector_index('SEC1')

            assert isinstance(result, list)
            assert len(result) == 2
            assert all(isinstance(record, dict) for record in result)
