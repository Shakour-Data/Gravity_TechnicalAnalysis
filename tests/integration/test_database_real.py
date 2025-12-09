"""
Integration tests for src/database.py using real TSE database

Tests database operations with actual TSE data from:
E:\\Shakour\\MyProjects\\GravityTseHisPrice\\data\\tse_data.db

These tests require the real database to be available.
"""

import os
import sqlite3
from datetime import datetime

import pytest

from src.database import TSEDatabaseConnector


@pytest.fixture
def real_db_connector():
    """Create a connector to the real TSE database."""
    # Use the actual TSE database path
    db_path = r"E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db"

    # Skip test if database doesn't exist
    if not os.path.exists(db_path):
        pytest.skip(f"Real TSE database not found at {db_path}")

    return TSEDatabaseConnector(db_path)


@pytest.mark.integration
class TestTSEDatabaseConnectorRealData:
    """Integration tests using real TSE database data."""

    def test_real_database_connection(self, real_db_connector):
        """Test connection to real TSE database."""
        conn = real_db_connector.get_connection()
        assert isinstance(conn, sqlite3.Connection)

        # Test that we can execute a simple query
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM price_data")
        count = cursor.fetchone()[0]
        assert count > 0
        assert count > 1000000  # Should have millions of records

        conn.close()

    def test_real_list_symbols(self, real_db_connector):
        """Test listing symbols from real database."""
        symbols = real_db_connector.list_symbols(limit=10, min_rows=100)

        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert len(symbols) <= 10
        assert all(isinstance(symbol, str) for symbol in symbols)

        # Should return Persian stock symbols
        assert any('آ' in symbol or 'ا' in symbol for symbol in symbols)

    def test_real_list_market_indices(self, real_db_connector):
        """Test listing market indices from real database."""
        indices = real_db_connector.list_market_indices(limit=5)

        assert isinstance(indices, list)
        assert len(indices) >= 0  # Might be empty if no indices

        if indices:  # Only check if there are indices
            assert all(isinstance(idx, str) for idx in indices)

    def test_real_list_sector_indices(self, real_db_connector):
        """Test listing sector indices from real database."""
        sectors = real_db_connector.list_sector_indices(limit=10)

        assert isinstance(sectors, list)
        assert len(sectors) >= 0

        if sectors:
            assert all(isinstance(sector, str) for sector in sectors)

    def test_real_fetch_price_data_known_symbol(self, real_db_connector):
        """Test fetching price data for a known symbol."""
        # Get a symbol from the database first
        symbols = real_db_connector.list_symbols(limit=1, min_rows=50)
        if not symbols:
            pytest.skip("No symbols with sufficient data found")

        symbol = symbols[0]

        # Fetch data for this symbol
        data = real_db_connector.fetch_price_data(symbol)

        # Limit to 20 records for testing
        data = data[:20] if len(data) > 20 else data

        assert isinstance(data, list)
        assert len(data) > 0
        assert all(isinstance(record, dict) for record in data)

        # Check that records have required fields
        for record in data:
            assert 'timestamp' in record
            assert 'open' in record
            assert 'high' in record
            assert 'low' in record
            assert 'close' in record
            assert 'volume' in record

    def test_real_fetch_price_data_with_date_range(self, real_db_connector):
        """Test fetching price data with date range."""
        symbols = real_db_connector.list_symbols(limit=1, min_rows=100)
        if not symbols:
            pytest.skip("No symbols with sufficient data found")

        symbol = symbols[0]

        # Fetch data for a specific date range
        data = real_db_connector.fetch_price_data(
            symbol,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )

        assert isinstance(data, list)

        if data:  # Only check if data exists for this range
            assert all(isinstance(record, dict) for record in data)
            # Check that dates are within range
            for record in data:
                record_date = record['timestamp'].date()
                assert record_date >= datetime(2023, 1, 1).date()
                assert record_date <= datetime(2023, 12, 31).date()

    def test_real_fetch_market_index(self, real_db_connector):
        """Test fetching market index data."""
        # Get available market indices
        indices = real_db_connector.list_market_indices(limit=1)
        if not indices:
            pytest.skip("No market indices found")

        index_code = indices[0]

        data = real_db_connector.fetch_market_index(index_code)

        # Limit to 10 records for testing
        data = data[:10] if len(data) > 10 else data

        assert isinstance(data, list)
        if data:
            assert all(isinstance(record, dict) for record in data)

    def test_real_fetch_sector_index(self, real_db_connector):
        """Test fetching sector index data."""
        # Get available sector indices
        sectors = real_db_connector.list_sector_indices(limit=1)
        if not sectors:
            pytest.skip("No sector indices found")

        sector_code = sectors[0]

        data = real_db_connector.fetch_sector_index(sector_code)

        # Limit to 10 records for testing
        data = data[:10] if len(data) > 10 else data

        assert isinstance(data, list)
        if data:
            assert all(isinstance(record, dict) for record in data)

    def test_real_price_data_structure(self, real_db_connector):
        """Test that real price data has correct structure."""
        symbols = real_db_connector.list_symbols(limit=1, min_rows=10)
        if not symbols:
            pytest.skip("No symbols found")

        symbol = symbols[0]
        data = real_db_connector.fetch_price_data(symbol)

        # Limit to 5 records for testing
        data = data[:5] if len(data) > 5 else data

        assert len(data) > 0

        for record in data:
            # Check OHLC relationships
            assert record['high'] >= record['open']
            assert record['high'] >= record['close']
            assert record['low'] <= record['open']
            assert record['low'] <= record['close']

            # Volume should be positive
            assert record['volume'] > 0

            # Prices should be reasonable (not zero or negative)
            assert record['open'] > 0
            assert record['high'] > 0
            assert record['low'] > 0
            assert record['close'] > 0

    def test_real_database_performance(self, real_db_connector):
        """Test that database queries perform reasonably."""
        import time

        symbols = real_db_connector.list_symbols(limit=1, min_rows=100)
        if not symbols:
            pytest.skip("No symbols found")

        symbol = symbols[0]

        # Time the query
        start_time = time.time()
        data = real_db_connector.fetch_price_data(symbol)
        end_time = time.time()

        query_time = end_time - start_time

        # Should complete in reasonable time (less than 1 second)
        assert query_time < 1.0
        assert len(data) > 0
