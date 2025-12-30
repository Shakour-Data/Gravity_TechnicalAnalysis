import os
import sqlite3
import sys
import unittest

# Add project root to sys.path to allow imports from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import init_price_data  # noqa: E402


class TestInitPriceData(unittest.TestCase):
    def setUp(self):
        # Use a temporary database for testing
        self.test_db = "test_tse_data.db"
        # Patch the get_connection method to use the test database
        self.original_get_connection = init_price_data.get_connection
        init_price_data.get_connection = staticmethod(lambda: sqlite3.connect(self.test_db))
        # Create all required tables (main, indices, USD)
        init_price_data.create_tables()
        init_price_data.create_indices_tables()
        init_price_data.create_usd_table()

    def tearDown(self):
        # Restore the original get_connection method
        init_price_data.get_connection = self.original_get_connection
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

    def test_create_tables(self):
        conn = sqlite3.connect(self.test_db)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            expected_tables = [
                "price_data",
                "companies",
                "sectors",
                "markets",
                "panels",
                "last_updates",
            ]
            for table in expected_tables:
                self.assertIn(table, tables)
        finally:
            conn.close()

    def test_insert_sectors(self):
        sectors = [{"SectorID": 1.0, "SectorName": "Test Sector"}]
        init_price_data.insert_sectors(sectors)
        conn = sqlite3.connect(self.test_db)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sectors")
            result = cursor.fetchall()
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], (1, "Test Sector", None, None))
        finally:
            conn.close()

    def test_insert_markets(self):
        markets = [{"MarketID": 1.0, "MarketName": "Test Market"}]
        init_price_data.insert_markets(markets)
        conn = sqlite3.connect(self.test_db)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM markets")
            result = cursor.fetchall()
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], (1, "Test Market"))
        finally:
            conn.close()

    def test_insert_panels(self):
        panels = [{"PanelID": 1.0, "PanelName": "Test Panel"}]
        init_price_data.insert_panels(panels)
        conn = sqlite3.connect(self.test_db)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM panels")
            result = cursor.fetchall()
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], (1, "Test Panel"))
        finally:
            conn.close()

    def test_insert_companies(self):
        companies = [
            {
                "CompanyID": "TEST001",
                "Ticker": "TEST",
                "Name": "Test Company",
                "SectorID": 1.0,
                "IndexName": "Test Index",
                "PanelID": 1.0,
                "MarketID": 1.0,
                "BoardID": 1.0,
                "BoardName": "Test Board",
                "IndustryGroupCode": "T1",
                "IndustryGroupName": "Test Group",
                "IndustryCode": "T2",
                "IndustryName": "Test Industry",
            }
        ]
        init_price_data.insert_companies(companies)
        conn = sqlite3.connect(self.test_db)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM companies WHERE company_id = ?", ("TEST001",))
            result = cursor.fetchone()
            self.assertIsNotNone(result)
            self.assertEqual(result[0], "TEST001")
        finally:
            conn.close()

    def test_insert_price_data_dict(self):
        records = [
            {
                "Date": "2023-01-01",
                "J-Date": "1401-01-01",
                "Adj Open": 100.0,
                "Adj High": 110.0,
                "Adj Low": 90.0,
                "Adj Close": 105.0,
                "Adj Final": 105.0,
                "Value": 100000.0,
                "Ticker": "TEST",
                "CompanyID": "TEST001",
            }
        ]
        init_price_data.insert_price_data(records)
        conn = sqlite3.connect(self.test_db)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM price_data")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)
        finally:
            conn.close()

    def test_insert_price_data_tuple(self):
        records = [
            (
                "2023-01-01",
                "1401-01-01",
                100.0,
                110.0,
                90.0,
                105.0,
                105.0,
                1000.0,
                "TEST",
                "TEST001",
                None,
            )
        ]
        init_price_data.insert_price_data(records)
        conn = sqlite3.connect(self.test_db)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM price_data")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)
        finally:
            conn.close()

    def test_insert_last_updates(self):
        updates = {"TEST": "1401-01-01"}
        init_price_data.insert_last_updates(updates)
        conn = sqlite3.connect(self.test_db)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM last_updates")
            result = cursor.fetchall()
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], ("TEST", "1401-01-01"))
        finally:
            conn.close()

    def test_get_last_update(self):
        updates = {"TEST": "1401-01-01"}
        init_price_data.insert_last_updates(updates)
        last_date = init_price_data.get_last_update("TEST")
        self.assertEqual(last_date, "1401-01-01")

    def test_update_last_update(self):
        init_price_data.update_last_update("TEST", "1401-01-02")
        last_date = init_price_data.get_last_update("TEST")
        self.assertEqual(last_date, "1401-01-02")

    def test_update_price_data_sectors(self):
        # Insert test data
        init_price_data.insert_sectors([{"SectorID": 1.0, "SectorName": "Test Sector"}])
        init_price_data.insert_companies(
            [
                {
                    "CompanyID": "TEST001",
                    "Ticker": "TEST",
                    "Name": "Test Company",
                    "SectorID": 1.0,
                    "IndexName": "Test Index",
                }
            ]
        )
        init_price_data.insert_price_data(
            [
                (
                    "2023-01-01",
                    "1401-01-01",
                    100.0,
                    110.0,
                    90.0,
                    105.0,
                    105.0,
                    1000.0,
                    "TEST",
                    "TEST001",
                    None,
                )
            ]
        )
        init_price_data.update_price_data_sectors()
        conn = sqlite3.connect(self.test_db)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT sector_id FROM price_data WHERE ticker = ?", ("TEST",))
            result = cursor.fetchone()
            self.assertEqual(result, (1,))
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
