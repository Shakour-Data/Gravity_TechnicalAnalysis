import os
import sqlite3
import unittest
from unittest.mock import patch

from src.database import init_price_data


class TestInitPriceData(unittest.TestCase):
    def setUp(self):
        self.test_db = "test_init_price_data.db"
        self.patcher = patch(
            "src.database.init_price_data.get_connection",
            side_effect=lambda: sqlite3.connect(self.test_db),
        )
        self.patcher.start()
        init_price_data.create_tables()

    def tearDown(self):
        self.patcher.stop()
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

    def test_insert_sector(self):
        sectors = [
            {
                "SectorCode": 1,
                "SectorName": "Test Sector",
                "SectorName_en": "Test",
                "US_Sector": "Test",
            }
        ]
        init_price_data.insert_sectors(sectors)
        conn = sqlite3.connect(self.test_db)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sectors")
            result = cursor.fetchall()
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0][1], "Test Sector")
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
