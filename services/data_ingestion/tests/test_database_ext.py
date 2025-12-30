import os
import sqlite3
import unittest

from src.database import init_price_data


class TestInitPriceDataExt(unittest.TestCase):
    def setUp(self):
        self.test_db = "test_ext_tse_data.db"
        init_price_data.get_connection = staticmethod(lambda: sqlite3.connect(self.test_db))
        # Create all required tables (main, indices, USD)
        init_price_data.create_tables()
        init_price_data.create_indices_tables()

    def tearDown(self):
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

    def test_insert_companies_and_fetch(self):
        companies = [
            {
                "CompanyID": "C1",
                "Ticker": "T1",
                "Name": "N1",
                "SectorID": 1,
                "PanelID": 1,
                "MarketID": 1,
            },
            {
                "CompanyID": "C2",
                "Ticker": "T2",
                "Name": "N2",
                "SectorID": 2,
                "PanelID": 2,
                "MarketID": 2,
            },
        ]
        init_price_data.insert_companies(companies)
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM companies")
        result = cursor.fetchall()
        self.assertEqual(len(result), 2)
        conn.close()

    def test_insert_and_fetch_sectors(self):
        sectors = [
            {"SectorID": 1, "SectorName": "Test Sector 1"},
            {"SectorID": 2, "SectorName": "Test Sector 2"},
        ]
        init_price_data.insert_sectors(sectors)
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sectors")
        result = cursor.fetchall()
        self.assertEqual(len(result), 2)
        conn.close()

    def test_insert_and_fetch_markets(self):
        markets = [
            {"MarketID": 1, "MarketName": "Test Market 1"},
            {"MarketID": 2, "MarketName": "Test Market 2"},
        ]
        init_price_data.insert_markets(markets)
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM markets")
        result = cursor.fetchall()
        self.assertEqual(len(result), 2)
        conn.close()

    def test_insert_and_fetch_panels(self):
        panels = [
            {"PanelID": 1, "PanelName": "Test Panel 1"},
            {"PanelID": 2, "PanelName": "Test Panel 2"},
        ]
        init_price_data.insert_panels(panels)
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM panels")
        result = cursor.fetchall()
        self.assertEqual(len(result), 2)
        conn.close()

    def test_insert_and_fetch_price_data(self):
        records = [
            {
                "date": "2023-01-01",
                "j_date": "1401-10-11",
                "adj_open": 100,
                "adj_high": 105,
                "adj_low": 95,
                "adj_close": 102,
                "adj_final": 102,
                "adj_volume": 1000,
                "sector_id": 1,
                "ticker": "T1",
                "company_id": "C1",
            },
            {
                "date": "2023-01-02",
                "j_date": "1401-10-12",
                "adj_open": 101,
                "adj_high": 106,
                "adj_low": 96,
                "adj_close": 103,
                "adj_final": 103,
                "adj_volume": 1100,
                "sector_id": 2,
                "ticker": "T2",
                "company_id": "C2",
            },
        ]
        init_price_data.insert_price_data(records)
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM price_data")
        result = cursor.fetchall()
        self.assertEqual(len(result), 2)
        conn.close()

    def test_update_and_get_last_update(self):
        init_price_data.update_last_update("T1", "2023-01-01")
        date = init_price_data.get_last_update("T1")
        self.assertEqual(date, "2023-01-01")


if __name__ == "__main__":
    unittest.main()
