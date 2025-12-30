import io
import os
import sqlite3
import unittest
from contextlib import redirect_stdout

from src import cli


class TestCLIExt(unittest.TestCase):
    def setUp(self):
        self.test_db = os.path.join(os.path.dirname(__file__), "test_cli_ext.db")
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        cli.get_connection = staticmethod(lambda: sqlite3.connect(self.test_db))
        conn = sqlite3.connect(self.test_db)
        cur = conn.cursor()
        cur.execute("CREATE TABLE sectors (sector_id INTEGER PRIMARY KEY, sector_name TEXT UNIQUE)")
        cur.execute(
            "CREATE TABLE companies (company_id TEXT PRIMARY KEY, ticker TEXT UNIQUE, name TEXT, sector_id INTEGER)"
        )
        cur.execute(
            "CREATE TABLE price_data (id INTEGER PRIMARY KEY, date TEXT, adj_close REAL, adj_volume REAL, ticker TEXT, company_id TEXT, sector_id INTEGER)"
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

    def test_list_sectors_output(self):
        conn = sqlite3.connect(self.test_db)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sectors (sector_id, sector_name) VALUES (?, ?)", (1, "فلزات اساسی")
        )
        conn.commit()
        conn.close()
        f = io.StringIO()
        with redirect_stdout(f):
            cli.list_sectors(None)
        output = f.getvalue()
        self.assertIn("فلزات اساسی", output)

    def test_list_companies_output(self):
        conn = sqlite3.connect(self.test_db)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sectors (sector_id, sector_name) VALUES (?, ?)", (1, "فلزات اساسی")
        )
        cur.execute(
            "INSERT INTO companies (company_id, ticker, name, sector_id) VALUES (?, ?, ?, ?)",
            ("C1", "T1", "شرکت تست", 1),
        )
        conn.commit()
        conn.close()
        args = type("A", (), {})()
        args.sector_id = 1
        f = io.StringIO()
        with redirect_stdout(f):
            cli.list_companies(args)
        output = f.getvalue()
        self.assertIn("شرکت تست", output)

    def test_get_price_data_output(self):
        conn = sqlite3.connect(self.test_db)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO price_data (date, adj_close, adj_volume, ticker, company_id, sector_id) VALUES (?, ?, ?, ?, ?, ?)",
            ("2023-01-01", 100, 1000, "T1", "C1", 1),
        )
        conn.commit()
        conn.close()
        args = type("A", (), {})()
        args.ticker = "T1"
        args.limit = 1
        f = io.StringIO()
        with redirect_stdout(f):
            cli.get_price_data(args)
        output = f.getvalue()
        self.assertIn("2023-01-01", output)
        self.assertIn("100", output)
        self.assertIn("1000", output)


if __name__ == "__main__":
    unittest.main()
