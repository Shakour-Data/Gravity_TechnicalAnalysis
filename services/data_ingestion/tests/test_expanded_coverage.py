import os
import sqlite3
import sys
import types
import unittest
from datetime import datetime

import pandas as pd
from src import cli
from src.database import init_price_data
from src.fetcher import DataFetcher


class BaseDBTest(unittest.TestCase):
    def setUp(self):
        self.db_path = os.path.join(os.path.dirname(__file__), "expanded_test.db")
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self._orig_get_conn = init_price_data.get_connection
        init_price_data.get_connection = staticmethod(lambda: sqlite3.connect(self.db_path))
        self._orig_cli_conn = cli.get_connection
        cli.get_connection = staticmethod(lambda: sqlite3.connect(self.db_path))
        init_price_data.create_tables()
        init_price_data.create_indices_tables()
        # stub gravity_tse to avoid network
        self.stub_module = types.SimpleNamespace()
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=3, freq="D"),
                "Open": [1, 2, 3],
                "High": [2, 3, 4],
                "Low": [0.5, 1.5, 2.5],
                "Close": [1.5, 2.5, 3.5],
            }
        )
        df.index = ["1398-10-11", "1398-10-12", "1398-10-13"]
        for name in [
            "Get_CWI_History",
            "Get_EWI_History",
            "Get_CWPI_History",
            "Get_EWPI_History",
            "Get_FFI_History",
            "Get_MKT1I_History",
            "Get_INDI_History",
            "Get_ACT50_History",
            "Get_LCI30_History",
            "Get_SectorIndex_History",
            "Get_Price_History",
            "Get_USD_RIAL",
        ]:
            setattr(self.stub_module, name, lambda *_, **__: df.copy())
        sys.modules["gravity_tse"] = self.stub_module

    def tearDown(self):
        init_price_data.get_connection = self._orig_get_conn
        cli.get_connection = self._orig_cli_conn
        if "gravity_tse" in sys.modules:
            sys.modules.pop("gravity_tse")
        if os.path.exists(self.db_path):
            os.remove(self.db_path)


class TestTablesAndSchema(BaseDBTest):
    def test_tables_created(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        tables = {
            r[0]
            for r in cur.execute("select name from sqlite_master where type='table'").fetchall()
        }
        expected = {
            "sectors",
            "markets",
            "panels",
            "companies",
            "price_data",
            "last_updates",
            "indices_info",
            "market_indices",
            "sector_indices",
            "usd_prices",
        }
        self.assertTrue(expected.issubset(tables))
        conn.close()

    def test_market_indices_columns(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cols = [c[1] for c in cur.execute("pragma table_info(market_indices)")]
        for col in ["index_code", "j_date", "date", "open", "high", "low", "close"]:
            self.assertIn(col, cols)
        conn.close()


class TestInsertBasicData(BaseDBTest):
    def test_insert_reference_tables(self):
        init_price_data.insert_sectors(
            [{"SectorCode": 1, "SectorName": "A", "SectorName_en": "A", "US_Sector": "X"}]
        )
        init_price_data.insert_markets([{"MarketID": 10, "MarketName": "M"}])
        init_price_data.insert_panels([{"PanelID": 5, "PanelName": "P"}])
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        self.assertEqual(cur.execute("select count(*) from sectors").fetchone()[0], 1)
        self.assertEqual(cur.execute("select count(*) from markets").fetchone()[0], 1)
        self.assertEqual(cur.execute("select count(*) from panels").fetchone()[0], 1)
        conn.close()

    def test_insert_companies_adds_missing_sectors(self):
        companies = [
            {
                "CompanyID": "c1",
                "Ticker": "T1",
                "Name": "N1",
                "SectorCode": 99,
                "PanelID": 5,
                "MarketID": 10,
            }
        ]
        init_price_data.insert_companies(companies)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        sectors = cur.execute("select sector_id from sectors where sector_id=99").fetchone()
        self.assertIsNotNone(sectors)
        conn.close()


def _price_record(idx, value=1000, adj_final=100, ticker=None):
    ticker = ticker or f"T{idx}"
    return {
        "Date": f"2025-01-{idx:02d}",
        "J-Date": f"1403-10-{idx:02d}",
        "Adj Open": 10,
        "Adj High": 11,
        "Adj Low": 9,
        "Adj Close": 10.5,
        "Adj Final": adj_final,
        "Value": value,
        "Ticker": ticker,
        "CompanyID": f"C{idx}",
    }


class TestInsertPriceData(BaseDBTest):
    def setUp(self):
        super().setUp()
        init_price_data.insert_companies(
            [
                {"CompanyID": f"C{i}", "Ticker": f"T{i}", "Name": "N", "SectorCode": 1}
                for i in range(1, 6)
            ]
        )


def _add_price_case(idx, record):
    def test(self):
        init_price_data.insert_price_data([record])
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        row = cur.execute(
            "select date, adj_volume from price_data where ticker=?",
            (record.get("Ticker"),),
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], record.get("Date"))
        # adj_volume derived from value / adj_final when available
        expected_volume = record.get("Value", 0) / record.get(
            "Adj Final", record.get("AdjFinal", 1)
        )
        self.assertAlmostEqual(row[1], expected_volume)
        conn.close()

    test.__name__ = f"test_insert_price_case_{idx}"
    return test


for i in range(1, 26):
    rec = _price_record(i, value=1000 + i, adj_final=10 + i)
    setattr(TestInsertPriceData, f"test_insert_price_case_{i}", _add_price_case(i, rec))


class TestInsertPriceDataTuples(BaseDBTest):
    def setUp(self):
        super().setUp()
        init_price_data.insert_companies(
            [{"CompanyID": "CT", "Ticker": "TT", "Name": "N", "SectorCode": 1}]
        )

    def test_tuple_insert(self):
        record = (
            "2025-01-01",
            "1403-10-01",
            1.0,
            2.0,
            0.5,
            1.5,
            1.5,
            10.0,
            "TT",
            "CT",
            "extra",
        )
        init_price_data.insert_price_data([record])
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        row = cur.execute(
            "select date, adj_close, adj_volume from price_data where ticker='TT'"
        ).fetchone()
        self.assertEqual(row[0], "2025-01-01")
        self.assertAlmostEqual(row[2], 10.0)
        conn.close()


class TestInsertIndicesData(BaseDBTest):
    def test_insert_market_indices(self):
        df = pd.DataFrame(
            {"Date": ["2025-01-01"], "Open": [1], "High": [2], "Low": [0.5], "Close": [1.5]},
            index=["1403-10-11"],
        )
        init_price_data.insert_market_indices("CWI", "شاخص کل", df)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        row = cur.execute("select close from market_indices where index_code='CWI'").fetchone()
        self.assertEqual(row[0], 1.5)
        conn.close()

    def test_insert_sector_indices(self):
        df = pd.DataFrame(
            {"Date": ["2025-01-01"], "Open": [1], "High": [2], "Low": [0.5], "Close": [1.5]},
            index=["1403-10-11"],
        )
        init_price_data.insert_sector_indices("27", "فلزات اساسی", df)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        row = cur.execute("select close from sector_indices where sector_code='27'").fetchone()
        self.assertEqual(row[0], 1.5)
        conn.close()

    def test_insert_usd_data(self):
        init_price_data.insert_usd_data(
            [
                {
                    "Date": "2025-01-01",
                    "J-Date": "1403-10-11",
                    "AdjOpen": 1,
                    "AdjHigh": 2,
                    "AdjLow": 0.5,
                    "AdjClose": 1.5,
                    "AdjFinal": 1.5,
                }
            ]
        )
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        count = cur.execute("select count(*) from usd_prices").fetchone()[0]
        self.assertEqual(count, 1)
        conn.close()


class TestDataFetcherUtils(unittest.TestCase):
    def test_normalize_date_value_valid(self):
        self.assertEqual(DataFetcher._normalize_date_value("2024-01-02"), "2024-01-02")

    def test_normalize_date_value_empty(self):
        self.assertEqual(DataFetcher._normalize_date_value(""), "")

    def test_to_float_handles_invalid(self):
        self.assertEqual(DataFetcher._to_float("bad", default=3.5), 3.5)

    def test_reindex_df_adds_datetime(self):
        df = pd.DataFrame({"Date": ["2024-01-01"]})
        out = DataFetcher.reindex_df(df)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(out["Date"]))


def _make_normalize_test(idx, value):
    def test(self):
        DataFetcher._normalize_date_value(value)

    test.__name__ = f"test_normalize_variation_{idx}"
    return test


for idx, val in enumerate(
    ["2020-01-01", "1399/01/01", None, datetime(2024, 1, 1), "2025-12-07", "invalid"]
):
    setattr(TestDataFetcherUtils, f"test_normalize_variation_{idx}", _make_normalize_test(idx, val))


class TestCLIFlows(BaseDBTest):
    def test_load_initial_missing_files(self):
        args = types.SimpleNamespace()
        with self.assertRaises(FileNotFoundError):
            cli.load_initial(args)

    def test_drop_table(self):
        args = types.SimpleNamespace(table="sectors")
        cli.drop_table(args)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("select name from sqlite_master where type='table' and name='sectors'")
        self.assertIsNone(cur.fetchone())
        conn.close()

    def test_create_indices_tables_idempotent(self):
        init_price_data.create_indices_tables()
        init_price_data.create_indices_tables()
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        count = cur.execute("select count(*) from indices_info").fetchone()[0]
        self.assertGreaterEqual(count, 1)
        conn.close()


def _make_cli_list_test(idx, sector_id):
    def test(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "insert into sectors (sector_id, sector_name) values (?, ?)",
            (sector_id, f"Sector {sector_id}"),
        )
        conn.commit()
        conn.close()
        sys_stdout = sys.stdout
        with open(os.devnull, "w") as sink:
            try:
                sys.stdout = sink
                cli.list_sectors(types.SimpleNamespace())
            finally:
                sys.stdout = sys_stdout

    test.__name__ = f"test_cli_list_sector_{idx}"
    return test


for idx, sector in enumerate(range(1, 11)):
    setattr(TestCLIFlows, f"test_cli_list_sector_{idx}", _make_cli_list_test(idx, sector))


class TestDataFetcherOps(BaseDBTest):
    def setUp(self):
        super().setUp()
        # reference data for companies and sectors
        init_price_data.insert_sectors(
            [
                {"SectorCode": 1, "SectorName": "A", "SectorName_en": "A", "US_Sector": "X"},
                {"SectorCode": 2, "SectorName": "B", "SectorName_en": "B", "US_Sector": "Y"},
            ]
        )
        init_price_data.insert_companies(
            [
                {"CompanyID": "C1", "Ticker": "T1", "Name": "N1", "SectorCode": 1},
                {"CompanyID": "C2", "Ticker": "T2", "Name": "N2", "SectorCode": 2},
            ]
        )

    def test_fetch_company_price_history(self):
        records = DataFetcher.fetch_company_price_history(
            {"Ticker": "T1", "CompanyID": "C1"}, "1395-01-01", "1404-01-01"
        )
        self.assertTrue(records)
        for r in records:
            self.assertIn("AdjFinal", r)

    def test_fetch_sector_index_history(self):
        records = DataFetcher.fetch_sector_index_history(
            {"SectorName": "A", "SectorCode": 1}, "1395-01-01", "1404-01-01"
        )
        self.assertTrue(records)
        self.assertIn("AdjFinal", records[0])

    def test_fetch_index_history(self):
        records = DataFetcher.fetch_index_history(
            sys.modules["gravity_tse"].Get_CWI_History, "شاخص کل", "1395-01-01", "1404-01-01"
        )
        self.assertTrue(records)
        self.assertIn("AdjFinal", records[0])

    def test_fetch_usd_price_history(self):
        records = DataFetcher.fetch_usd_price_history("1395-01-01", "1404-01-01")
        self.assertTrue(records)
        self.assertEqual(records[0]["Ticker"], "USD")


def _generate_float_tests():
    values = [0, 1, -1, 1.5, "2.5", None, "bad", float("nan")]
    for idx, val in enumerate(values):

        def test(self, v=val):
            DataFetcher._to_float(v, default=0)

        test.__name__ = f"test_to_float_var_{idx}"
        setattr(TestDataFetcherUtils, test.__name__, test)


_generate_float_tests()


def _generate_company_tests():
    scenarios = []
    for i in range(20):
        scenarios.append(
            {
                "CompanyID": f"CX{i}",
                "Ticker": f"TX{i}",
                "Name": "N",
                "SectorCode": i,
                "PanelID": i % 3,
                "MarketID": i % 2,
            }
        )
    for idx, company in enumerate(scenarios):

        def test(self, comp=company):
            init_price_data.insert_companies([comp])
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            found = cur.execute(
                "select ticker from companies where ticker=?",
                (comp["Ticker"],),
            ).fetchone()
            self.assertIsNotNone(found)
            conn.close()

        test.__name__ = f"test_insert_company_variant_{idx}"
        setattr(TestInsertBasicData, test.__name__, test)


_generate_company_tests()


def _generate_price_volume_edge_tests():
    edges = [
        {"Value": 0, "AdjFinal": 10},
        {"Value": 100, "AdjFinal": 0.1},
        {"Value": None, "AdjFinal": 5},
        {"Value": 50, "AdjFinal": None},
    ]
    for idx, edge in enumerate(edges):
        rec = _price_record(
            idx + 30, value=edge["Value"], adj_final=edge["AdjFinal"] or 1, ticker=f"EDGE{idx}"
        )

        def test(self, record=rec):
            init_price_data.insert_price_data([record])
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            row = cur.execute(
                "select adj_volume from price_data where ticker=?", (record["Ticker"],)
            ).fetchone()
            self.assertIsNotNone(row)
            conn.close()

        test.__name__ = f"test_price_volume_edge_{idx}"
        setattr(TestInsertPriceData, test.__name__, test)


_generate_price_volume_edge_tests()


def _generate_last_update_tests():
    for idx in range(5):

        def test(self, i=idx):
            init_price_data.insert_last_updates({f"S{i}": f"2025-01-{i + 1:02d}"})
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            val = cur.execute(
                "select last_date from last_updates where symbol=?", (f"S{i}",)
            ).fetchone()
            self.assertIsNotNone(val)
            conn.close()

        test.__name__ = f"test_last_update_insert_{idx}"
        setattr(TestInsertIndicesData, test.__name__, test)


_generate_last_update_tests()


def _generate_dummy_cli_tests():
    for idx in range(5):

        def test(self, i=idx):
            args = types.SimpleNamespace(table="markets")
            cli.drop_table(args)
            init_price_data.create_tables()

        test.__name__ = f"test_cli_drop_table_cycle_{idx}"
        setattr(TestCLIFlows, test.__name__, test)


_generate_dummy_cli_tests()


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(tests)
