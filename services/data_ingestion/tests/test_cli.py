import unittest
import sqlite3
import os
import sys
import json
import io
from contextlib import redirect_stdout

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import cli
from src.database import init_price_data

class TestCLI(unittest.TestCase):
    def setUp(self):
        # Create a temporary sqlite file
        self.test_db = os.path.join(os.path.dirname(__file__), 'test_cli.db')
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        # Patch cli.get_connection to use our test DB
        self.conn = sqlite3.connect(self.test_db)
        cli.get_connection = staticmethod(lambda: sqlite3.connect(self.test_db))
        # Create necessary tables
        cur = self.conn.cursor()
        cur.execute('''CREATE TABLE sectors (sector_id INTEGER PRIMARY KEY, sector_name TEXT UNIQUE)''')
        cur.execute('''CREATE TABLE companies (company_id TEXT PRIMARY KEY, ticker TEXT UNIQUE, name TEXT, sector_id INTEGER)''')
        cur.execute('''CREATE TABLE price_data (id INTEGER PRIMARY KEY, date TEXT, adj_close REAL, adj_volume REAL, ticker TEXT, company_id TEXT, sector_id INTEGER)''')
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

    def test_create_db_calls_create_tables(self):
        called = {'flag': False}
        original = init_price_data.create_tables
        init_price_data.create_tables = staticmethod(lambda: called.update({'flag': True}))
        cli.create_db(None)
        self.assertTrue(called['flag'])
        init_price_data.create_tables = original

    def test_reload_table_calls_insert(self):
        # prepare a json file
        tmpfile = os.path.join(os.path.dirname(__file__), 'tmp_companies.json')
        data = [{'CompanyID': 'C1', 'Ticker': 'T1', 'Name': 'N1'}]
        with open(tmpfile, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        called = {'data': None}
        original = init_price_data.insert_companies
        init_price_data.insert_companies = staticmethod(lambda d: called.update({'data': d}))
        args = type('A', (), {})()
        args.table = 'companies'
        args.file = tmpfile
        try:
            cli.reload_table(args)
            self.assertIsNotNone(called['data'])
            self.assertEqual(called['data'], data)
        finally:
            init_price_data.insert_companies = original
            os.remove(tmpfile)

    def test_drop_table(self):
        # create a table then drop
        conn = sqlite3.connect(self.test_db)
        cur = conn.cursor()
        cur.execute('CREATE TABLE temp_table (id INTEGER)')
        conn.commit()
        conn.close()
        args = type('A', (), {})()
        args.table = 'temp_table'
        # capture stdout
        f = io.StringIO()
        with redirect_stdout(f):
            cli.drop_table(args)
        # ensure table does not exist
        conn2 = sqlite3.connect(self.test_db)
        cur2 = conn2.cursor()
        cur2.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='temp_table'")
        res = cur2.fetchall()
        conn2.close()
        self.assertEqual(res, [])

    def test_list_and_get_price_data_output(self):
        # insert sector, company and price data
        conn = sqlite3.connect(self.test_db)
        cur = conn.cursor()
        cur.execute('INSERT INTO sectors (sector_id, sector_name) VALUES (?, ?)', (1, 'TestSector'))
        cur.execute('INSERT INTO companies (company_id, ticker, name, sector_id) VALUES (?, ?, ?, ?)', ('C1', 'T1', 'Name1', 1))
        cur.execute('INSERT INTO price_data (date, adj_close, adj_volume, ticker, company_id, sector_id) VALUES (?, ?, ?, ?, ?, ?)', ('2023-01-01', 100.0, 200.0, 'T1', 'C1', 1))
        conn.commit()
        conn.close()

        # capture list_sectors output
        f = io.StringIO()
        args = type('A', (), {})()
        with redirect_stdout(f):
            cli.list_sectors(args)
        output = f.getvalue().strip()
        self.assertIn('1: TestSector', output)

        # capture list_companies output
        f2 = io.StringIO()
        args2 = type('A', (), {})()
        args2.sector_id = 1
        with redirect_stdout(f2):
            cli.list_companies(args2)
        out2 = f2.getvalue().strip()
        self.assertIn('T1: Name1', out2)

        # capture get_price_data output
        f3 = io.StringIO()
        args3 = type('A', (), {})()
        args3.ticker = 'T1'
        args3.limit = 1
        with redirect_stdout(f3):
            cli.get_price_data(args3)
        out3 = f3.getvalue().strip()
        self.assertIn('Date: 2023-01-01, Close: 100.0, Volume: 200.0', out3)

    def test_main_load_market_and_sector_indices(self):
        import types
        import sys
        import pandas as pd
        # prepare dummy gravity_tse module
        gpy = types.SimpleNamespace()
        def make_df():
            return pd.DataFrame({
                'Date': ['2023-01-01'],
                'J-Date': ['1401-01-01'],
                'Open': [1], 'High': [2], 'Low': [3], 'Close': [4]
            })
        gpy.Get_CWI_History = lambda *a, **k: make_df()
        gpy.Get_EWI_History = lambda *a, **k: make_df()
        gpy.Get_CWPI_History = lambda *a, **k: make_df()
        gpy.Get_EWPI_History = lambda *a, **k: make_df()
        gpy.Get_FFI_History = lambda *a, **k: make_df()
        gpy.Get_MKT1I_History = lambda *a, **k: make_df()
        gpy.Get_INDI_History = lambda *a, **k: make_df()
        gpy.Get_ACT50_History = lambda *a, **k: make_df()
        gpy.Get_LCI30_History = lambda *a, **k: make_df()
        gpy.Get_SectorIndex_History = lambda *a, **k: make_df()
        # inject dummy module
        sys.modules['gravity_tse'] = gpy

        # patch insert_market_indices and insert_sector_indices to capture calls
        called_market = {'count': 0}
        orig_market = init_price_data.insert_market_indices
        init_price_data.insert_market_indices = staticmethod(lambda code, name, df: called_market.update({'count': called_market['count'] + 1}))

        # Prepare sectors json for load-sector-indices
        from src.config import SECTORS_FILE
        os.makedirs(os.path.dirname(SECTORS_FILE), exist_ok=True)
        with open(SECTORS_FILE, 'w', encoding='utf-8') as f:
            json.dump([{'SectorName': 'فلزات اساسی', 'SectorCode': 27}], f)

        called_sector = {'count': 0}
        orig_sector = init_price_data.insert_sector_indices
        init_price_data.insert_sector_indices = staticmethod(lambda code, name, df: called_sector.update({'count': called_sector['count'] + 1}))

        # Run main for market indices
        old_argv = sys.argv
        try:
            sys.argv = ['prog', 'load-market-indices']
            cli.main()
            self.assertGreater(called_market['count'], 0)

            # Run main for sector indices
            sys.argv = ['prog', 'load-sector-indices']
            cli.main()
            self.assertGreater(called_sector['count'], 0)
        finally:
            sys.argv = old_argv
            # restore
            init_price_data.insert_market_indices = orig_market
            init_price_data.insert_sector_indices = orig_sector
            try:
                os.remove(SECTORS_FILE)
            except Exception:
                pass
            # remove dummy gravity_tse module if present
            try:
                del sys.modules['gravity_tse']
            except Exception:
                pass

    def test_main_various_commands(self):
        import sys
        # ensure cli uses our test DB connection
        cli.get_connection = staticmethod(lambda: sqlite3.connect(self.test_db))

        # create-db
        called = {'create': False}
        orig_create = init_price_data.create_tables
        init_price_data.create_tables = staticmethod(lambda: called.update({'create': True}))
        old_argv = sys.argv
        try:
            sys.argv = ['prog', 'create-db']
            cli.main()
            self.assertTrue(called['create'])
        finally:
            sys.argv = old_argv
            init_price_data.create_tables = orig_create

        # reload-table via main (companies)
        tmpfile = os.path.join(os.path.dirname(__file__), 'tmp_companies.json')
        data = [{'CompanyID': 'C2', 'Ticker': 'T2', 'Name': 'N2'}]
        with open(tmpfile, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        called_reload = {'data': None}
        orig_insert_companies = init_price_data.insert_companies
        init_price_data.insert_companies = staticmethod(lambda d: called_reload.update({'data': d}))
        try:
            sys.argv = ['prog', 'reload-table', 'companies', tmpfile]
            cli.main()
            self.assertIsNotNone(called_reload['data'])
        finally:
            init_price_data.insert_companies = orig_insert_companies
            if os.path.exists(tmpfile):
                os.remove(tmpfile)

        # drop-table via main for an allowed table name (price_data)
        # ensure price_data table exists then drop it
        conn = sqlite3.connect(self.test_db)
        cur = conn.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS price_data (id INTEGER)')
        conn.commit()
        conn.close()
        try:
            sys.argv = ['prog', 'drop-table', 'price_data']
            cli.main()
            conn2 = sqlite3.connect(self.test_db)
            cur2 = conn2.cursor()
            cur2.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='price_data'")
            res = cur2.fetchall()
            conn2.close()
            self.assertEqual(res, [])
        finally:
            pass

        # update-db should call load_initial (patch it)
        called_upd = {'flag': False}
        orig_load_initial = cli.load_initial
        cli.load_initial = staticmethod(lambda args: called_upd.update({'flag': True}))
        try:
            sys.argv = ['prog', 'update-db']
            cli.main()
            self.assertTrue(called_upd['flag'])
        finally:
            cli.load_initial = orig_load_initial

        # update-table via main
        tmpfile2 = os.path.join(os.path.dirname(__file__), 'tmp_companies2.json')
        with open(tmpfile2, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        called_update_table = {'data': None}
        orig_insert_companies = init_price_data.insert_companies
        init_price_data.insert_companies = staticmethod(lambda d: called_update_table.update({'data': d}))
        try:
            sys.argv = ['prog', 'update-table', 'companies', tmpfile2]
            cli.main()
            self.assertIsNotNone(called_update_table['data'])
        finally:
            init_price_data.insert_companies = orig_insert_companies
            if os.path.exists(tmpfile2):
                os.remove(tmpfile2)

        # init-all and load-all-prices should call DataFetcher.run (patch it)
        from src.fetcher import DataFetcher
        called_run = {'flag': False}
        orig_run = DataFetcher.run
        DataFetcher.run = staticmethod(lambda: called_run.update({'flag': True}))
        try:
            sys.argv = ['prog', 'load-all-prices']
            cli.main()
            self.assertTrue(called_run['flag'])
            called_run['flag'] = False
            sys.argv = ['prog', 'init-all']
            # patch load_initial to avoid file IO during init-all
            orig_load_initial = cli.load_initial
            cli.load_initial = staticmethod(lambda args: None)
            try:
                cli.main()
                self.assertTrue(called_run['flag'])
            finally:
                cli.load_initial = orig_load_initial
        finally:
            DataFetcher.run = orig_run

if __name__ == '__main__':
    unittest.main()
