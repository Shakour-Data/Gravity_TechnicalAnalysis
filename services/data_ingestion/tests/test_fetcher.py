import unittest
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
import sys
import os
import sqlite3

# Add project root to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fetcher import DataFetcher
from src.database import init_price_data
from src.config import INITIAL_START_DATE

class TestDataFetcher(unittest.TestCase):

    def setUp(self):
        # Use a temporary database for testing
        self.test_db = 'test_fetcher.db'
        # Patch the get_connection method to use the test database
        self.original_get_connection = init_price_data.get_connection
        init_price_data.get_connection = staticmethod(lambda: sqlite3.connect(self.test_db))
        # Create all required tables (main, indices, USD)
        init_price_data.create_tables()
        init_price_data.create_indices_tables()
        init_price_data.create_usd_table()
        
        # Mock data
        self.sample_company = {'Ticker': 'TEST', 'CompanyID': 123}
        self.sample_sector = {'SectorName': 'TestSector', 'SectorCode': 456}
        self.sample_df = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02'],
            'J-Date': ['1401-10-11', '1401-10-12'],
            'AdjOpen': [100, 101],
            'AdjHigh': [105, 106],
            'AdjLow': [95, 96],
            'AdjClose': [102, 103],
            'AdjFinal': [102, 103]
        })

    def tearDown(self):
        # Restore the original get_connection method
        init_price_data.get_connection = self.original_get_connection
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

    @patch('builtins.open', new_callable=mock_open, read_data='{"key": "value"}')
    def test_load_json(self, mock_file):
        result = DataFetcher.load_json('test.json')
        self.assertEqual(result, {"key": "value"})
        mock_file.assert_called_once_with('test.json', 'r', encoding='utf-8')

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_json(self, mock_json_dump, mock_file):
        data = {"key": "value"}
        DataFetcher.save_json(data, 'test.json')
        mock_file.assert_called_once_with('test.json', 'w', encoding='utf-8')
        mock_json_dump.assert_called_once_with(data, mock_file(), ensure_ascii=False, indent=2)

    @patch('src.database.init_price_data.get_last_update', return_value='1400-01-01')
    def test_get_last_update_existing(self, mock_get):
        result = DataFetcher.get_last_update('test_symbol')
        self.assertEqual(result, '1400-01-01')

    @patch('src.database.init_price_data.get_last_update', return_value=INITIAL_START_DATE)
    def test_get_last_update_not_existing(self, mock_get):
        result = DataFetcher.get_last_update('test_symbol')
        self.assertEqual(result, INITIAL_START_DATE)

    @patch('src.database.init_price_data.update_last_update')
    def test_update_last_update_new_file(self, mock_update):
        DataFetcher.update_last_update('test_symbol', '1401-01-01')
        mock_update.assert_called_once_with('test_symbol', '1401-01-01')

    @patch('src.database.init_price_data.update_last_update')
    def test_update_last_update_existing(self, mock_update):
        DataFetcher.update_last_update('existing', 'new')
        mock_update.assert_called_once_with('existing', 'new')

    def test_reindex_df(self):
        df = self.sample_df.copy()
        df.set_index('Date', inplace=True)
        result = DataFetcher.reindex_df(df)
        self.assertIn('Date', result.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['Date']))

    @patch('gravity_tse.Get_Price_History', return_value=MagicMock())
    def test_fetch_company_price_history_success(self, mock_get_price):
        mock_get_price.return_value = self.sample_df
        result = DataFetcher.fetch_company_price_history(self.sample_company, '1400-01-01', '1401-01-01')
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0]['CompanyID'], 123)
        self.assertEqual(result[0]['Ticker'], 'TEST')

    @patch('gravity_tse.Get_Price_History', return_value=pd.DataFrame())
    def test_fetch_company_price_history_empty_df(self, mock_get_price):
        result = DataFetcher.fetch_company_price_history(self.sample_company, '1400-01-01', '1401-01-01')
        self.assertEqual(result, [])

    @patch('gravity_tse.Get_Price_History', side_effect=Exception('Test error'))
    def test_fetch_company_price_history_error(self, mock_get_price):
        result = DataFetcher.fetch_company_price_history(self.sample_company, '1400-01-01', '1401-01-01')
        self.assertEqual(result, [])

    @patch('gravity_tse.Get_SectorIndex_History', return_value=MagicMock())
    def test_fetch_sector_index_history_success(self, mock_get_sector):
        mock_get_sector.return_value = self.sample_df
        result = DataFetcher.fetch_sector_index_history(self.sample_sector, '1400-01-01', '1401-01-01')
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0]['SectorCode'], 456)

    @patch('gravity_tse.Get_SectorIndex_History', return_value=pd.DataFrame())
    def test_fetch_sector_index_history_empty_df(self, mock_get_sector):
        result = DataFetcher.fetch_sector_index_history(self.sample_sector, '1400-01-01', '1401-01-01')
        self.assertEqual(result, [])

    @patch('gravity_tse.Get_SectorIndex_History', side_effect=Exception('Test error'))
    def test_fetch_sector_index_history_error(self, mock_get_sector):
        result = DataFetcher.fetch_sector_index_history(self.sample_sector, '1400-01-01', '1401-01-01')
        self.assertEqual(result, [])

    @patch('gravity_tse.Get_CWI_History', return_value=MagicMock())
    def test_fetch_index_history_success(self, mock_get_index):
        mock_get_index.return_value = self.sample_df
        result = DataFetcher.fetch_index_history(mock_get_index, 'CWI', '1400-01-01', '1401-01-01')
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0]['IndexName'], 'CWI')

    @patch('gravity_tse.Get_CWI_History', return_value=pd.DataFrame())
    def test_fetch_index_history_empty_df(self, mock_get_index):
        result = DataFetcher.fetch_index_history(mock_get_index, 'CWI', '1400-01-01', '1401-01-01')
        self.assertEqual(result, [])

    @patch('gravity_tse.Get_CWI_History', side_effect=Exception('Test error'))
    def test_fetch_index_history_error(self, mock_get_index):
        result = DataFetcher.fetch_index_history(mock_get_index, 'CWI', '1400-01-01', '1401-01-01')
        self.assertEqual(result, [])

    @patch('src.database.init_price_data.update_price_data_sectors')
    @patch('src.database.init_price_data.insert_price_data')
    @patch('src.database.init_price_data.get_last_update', return_value=INITIAL_START_DATE)
    @patch('src.database.init_price_data.update_last_update')
    @patch('src.fetcher.DataFetcher.fetch_company_price_history', return_value=[])
    @patch('src.fetcher.DataFetcher.fetch_sector_index_history', return_value=[])
    @patch('src.fetcher.DataFetcher.fetch_index_history', return_value=[])
    @patch('gravity_tse.Get_USD_RIAL', return_value=MagicMock())
    @patch('jdatetime.datetime')
    def test_run(self, mock_datetime, mock_usd, mock_index_hist, mock_sector_hist, mock_comp_hist, mock_update, mock_get_last, mock_insert, mock_update_sectors):
        def load_json_side_effect(filepath):
            from src.config import COMPANIES_FILE, SECTORS_FILE
            if filepath == COMPANIES_FILE:
                return [self.sample_company]
            elif filepath == SECTORS_FILE:
                return [self.sample_sector]
            return {}
        with patch('src.fetcher.DataFetcher.load_json', side_effect=load_json_side_effect):
            mock_datetime.now.return_value.strftime.return_value = '1401-01-01'
            mock_usd.return_value = self.sample_df
            DataFetcher.run()
        # Assert calls were made
        self.assertTrue(mock_insert.called)
        self.assertTrue(mock_update_sectors.called)

if __name__ == '__main__':
    unittest.main()
