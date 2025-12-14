import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.fetcher import DataFetcher

class TestDataFetcherExt(unittest.TestCase):
    def setUp(self):
        self.company = {'Ticker': 'TEST', 'CompanyID': 1}
        self.sector = {'SectorName': 'فلزات اساسی', 'SectorCode': 27}
        self.df = pd.DataFrame({
            'Date': ['2023-01-01'],
            'J-Date': ['1401-10-11'],
            'AdjOpen': [100],
            'AdjHigh': [105],
            'AdjLow': [95],
            'AdjClose': [102],
            'AdjFinal': [102],
            'Value': [10000],
            'Close': [102],
            'Final': [102]
        })

    @patch('src.fetcher.pd.to_datetime', side_effect=lambda x: pd.Timestamp('2023-01-01'))
    def test_normalize_date_value_valid(self, mock_dt):
        self.assertEqual(DataFetcher._normalize_date_value('2023-01-01'), '2023-01-01')

    def test_normalize_date_value_empty(self):
        self.assertEqual(DataFetcher._normalize_date_value(''), '')
        self.assertEqual(DataFetcher._normalize_date_value(None), '')

    def test_to_float_valid(self):
        self.assertEqual(DataFetcher._to_float('123.45'), 123.45)
        self.assertEqual(DataFetcher._to_float(99), 99.0)

    def test_to_float_invalid(self):
        self.assertEqual(DataFetcher._to_float('bad', 5.5), 5.5)
        self.assertEqual(DataFetcher._to_float(None, 1.1), 1.1)

    @patch('src.fetcher.init_price_data.get_last_update', return_value='2022-01-01')
    def test_get_last_update(self, mock_get):
        self.assertEqual(DataFetcher.get_last_update('TEST'), '2022-01-01')

    @patch('src.fetcher.init_price_data.update_last_update')
    def test_update_last_update(self, mock_update):
        DataFetcher.update_last_update('TEST', '2022-01-02')
        mock_update.assert_called_once_with('TEST', '2022-01-02')

    @patch('src.fetcher.pd.DataFrame.reset_index', return_value=pd.DataFrame({'Date': ['2023-01-01']}))
    def test_reindex_df(self, mock_reset):
        df = pd.DataFrame({'Date': ['2023-01-01']})
        result = DataFetcher.reindex_df(df)
        self.assertIn('Date', result.columns)

    @patch('src.fetcher.init_price_data.insert_usd_data')
    @patch('src.fetcher.init_price_data.insert_price_data')
    @patch('src.fetcher.init_price_data.get_last_update', return_value='2023-01-01')
    @patch('src.fetcher.DataFetcher.fetch_company_price_history', return_value=[{'Date': '2023-01-02'}])
    @patch('src.fetcher.DataFetcher.load_json', return_value=[{'Ticker': 'TEST', 'CompanyID': 1}])
    @patch('src.fetcher.jdatetime.datetime.now', return_value=MagicMock(strftime=lambda fmt: '1401-10-12'))
    def test_fetch_all_prices_to_json(self, mock_now, mock_load_json, mock_fetch, mock_get_last, mock_insert_price, mock_insert_usd):
        DataFetcher.fetch_all_prices_to_json()
        mock_insert_price.assert_called()

    @patch('src.fetcher.init_price_data.insert_sector_indices')
    @patch('src.fetcher.DataFetcher.fetch_sector_index_history', return_value=[{'J-Date': '1401-10-11', 'Open': 100}])
    @patch('src.fetcher.DataFetcher.load_json', return_value=[{'SectorName': 'فلزات اساسی', 'SectorCode': 27}])
    @patch('src.fetcher.jdatetime.datetime.now', return_value=MagicMock(strftime=lambda fmt: '1401-10-12'))
    def test_fetch_all_sector_indices_to_json(self, mock_now, mock_load_json, mock_fetch, mock_insert):
        DataFetcher.fetch_all_sector_indices_to_json()
        mock_insert.assert_called()

if __name__ == '__main__':
    unittest.main()
