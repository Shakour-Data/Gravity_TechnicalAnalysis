import unittest
from src.fetcher import DataFetcher

class TestDataFetcher(unittest.TestCase):
    def test_load_json(self):
        # You can add more detailed tests here
        self.assertTrue(callable(DataFetcher.load_json))

    def test_save_json(self):
        self.assertTrue(callable(DataFetcher.save_json))

if __name__ == "__main__":
    unittest.main()
