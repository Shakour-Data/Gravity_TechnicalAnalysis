"""
TSE Data Source Module

Provides access to TSE market data.
"""

from config import TSE_DB_FILE

from database import TSEDatabaseConnector

# Create a singleton instance
tse_data_source = TSEDatabaseConnector(TSE_DB_FILE)
