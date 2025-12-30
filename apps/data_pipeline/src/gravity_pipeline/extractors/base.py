"""
Base extractor interface for pipeline

All extractors (TSE, Binance, CSV, etc) implement this contract.
"""

from abc import ABC, abstractmethod
from typing import Any

import structlog

logger = structlog.get_logger()


class ExtractorConfig:
    """Base configuration for extractors"""

    def __init__(self):
        self.timeout: int = 30
        self.retry_count: int = 3
        self.retry_delay: int = 5


class Extractor(ABC):
    """Abstract base class for data extractors"""

    def __init__(self, config: ExtractorConfig | None = None):
        """Initialize extractor"""
        self.config = config or ExtractorConfig()
        self.extracted_count = 0
        self.error_count = 0

    @abstractmethod
    async def extract(
        self,
        symbols: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Extract data from source

        Args:
            symbols: List of symbols to extract
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Max records to extract

        Returns:
            List of candle dictionaries with keys:
            - timestamp: Candle timestamp
            - open: Opening price
            - high: High price
            - low: Low price
            - close: Closing price
            - volume: Trading volume
            - symbol: (optional) Asset symbol
        """
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """
        Test connection to data source

        Returns:
            True if connection valid, False otherwise
        """
        pass

    @abstractmethod
    async def get_available_symbols(self) -> list[str]:
        """
        Get list of available symbols from source

        Returns:
            List of symbol names
        """
        pass

    def get_stats(self) -> dict[str, Any]:
        """Get extraction statistics"""
        return {
            "extracted": self.extracted_count,
            "errors": self.error_count,
        }

    async def close(self):
        """Cleanup resources (override if needed)"""
        logger.info("extractor_closed", type=self.__class__.__name__)
