"""
TSE (Tehran Stock Exchange) data extractor

Extracts OHLCV data from TSE sources (API or local database).
"""

from typing import Any

import aiohttp
import structlog

from gravity_pipeline.extractors.base import Extractor, ExtractorConfig

logger = structlog.get_logger()


class TSEExtractorConfig(ExtractorConfig):
    """TSE-specific configuration"""

    def __init__(
        self,
        api_base_url: str = "https://api.tse.ir/api",
        api_key: str | None = None,
        use_local_db: bool = True,
        local_db_url: str | None = None,
    ):
        super().__init__()
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.use_local_db = use_local_db
        self.local_db_url = local_db_url


class TSEExtractor(Extractor):
    """Extract OHLCV data from TSE"""

    def __init__(self, config: TSEExtractorConfig | None = None):
        """Initialize TSE extractor"""
        self.config = config or TSEExtractorConfig()
        super().__init__(self.config)
        self.session: aiohttp.ClientSession | None = None
        self.available_symbols: list[str] | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session

    async def validate_connection(self) -> bool:
        """Test connection to TSE API"""
        try:
            session = await self._get_session()

            async with session.get(
                f"{self.config.api_base_url}/tickers",
                timeout=aiohttp.ClientTimeout(seconds=self.config.timeout),
            ) as resp:
                if resp.status == 200:
                    logger.info("tse_connection_valid")
                    return True
                else:
                    logger.warning("tse_connection_invalid", status=resp.status)
                    return False

        except Exception as e:
            logger.error("tse_connection_error", error=str(e))
            return False

    async def get_available_symbols(self) -> list[str]:
        """Get list of available symbols from TSE"""

        if self.available_symbols:
            return self.available_symbols

        try:
            session = await self._get_session()

            async with session.get(
                f"{self.config.api_base_url}/tickers",
                timeout=aiohttp.ClientTimeout(seconds=self.config.timeout),
            ) as resp:
                if resp.status != 200:
                    logger.error("get_symbols_failed", status=resp.status)
                    return []

                data = await resp.json()
                self.available_symbols = [item.get("symbol") for item in data.get("tickers", [])]

                logger.info("symbols_fetched", count=len(self.available_symbols))
                return self.available_symbols

        except Exception as e:
            logger.error("get_symbols_error", error=str(e))
            return []

    async def extract(
        self,
        symbols: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Extract OHLCV data from TSE

        Args:
            symbols: List of symbols (None = all available)
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD
            limit: Max candles per symbol

        Returns:
            List of candle dicts
        """

        logger.info("tse_extract_starting", symbols=symbols, start_date=start_date, limit=limit)

        try:
            # Get symbols to extract
            if not symbols:
                symbols = await self.get_available_symbols()
                if limit:
                    symbols = symbols[:limit]

            logger.info("extracting_symbols", count=len(symbols))

            candles = []

            # Extract each symbol
            for symbol in symbols:
                try:
                    symbol_candles = await self._extract_symbol(symbol, start_date, end_date)
                    candles.extend(symbol_candles)
                    self.extracted_count += len(symbol_candles)

                    logger.info("symbol_extracted", symbol=symbol, candles=len(symbol_candles))

                except Exception as e:
                    self.error_count += 1
                    logger.error("symbol_extraction_failed", symbol=symbol, error=str(e))
                    continue

            logger.info("tse_extract_complete", total=len(candles), errors=self.error_count)

            return candles

        except Exception as e:
            logger.error("tse_extract_error", error=str(e))
            self.error_count += 1
            raise

    async def _extract_symbol(
        self, symbol: str, start_date: str | None, end_date: str | None
    ) -> list[dict[str, Any]]:
        """Extract candles for single symbol"""

        try:
            session = await self._get_session()

            # Build request parameters
            params = {"symbol": symbol}
            if start_date:
                params["from"] = start_date
            if end_date:
                params["to"] = end_date

            # Request from API
            async with session.get(
                f"{self.config.api_base_url}/candles",
                params=params,
                timeout=aiohttp.ClientTimeout(seconds=self.config.timeout),
                headers={"Authorization": f"Bearer {self.config.api_key}"}
                if self.config.api_key
                else {},
            ) as resp:
                if resp.status != 200:
                    logger.warning("symbol_request_failed", symbol=symbol, status=resp.status)
                    return []

                data = await resp.json()
                candles = []

                for candle_data in data.get("candles", []):
                    candle = {
                        "symbol": symbol,
                        "timestamp": candle_data.get("timestamp"),
                        "open": float(candle_data.get("o")),
                        "high": float(candle_data.get("h")),
                        "low": float(candle_data.get("l")),
                        "close": float(candle_data.get("c")),
                        "volume": float(candle_data.get("v")),
                    }
                    candles.append(candle)

                return candles

        except TimeoutError:
            logger.error("symbol_extraction_timeout", symbol=symbol)
            raise
        except Exception as e:
            logger.error("symbol_extraction_error", symbol=symbol, error=str(e))
            raise

    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
        logger.info("tse_extractor_closed")
