"""
Data Service Client - Interface to Gravity Data Ingestion Microservice

This client handles communication with the Data Ingestion Service to retrieve
adjusted price data (splits/dividends adjusted) and adjusted volume data.

Responsibilities:
- Request candle data from Data Service
- Handle API errors and retries
- Cache responses for performance
- Validate data quality

NOT responsible for:
- Raw data fetching (Alpha Vantage, CODAL, etc.)
- Data cleaning or validation

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
- Adjustment calculations
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, Field
from redis import asyncio as aioredis

logger = structlog.get_logger()


class CandleData(BaseModel):
    """Single candle/bar data point with adjusted values."""
    timestamp: datetime
    adjusted_open: float = Field(gt=0, description="Open price adjusted for splits/dividends")
    adjusted_high: float = Field(gt=0, description="High price adjusted for splits/dividends")
    adjusted_low: float = Field(gt=0, description="Low price adjusted for splits/dividends")
    adjusted_close: float = Field(gt=0, description="Close price adjusted for splits/dividends")
    adjusted_volume: int = Field(ge=0, description="Volume adjusted for splits")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DataServiceResponse(BaseModel):
    """Response from Data Ingestion Service."""
    symbol: str
    timeframe: str
    candles: list[CandleData]
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataServiceClient:
    """
    Client for Gravity Data Ingestion Microservice.

    Features:
    - Async HTTP communication
    - Automatic retry with exponential backoff
    - Redis caching (6 hours TTL)
    - Request validation
    - Error handling

    Example:
        ```python
        client = DataServiceClient(base_url="http://data-service:8080")
        candles = await client.get_candles("AAPL", "1d", start_date, end_date)
        ```
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 30.0,
        max_retries: int = 3,
        redis_url: str | None = None,
        cache_ttl: int = 21600  # 6 hours
    ):
        """
        Initialize Data Service client.

        Args:
            base_url: Base URL of Data Ingestion Service
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            redis_url: Redis connection URL for caching
            cache_ttl: Cache TTL in seconds (default: 6 hours)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_ttl = cache_ttl

        self._valid_timeframes = {"1m", "5m", "15m", "1h", "4h", "1d", "1w"}
        self._min_candles = 30
        self._max_days = 1095  # cap requests to ~3 years

        # HTTP client with retry
        self.client = httpx.AsyncClient(
            timeout=timeout,
            transport=httpx.AsyncHTTPTransport(retries=max_retries)
        )

        # Redis cache (optional)
        self.redis: aioredis.Redis | None = None
        if redis_url:
            self.redis = aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)

        logger.info(
            "data_service_client_initialized",
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            cache_enabled=bool(redis_url)
        )

    async def get_candles(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        use_cache: bool = True
    ) -> list[CandleData]:
        """
        Get adjusted candle data from Data Service.

        Args:
            symbol: Stock/crypto symbol (e.g., "AAPL", "BTC-USD")
            timeframe: Timeframe (e.g., "1m", "5m", "1h", "1d")
            start_date: Start date (default: 1 year ago)
            end_date: End date (default: today)
            use_cache: Whether to use Redis cache

        Returns:
            List of CandleData objects with adjusted prices/volume

        Raises:
            httpx.HTTPError: If request fails
            ValueError: If response is invalid
        """
        # Default date range
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=365)

        self._validate_request(symbol, timeframe, start_date, end_date)

        # Generate cache key
        cache_key = f"candles:{self.base_url}:{symbol}:{timeframe}:{start_date.isoformat()}:{end_date.isoformat()}"

        # Try cache first
        if use_cache and self.redis:
            cached = await self._get_from_cache(cache_key)
            if cached:
                logger.info("cache_hit", symbol=symbol, timeframe=timeframe)
                return cached
            logger.debug("cache_miss", symbol=symbol, timeframe=timeframe)

        # Request from Data Service
        logger.info(
            "requesting_candles",
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date.date(),
            end_date=end_date.date()
        )

        try:
            response = await self.client.get(
                f"{self.base_url}/api/v1/candles/{symbol}",
                params={
                    "timeframe": timeframe,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "fields": "adjusted_open,adjusted_high,adjusted_low,adjusted_close,adjusted_volume"
                }
            )
            response.raise_for_status()

            # Parse response
            data = response.json()
            service_response = DataServiceResponse(**data)

            # Validate data quality
            self._validate_candles(service_response.candles)

            # Cache result
            if use_cache and self.redis:
                await self._save_to_cache(cache_key, service_response.candles)

            logger.info(
                "candles_received",
                symbol=symbol,
                timeframe=timeframe,
                count=len(service_response.candles),
                data_quality=service_response.metadata.get("data_quality_score", "N/A")
            )

            return service_response.candles

        except httpx.HTTPStatusError as e:
            logger.error(
                "data_service_error",
                symbol=symbol,
                status_code=e.response.status_code,
                error=str(e)
            )
            raise
        except Exception as e:
            logger.error(
                "unexpected_error",
                symbol=symbol,
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def _validate_candles(self, candles: list[CandleData]) -> None:
        """
        Validate candle data quality.

        Args:
            candles: List of candles to validate

        Raises:
            ValueError: If data quality issues detected
        """
        if not candles:
            raise ValueError("No candle data received")
        if len(candles) < self._min_candles:
            raise ValueError(f"Received only {len(candles)} candles (min {self._min_candles} required)")

        # Check for required fields
        for i, candle in enumerate(candles):
            values = [
                candle.adjusted_open,
                candle.adjusted_high,
                candle.adjusted_low,
                candle.adjusted_close,
                candle.adjusted_volume,
            ]
            if not all(np.isfinite(val) for val in values):
                raise ValueError(f"Non-finite OHLCV at index {i}")
            if candle.adjusted_high < candle.adjusted_low:
                raise ValueError(f"Invalid candle at index {i}: high < low")
            if not (candle.adjusted_low <= candle.adjusted_close <= candle.adjusted_high):
                logger.warning(
                    "suspicious_candle",
                    index=i,
                    close=candle.adjusted_close,
                    low=candle.adjusted_low,
                    high=candle.adjusted_high
                )

        # Check chronological order
        for i in range(1, len(candles)):
            if candles[i].timestamp <= candles[i-1].timestamp:
                raise ValueError(f"Candles not in chronological order at index {i}")

    def _validate_request(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> None:
        """Validate request parameters before hitting the service."""
        if not symbol or len(symbol) < 2:
            raise ValueError("symbol is required and must be at least 2 characters")
        if not all(ch.isalnum() or ch in "-_." for ch in symbol):
            raise ValueError("symbol may only contain letters, numbers, '-', '_', '.'")
        if timeframe not in self._valid_timeframes:
            raise ValueError(f"timeframe must be one of {sorted(self._valid_timeframes)}")
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")
        if (end_date - start_date).days > self._max_days:
            raise ValueError(f"date range too large (>{self._max_days} days)")

        logger.debug("candle_validation_passed", count=len(candles))

    async def _get_from_cache(self, key: str) -> list[CandleData] | None:
        """Get candles from Redis cache."""
        if not self.redis:
            return None

        try:
            cached_json = await self.redis.get(key)
            if cached_json:
                import json
                candles_data = json.loads(cached_json)
                return [CandleData(**c) for c in candles_data]
        except Exception as e:
            logger.warning("cache_read_error", key=key, error=str(e))

        return None

    async def _save_to_cache(self, key: str, candles: list[CandleData]) -> None:
        """Save candles to Redis cache."""
        if not self.redis:
            return

        try:
            import json
            candles_json = json.dumps([c.dict() for c in candles], default=str)
            await self.redis.setex(key, self.cache_ttl, candles_json)
            logger.debug("cache_saved", key=key, ttl=self.cache_ttl)
        except Exception as e:
            logger.warning("cache_write_error", key=key, error=str(e))

    async def close(self):
        """Close HTTP client and Redis connection."""
        await self.client.aclose()
        if self.redis:
            await self.redis.close()
        logger.info("data_service_client_closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
