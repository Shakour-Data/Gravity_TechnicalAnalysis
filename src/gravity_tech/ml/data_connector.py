"""
Data Connector for Historical Market Data

Provides a resilient client for the historical-data microservice with retry/backoff
and an optional mock-data fallback for offline development.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import requests
from gravity_tech.config.settings import settings
from gravity_tech.core.domain.entities import Candle

logger = logging.getLogger(__name__)


class DataConnector:
    """
    Connects to the historical data microservice (or generates mock data if needed).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        backoff_factor: float = 0.5,
        allow_mock_on_failure: bool = True,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = (base_url or settings.DATA_SERVICE_URL).rstrip("/")
        self.timeout = timeout or settings.DATA_SERVICE_TIMEOUT
        self.max_retries = max(
            0, max_retries if max_retries is not None else settings.DATA_SERVICE_MAX_RETRIES
        )
        self.backoff_factor = backoff_factor
        self.allow_mock = allow_mock_on_failure
        self._session = session or requests.Session()
        self.last_data_source: str = "remote"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fetch_daily_candles(
        self,
        symbol: str = "BTCUSDT",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[Candle]:
        """
        Fetch daily candles from the remote data service (or mock fallback).
        """
        return self.fetch_candles(
            symbol=symbol,
            interval="1d",
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

    def fetch_candles(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 500,
    ) -> list[Candle]:
        """
        Fetch candles for an arbitrary interval.
        """
        interval_delta = self._interval_to_timedelta(interval)
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - interval_delta * max(1, limit)

        params = {
            "symbol": symbol,
            "interval": interval,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "limit": limit,
        }

        try:
            payload = self._perform_request("/api/v1/candles", params=params)
            candles = [self._parse_candle(item) for item in payload.get("candles", [])]
            self.last_data_source = "remote"
            return candles
        except requests.RequestException as exc:
            logger.warning(
                "data_connector.remote_fetch_failed",
                extra={
                    "symbol": symbol,
                    "base_url": self.base_url,
                    "retries": self.max_retries,
                    "error": str(exc),
                },
            )
            if not self.allow_mock:
                raise

        # Mock fallback for offline scenarios
        self.last_data_source = "mock"
        logger.info("data_connector.generating_mock_data", extra={"symbol": symbol})
        return self._generate_mock_data(symbol, start_date, limit, interval_delta)

    def fetch_multiple_symbols(
        self,
        symbols: list[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000,
    ) -> dict[str, list[Candle]]:
        """
        Fetch data for multiple symbols sequentially.
        """
        return {
            symbol: self.fetch_daily_candles(symbol, start_date, end_date, limit)
            for symbol in symbols
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _perform_request(self, path: str, params: dict[str, Any]) -> dict:
        url = f"{self.base_url}{path}"
        last_exc: Optional[requests.RequestException] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                sleep_duration = self.backoff_factor * (2 ** attempt)
                logger.debug(
                    "data_connector.retrying",
                    extra={
                        "url": url,
                        "attempt": attempt + 1,
                        "max_retries": self.max_retries,
                        "sleep_seconds": sleep_duration,
                    },
                )
                time.sleep(sleep_duration)

        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _parse_candle(item: dict[str, Any]) -> Candle:
        return Candle(
            open=float(item["open"]),
            high=float(item["high"]),
            low=float(item["low"]),
            close=float(item["close"]),
            volume=float(item["volume"]),
            timestamp=datetime.fromisoformat(item["timestamp"]),
        )

    def _generate_mock_data(
        self,
        symbol: str,
        start_date: datetime,
        limit: int,
        interval_delta: timedelta,
    ) -> list[Candle]:
        steps = max(1, limit)
        base_price = 40000.0

        rng = np.random.default_rng(seed=42)
        returns = rng.normal(0.001, 0.03, steps)
        half = steps // 2
        trend = np.concatenate(
            [np.linspace(0, 0.002, half, endpoint=False), np.linspace(0.002, -0.001, steps - half)]
        )
        returns += trend

        price_multipliers = np.cumprod(1 + returns)
        closes = base_price * price_multipliers

        candles: list[Candle] = []
        current_date = start_date

        for close in closes:
            daily_range = close * rng.uniform(0.02, 0.05)
            open_price = close + rng.uniform(-daily_range / 2, daily_range / 2)
            high = max(open_price, close) + rng.uniform(0, daily_range / 2)
            low = min(open_price, close) - rng.uniform(0, daily_range / 2)

            price_move = abs(close - open_price) / max(open_price, 1e-6)
            base_volume = 25000 + rng.uniform(-5000, 5000)
            volume = base_volume * (1 + price_move * 10)

            candles.append(
                Candle(
                    open=round(open_price, 2),
                    high=round(high, 2),
                    low=round(low, 2),
                    close=round(close, 2),
                    volume=round(volume, 2),
                    timestamp=current_date,
                )
            )
            current_date += interval_delta

        logger.info(
            "data_connector.mock_data_ready",
            extra={"symbol": symbol, "count": len(candles)},
        )
        return candles

    @staticmethod
    def _interval_to_timedelta(interval: str) -> timedelta:
        """Convert strings like 15m, 1h, 4h, 1d to timedeltas."""
        if not interval:
            return timedelta(hours=1)
        unit = interval[-1]
        try:
            value = int(interval[:-1]) if unit.isalpha() else int(interval)
            unit = unit if unit.isalpha() else "m"
        except ValueError:
            value = 1
            unit = "h"

        unit = unit.lower()
        if unit == "m":
            return timedelta(minutes=value)
        if unit == "h":
            return timedelta(hours=value)
        if unit == "d":
            return timedelta(days=value)
        if unit == "w":
            return timedelta(weeks=value)
        return timedelta(hours=value)
