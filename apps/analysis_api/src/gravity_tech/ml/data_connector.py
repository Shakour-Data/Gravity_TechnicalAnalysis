"""
Data Connector for Historical Market Data

Provides a resilient client for the historical-data microservice with retry/backoff.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from gravity_tech.config.settings import settings
from gravity_tech.core.domain.entities import Candle
from datetime import timezone

try:
    from prometheus_client import Counter, Histogram
except Exception:  # pragma: no cover - fallback when prometheus_client is unavailable
    class _Noop:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            return self

        def observe(self, *args, **kwargs):
            return self

    Counter = Histogram = lambda *args, **kwargs: _Noop()

DATA_CONNECTOR_REQUESTS = Counter(
    "data_connector_requests_total",
    "Total DataConnector fetches",
    ["source", "outcome", "interval"],
)
DATA_CONNECTOR_LATENCY = Histogram(
    "data_connector_latency_seconds",
    "Latency of DataConnector fetches",
    ["source", "interval"],
)

logger = logging.getLogger(__name__)


class DataConnector:
    """Connects to the historical data microservice."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        backoff_factor: float = 0.5,
        allow_mock_on_failure: bool = False,
        session: requests.Session | None = None,
    ):
        self.base_url = (base_url or settings.DATA_SERVICE_URL).rstrip("/")
        self.timeout = timeout or settings.DATA_SERVICE_TIMEOUT
        self.max_retries = max(
            0, max_retries if max_retries is not None else settings.DATA_SERVICE_MAX_RETRIES
        )
        self.backoff_factor = backoff_factor
        self.allow_mock = allow_mock_on_failure  # kept for backward compatibility; no-op
        self._session = session or requests.Session()
        self.last_data_source: str = "remote"
        # Telemetry counters
        self.success_count = 0
        self.failure_count = 0
        self.last_latency_ms: float | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fetch_daily_candles(
        self,
        symbol: str = "BTCUSDT",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 1000,
    ) -> list[Candle]:
        """
        Fetch daily candles from the remote data service.
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
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 500,
    ) -> list[Candle]:
        """
        Fetch candles for an arbitrary interval.
        """
        interval_delta = self._interval_to_timedelta(interval)
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - interval_delta * max(1, limit)

        params = {
            "symbol": symbol,
            "interval": interval,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "limit": limit,
        }

        start_ts = time.perf_counter()
        try:
            payload = self._perform_request("/api/v1/candles", params=params)
            candles = [self._parse_candle(item) for item in payload.get("candles", [])]
            self.last_data_source = "remote"
            self.success_count += 1
            self.last_latency_ms = (time.perf_counter() - start_ts) * 1000
            duration_seconds = self.last_latency_ms / 1000
            DATA_CONNECTOR_REQUESTS.labels("remote", "success", interval).inc()
            DATA_CONNECTOR_LATENCY.labels("remote", interval).observe(duration_seconds)
            logger.info(
                "data_connector.fetch_success",
                extra={
                    "symbol": symbol,
                    "interval": interval,
                    "limit": limit,
                    "latency_ms": self.last_latency_ms,
                    "source": self.last_data_source,
                },
            )
            return candles
        except requests.RequestException as exc:
            self.failure_count += 1
            DATA_CONNECTOR_REQUESTS.labels("remote", "failure", interval).inc()
            logger.warning(
                "data_connector.remote_fetch_failed",
                extra={
                    "symbol": symbol,
                    "base_url": self.base_url,
                    "retries": self.max_retries,
                    "error": str(exc),
                },
            )
            if self.allow_mock:
                logger.info("data_connector.mock_fallback_enabled", symbol=symbol, interval=interval, limit=limit)
                mock_candles = self._generate_mock_candles(symbol, interval, start_date, end_date, limit)
                self.last_data_source = "mock"
                return mock_candles
            raise

    def fetch_multiple_symbols(
        self,
        symbols: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 1000,
        max_workers: int = 4,
    ) -> dict[str, list[Candle]]:
        """
        Fetch data for multiple symbols concurrently.
        """
        results: dict[str, list[Candle]] = {}
        if not symbols:
            return results

        def _task(sym: str) -> tuple[str, list[Candle]]:
            candles = self.fetch_daily_candles(sym, start_date, end_date, limit)
            return sym, candles

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_task, sym): sym for sym in symbols}
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    sym_key, candles = fut.result()
                    results[sym_key] = candles
                except Exception as exc:
                    logger.warning(
                        "data_connector.fetch_symbol_failed",
                        extra={"symbol": sym, "error": str(exc)},
                    )
                    results[sym] = []

        return results

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _perform_request(self, path: str, params: dict[str, Any]) -> dict:
        url = f"{self.base_url}{path}"
        last_exc: requests.RequestException | None = None

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

    def _generate_mock_candles(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
        limit: int,
    ) -> list[Candle]:
        """Generate simple mock candles when remote fetch is unavailable."""
        step = self._interval_to_timedelta(interval)
        candles: list[Candle] = []
        current = start_date
        for i in range(limit):
            if current > end_date:
                break
            base = 100.0 + i
            candles.append(
                Candle(
                    timestamp=current,
                    open=base,
                    high=base + 1.0,
                    low=base - 1.0,
                    close=base + 0.5,
                    volume=1_000 + i * 10,
                    symbol=symbol,
                    timeframe=interval,
                )
            )
            current += step
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
