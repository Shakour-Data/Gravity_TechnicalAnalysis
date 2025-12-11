"""
Data Ingestor Service

این سرویس eventهای تحلیل را consume کرده و در دیتابیس ذخیره می‌کند.
برای پشتیبانی از رویکرد هیبریدی: API real-time + Database historical.

Author: Gravity Tech Team
Date: November 20, 2025
Version: 1.0.0
License: MIT
"""

import asyncio
import hashlib
import json
import math
import time
from collections import deque
from datetime import UTC, datetime
from typing import Any

import structlog
from gravity_tech.config.settings import settings
from gravity_tech.database.database_manager import DatabaseManager
from gravity_tech.database.historical_manager import HistoricalScoreManager
from gravity_tech.middleware.events import EventConsumer, MessageType
from prometheus_client import Counter, Histogram

logger = structlog.get_logger()

INGEST_EVENTS = Counter(
    "data_ingestion_events_total",
    "Total ingestion events handled",
    ["status", "mode"],  # status: success|error|skipped, mode: broker|direct
)
INGEST_LATENCY = Histogram(
    "data_ingestion_latency_seconds",
    "Latency of ingestion handling",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
INGEST_RETRIES = Counter(
    "data_ingestion_retries_total",
    "Total retries for ingestion",
    ["reason"],
)
INGEST_CIRCUIT_BREAKER = Counter(
    "data_ingestion_circuit_breaker_total",
    "Circuit breaker activations",
    ["state"],
)


class DataIngestorService:
    """
    سرویس مصرف eventها و ذخیره در دیتابیس

    این سرویس به صورت background اجرا می‌شود و eventهای ANALYSIS_COMPLETED
    را دریافت کرده و در historical database ذخیره می‌کند.
    """

    def __init__(self):
        self.consumer: EventConsumer | None = None
        self.database_url: str | None = None
        self.running = False
        self.circuit_breaker_failures = 0
        self.max_failures = 5  # Circuit breaker threshold
        self._recent_keys: deque[str] = deque(maxlen=500)  # simple dedup window

    def _validate_payload(self, data: dict[str, Any]) -> bool:
        """
        Validate incoming payload for required fields and data integrity.
        """
        required_fields = ["symbol", "timeframe"]
        for field in required_fields:
            if field not in data:
                logger.warning("missing_required_field", field=field)
                return False

        symbol = data["symbol"]
        timeframe = data["timeframe"]

        # Validate symbol format (basic check)
        if not isinstance(symbol, str) or len(symbol) == 0 or len(symbol) > 10:
            logger.warning("invalid_symbol", symbol=symbol)
            return False

        # Validate timeframe (align with API)
        valid_timeframes = {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"}
        if timeframe not in valid_timeframes:
            logger.warning("invalid_timeframe", timeframe=timeframe)
            return False

        # Check for NaN/Inf in numeric fields
        numeric_fields = [
            "trend_score", "momentum_score", "combined_score",
            "price_at_analysis", "trend_confidence", "momentum_confidence"
        ]
        for field in numeric_fields:
            value = data.get(field)
            if value is not None:
                if not isinstance(value, int | float) or math.isnan(value) or math.isinf(value):
                    logger.warning("invalid_numeric_value", field=field, value=value)
                    return False

        # Validate candles if present
        candles = data.get("candles", [])
        if candles:
            for i, candle in enumerate(candles):
                if not isinstance(candle, dict):
                    logger.warning("invalid_candle_format", index=i)
                    return False
                for price_field in ["open", "high", "low", "close"]:
                    price = candle.get(price_field)
                    if price is None or not isinstance(price, int | float) or price <= 0 or math.isnan(price) or math.isinf(price):
                        logger.warning("invalid_candle_price", index=i, field=price_field, value=price)
                        return False
                # Check chronological order
                if i > 0 and candle.get("timestamp") <= candles[i-1].get("timestamp"):
                    logger.warning("non_chronological_candles", index=i)
                    return False

        # Ensure results section exists
        if not data.get("results") and "results" in data:
            logger.warning("empty_results_section")
            return False

        # Size limit check (10MB max)
        payload_size = len(json.dumps(data).encode('utf-8'))
        if payload_size > 10 * 1024 * 1024:
            logger.warning("payload_too_large", size=payload_size)
            return False

        return True

    def _generate_unique_key(self, data: dict[str, Any]) -> str:
        """
        Generate a unique key for deduplication based on symbol, timeframe, and analysis timestamp.
        """
        key_data = {
            "symbol": data["symbol"],
            "timeframe": data["timeframe"],
            "timestamp": data.get("analysis_timestamp", datetime.now(UTC).isoformat())
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _authenticate_request(self, payload: dict[str, Any]) -> bool:
        """
        Authenticate incoming request using a simple token check.
        In production, use proper JWT or OAuth.
        """
        expected_token = settings.ingestion_auth_token
        # If no token configured, skip auth (internal usage)
        if not expected_token:
            return True
        token = payload.get("auth_token")
        if token != expected_token:
            logger.warning("invalid_auth_token")
            return False
        return True

    def _check_deduplication(self, unique_key: str) -> bool:
        """
        Simple in-memory deduplication window.
        Returns True if key already seen recently.
        """
        if unique_key in self._recent_keys:
            return True
        self._recent_keys.append(unique_key)
        return False

    async def initialize(self):
        """راه‌اندازی سرویس"""
        try:
            # بررسی فعال بودن event messaging
            if not (settings.kafka_enabled or settings.rabbitmq_enabled):
                logger.warning("no_event_broker_enabled",
                             kafka=settings.kafka_enabled,
                             rabbitmq=settings.rabbitmq_enabled)
                # بدون consumer ادامه می‌دهیم
                self.consumer = None
            else:
                # راه‌اندازی consumer
                broker_type = "kafka" if settings.kafka_enabled else "rabbitmq"
                self.consumer = EventConsumer()
                await self.consumer.initialize(broker_type)

            # راه‌اندازی historical manager
            self.database_url = settings.database_url

            # Health check for database
            await self._check_database_health()

            logger.info("data_ingestor_service_initialized")

        except Exception as e:
            logger.error("data_ingestor_initialization_failed", error=str(e))
            raise

    async def _check_database_health(self):
        """Check database connectivity and basic operations."""
        if not self.database_url:
            raise RuntimeError("Database URL not configured")

        try:
            # Test connection by initializing HistoricalScoreManager
            HistoricalScoreManager(self.database_url)
            logger.info("database_health_check_passed")
        except Exception as e:
            logger.error("database_health_check_failed", error=str(e))
            raise RuntimeError(f"Database health check failed: {e}") from e

    async def start_consuming(self):
        """شروع مصرف eventها"""
        if not self.database_url:
            raise RuntimeError("Service not initialized")

        if not self.consumer:
            logger.info("event_consumer_disabled", reason="no_broker_enabled")
            return

        self.running = True
        logger.info("data_ingestor_started")

        try:
            # Subscribe به eventهای ANALYSIS_COMPLETED
            await self.consumer.subscribe(
                MessageType.ANALYSIS_COMPLETED,
                self._handle_analysis_completed
            )

            # شروع consuming
            await self.consumer.start_consuming()

        except Exception as e:
            logger.error("data_ingestor_consuming_failed", error=str(e))
            self.running = False
            raise

    async def stop_consuming(self):
        """توقف مصرف eventها"""
        self.running = False
        if self.consumer:
            await self.consumer.close()
        logger.info("data_ingestor_stopped")

    async def _handle_analysis_completed(self, message: dict[str, Any]):
        """
        هندل کردن event ANALYSIS_COMPLETED

        Args:
            message: پیام event شامل نتایج تحلیل
        """
        try:
            data = message.get("data", {})

            # Validate payload
            if not self._validate_payload(data):
                INGEST_EVENTS.labels(status="skipped", mode="broker").inc()
                logger.warning("payload_validation_failed", message_keys=list(message.keys()))
                return

            symbol = data.get("symbol")
            timeframe = data.get("timeframe")
            results = data.get("results", {})

            if not symbol or not results:
                logger.warning("invalid_analysis_event_data", data_keys=list(data.keys()))
                INGEST_EVENTS.labels(status="skipped", mode="broker").inc()
                return

            # Check deduplication
            unique_key = self._generate_unique_key(data)
            if self._check_deduplication(unique_key):
                logger.info("duplicate_analysis_skipped", symbol=symbol, timeframe=timeframe)
                INGEST_EVENTS.labels(status="skipped", mode="broker").inc()
                return

            # Circuit breaker check
            if self.circuit_breaker_failures >= self.max_failures:
                INGEST_CIRCUIT_BREAKER.labels(state="open").inc()
                logger.warning("circuit_breaker_open", failures=self.circuit_breaker_failures)
                return

            # تبدیل نتایج به HistoricalScoreEntry
            entry = self._convert_to_historical_entry(symbol, timeframe, results)

            horizon_scores = results.get("horizon_scores") or results.get("multi_horizon_scores")

            indicator_scores = (
                results.get("indicator_scores")
                or results.get("indicators")
                or self._build_indicator_scores(results)
            )
            patterns = results.get("patterns")
            volume_analysis = results.get("volume_analysis") or results.get("volume")
            price_targets = results.get("price_targets")

            pattern_detections = self._build_pattern_detections(symbol, timeframe, results)

            # ذخیره در دیتابیس (synchronous operation in thread pool)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,  # Use default executor
                lambda: self._persist_entry(
                    entry,
                    horizon_scores=horizon_scores,
                    indicator_scores=indicator_scores,
                    patterns=patterns,
                    volume_analysis=volume_analysis,
                    price_targets=price_targets,
                    pattern_detections=pattern_detections,
                )
            )

            # Reset circuit breaker on success
            self.circuit_breaker_failures = 0
            INGEST_EVENTS.labels(status="success", mode="broker").inc()

            logger.info(
                "analysis_result_saved",
                symbol=symbol,
                timeframe=timeframe,
                score=entry.combined_score
            )

        except Exception as e:
            self.circuit_breaker_failures += 1
            INGEST_EVENTS.labels(status="error", mode="broker").inc()
            logger.error(
                "analysis_event_handling_failed",
                error=str(e),
                message_data=message
            )

    def _convert_to_historical_entry(self, symbol: str, timeframe: str, results: dict[str, Any]):
        """
        تبدیل نتایج تحلیل به HistoricalScoreEntry

        Args:
            symbol: نماد معاملاتی
            timeframe: تایم‌فریم
            results: نتایج تحلیل از API

        Returns:
            HistoricalScoreEntry object
        """
        from gravity_tech.database.historical_manager import HistoricalScoreEntry

        # استخراج داده‌ها از results
        # این قسمت بسته به ساختار TechnicalAnalysisResult تنظیم شود
        trend_score = results.get("trend_score", 0.0)
        trend_confidence = results.get("trend_confidence", 0.0)
        momentum_score = results.get("momentum_score", 0.0)
        momentum_confidence = results.get("momentum_confidence", 0.0)
        combined_score = results.get("combined_score", 0.0)
        combined_confidence = results.get("combined_confidence", 0.0)

        # سیگنال‌ها
        trend_signal = results.get("trend_signal", "NEUTRAL")
        momentum_signal = results.get("momentum_signal", "NEUTRAL")
        combined_signal = results.get("combined_signal", "NEUTRAL")

        # وزن‌ها
        trend_weight = results.get("trend_weight", 0.5)
        momentum_weight = results.get("momentum_weight", 0.5)

        volume_score = results.get("volume_score", 0.0)
        volatility_score = results.get("volatility_score", 0.0)
        cycle_score = results.get("cycle_score", 0.0)
        support_resistance_score = results.get("support_resistance_score", 0.0)

        price_at_analysis = results.get("price_at_analysis") or results.get("close") or 0.0
        recommendation = results.get("recommendation") or ("BUY" if combined_score > 0 else "HOLD")
        action = results.get("action") or "HOLD"

        return HistoricalScoreEntry(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            timeframe=timeframe,
            trend_score=trend_score,
            trend_confidence=trend_confidence,
            momentum_score=momentum_score,
            momentum_confidence=momentum_confidence,
            combined_score=combined_score,
            combined_confidence=combined_confidence,
            trend_weight=trend_weight,
            momentum_weight=momentum_weight,
            trend_signal=trend_signal,
            momentum_signal=momentum_signal,
            combined_signal=combined_signal,
            volume_score=volume_score,
            volatility_score=volatility_score,
            cycle_score=cycle_score,
            support_resistance_score=support_resistance_score,
            raw_data=results,
            recommendation=recommendation,
            action=action,
            price_at_analysis=float(price_at_analysis)
        )

    def _persist_entry(
        self,
        entry,
        *,
        horizon_scores=None,
        indicator_scores=None,
        patterns=None,
        volume_analysis=None,
        price_targets=None,
        pattern_detections=None,
    ):
        if not self.database_url:
            return
        manager = HistoricalScoreManager(self.database_url)
        with manager:
            manager.save_score(
                entry,
                horizon_scores=horizon_scores,
                indicator_scores=indicator_scores,
                patterns=patterns,
                volume_analysis=volume_analysis,
                price_targets=price_targets,
            )

        # ثبت الگوها در جدول pattern_detection_results (بدون وابستگی به score_id)
        if pattern_detections:
            dbm = DatabaseManager(connection_string=self.database_url, auto_setup=False)
            dbm.save_pattern_detections(pattern_detections)

    def persist_direct(self, payload: dict[str, Any]):
        """
        Persist analysis results directly without the event broker.

        Useful when enable_data_ingestion=True but Kafka/RabbitMQ are disabled.
        Includes validation, deduplication, and retry logic.
        """
        # Authenticate
        if not self._authenticate_request(payload):
            raise ValueError("Authentication failed")

        data = payload.get("results") if "results" in payload else payload

        # Validate payload
        if not self._validate_payload(data):
            INGEST_EVENTS.labels(status="skipped", mode="direct").inc()
            logger.warning("payload_validation_failed_direct", payload_keys=list(payload.keys()))
            raise ValueError("Invalid payload data")

        symbol = data.get("symbol")
        timeframe = data.get("timeframe")

        if not symbol or not timeframe:
            logger.warning("persist_direct_missing_keys", keys=list(data.keys()))
            raise ValueError("Missing required fields: symbol or timeframe")

        # Check deduplication
        unique_key = self._generate_unique_key(data)
        if self._check_deduplication(unique_key):
            logger.info("duplicate_analysis_skipped_direct", symbol=symbol, timeframe=timeframe)
            INGEST_EVENTS.labels(status="skipped", mode="direct").inc()
            return

        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                entry = self._convert_to_historical_entry(symbol, timeframe, data)
                horizon_scores = data.get("horizon_scores") or data.get("multi_horizon_scores")
                indicator_scores = (
                    data.get("indicator_scores")
                    or data.get("indicators")
                    or self._build_indicator_scores(data)
                )
                patterns = data.get("patterns")
                volume_analysis = data.get("volume_analysis") or data.get("volume")
                price_targets = data.get("price_targets")
                pattern_detections = self._build_pattern_detections(symbol, timeframe, data)

                self._persist_entry(
                    entry,
                    horizon_scores=horizon_scores,
                    indicator_scores=indicator_scores,
                    patterns=patterns,
                    volume_analysis=volume_analysis,
                    price_targets=price_targets,
                    pattern_detections=pattern_detections,
                )

                INGEST_EVENTS.labels(status="success", mode="direct").inc()
                logger.info("direct_persist_success", symbol=symbol, timeframe=timeframe)
                return

            except Exception as e:
                INGEST_RETRIES.labels(reason="persistence_error").inc()
                if attempt == max_retries - 1:
                    INGEST_EVENTS.labels(status="error", mode="direct").inc()
                    logger.error("direct_persist_failed_after_retries", error=str(e), attempts=max_retries)
                    raise
                else:
                    logger.warning("direct_persist_retry", attempt=attempt+1, error=str(e))
                    time.sleep(1)  # Simple backoff

    def _build_indicator_scores(self, results: dict[str, Any]) -> list[dict] | None:
        """Extract indicator scores from analysis result structure.

        This helper supports both dict-like payloads (e.g., JSON from API)
        and IndicatorResult objects produced by the analysis pipeline.
        """

        buckets = [
            results.get("trend_indicators"),
            results.get("momentum_indicators"),
            results.get("cycle_indicators"),
            results.get("volume_indicators"),
            results.get("volatility_indicators"),
            results.get("support_resistance_indicators"),
        ]

        flat: list[Any] = []
        for bucket in buckets:
            if bucket:
                flat.extend(bucket)

        if not flat:
            return None

        normalized: list[dict] = []
        for item in flat:
            # Handle Pydantic/BaseModel or dataclass objects
            if hasattr(item, "indicator_name"):
                name = item.indicator_name
                category = getattr(item, "category", None)
                signal = getattr(item, "signal", None)
                confidence = getattr(item, "confidence", None)
                value = getattr(item, "value", None)
                additional = getattr(item, "additional_values", None)
            else:
                # Assume dict-like
                name = item.get("indicator_name") or item.get("name")
                category = item.get("category")
                signal = item.get("signal")
                confidence = item.get("confidence")
                value = item.get("value")
                additional = item.get("additional_values") or item.get("params")

            if not name:
                continue

            norm_category = getattr(category, "value", category) or "UNKNOWN"
            norm_signal = getattr(signal, "value", signal) or "NEUTRAL"

            normalized.append(
                {
                    "name": name,
                    "category": norm_category,
                    "params": additional or {},
                    "score": value,
                    "confidence": confidence if confidence is not None else 0.0,
                    "signal": norm_signal,
                    "raw_value": value,
                }
            )

        return normalized if normalized else None

    def _build_pattern_detections(self, symbol: str, timeframe: str, results: dict[str, Any]) -> list[dict] | None:
        """Normalize pattern results (classical/candlestick) to storage schema."""

        buckets = [
            results.get("classical_patterns"),
            results.get("candlestick_patterns"),
        ]

        flat: list[Any] = []
        for bucket in buckets:
            if bucket:
                flat.extend(bucket)

        if not flat:
            return None

        analysis_ts = results.get("analysis_timestamp") or datetime.now(UTC)

        detections: list[dict] = []
        for p in flat:
            if hasattr(p, "pattern_name"):
                name = p.pattern_name
                ptype = getattr(p, "pattern_type", None)
                signal = getattr(p, "signal", None)
                confidence = getattr(p, "confidence", None)
                start_time = getattr(p, "start_time", None)
                end_time = getattr(p, "end_time", None)
                price_target = getattr(p, "price_target", None)
                stop_loss = getattr(p, "stop_loss", None)
                description = getattr(p, "description", None)
            else:
                name = p.get("pattern_name") or p.get("name")
                ptype = p.get("pattern_type") or p.get("type")
                signal = p.get("signal")
                confidence = p.get("confidence")
                start_time = p.get("start_time")
                end_time = p.get("end_time")
                price_target = p.get("price_target") or p.get("target_price")
                stop_loss = p.get("stop_loss")
                description = p.get("description")

            if not name:
                continue

            pattern_timestamp = end_time or start_time or analysis_ts
            norm_signal = getattr(signal, "value", signal)
            norm_type = getattr(ptype, "value", ptype) or "UNKNOWN"

            detections.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": pattern_timestamp,
                    "pattern_type": norm_type,
                    "pattern_name": name,
                    "confidence": confidence,
                    "strength": norm_signal,
                    "start_time": start_time,
                    "end_time": end_time,
                    "start_price": None,
                    "end_price": None,
                    "prediction": norm_signal,
                    "target_price": price_target,
                    "stop_loss": stop_loss,
                    "metadata": {
                        "description": description,
                    },
                }
            )

        return detections if detections else None


# Global instance
data_ingestor = DataIngestorService()


async def start_data_ingestor():
    """راه‌اندازی data ingestor (برای استفاده در main.py)"""
    if settings.enable_data_ingestion:
        await data_ingestor.initialize()
        # اجرای در background
        asyncio.create_task(data_ingestor.start_consuming())
        logger.info("data_ingestor_background_task_started")


async def stop_data_ingestor():
    """توقف data ingestor"""
    await data_ingestor.stop_consuming()
