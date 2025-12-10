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
from datetime import UTC, datetime
from typing import Any

import structlog
from gravity_tech.config.settings import settings
from gravity_tech.database.database_manager import DatabaseManager
from gravity_tech.database.historical_manager import HistoricalScoreManager
from gravity_tech.middleware.events import EventConsumer, MessageType

logger = structlog.get_logger()


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

            logger.info("data_ingestor_service_initialized")

        except Exception as e:
            logger.error("data_ingestor_initialization_failed", error=str(e))
            raise

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
            symbol = data.get("symbol")
            timeframe = data.get("timeframe")
            results = data.get("results", {})

            if not symbol or not results:
                logger.warning("invalid_analysis_event_data", data_keys=list(data.keys()))
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

            logger.info(
                "analysis_result_saved",
                symbol=symbol,
                timeframe=timeframe,
                score=entry.combined_score
            )

        except Exception as e:
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
        """
        data = payload.get("results") if "results" in payload else payload
        symbol = data.get("symbol")
        timeframe = data.get("timeframe")
        results = data

        if not symbol or not timeframe:
            logger.warning("persist_direct_missing_keys", keys=list(data.keys()))
            return

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

        self._persist_entry(
            entry,
            horizon_scores=horizon_scores,
            indicator_scores=indicator_scores,
            patterns=patterns,
            volume_analysis=volume_analysis,
            price_targets=price_targets,
            pattern_detections=pattern_detections,
        )

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
