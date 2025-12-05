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
import json
from datetime import datetime
from typing import Any, Optional

import structlog
from gravity_tech.config.settings import settings
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
        self.consumer: Optional[EventConsumer] = None
        self.historical_manager: Optional[HistoricalScoreManager] = None
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
            self.historical_manager = HistoricalScoreManager()

            logger.info("data_ingestor_service_initialized")

        except Exception as e:
            logger.error("data_ingestor_initialization_failed", error=str(e))
            raise

    async def start_consuming(self):
        """شروع مصرف eventها"""
        if not self.historical_manager:
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

            # ذخیره در دیتابیس (synchronous operation in thread pool)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,  # Use default executor
                lambda: self.historical_manager.save_score(entry)
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

        # سایر فیلدها
        volume_score = results.get("volume_score", 0.0)
        volatility_score = results.get("volatility_score", 0.0)
        cycle_score = results.get("cycle_score", 0.0)
        support_resistance_score = results.get("support_resistance_score", 0.0)

        return HistoricalScoreEntry(
            symbol=symbol,
            timestamp=datetime.utcnow(),
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
            raw_data=json.dumps(results)  # ذخیره کامل نتایج به صورت JSON
        )


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
