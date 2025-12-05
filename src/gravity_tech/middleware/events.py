"""
Event-Driven Messaging - Kafka & RabbitMQ Integration

ارسال و دریافت پیام‌های async برای communication بین میکروسرویس‌ها

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import structlog

# Make aiokafka optional
try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    if not TYPE_CHECKING:
        AIOKafkaProducer = None
        AIOKafkaConsumer = None

# Make aio_pika optional
try:
    from aio_pika import Channel, Connection, DeliveryMode, Message, connect_robust
    from aio_pika.pool import Pool
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False
    if not TYPE_CHECKING:
        Channel = None
        Connection = None
        Pool = None
    connect_robust = None
    Message = None
    DeliveryMode = None
    Channel = None
    Connection = None
    Pool = None

from gravity_tech.config.settings import settings

logger = structlog.get_logger()


class MessageType(Enum):
    """انواع پیام‌های event"""
    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_COMPLETED = "analysis.completed"
    ANALYSIS_FAILED = "analysis.failed"
    SIGNAL_GENERATED = "signal.generated"
    ALERT_TRIGGERED = "alert.triggered"


class EventPublisher:
    """
    ناشر رویدادها (Events) برای ارتباط async

    پشتیبانی از Kafka و RabbitMQ

    Example:
        >>> publisher = EventPublisher()
        >>> await publisher.initialize()
        >>> await publisher.publish(
        ...     MessageType.ANALYSIS_COMPLETED,
        ...     {"symbol": "BTCUSDT", "signal": "BUY"}
        ... )
    """

    def __init__(self):
        if TYPE_CHECKING:
            self.kafka_producer: Optional[AIOKafkaProducer] = None
            self.rabbitmq_connection_pool: Optional[Pool] = None  # type: ignore[valid-type]
            self.rabbitmq_channel_pool: Optional[Pool] = None  # type: ignore[valid-type]
        else:
            self.kafka_producer = None
            self.rabbitmq_connection_pool = None
            self.rabbitmq_channel_pool = None
        self.broker_type: Optional[str] = None

    async def initialize(self, broker_type: str = "kafka"):
        """
        راه‌اندازی اولیه publisher

        Args:
            broker_type: نوع message broker ("kafka" یا "rabbitmq")
        """
        if not settings.kafka_enabled and not settings.rabbitmq_enabled:
            logger.info("event_messaging_disabled")
            return

        self.broker_type = broker_type

        if broker_type == "kafka" and KAFKA_AVAILABLE:
            await self._init_kafka()
        elif broker_type == "rabbitmq":
            await self._init_rabbitmq()
        else:
            raise ValueError(f"Unsupported broker type: {broker_type}")

    async def _init_kafka(self):
        """راه‌اندازی Kafka Producer"""
        try:
            self.kafka_producer = AIOKafkaProducer(
                bootstrap_servers=getattr(settings, 'kafka_bootstrap_servers', 'localhost:9092'),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                compression_type='gzip',
                max_batch_size=16384,
                linger_ms=10,
            )

            await self.kafka_producer.start()
            logger.info("kafka_producer_started")

        except Exception as e:
            logger.error("kafka_producer_initialization_failed", error=str(e))
            raise

    async def _init_rabbitmq(self):
        """راه‌اندازی RabbitMQ Connection Pool"""
        try:
            rabbitmq_url = getattr(settings, 'rabbitmq_url', 'amqp://guest:guest@localhost/')

            async def get_connection() -> Connection:  # type: ignore[valid-type]
                return await connect_robust(rabbitmq_url)

            async def get_channel() -> Channel:  # type: ignore[valid-type]
                async with self.rabbitmq_connection_pool.acquire() as connection:
                    return await connection.channel()

            self.rabbitmq_connection_pool = Pool(get_connection, max_size=10)
            self.rabbitmq_channel_pool = Pool(get_channel, max_size=100)

            logger.info("rabbitmq_connection_pool_created")

        except Exception as e:
            logger.error("rabbitmq_initialization_failed", error=str(e))
            raise

    async def publish(
        self,
        event_type: MessageType,
        data: dict[str, Any],
        routing_key: Optional[str] = None
    ):
        """
        انتشار یک event

        Args:
            event_type: نوع event
            data: داده‌های event
            routing_key: کلید routing (برای RabbitMQ)
        """
        message = {
            "event_type": event_type.value,
            "data": data,
            "timestamp": str(asyncio.get_event_loop().time()),
            "service": settings.app_name,
            "version": settings.app_version
        }

        try:
            if self.broker_type == "kafka":
                await self._publish_kafka(event_type.value, message)
            elif self.broker_type == "rabbitmq":
                await self._publish_rabbitmq(
                    event_type.value,
                    message,
                    routing_key or event_type.value
                )

            logger.info(
                "event_published",
                event_type=event_type.value,
                data_keys=list(data.keys())
            )

        except Exception as e:
            logger.error(
                "event_publish_failed",
                event_type=event_type.value,
                error=str(e)
            )
            raise

    async def _publish_kafka(self, topic: str, message: dict):
        """ارسال به Kafka"""
        if not self.kafka_producer:
            raise RuntimeError("Kafka producer not initialized")

        await self.kafka_producer.send(topic, message)

    async def _publish_rabbitmq(
        self,
        exchange: str,
        message: dict,
        routing_key: str
    ):
        """ارسال به RabbitMQ"""
        if not self.rabbitmq_channel_pool:
            raise RuntimeError("RabbitMQ not initialized")

        async with self.rabbitmq_channel_pool.acquire() as channel:
            # اعلام exchange
            await channel.declare_exchange(
                exchange,
                type='topic',
                durable=True
            )

            # ارسال پیام
            await channel.default_exchange.publish(
                Message(
                    body=json.dumps(message).encode(),
                    delivery_mode=DeliveryMode.PERSISTENT,
                    content_type='application/json',
                ),
                routing_key=routing_key
            )

    async def close(self):
        """بستن اتصالات"""
        try:
            if self.kafka_producer:
                await self.kafka_producer.stop()
                logger.info("kafka_producer_stopped")

            if self.rabbitmq_connection_pool:
                await self.rabbitmq_connection_pool.close()
                logger.info("rabbitmq_connection_pool_closed")

        except Exception as e:
            logger.error("event_publisher_close_error", error=str(e))


class EventConsumer:
    """
    مصرف‌کننده رویدادها

    Example:
        >>> consumer = EventConsumer()
        >>> await consumer.initialize()
        >>>
        >>> async def handle_analysis_completed(data):
        ...     print(f"Analysis completed: {data}")
        >>>
        >>> await consumer.subscribe(
        ...     MessageType.ANALYSIS_COMPLETED,
        ...     handle_analysis_completed
        ... )
    """

    def __init__(self):
        self.kafka_consumer: Optional[AIOKafkaConsumer] = None
        self.rabbitmq_connection_pool: Optional[Pool] = None  # type: ignore[valid-type]
        self.broker_type: Optional[str] = None
        self.handlers: dict[str, Callable] = {}

    async def initialize(self, broker_type: str = "kafka"):
        """راه‌اندازی consumer"""
        self.broker_type = broker_type

        if broker_type == "kafka":
            await self._init_kafka()
        elif broker_type == "rabbitmq":
            await self._init_rabbitmq()

    async def _init_kafka(self):
        """راه‌اندازی Kafka Consumer"""
        self.kafka_consumer = AIOKafkaConsumer(
            bootstrap_servers=getattr(settings, 'kafka_bootstrap_servers', 'localhost:9092'),
            group_id=settings.app_name,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True,
        )

        await self.kafka_consumer.start()
        logger.info("kafka_consumer_started")

    async def _init_rabbitmq(self):
        """راه‌اندازی RabbitMQ Consumer"""
        rabbitmq_url = getattr(settings, 'rabbitmq_url', 'amqp://guest:guest@localhost/')

        async def get_connection() -> Connection:  # type: ignore[valid-type]
            return await connect_robust(rabbitmq_url)

        self.rabbitmq_connection_pool = Pool(get_connection, max_size=10)
        logger.info("rabbitmq_consumer_initialized")

    async def subscribe(
        self,
        event_type: MessageType,
        handler: Callable[[dict], Any]
    ):
        """
        اشتراک در یک event و تعریف handler

        Args:
            event_type: نوع event
            handler: تابع برای پردازش event
        """
        self.handlers[event_type.value] = handler

        if self.broker_type == "kafka":
            self.kafka_consumer.subscribe([event_type.value])

        logger.info("subscribed_to_event", event_type=event_type.value)

    async def start_consuming(self):
        """شروع مصرف پیام‌ها (blocking)"""
        if self.broker_type == "kafka":
            await self._consume_kafka()
        elif self.broker_type == "rabbitmq":
            await self._consume_rabbitmq()

    async def _consume_kafka(self):
        """مصرف پیام‌ها از Kafka"""
        async for message in self.kafka_consumer:
            event_type = message.topic
            data = message.value

            if event_type in self.handlers:
                try:
                    await self.handlers[event_type](data)
                    logger.info("event_processed", event_type=event_type)
                except Exception as e:
                    logger.error(
                        "event_processing_failed",
                        event_type=event_type,
                        error=str(e)
                    )

    async def _consume_rabbitmq(self):
        """مصرف پیام‌ها از RabbitMQ"""
        async with self.rabbitmq_connection_pool.acquire() as connection:
            channel = await connection.channel()

            for event_type, handler in self.handlers.items():
                queue = await channel.declare_queue(
                    f"{settings.app_name}.{event_type}",
                    durable=True
                )

                async def on_message(message):
                    async with message.process():
                        data = json.loads(message.body.decode())
                        try:
                            await handler(data)
                            logger.info("event_processed", event_type=event_type)
                        except Exception as e:
                            logger.error(
                                "event_processing_failed",
                                event_type=event_type,
                                error=str(e)
                            )

                await queue.consume(on_message)

            # نگه داشتن connection
            await asyncio.Future()

    async def close(self):
        """بستن consumer"""
        if self.kafka_consumer:
            await self.kafka_consumer.stop()

        if self.rabbitmq_connection_pool:
            await self.rabbitmq_connection_pool.close()


# Global instances
event_publisher = EventPublisher()
event_consumer = EventConsumer()
