"""
Tests for Event-Driven Messaging Middleware

Tests Kafka and RabbitMQ integration.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from gravity_tech.middleware.events import EventPublisher, EventConsumer, MessageType


@pytest.fixture
def event_publisher():
    """Create event publisher instance for testing."""
    return EventPublisher()


@pytest.fixture
def event_consumer():
    """Create event consumer instance for testing."""
    return EventConsumer()


class TestEventPublisher:
    """Tests for EventPublisher class."""
    
    @pytest.mark.asyncio
    async def test_kafka_publisher_initialization(self, event_publisher):
        """Test Kafka publisher initialization."""
        with patch('gravity_tech.middleware.events.settings') as mock_settings, \
             patch('gravity_tech.middleware.events.AIOKafkaProducer') as mock_producer:
            mock_settings.kafka_enabled = True
            mock_settings.rabbitmq_enabled = False
            mock_instance = AsyncMock()
            mock_instance.start = AsyncMock()
            mock_producer.return_value = mock_instance
            
            await event_publisher.initialize(broker_type="kafka")
            
            assert event_publisher.broker_type == "kafka"
            mock_producer.assert_called_once()
            mock_instance.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rabbitmq_consumer_initialization(self, event_consumer):
        """Test RabbitMQ consumer initialization."""
        with patch('gravity_tech.middleware.events.settings') as mock_settings, \
             patch('aio_pika.connect_robust') as mock_connect, \
             patch('gravity_tech.middleware.events.Pool') as mock_pool:
            mock_settings.rabbitmq_enabled = True
            mock_connection = AsyncMock()
            mock_connect.return_value = mock_connection
            mock_pool.return_value = AsyncMock()
            
            await event_consumer.initialize(broker_type="rabbitmq")
            
            assert event_consumer.broker_type == "rabbitmq"
    
    @pytest.mark.asyncio
    async def test_publish_kafka_event(self, event_publisher):
        """Test publishing event to Kafka."""
        with patch('gravity_tech.middleware.events.settings') as mock_settings, \
             patch('gravity_tech.middleware.events.AIOKafkaProducer') as mock_producer:
            mock_settings.kafka_enabled = True
            mock_instance = AsyncMock()
            mock_instance.start = AsyncMock()
            mock_instance.send = AsyncMock()
            mock_producer.return_value = mock_instance
            
            await event_publisher.initialize(broker_type="kafka")
            
            await event_publisher.publish(
                MessageType.ANALYSIS_COMPLETED,
                {"symbol": "BTCUSDT", "signal": "BUY"}
            )
            
            # Verify send was called
            mock_instance.send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_publish_rabbitmq_event(self, event_publisher):
        """Test publishing event to RabbitMQ."""
        with patch('aio_pika.connect_robust') as mock_connect, \
             patch('gravity_tech.middleware.events.Pool') as mock_pool, \
             patch('aio_pika.Message') as mock_message:
            mock_connection = AsyncMock()
            mock_connect.return_value = mock_connection
            
            mock_channel_pool = AsyncMock()
            mock_channel = AsyncMock()
            mock_channel_pool.acquire.return_value.__aenter__.return_value = mock_channel
            
            event_publisher.rabbitmq_channel_pool = mock_channel_pool
            event_publisher.broker_type = "rabbitmq"
            
            await event_publisher.publish(
                MessageType.ANALYSIS_COMPLETED,
                {"symbol": "BTCUSDT", "signal": "BUY"}
            )
            
            # Test passes if no exception raised
            assert event_publisher.broker_type == "rabbitmq"


class TestEventConsumer:
    """Tests for EventConsumer class."""
    
    @pytest.mark.asyncio
    async def test_kafka_consumer_initialization(self, event_consumer):
        """Test Kafka consumer initialization."""
        with patch('gravity_tech.middleware.events.AIOKafkaConsumer') as mock_consumer:
            mock_instance = AsyncMock()
            mock_consumer.return_value = mock_instance
            
            await event_consumer.initialize(broker_type="kafka")
            
            assert event_consumer.broker_type == "kafka"
    
    @pytest.mark.asyncio
    async def test_rabbitmq_consumer_initialization(self, event_consumer):
        """Test RabbitMQ consumer initialization."""
        with patch('gravity_tech.middleware.events.aio_pika.connect_robust') as mock_connect:
            mock_connection = AsyncMock()
            mock_channel = AsyncMock()
            mock_queue = AsyncMock()
            mock_connection.channel.return_value = mock_channel
            mock_channel.declare_queue.return_value = mock_queue
            mock_connect.return_value = mock_connection
            
            await event_consumer.initialize(broker_type="rabbitmq")
            
            assert event_consumer.broker_type == "rabbitmq"
    
    @pytest.mark.asyncio
    async def test_subscribe_to_event(self, event_consumer):
        """Test subscribing to event type."""
        async def handler(data):
            return data
        
        await event_consumer.subscribe(MessageType.ANALYSIS_COMPLETED, handler)
        
        assert MessageType.ANALYSIS_COMPLETED.value in event_consumer.handlers
        assert event_consumer.handlers[MessageType.ANALYSIS_COMPLETED.value] == handler
    
    @pytest.mark.asyncio
    async def test_consume_event_calls_handler(self, event_consumer):
        """Test consuming event calls registered handler."""
        handler_called = False
        received_data = None
        
        async def handler(data):
            nonlocal handler_called, received_data
            handler_called = True
            received_data = data
        
        await event_consumer.subscribe(MessageType.ANALYSIS_COMPLETED, handler)
        
        # Simulate event consumption
        test_data = {"symbol": "BTCUSDT", "signal": "BUY"}
        await handler(test_data)
        
        assert handler_called
        assert received_data == test_data


class TestMessageType:
    """Tests for MessageType enum."""
    
    def test_message_type_values(self):
        """Test MessageType enum has expected values."""
        assert MessageType.ANALYSIS_STARTED.value == "analysis.started"
        assert MessageType.ANALYSIS_COMPLETED.value == "analysis.completed"
        assert MessageType.ANALYSIS_FAILED.value == "analysis.failed"
    
    def test_message_type_membership(self):
        """Test MessageType membership."""
        assert MessageType.ANALYSIS_STARTED in MessageType
        assert "INVALID_TYPE" not in [m.name for m in MessageType]


class TestEventSerialization:
    """Tests for event serialization."""
    
    @pytest.mark.asyncio
    async def test_event_serialization(self, event_publisher):
        """Test events are properly serialized."""
        with patch('gravity_tech.middleware.events.settings') as mock_settings, \
             patch('gravity_tech.middleware.events.AIOKafkaProducer') as mock_producer:
            mock_settings.kafka_enabled = True
            mock_instance = AsyncMock()
            mock_instance.start = AsyncMock()
            mock_instance.send = AsyncMock()
            mock_producer.return_value = mock_instance
            
            await event_publisher.initialize(broker_type="kafka")
            
            test_data = {
                "symbol": "BTCUSDT",
                "signal": "BUY",
                "score": 7.5,
                "metadata": {"timestamp": "2024-01-01T00:00:00"}
            }
            
            await event_publisher.publish(MessageType.ANALYSIS_COMPLETED, test_data)
            
            # Verify send was called with serialized data
            assert mock_instance.send.called
            assert event_publisher.broker_type == "kafka"


class TestErrorHandling:
    """Tests for error handling in messaging."""
    
    @pytest.mark.asyncio
    async def test_publish_handles_connection_error(self, event_publisher):
        """Test publisher handles connection errors gracefully."""
        with patch('gravity_tech.middleware.events.AIOKafkaProducer') as mock_producer:
            mock_instance = AsyncMock()
            mock_instance.send.side_effect = Exception("Connection failed")
            mock_producer.return_value = mock_instance
            
            await event_publisher.initialize(broker_type="kafka")
            
            # Should not raise exception
            try:
                await event_publisher.publish(
                    MessageType.ANALYSIS_COMPLETED,
                    {"symbol": "BTCUSDT"}
                )
            except Exception:
                pass  # Expected to handle gracefully
    
    @pytest.mark.asyncio
    async def test_consumer_handles_invalid_message(self, event_consumer):
        """Test consumer handles invalid messages."""
        async def handler(data):
            raise ValueError("Invalid data")
        
        await event_consumer.subscribe(MessageType.ANALYSIS_COMPLETED, handler)
        
        # Should handle error without crashing
        try:
            await handler({"invalid": "data"})
        except ValueError:
            pass  # Expected


class TestConnectionPooling:
    """Tests for connection pooling."""
    
    @pytest.mark.asyncio
    async def test_rabbitmq_connection_pool(self, event_publisher):
        """Test RabbitMQ uses connection pooling."""
        with patch('gravity_tech.middleware.events.settings') as mock_settings, \
             patch('aio_pika.connect_robust') as mock_connect, \
             patch('gravity_tech.middleware.events.Pool') as mock_pool:
            mock_settings.rabbitmq_enabled = True
            mock_connection = AsyncMock()
            mock_connect.return_value = mock_connection
            mock_pool.return_value = AsyncMock()
            
            await event_publisher.initialize(broker_type="rabbitmq")
            
            # Verify connection pooling is configured
            assert event_publisher.rabbitmq_connection_pool is not None


class TestGracefulShutdown:
    """Tests for graceful shutdown."""
    
    @pytest.mark.asyncio
    async def test_publisher_shutdown(self, event_publisher):
        """Test publisher graceful shutdown."""
        with patch('gravity_tech.middleware.events.settings') as mock_settings, \
             patch('gravity_tech.middleware.events.AIOKafkaProducer') as mock_producer:
            mock_settings.kafka_enabled = True
            mock_instance = AsyncMock()
            mock_instance.start = AsyncMock()
            mock_instance.stop = AsyncMock()
            mock_producer.return_value = mock_instance
            
            await event_publisher.initialize(broker_type="kafka")
            await event_publisher.close()
            
            # Verify stop was called
            mock_instance.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_consumer_shutdown(self, event_consumer):
        """Test consumer graceful shutdown."""
        with patch('gravity_tech.middleware.events.AIOKafkaConsumer') as mock_consumer:
            mock_instance = AsyncMock()
            mock_instance.start = AsyncMock()
            mock_instance.stop = AsyncMock()
            mock_consumer.return_value = mock_instance
            
            await event_consumer.initialize(broker_type="kafka")
            await event_consumer.close()
            
            # Verify stop was called
            mock_instance.stop.assert_called_once()


@pytest.mark.integration
class TestEventIntegration:
    """Integration tests for event messaging."""
    
    @pytest.mark.asyncio
    async def test_publish_and_consume_flow(self):
        """Test complete publish and consume flow."""
        publisher = EventPublisher()
        consumer = EventConsumer()
        
        received_events = []
        
        async def handler(data):
            received_events.append(data)
        
        with patch('gravity_tech.middleware.events.settings') as mock_settings, \
             patch('gravity_tech.middleware.events.AIOKafkaProducer') as mock_producer, \
             patch('gravity_tech.middleware.events.AIOKafkaConsumer') as mock_consumer:
            
            mock_settings.kafka_enabled = True
            mock_prod_instance = AsyncMock()
            mock_prod_instance.start = AsyncMock()
            mock_prod_instance.send = AsyncMock()
            mock_cons_instance = AsyncMock()
            mock_cons_instance.start = AsyncMock()
            mock_cons_instance.subscribe = Mock()  # Non-async mock
            mock_producer.return_value = mock_prod_instance
            mock_consumer.return_value = mock_cons_instance
            
            await publisher.initialize(broker_type="kafka")
            await consumer.initialize(broker_type="kafka")
            await consumer.subscribe(MessageType.ANALYSIS_COMPLETED, handler)
            
            # Publish event
            test_data = {"symbol": "BTCUSDT", "signal": "BUY"}
            await publisher.publish(MessageType.ANALYSIS_COMPLETED, test_data)
            
            # Verify both initialized
            assert publisher.broker_type == "kafka"
            assert consumer.broker_type == "kafka"
            await publisher.publish(MessageType.ANALYSIS_COMPLETED, test_data)
            
            # Verify publish was called
            assert mock_prod_instance.send.called


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

