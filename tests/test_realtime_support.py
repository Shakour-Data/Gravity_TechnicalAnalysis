"""
Comprehensive Tests for Real-time Support Features

Tests cover:
- WebSocket handler functionality
- Server-Sent Events (SSE) handler
- Real-time data streaming
- Connection management
- Integration with TSE database
- Error handling and reconnection

Author: Gravity Tech Team
Date: December 4, 2025
Version: 1.0.0
License: MIT
"""

import pytest
import asyncio
import sqlite3
import json
import websockets
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np

from gravity_tech.core.domain.entities import Candle
from gravity_tech.api.websocket_handler import WebSocketHandler
from gravity_tech.api.sse_handler import SSEHandler


class TestRealTimeSupport:
    """Test suite for real-time support features with TSE data integration."""

    @pytest.fixture
    def tse_db_connection(self):
        """Fixture to provide TSE database connection."""
        db_path = project_root / "data" / "tse_data.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        yield conn
        conn.close()

    @pytest.fixture
    def sample_tse_candles(self, tse_db_connection) -> List[Candle]:
        """Load real TSE candle data for testing."""
        cursor = tse_db_connection.cursor()
        cursor.execute("""
            SELECT * FROM candles
            WHERE symbol = 'شستا'
            ORDER BY timestamp ASC
            LIMIT 50
        """)

        candles = []
        for row in cursor.fetchall():
            candles.append(Candle(
                timestamp=datetime.fromisoformat(row['timestamp']),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume'])
            ))

        return candles

    @pytest.fixture
    def websocket_handler(self):
        """Fixture to provide WebSocket handler instance."""
        return WebSocketHandler()

    @pytest.fixture
    def sse_handler(self):
        """Fixture to provide SSE handler instance."""
        return SSEHandler()

    @pytest.mark.asyncio
    async def test_websocket_handler_initialization(self, websocket_handler):
        """Test WebSocket handler initialization."""
        assert websocket_handler is not None
        assert websocket_handler.host == "localhost"
        assert websocket_handler.port == 8765
        assert hasattr(websocket_handler, 'start_server')
        assert hasattr(websocket_handler, 'stop_server')
        assert hasattr(websocket_handler, 'broadcast_message')

    @pytest.mark.asyncio
    async def test_sse_handler_initialization(self, sse_handler):
        """Test SSE handler initialization."""
        assert sse_handler is not None
        assert sse_handler.host == "localhost"
        assert sse_handler.port == 8000
        assert hasattr(sse_handler, 'start_server')
        assert hasattr(sse_handler, 'stop_server')
        assert hasattr(sse_handler, 'send_event')

    def test_websocket_message_formatting(self, websocket_handler, sample_tse_candles):
        """Test WebSocket message formatting with TSE data."""
        # Format candle data as WebSocket message
        candle = sample_tse_candles[0]
        message = websocket_handler._format_candle_message(candle)

        assert isinstance(message, str)

        # Parse JSON message
        data = json.loads(message)
        assert 'type' in data
        assert 'symbol' in data
        assert 'timestamp' in data
        assert 'price_data' in data

        price_data = data['price_data']
        assert price_data['open'] == candle.open
        assert price_data['high'] == candle.high
        assert price_data['low'] == candle.low
        assert price_data['close'] == candle.close
        assert price_data['volume'] == candle.volume

    def test_sse_event_formatting(self, sse_handler, sample_tse_candles):
        """Test SSE event formatting with TSE data."""
        candle = sample_tse_candles[0]
        event_data = sse_handler._format_candle_event(candle)

        assert isinstance(event_data, dict)
        assert 'event' in event_data
        assert 'data' in event_data
        assert 'id' in event_data

        # Parse the data field
        candle_data = json.loads(event_data['data'])
        assert candle_data['open'] == candle.open
        assert candle_data['close'] == candle.close

    @pytest.mark.asyncio
    async def test_websocket_broadcast_simulation(self, websocket_handler):
        """Test WebSocket broadcast functionality simulation."""
        # Mock connected clients
        websocket_handler.connected_clients = {Mock(), Mock(), Mock()}

        test_message = {"type": "price_update", "symbol": "شستا", "price": 1000.0}

        # Mock send method for all clients
        for client in websocket_handler.connected_clients:
            client.send = AsyncMock()

        # Broadcast message
        await websocket_handler.broadcast_message(test_message)

        # Verify all clients received the message
        for client in websocket_handler.connected_clients:
            client.send.assert_called_once_with(json.dumps(test_message))

    @pytest.mark.asyncio
    async def test_sse_event_broadcast(self, sse_handler):
        """Test SSE event broadcasting."""
        # Mock connected clients
        sse_handler.connected_clients = {Mock(), Mock()}

        test_event = {
            "event": "price_update",
            "data": json.dumps({"symbol": "شستا", "price": 1000.0}),
            "id": "123"
        }

        # Mock send method for all clients
        for client in sse_handler.connected_clients:
            client.send = AsyncMock()

        # Send event
        await sse_handler.send_event(test_event)

        # Verify all clients received the event
        for client in sse_handler.connected_clients:
            expected_data = f"event: {test_event['event']}\ndata: {test_event['data']}\nid: {test_event['id']}\n\n"
            client.send.assert_called_once_with(expected_data)

    def test_realtime_data_streaming_from_tse(self, websocket_handler, tse_db_connection):
        """Test real-time data streaming from TSE database."""
        cursor = tse_db_connection.cursor()

        # Get recent candles for streaming simulation
        cursor.execute("""
            SELECT * FROM candles
            WHERE symbol = 'شستا'
            ORDER BY timestamp DESC
            LIMIT 10
        """)

        recent_candles = []
        for row in cursor.fetchall():
            recent_candles.append(Candle(
                timestamp=datetime.fromisoformat(row['timestamp']),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume'])
            ))

        # Simulate streaming these candles
        messages = []
        for candle in recent_candles:
            message = websocket_handler._format_candle_message(candle, symbol='شستا')
            messages.append(json.loads(message))

        assert len(messages) == len(recent_candles)

        # Verify message structure
        for msg in messages:
            assert msg['type'] == 'candle_update'
            assert msg['symbol'] == 'شستا'
            assert 'price_data' in msg

    @pytest.mark.asyncio
    async def test_connection_management(self, websocket_handler):
        """Test WebSocket connection management."""
        # Simulate client connections
        mock_client1 = Mock()
        mock_client2 = Mock()

        # Add connections
        websocket_handler.connected_clients.add(mock_client1)
        websocket_handler.connected_clients.add(mock_client2)

        assert len(websocket_handler.connected_clients) == 2

        # Simulate disconnection
        websocket_handler.connected_clients.remove(mock_client1)

        assert len(websocket_handler.connected_clients) == 1

    @pytest.mark.asyncio
    async def test_error_handling_connection_loss(self, websocket_handler):
        """Test error handling when connections are lost."""
        # Mock clients with failing send methods
        failing_client = Mock()
        close_frame = websockets.frames.Close(1000, "Connection closed")
        failing_client.send = AsyncMock(side_effect=websockets.exceptions.ConnectionClosedError(close_frame, close_frame))

        working_client = Mock()
        working_client.send = AsyncMock()

        websocket_handler.connected_clients = {failing_client, working_client}

        test_message = {"type": "test", "data": "test"}

        # Broadcast should handle failed connections gracefully
        await websocket_handler.broadcast_message(test_message)

        # Working client should still receive message
        working_client.send.assert_called_once()

        # Failing client should have been called but failed
        failing_client.send.assert_called_once()

        # Failing client should be removed from connected clients
        assert failing_client not in websocket_handler.connected_clients
        assert working_client in websocket_handler.connected_clients

    def test_data_filtering_by_symbol(self, websocket_handler, tse_db_connection):
        """Test data filtering by symbol for real-time updates."""
        cursor = tse_db_connection.cursor()

        # Get data for multiple symbols
        symbols = ['شستا', 'فملی']
        symbol_messages = {}

        for symbol in symbols:
            cursor.execute("""
                SELECT * FROM candles
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 5
            """, (symbol,))

            candles = []
            for row in cursor.fetchall():
                candles.append(Candle(
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(row['volume'])
                ))

            # Format messages for this symbol
            messages = []
            for candle in candles:
                message = websocket_handler._format_candle_message(candle, symbol=symbol)
                messages.append(json.loads(message))

            symbol_messages[symbol] = messages

        # Verify each symbol has its own data stream
        for symbol, messages in symbol_messages.items():
            assert len(messages) > 0
            for msg in messages:
                assert msg['symbol'] == symbol

    def test_realtime_price_alerts(self, websocket_handler, sample_tse_candles):
        """Test real-time price alert functionality."""
        # Set up price alerts
        alerts = {
            'شستا': {
                'upper_limit': 15000.0,
                'lower_limit': 5000.0
            }
        }

        # Simulate price monitoring
        alerts_triggered = []

        for candle in sample_tse_candles:
            if candle.close >= alerts['شستا']['upper_limit']:
                alerts_triggered.append({
                    'type': 'price_alert',
                    'symbol': 'شستا',
                    'alert_type': 'upper_limit',
                    'price': candle.close,
                    'timestamp': candle.timestamp.isoformat()
                })
            elif candle.close <= alerts['شستا']['lower_limit']:
                alerts_triggered.append({
                    'type': 'price_alert',
                    'symbol': 'شستا',
                    'alert_type': 'lower_limit',
                    'price': candle.close,
                    'timestamp': candle.timestamp.isoformat()
                })

        # Format alerts as WebSocket messages
        alert_messages = []
        for alert in alerts_triggered:
            message = websocket_handler._format_alert_message(alert)
            alert_messages.append(json.loads(message))

        # Verify alert message structure
        for msg in alert_messages:
            assert msg['type'] == 'price_alert'
            assert 'alert_type' in msg
            assert 'price' in msg

    @pytest.mark.asyncio
    async def test_concurrent_connections_simulation(self, websocket_handler):
        """Test handling multiple concurrent connections."""
        # Simulate many concurrent connections
        num_clients = 100
        mock_clients = [Mock() for _ in range(num_clients)]

        for client in mock_clients:
            client.send = AsyncMock()

        websocket_handler.connected_clients = set(mock_clients)

        test_message = {"type": "broadcast_test", "data": "concurrent_test"}

        # Broadcast to all clients
        await websocket_handler.broadcast_message(test_message)

        # Verify all clients received the message
        for client in mock_clients:
            client.send.assert_called_once_with(json.dumps(test_message))

    def test_data_compression_for_realtime(self, websocket_handler, sample_tse_candles):
        """Test data compression for efficient real-time transmission."""
        # Create a batch of candle updates
        batch_candles = sample_tse_candles[:10]

        # Format batch message
        batch_message = websocket_handler._format_batch_candle_message(batch_candles, symbol='شستا')

        # Verify batch format
        data = json.loads(batch_message)
        assert data['type'] == 'batch_candle_update'
        assert data['symbol'] == 'شستا'
        assert 'candles' in data
        assert len(data['candles']) == len(batch_candles)

    def test_realtime_technical_indicators(self, websocket_handler, sample_tse_candles):
        """Test real-time technical indicator calculations."""
        if len(sample_tse_candles) >= 14:  # Need enough data for RSI
            # Calculate RSI for real-time streaming
            closes = [c.close for c in sample_tse_candles]

            # Simple RSI calculation
            def calculate_rsi(prices, period=14):
                deltas = np.diff(prices)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)

                avg_gain = np.mean(gains[:period])
                avg_loss = np.mean(losses[:period])

                for i in range(period, len(gains)):
                    avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                    avg_loss = (avg_loss * (period - 1) + losses[i]) / period

                rs = avg_gain / avg_loss if avg_loss != 0 else 0
                rsi = 100 - (100 / (1 + rs))

                return rsi

            rsi_value = calculate_rsi(closes)

            # Format RSI update message
            rsi_message = websocket_handler._format_indicator_message(
                'شستا', 'RSI', rsi_value, sample_tse_candles[-1].timestamp
            )

            data = json.loads(rsi_message)
            assert data['type'] == 'indicator_update'
            assert data['symbol'] == 'شستا'
            assert data['indicator'] == 'RSI'
            assert 'value' in data
            assert 'timestamp' in data

    def test_connection_reconnection_logic(self, websocket_handler):
        """Test connection reconnection logic."""
        # Simulate connection states
        connection_states = ['connecting', 'connected', 'disconnected', 'reconnecting']

        for state in connection_states:
            websocket_handler.connection_state = state

            # Test state-based behavior
            if state == 'connected':
                assert websocket_handler._can_broadcast()
            elif state == 'disconnected':
                assert not websocket_handler._can_broadcast()

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, websocket_handler, sse_handler):
        """Test graceful shutdown of real-time services."""
        # Start services (mock)
        websocket_handler.server = Mock()
        websocket_handler.server.close = AsyncMock()
        websocket_handler.server.wait_closed = AsyncMock()

        sse_handler.server = Mock()
        sse_handler.server.close = AsyncMock()
        sse_handler.server.wait_closed = AsyncMock()

        # Shutdown WebSocket handler
        await websocket_handler.stop_server()

        websocket_handler.server.close.assert_called_once()
        websocket_handler.server.wait_closed.assert_called_once()

        # Shutdown SSE handler
        await sse_handler.stop_server()

        sse_handler.server.close.assert_called_once()
        sse_handler.server.wait_closed.assert_called_once()

    def test_memory_management_large_datasets(self, websocket_handler, tse_db_connection):
        """Test memory management with large real-time datasets."""
        cursor = tse_db_connection.cursor()

        # Get large dataset
        cursor.execute("""
            SELECT COUNT(*) as count FROM candles
        """)

        total_count = cursor.fetchone()['count']

        # Simulate processing large dataset in chunks
        chunk_size = 100
        processed_count = 0

        for offset in range(0, min(total_count, 1000), chunk_size):  # Limit for test
            cursor.execute("""
                SELECT * FROM candles
                ORDER BY timestamp ASC
                LIMIT ? OFFSET ?
            """, (chunk_size, offset))

            chunk = cursor.fetchall()
            processed_count += len(chunk)

            # Simulate real-time processing
            messages = []
            for row in chunk:
                candle = Candle(
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(row['volume'])
                )
                message = websocket_handler._format_candle_message(candle, symbol=row['symbol'])
                messages.append(message)

            # Verify chunk processing
            assert len(messages) == len(chunk)

        assert processed_count > 0

    def test_realtime_data_validation(self, websocket_handler, sample_tse_candles):
        """Test real-time data validation."""
        # Test valid candle data
        valid_candle = sample_tse_candles[0]
        assert websocket_handler._validate_candle_data(valid_candle)

        # Test invalid candle data
        invalid_candle = Candle(
            timestamp=datetime.now(),
            open=100.0,
            high=90.0,  # High lower than open (invalid)
            low=80.0,
            close=95.0,
            volume=1000
        )

        assert not websocket_handler._validate_candle_data(invalid_candle)

    @pytest.mark.asyncio
    async def test_rate_limiting(self, websocket_handler):
        """Test rate limiting for real-time updates."""
        # Mock rate limiter
        websocket_handler.rate_limiter = Mock()
        websocket_handler.rate_limiter.allow = Mock(return_value=True)

        mock_client = Mock()
        mock_client.send = AsyncMock()

        websocket_handler.connected_clients = {mock_client}

        # Send multiple messages quickly
        for i in range(10):
            await websocket_handler.broadcast_message({"test": i})

        # Rate limiter should be checked for each message
        assert websocket_handler.rate_limiter.allow.call_count == 10

    def test_cross_platform_compatibility(self, websocket_handler, sse_handler):
        """Test cross-platform compatibility of real-time features."""
        # Test message formatting for different platforms
        test_candle = Candle(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            open=1000.0,
            high=1010.0,
            low=990.0,
            close=1005.0,
            volume=10000
        )

        # WebSocket message (JSON)
        ws_message = websocket_handler._format_candle_message(test_candle)
        ws_data = json.loads(ws_message)

        # SSE message
        sse_event = sse_handler._format_candle_event(test_candle)

        # Both should contain same core data
        assert ws_data['price_data']['close'] == test_candle.close
        assert json.loads(sse_event['data'])['close'] == test_candle.close