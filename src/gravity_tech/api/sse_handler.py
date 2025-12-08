"""
Server-Sent Events (SSE) Handler for Real-time Data Streaming

This module implements SSE connections for:
- Real-time price updates (lightweight alternative to WebSocket)
- Live indicator streaming
- Event-driven notifications
- Server push notifications

Author: Gravity Tech Team
Date: December 4, 2025
Version: 1.0.0
License: MIT
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from datetime import datetime
from decimal import Decimal
from typing import Any

from fastapi import Request
from gravity_tech.core.indicators import MomentumIndicators, TrendIndicators
from gravity_tech.models.schemas import Candle, MarketData, SubscriptionType
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)


class SSEConnectionManager:
    """Manages SSE connections and subscriptions"""

    def __init__(self):
        self.active_connections: dict[str, asyncio.Queue] = {}
        self.subscriptions: dict[str, set[SubscriptionType]] = {}
        self.client_data: dict[str, dict[str, Any]] = {}
        self.broadcast_queues: dict[SubscriptionType, asyncio.Queue] = {}

        # Initialize broadcast queues for each subscription type
        for sub_type in SubscriptionType:
            self.broadcast_queues[sub_type] = asyncio.Queue()

    async def connect(self, client_id: str) -> asyncio.Queue:
        """Create and register a new SSE connection"""
        queue = asyncio.Queue()
        self.active_connections[client_id] = queue
        self.subscriptions[client_id] = set()
        self.client_data[client_id] = {
            'connected_at': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'message_count': 0,
            'subscriptions': set()
        }
        logger.info(f"SSE Client {client_id} connected. Total connections: {len(self.active_connections)}")
        return queue

    def disconnect(self, client_id: str) -> None:
        """Remove an SSE connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        if client_id in self.client_data:
            del self.client_data[client_id]
        logger.info(f"SSE Client {client_id} disconnected. Total connections: {len(self.active_connections)}")

    async def subscribe(self, client_id: str, subscription_type: SubscriptionType) -> None:
        """Subscribe client to a data stream"""
        if client_id not in self.subscriptions:
            self.subscriptions[client_id] = set()
        self.subscriptions[client_id].add(subscription_type)

        # Mirror subscription state in client metadata for quick access
        if client_id in self.client_data:
            self.client_data[client_id].setdefault('subscriptions', set()).add(subscription_type)

        logger.info(f"SSE Client {client_id} subscribed to {subscription_type}")

    async def unsubscribe(self, client_id: str, subscription_type: SubscriptionType) -> None:
        """Unsubscribe client from a data stream"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].discard(subscription_type)
            if client_id in self.client_data:
                self.client_data[client_id].setdefault('subscriptions', set()).discard(subscription_type)
            logger.info(f"SSE Client {client_id} unsubscribed from {subscription_type}")

    async def broadcast_to_subscribers(self, subscription_type: SubscriptionType,
                                     message: dict[str, Any]) -> None:
        """Broadcast message to all subscribers of a type"""
        recipients = [
            (client_id, self.active_connections[client_id])
            for client_id, subs in self.subscriptions.items()
            if subscription_type in subs and client_id in self.active_connections
        ]

        if not recipients:
            return

        for client_id, queue in recipients:
            await queue.put(message)

            # Track activity for the client when delivering broadcasts
            client_meta = self.client_data.get(client_id)
            if client_meta is not None:
                client_meta['last_activity'] = datetime.utcnow()
                client_meta['message_count'] += 1

    async def send_personal_message(self, client_id: str, message: dict[str, Any]) -> None:
        """Send message to specific client"""
        if client_id in self.active_connections:
            queue = self.active_connections[client_id]
            await queue.put(message)
            self.client_data[client_id]['last_activity'] = datetime.utcnow()
            self.client_data[client_id]['message_count'] += 1

    def get_connection_stats(self) -> dict[str, Any]:
        """Get connection statistics"""
        return {
            'total_connections': len(self.active_connections),
            'total_subscriptions': sum(len(subs) for subs in self.subscriptions.values()),
            'subscription_breakdown': {
                sub_type.value: sum(1 for subs in self.subscriptions.values() if sub_type in subs)
                for sub_type in SubscriptionType
            }
        }

    def get_client_data(self, client_id: str) -> dict[str, Any] | None:
        """Return stored metadata for a client, if connected"""
        return self.client_data.get(client_id)


class SSEHandler:
    """Handles SSE connections and real-time data streaming"""

    def __init__(self):
        self.connection_manager = SSEConnectionManager()
        self.trend_indicators = TrendIndicators()
        self.momentum_indicators = MomentumIndicators()
        self.market_data_buffer: dict[str, list[Candle]] = {}
        self.is_streaming = False
        self.broadcast_task = None

    async def handle_connection(self, request: Request, client_id: str,
                              subscriptions: list[SubscriptionType]) -> AsyncGenerator[str, None]:
        """Handle individual SSE connection"""
        queue = await self.connection_manager.connect(client_id)

        # Subscribe to requested types
        for sub_type in subscriptions:
            await self.connection_manager.subscribe(client_id, sub_type)

        logger.info(f"SSE connection established for {client_id} with subscriptions: {subscriptions}")

        try:
            # Send initial connection message
            yield self._format_sse_message({
                "type": "connection_established",
                "client_id": client_id,
                "subscriptions": [s.value for s in subscriptions],
                "timestamp": datetime.utcnow().isoformat()
            })

            while True:
                # Wait for messages
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)

                    # Send heartbeat if no message
                    if message is None:
                        yield self._format_sse_message({
                            "type": "heartbeat",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        continue

                    yield self._format_sse_message(message)

                except TimeoutError:
                    # Send heartbeat
                    yield self._format_sse_message({
                        "type": "heartbeat",
                        "timestamp": datetime.utcnow().isoformat()
                    })

        except Exception as e:
            logger.error(f"Error in SSE connection for {client_id}: {e}")
        finally:
            self.connection_manager.disconnect(client_id)

    async def start_market_data_stream(self, symbol: str, interval: str = "1m") -> None:
        """Start streaming market data for a symbol"""
        self.is_streaming = True

        # Initialize buffer for symbol
        if symbol not in self.market_data_buffer:
            self.market_data_buffer[symbol] = []

        logger.info(f"Started SSE market data stream for {symbol}")

        # Start broadcast task if not running
        if self.broadcast_task is None:
            self.broadcast_task = asyncio.create_task(self._broadcast_worker())

        # In a real implementation, this would connect to a market data feed
        # For now, we'll simulate streaming
        while self.is_streaming:
            try:
                # Simulate receiving new candle data
                new_candle = self._simulate_new_candle(symbol)

                # Add to buffer
                self.market_data_buffer[symbol].append(new_candle)

                # Keep only recent candles (last 1000)
                if len(self.market_data_buffer[symbol]) > 1000:
                    self.market_data_buffer[symbol] = self.market_data_buffer[symbol][-1000:]

                # Process and broadcast data
                await self._process_and_broadcast_candle(symbol, new_candle)

                # Wait for next update (simulate real-time)
                await asyncio.sleep(1)  # 1 second intervals

            except Exception as e:
                logger.error(f"Error in SSE market data stream for {symbol}: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def stop_market_data_stream(self, symbol: str) -> None:
        """Stop streaming market data for a symbol"""
        self.is_streaming = False
        logger.info(f"Stopped SSE market data stream for {symbol}")

    async def _broadcast_worker(self) -> None:
        """Worker task to handle broadcasting messages to subscribers"""
        while True:
            try:
                # Check all broadcast queues
                for sub_type, queue in self.connection_manager.broadcast_queues.items():
                    try:
                        message = queue.get_nowait()
                        await self.connection_manager.broadcast_to_subscribers(sub_type, message)
                    except asyncio.QueueEmpty:
                        continue

                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

            except Exception as e:
                logger.error(f"Error in broadcast worker: {e}")
                await asyncio.sleep(1)

    async def _process_and_broadcast_candle(self, symbol: str, candle: Candle) -> None:
        """Process new candle and broadcast to subscribers"""
        try:
            # Get recent candles for analysis
            recent_candles = self.market_data_buffer[symbol][-100:]  # Last 100 candles

            # Calculate indicators
            indicators = await self._calculate_indicators(recent_candles)

            # Detect patterns
            patterns = await self._detect_patterns(recent_candles)

            # Prepare market data message
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                price=Decimal(str(candle.close)),
                open_price=Decimal(str(candle.open)),
                high_price=Decimal(str(candle.high)),
                low_price=Decimal(str(candle.low)),
                volume=Decimal(str(candle.volume))
            )

            # Broadcast to subscribers
            await self.connection_manager.broadcast_to_subscribers(
                SubscriptionType.MARKET_DATA,
                {
                    "type": "market_data",
                    "data": market_data.to_dict(),
                    "indicators": indicators,
                    "patterns": patterns,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

        except Exception as e:
            logger.error(f"Error processing candle for {symbol}: {e}")

    async def _calculate_indicators(self, candles: list[Candle]) -> dict[str, Any]:
        """Calculate technical indicators for broadcasting"""
        try:
            indicators = {}

            # Calculate key indicators
            if len(candles) >= 50:
                # SMA 20 and 50
                try:
                    sma_20 = self.trend_indicators.sma(candles, 20)
                    sma_50 = self.trend_indicators.sma(candles, 50)
                    indicators['sma'] = {
                        'sma_20': sma_20.value,
                        'sma_50': sma_50.value
                    }
                except Exception as e:
                    logger.warning(f"Error calculating SMA: {e}")

                # MACD
                try:
                    macd_result = self.trend_indicators.macd(candles)
                    indicators['macd'] = {
                        'macd': macd_result.value,
                        'signal': macd_result.additional_values.get('signal', 0) if macd_result.additional_values else 0,
                        'histogram': macd_result.additional_values.get('histogram', 0) if macd_result.additional_values else 0
                    }
                except Exception as e:
                    logger.warning(f"Error calculating MACD: {e}")

            # RSI from momentum indicators (if available)
            if len(candles) >= 14:
                try:
                    closes = [c.close for c in candles]
                    # Use a simple RSI calculation for now
                    rsi = self._calculate_simple_rsi(closes, 14)
                    indicators['rsi'] = rsi
                except Exception as e:
                    logger.warning(f"Error calculating RSI: {e}")

            return indicators

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}

    def _calculate_simple_rsi(self, prices: list[float], period: int = 14) -> float:
        """Simple RSI calculation for SSE"""
        if len(prices) < period + 1:
            return 50.0

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    async def _detect_patterns(self, candles: list[Candle]) -> list[dict[str, Any]]:
        """Detect chart patterns for broadcasting"""
        try:
            patterns = []

            # Pattern detection not yet implemented
            # TODO: Implement pattern recognition when available

            return patterns

        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []

    def _simulate_new_candle(self, symbol: str) -> Candle:
        """Simulate new candle data (for testing)"""
        import random

        # Get last candle or create initial one
        if self.market_data_buffer[symbol]:
            last_candle = self.market_data_buffer[symbol][-1]
            base_price = last_candle.close
        else:
            base_price = 100.0  # Starting price

        # Simulate price movement
        change_percent = random.uniform(-0.02, 0.02)  # -2% to +2%
        close_price = base_price * (1 + change_percent)

        # Generate OHLC
        high = max(close_price, base_price * (1 + random.uniform(0, 0.01)))
        low = min(close_price, base_price * (1 - random.uniform(0, 0.01)))
        open_price = base_price

        volume = random.randint(1000, 10000)

        return Candle(
            timestamp=datetime.utcnow(),
            open=open_price,
            high=high,
            low=low,
            close=close_price,
            volume=volume
        )

    def _format_sse_message(self, data: dict[str, Any]) -> str:
        """Format data as SSE message"""
        return f"data: {json.dumps(data)}\n\n"

    def get_stats(self) -> dict[str, Any]:
        """Get SSE handler statistics"""
        return {
            'connections': self.connection_manager.get_connection_stats(),
            'streaming_symbols': list(self.market_data_buffer.keys()),
            'is_streaming': self.is_streaming,
            'broadcast_task_running': self.broadcast_task is not None and not self.broadcast_task.done()
        }


# Global instance
sse_handler = SSEHandler()


async def handle_sse_connection(request: Request, client_id: str,
                              subscriptions: list[str]) -> EventSourceResponse:
    """FastAPI endpoint handler for SSE connections"""
    # Convert string subscriptions to enum
    sub_types = []
    for sub in subscriptions:
        try:
            sub_types.append(SubscriptionType(sub))
        except ValueError:
            continue  # Skip invalid subscription types

    async def event_generator():
        async for message in sse_handler.handle_connection(request, client_id, sub_types):
            yield message

    return EventSourceResponse(event_generator())
