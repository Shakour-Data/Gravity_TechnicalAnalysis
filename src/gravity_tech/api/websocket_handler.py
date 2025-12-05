"""
WebSocket Handler for Real-time Data Streaming

This module implements WebSocket connections for:
- Real-time price updates
- Live indicator calculations
- Streaming analysis results
- Market data broadcasting

Author: Gravity Tech Team
Date: December 4, 2025
Version: 1.0.0
License: MIT
"""

import asyncio
import logging
import random
from datetime import datetime
from decimal import Decimal
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from gravity_tech.core.domain.entities import (
    Candle,
    MarketData,
    SubscriptionType,
    WebSocketMessage,
)
from gravity_tech.core.indicators.momentum import MomentumIndicators
from gravity_tech.core.indicators.trend import TrendIndicators

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and subscriptions"""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.subscriptions: dict[str, set[SubscriptionType]] = {}
        self.client_data: dict[str, dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = set()
        self.client_data[client_id] = {
            'connected_at': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'message_count': 0
        }
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, client_id: str) -> None:
        """Remove a WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        if client_id in self.client_data:
            del self.client_data[client_id]
        logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")

    async def subscribe(self, client_id: str, subscription_type: SubscriptionType) -> None:
        """Subscribe client to a data stream"""
        if client_id not in self.subscriptions:
            self.subscriptions[client_id] = set()
        self.subscriptions[client_id].add(subscription_type)
        logger.info(f"Client {client_id} subscribed to {subscription_type}")

    async def unsubscribe(self, client_id: str, subscription_type: SubscriptionType) -> None:
        """Unsubscribe client from a data stream"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].discard(subscription_type)
            logger.info(f"Client {client_id} unsubscribed from {subscription_type}")

    async def broadcast_to_subscribers(self, subscription_type: SubscriptionType,
                                     message: dict[str, Any]) -> None:
        """Broadcast message to all subscribers of a type"""
        disconnected_clients = []

        for client_id, subscriptions in self.subscriptions.items():
            if subscription_type in subscriptions:
                try:
                    await self.send_personal_message(client_id, message)
                except Exception as e:
                    logger.error(f"Failed to send message to {client_id}: {e}")
                    disconnected_clients.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)

    async def send_personal_message(self, client_id: str, message: dict[str, Any]) -> None:
        """Send message to specific client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_json(message)
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


class WebSocketHandler:
    """Handles WebSocket connections and real-time data streaming"""

    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.trend_indicators = TrendIndicators()
        self.momentum_indicators = MomentumIndicators()
        self.market_data_buffer: dict[str, list[Candle]] = {}
        self.is_streaming = False

    async def handle_connection(self, websocket: WebSocket, client_id: str) -> None:
        """Handle individual WebSocket connection"""
        await self.connection_manager.connect(websocket, client_id)

        try:
            while True:
                # Receive message from client
                data = await websocket.receive_json()
                message = WebSocketMessage(**data)

                # Process message based on type
                if message.message_type == "subscribe":
                    if message.subscription_type:
                        await self.connection_manager.subscribe(client_id, message.subscription_type)

                    # Send confirmation
                    await self.connection_manager.send_personal_message(
                        client_id,
                        {
                            "type": "subscription_confirmed",
                            "subscription_type": message.subscription_type.value if message.subscription_type else None,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )

                elif message.message_type == "unsubscribe":
                    if message.subscription_type:
                        await self.connection_manager.unsubscribe(client_id, message.subscription_type)

                    # Send confirmation
                    await self.connection_manager.send_personal_message(
                        client_id,
                        {
                            "type": "unsubscription_confirmed",
                            "subscription_type": message.subscription_type.value if message.subscription_type else None,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )

                elif message.message_type == "ping":
                    await self.connection_manager.send_personal_message(
                        client_id,
                        {
                            "type": "pong",
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )

        except WebSocketDisconnect:
            self.connection_manager.disconnect(client_id)
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
            self.connection_manager.disconnect(client_id)

    async def start_market_data_stream(self, symbol: str, interval: str = "1m") -> None:
        """Start streaming market data for a symbol"""
        self.is_streaming = True

        # Initialize buffer for symbol
        if symbol not in self.market_data_buffer:
            self.market_data_buffer[symbol] = []

        logger.info(f"Started market data stream for {symbol}")

        # In a real implementation, this would connect to a market data feed
        # For now, we'll simulate streaming
        while self.is_streaming:
            try:
                # Simulate receiving new candle data
                # In production, this would come from your data provider
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
                logger.error(f"Error in market data stream for {symbol}: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def stop_market_data_stream(self, symbol: str) -> None:
        """Stop streaming market data for a symbol"""
        self.is_streaming = False
        logger.info(f"Stopped market data stream for {symbol}")

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
            market_data = MarketData.from_candle_data(
                symbol=symbol,
                timestamp=candle.timestamp,
                open_price=Decimal(str(candle.open)),
                high_price=Decimal(str(candle.high)),
                low_price=Decimal(str(candle.low)),
                close_price=Decimal(str(candle.close)),
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
                # RSI
                try:
                    rsi = self.momentum_indicators.rsi(candles, 14)
                    indicators['rsi'] = {
                        'name': rsi.indicator_name,
                        'value': rsi.value,
                        'signal': rsi.signal.value,
                        'confidence': rsi.confidence
                    }
                except Exception:
                    pass

                # Moving averages
                try:
                    sma_20 = self.trend_indicators.sma(candles, 20)
                    sma_50 = self.trend_indicators.sma(candles, 50)
                    indicators['sma'] = {
                        'sma_20': sma_20.value if sma_20 else None,
                        'sma_50': sma_50.value if sma_50 else None
                    }
                except Exception:
                    pass

            return indicators

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}

    async def _detect_patterns(self, candles: list[Candle]) -> list[dict[str, Any]]:
        """Detect chart patterns for broadcasting"""
        try:
            patterns = []

            # Detect basic patterns
            if len(candles) >= 2:
                # Check for bullish engulfing
                try:
                    last_two = candles[-2:]
                    if (last_two[0].close < last_two[0].open and  # First candle is bearish
                        last_two[1].close > last_two[1].open and  # Second candle is bullish
                        last_two[1].low <= last_two[0].low and    # Engulfs low
                        last_two[1].high >= last_two[0].high):    # Engulfs high
                        patterns.append({
                            'name': 'Bullish Engulfing',
                            'type': 'reversal',
                            'direction': 'bullish',
                            'confidence': 0.7
                        })
                except Exception:
                    pass

                # Check for bearish engulfing
                try:
                    last_two = candles[-2:]
                    if (last_two[0].close > last_two[0].open and  # First candle is bullish
                        last_two[1].close < last_two[1].open and  # Second candle is bearish
                        last_two[1].low <= last_two[0].low and    # Engulfs low
                        last_two[1].high >= last_two[0].high):    # Engulfs high
                        patterns.append({
                            'name': 'Bearish Engulfing',
                            'type': 'reversal',
                            'direction': 'bearish',
                            'confidence': 0.7
                        })
                except Exception:
                    pass

            return patterns

        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []

    def _simulate_new_candle(self, symbol: str) -> Candle:
        """Simulate new candle data (for testing)"""
        # This is a placeholder - in production, you'd get real market data

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

    def get_stats(self) -> dict[str, Any]:
        """Get WebSocket handler statistics"""
        return {
            'connections': self.connection_manager.get_connection_stats(),
            'streaming_symbols': list(self.market_data_buffer.keys()),
            'is_streaming': self.is_streaming
        }


# Global instance
websocket_handler = WebSocketHandler()


async def handle_websocket_connection(websocket: WebSocket, client_id: str):
    """FastAPI endpoint handler for WebSocket connections"""
    await websocket_handler.handle_connection(websocket, client_id)
