"""
================================================================================
Subscription Type Enumeration

Clean Architecture - Domain Layer
Defines subscription types for real-time data streaming.

Last Updated: 2025-11-07 (Phase 2.1 - Task 1.4)
================================================================================
"""

from enum import Enum


class SubscriptionType(Enum):
    """
    Subscription types for real-time data streaming.

    Used by WebSocket and SSE handlers to manage client subscriptions
    for different types of market data and analysis updates.
    """

    # Market Data Subscriptions
    MARKET_DATA = "market_data"
    """Real-time market data (price, volume, etc.)"""

    CANDLE_DATA = "candle_data"
    """Real-time candle/OHLCV data"""

    # Analysis Subscriptions
    TECHNICAL_ANALYSIS = "technical_analysis"
    """Real-time technical analysis results"""

    PATTERN_RECOGNITION = "pattern_recognition"
    """Real-time pattern recognition updates"""

    ELLIOTT_WAVE = "elliott_wave"
    """Real-time Elliott Wave analysis updates"""

    FIBONACCI_LEVELS = "fibonacci_levels"
    """Real-time Fibonacci level calculations"""

    # Indicator Subscriptions
    TREND_INDICATORS = "trend_indicators"
    """Real-time trend indicator updates"""

    MOMENTUM_INDICATORS = "momentum_indicators"
    """Real-time momentum indicator updates"""

    VOLATILITY_INDICATORS = "volatility_indicators"
    """Real-time volatility indicator updates"""

    VOLUME_INDICATORS = "volume_indicators"
    """Real-time volume indicator updates"""

    # ML Subscriptions
    ML_PREDICTIONS = "ml_predictions"
    """Real-time ML model predictions"""

    # System Subscriptions
    SYSTEM_STATUS = "system_status"
    """System status and health updates"""

    ALERTS = "alerts"
    """Trading alerts and notifications"""

    def __str__(self) -> str:
        """String representation of subscription type."""
        return self.value

    @classmethod
    def from_string(cls, value: str) -> 'SubscriptionType':
        """
        Create SubscriptionType from string value.

        Args:
            value: String representation of subscription type

        Returns:
            Corresponding SubscriptionType enum value

        Raises:
            ValueError: If value is not a valid subscription type
        """
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Invalid subscription type: {value}")

    @property
    def is_market_data(self) -> bool:
        """Check if this is a market data subscription."""
        return self in {
            SubscriptionType.MARKET_DATA,
            SubscriptionType.CANDLE_DATA
        }

    @property
    def is_analysis(self) -> bool:
        """Check if this is an analysis subscription."""
        return self in {
            SubscriptionType.TECHNICAL_ANALYSIS,
            SubscriptionType.PATTERN_RECOGNITION,
            SubscriptionType.ELLIOTT_WAVE,
            SubscriptionType.FIBONACCI_LEVELS
        }

    @property
    def is_indicator(self) -> bool:
        """Check if this is an indicator subscription."""
        return self in {
            SubscriptionType.TREND_INDICATORS,
            SubscriptionType.MOMENTUM_INDICATORS,
            SubscriptionType.VOLATILITY_INDICATORS,
            SubscriptionType.VOLUME_INDICATORS
        }

    @property
    def is_ml(self) -> bool:
        """Check if this is an ML subscription."""
        return self == SubscriptionType.ML_PREDICTIONS

    @property
    def is_system(self) -> bool:
        """Check if this is a system subscription."""
        return self in {
            SubscriptionType.SYSTEM_STATUS,
            SubscriptionType.ALERTS
        }
