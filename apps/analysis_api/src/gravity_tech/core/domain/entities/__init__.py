"""
================================================================================
Core Domain Entities Package

Clean Architecture - Domain Layer
All business entities are immutable and framework-independent.

Exported Entities:
- Candle: OHLCV price data (Phase 2)
- Signal: Trading signal entity (Phase 2)
- Decision: Trading decision entity (Phase 2)
- SignalStrength: Signal strength enumeration - 7 levels (Phase 2.1)
- IndicatorCategory: Indicator category enumeration - 6 categories (Phase 2.1)
- IndicatorResult: Single indicator calculation result (Phase 2.1)
- PatternType: Pattern type enumeration - 2 types (Phase 2.1)
- PatternResult: Pattern recognition result (Phase 2.1)
- WavePoint: Elliott Wave point (Phase 2.1)
- ElliottWaveResult: Complete Elliott Wave analysis (Phase 2.1)

Last Updated: 2025-11-07 (Phase 2.1 - Task 1.3)
================================================================================
"""

from .candle import Candle, CandleType
from .decision import ConfidenceLevel, Decision, DecisionType
from .elliott_wave_result import ElliottWaveResult
from .fibonacci_level import FibonacciLevel
from .fibonacci_result import FibonacciResult
from .indicator_category import IndicatorCategory
from .indicator_result import IndicatorResult
from .lstm_result import LSTMResult
from .market_data import MarketData
from .pattern_result import PatternResult
from .pattern_type import PatternType
from .prediction_result import PredictionResult, PredictionSignal
from .signal import Signal, SignalType
from .signal_strength import SignalStrength as CoreSignalStrength
from .sse_message import SSEMessage
from .subscription_type import SubscriptionType
from .transformer_result import TransformerResult
from .wave_point import WavePoint
from .websocket_message import WebSocketMessage

# Note: CoreSignalStrength (7-level) from signal_strength.py is the new standard
# OldSignalStrength (3-level) from signal.py is kept for legacy Signal class

# Alias for backward compatibility
SignalStrength = CoreSignalStrength

__all__ = [
    # Existing (Phase 2)
    "Candle",
    "CandleType",
    "Signal",
    "SignalType",
    "Decision",
    "DecisionType",
    "ConfidenceLevel",

    # NEW (Phase 2.1)
    "CoreSignalStrength",  # From signal_strength.py (new, with Persian labels, 7 levels)
    "SignalStrength",      # Alias for backward compatibility
    "IndicatorCategory",
    "IndicatorResult",
    "PatternType",
    "PatternResult",
    "WavePoint",
    "ElliottWaveResult",
    "FibonacciLevel",
    "FibonacciResult",

    # NEW (Phase 2.1 - Task 1.4 - Real-time entities)
    "SubscriptionType",
    "WebSocketMessage",
    "SSEMessage",
    "MarketData",

    # NEW (Phase 2.1 - Task 1.5 - ML entities)
    "LSTMResult",
    "PredictionResult",
    "PredictionSignal",
    "TransformerResult",
]
