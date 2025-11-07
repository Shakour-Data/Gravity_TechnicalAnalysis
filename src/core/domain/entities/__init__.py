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

# Existing exports (Phase 2)
from .candle import Candle, CandleType
from .signal import Signal, SignalType, SignalStrength as OldSignalStrength
from .decision import Decision, DecisionType, ConfidenceLevel

# NEW exports (Phase 2.1 - Task 1.3)
from .signal_strength import SignalStrength as CoreSignalStrength
from .indicator_category import IndicatorCategory
from .indicator_result import IndicatorResult
from .pattern_type import PatternType
from .pattern_result import PatternResult
from .wave_point import WavePoint
from .elliott_wave_result import ElliottWaveResult

# Note: CoreSignalStrength (7-level) from signal_strength.py is the new standard
# OldSignalStrength (3-level) from signal.py is kept for legacy Signal class

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
    "IndicatorCategory",
    "IndicatorResult",
    "PatternType",
    "PatternResult",
    "WavePoint",
    "ElliottWaveResult",
]
