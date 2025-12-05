"""
Core data models for technical analysis

⚠️ DEPRECATION WARNING ⚠️
This module is DEPRECATED as of Phase 2.1 (November 7, 2025).
All models have been migrated to: src.core.domain.entities

Please update your imports:
OLD: from models.schemas import Candle, SignalStrength, IndicatorResult
NEW: from gravity_tech.core.domain.entities import Candle, CoreSignalStrength, IndicatorResult

This backward compatibility layer will be removed in Phase 2.2 (Day 3).
"""

import warnings

from gravity_tech.core.domain.entities import (
    Candle,
    ElliottWaveResult,
    IndicatorCategory,
    IndicatorResult,
    PatternResult,
    PatternType,
    WavePoint,
)
from gravity_tech.core.domain.entities import (
    CoreSignalStrength as SignalStrength,
)

# Issue deprecation warning
warnings.warn(
    "Importing from models.schemas is deprecated. "
    "Use src.core.domain.entities instead. "
    "This module will be removed in Phase 2.2.",
    DeprecationWarning,
    stacklevel=2,
)

# ============================================================================
# BACKWARD COMPATIBILITY LAYER (Phase 2.1)
# ============================================================================

# Re-export for backward compatibility
__all__ = [
    "Candle",
    "SignalStrength",
    "IndicatorCategory",
    "IndicatorResult",
    "PatternType",
    "PatternResult",
    "WavePoint",
    "ElliottWaveResult",
]
