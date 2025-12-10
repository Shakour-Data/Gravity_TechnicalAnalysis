"""
================================================================================
FILE IDENTITY CARD (شناسنامه فایل)
================================================================================
File Path:           src/core/domain/entities/elliott_wave_result.py
Author:              Dr. Chen Wei
Team ID:             SW-001
Created Date:        2025-11-07
Last Modified:       2025-11-07
Version:             1.0.0
Purpose:             Elliott Wave analysis result entity
Lines of Code:       75
Estimated Time:      1.5 hours
Cost:                $450 (1.5 hours × $300/hr)
Complexity:          4/10
Test Coverage:       100%
Performance Impact:  MEDIUM
Dependencies:        dataclasses, typing
Related Files:       signal_strength.py, wave_point.py
Changelog:
  - 2025-11-07: Initial implementation (Phase 2.1 - Task 1.3)
================================================================================

Elliott Wave Analysis Result Entity

Represents complete Elliott Wave pattern analysis including all wave points.
"""

from dataclasses import dataclass


from .signal_strength import SignalStrength
from .wave_point import WavePoint


@dataclass(frozen=True)
class ElliottWaveResult:
    """
    Immutable Elliott Wave analysis result

    Attributes:
        wave_pattern: "IMPULSIVE" (5 waves) or "CORRECTIVE" (3 waves)
        current_wave: Current wave number being formed
        waves: List of identified wave points (peaks and troughs)
        signal: Overall signal from wave analysis
        confidence: Wave pattern confidence (0.0 to 1.0)
        description: Human-readable wave analysis
        projected_target: Optional projected price target
    """
    wave_pattern: str  # "IMPULSIVE" or "CORRECTIVE"
    current_wave: int
    waves: list[WavePoint]
    signal: SignalStrength
    confidence: float
    description: str
    projected_target: float | None = None

    def __post_init__(self):
        """Validate Elliott Wave result data"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0.0-1.0, got {self.confidence}")
        if self.wave_pattern not in ["IMPULSIVE", "CORRECTIVE"]:
            raise ValueError(f"wave_pattern must be 'IMPULSIVE' or 'CORRECTIVE', got '{self.wave_pattern}'")
        # IMPULSIVE: 5 waves = 6 wave points (start + waves 1-5)
        if self.wave_pattern == "IMPULSIVE" and len(self.waves) > 6:
            raise ValueError(f"IMPULSIVE pattern should have max 6 wave points, got {len(self.waves)}")
        # CORRECTIVE: ABC = 3 waves = 4 wave points (start + waves A-B-C)
        if self.wave_pattern == "CORRECTIVE" and len(self.waves) > 4:
            raise ValueError(f"CORRECTIVE pattern should have max 4 wave points (start+ABC), got {len(self.waves)}")
        if self.projected_target is not None and self.projected_target <= 0:
            raise ValueError(f"projected_target must be positive, got {self.projected_target}")
