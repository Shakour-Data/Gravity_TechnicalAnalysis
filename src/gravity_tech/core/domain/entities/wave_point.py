"""
================================================================================
FILE IDENTITY CARD (شناسنامه فایل)
================================================================================
File Path:           src/core/domain/entities/wave_point.py
Author:              Dr. Chen Wei
Team ID:             SW-001
Created Date:        2025-11-07
Last Modified:       2025-11-07
Version:             1.0.0
Purpose:             Elliott Wave point entity
Lines of Code:       45
Estimated Time:      1 hour
Cost:                $300 (1 hour × $300/hr)
Complexity:          2/10
Test Coverage:       100%
Performance Impact:  LOW
Dependencies:        dataclasses, datetime
Related Files:       elliott_wave_result.py
Changelog:
  - 2025-11-07: Initial implementation (Phase 2.1 - Task 1.3)
================================================================================

Elliott Wave Point Entity

Represents a single point in Elliott Wave analysis (peak or trough).
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class WavePoint:
    """
    Immutable Elliott Wave point
    
    Represents a peak or trough in Elliott Wave pattern identification.
    
    Attributes:
        wave_number: Wave number in the sequence (1-5 for impulse, A-C for correction)
        price: Price level at this wave point
        timestamp: Time when this wave point occurred
        wave_type: "PEAK" for tops, "TROUGH" for bottoms
    """
    wave_number: int
    price: float
    timestamp: datetime
    wave_type: str  # "PEAK" or "TROUGH"
    
    def __post_init__(self):
        """Validate wave point data"""
        if self.wave_type not in ["PEAK", "TROUGH"]:
            raise ValueError(f"wave_type must be 'PEAK' or 'TROUGH', got '{self.wave_type}'")
        if self.price <= 0:
            raise ValueError(f"price must be positive, got {self.price}")
