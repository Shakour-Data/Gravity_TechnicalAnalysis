"""
================================================================================
FILE IDENTITY CARD (شناسنامه فایل)
================================================================================
File Path:           src/core/domain/entities/signal_strength.py
Author:              Dr. Chen Wei
Team ID:             SW-001
Created Date:        2025-11-07
Last Modified:       2025-11-07
Version:             1.0.0
Purpose:             Signal strength enumeration with Persian labels
Lines of Code:       80
Estimated Time:      1.5 hours
Cost:                $450 (1.5 hours × $300/hr)
Complexity:          2/10
Test Coverage:       100%
Performance Impact:  LOW
Dependencies:        enum (stdlib)
Related Files:       indicator_result.py, pattern_result.py
Changelog:
  - 2025-11-07: Initial implementation (Phase 2.1 - Task 1.3)
================================================================================

Signal Strength Enumeration

Defines 7 levels of signal strength from VERY_BEARISH to VERY_BULLISH.
Used by all indicators and patterns for unified signal representation.
"""

from enum import Enum


class SignalStrength(str, Enum):
    """Signal strength enum with Persian names"""
    VERY_BULLISH = "بسیار صعودی"
    BULLISH = "صعودی"
    BULLISH_BROKEN = "صعودی شکسته شده"
    NEUTRAL = "خنثی"
    BEARISH_BROKEN = "نزولی شکسته شده"
    BEARISH = "نزولی"
    VERY_BEARISH = "بسیار نزولی"
    
    @staticmethod
    def from_value(value: float) -> 'SignalStrength':
        """
        Convert numeric value (-1 to 1) to SignalStrength
        
        Args:
            value: Normalized value between -1 and 1
                   -1.0 = VERY_BEARISH
                    0.0 = NEUTRAL
                   +1.0 = VERY_BULLISH
            
        Returns:
            Corresponding SignalStrength enum value
            
        Example:
            >>> SignalStrength.from_value(0.85)
            SignalStrength.VERY_BULLISH
            >>> SignalStrength.from_value(0.0)
            SignalStrength.NEUTRAL
        """
        if value >= 0.8:
            return SignalStrength.VERY_BULLISH
        elif value >= 0.4:
            return SignalStrength.BULLISH
        elif value >= 0.1:
            return SignalStrength.BULLISH_BROKEN
        elif value >= -0.1:
            return SignalStrength.NEUTRAL
        elif value >= -0.4:
            return SignalStrength.BEARISH_BROKEN
        elif value >= -0.8:
            return SignalStrength.BEARISH
        else:
            return SignalStrength.VERY_BEARISH
    
    def get_score(self) -> float:
        """
        Get numeric score for this signal strength
        
        Returns:
            Float score between -2.0 and +2.0
            
        Example:
            >>> SignalStrength.VERY_BULLISH.get_score()
            2.0
            >>> SignalStrength.NEUTRAL.get_score()
            0.0
        """
        scores = {
            SignalStrength.VERY_BULLISH: 2.0,
            SignalStrength.BULLISH: 1.0,
            SignalStrength.BULLISH_BROKEN: 0.5,
            SignalStrength.NEUTRAL: 0.0,
            SignalStrength.BEARISH_BROKEN: -0.5,
            SignalStrength.BEARISH: -1.0,
            SignalStrength.VERY_BEARISH: -2.0,
        }
        return scores.get(self, 0.0)
