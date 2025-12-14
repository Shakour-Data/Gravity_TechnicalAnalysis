"""
Pattern Utilities Module

Utility functions for pattern detection and analysis.

Author: Gravity Tech Team
Date: December 5, 2025
Version: 1.0.0
License: MIT
"""




def calculate_pattern_strength(
    ratio: float,
    completion_percentage: float,
    volume_confirmation: bool
) -> float:
    """
    Calculate pattern strength based on various factors.

    Args:
        ratio: Fibonacci ratio accuracy (0-1)
        completion_percentage: Pattern completion percentage (0-100)
        volume_confirmation: Whether volume confirms the pattern

    Returns:
        Pattern strength score (0-1)
    """
    # Base strength from ratio and completion
    base_strength = (ratio + completion_percentage / 100) / 2

    # Volume confirmation bonus
    volume_bonus = 0.1 if volume_confirmation else 0.0

    return min(base_strength + volume_bonus, 1.0)


def get_reliability_score(pattern_name: str) -> float | None:
    """
    Get reliability score for a pattern type.

    Args:
        pattern_name: Name of the pattern

    Returns:
        Reliability score (0-1) or None if unknown
    """
    reliability_scores = {
        "GARTLEY": 0.85,
        "BUTTERFLY": 0.80,
        "BAT": 0.75,
        "CRAB": 0.70,
        "HEAD_AND_SHOULDERS": 0.90,
        "DOUBLE_TOP": 0.85,
        "DOUBLE_BOTTOM": 0.85,
        "TRIANGLE": 0.75,
        "FLAG": 0.70,
        "WEDGE": 0.65,
        "DOJI": 0.60,
        "HAMMER": 0.75,
        "HANGING_MAN": 0.75,
        "ENGULFING": 0.80,
        "HARAMI": 0.70,
        "MORNING_STAR": 0.85,
        "EVENING_STAR": 0.85,
        "DIVERGENCE": 0.95
    }

    return reliability_scores.get(pattern_name.upper())


def get_pattern_hierarchy(pattern_name: str) -> int | None:
    """
    Get pattern hierarchy level.

    Args:
        pattern_name: Name of the pattern

    Returns:
        Hierarchy level (1-5, lower is more reliable) or None
    """
    hierarchy = {
        "DIVERGENCE": 1,
        "HEAD_AND_SHOULDERS": 1,
        "DOUBLE_TOP": 2,
        "DOUBLE_BOTTOM": 2,
        "MORNING_STAR": 2,
        "EVENING_STAR": 2,
        "ENGULFING": 3,
        "GARTLEY": 3,
        "BUTTERFLY": 3,
        "TRIANGLE": 4,
        "FLAG": 4,
        "WEDGE": 4,
        "HAMMER": 5,
        "DOJI": 5
    }

    return hierarchy.get(pattern_name.upper())


def validate_harmonic_ratios(ratios: list[float]) -> bool | None:
    """
    Validate harmonic ratios against Fibonacci levels.

    Args:
        ratios: List of ratios to validate

    Returns:
        True if valid, False if invalid, None if cannot determine
    """
    if not ratios:
        return None

    fibonacci_levels = [0.382, 0.500, 0.618, 0.786, 0.886, 1.272, 1.414, 1.618, 2.618]

    tolerance = 0.05  # 5% tolerance

    for ratio in ratios:
        # Check if ratio is close to any Fibonacci level
        if not any(abs(ratio - fib) <= tolerance for fib in fibonacci_levels):
            return False

    return True


def detect_pattern_multiframe(candles: list, timeframe: str) -> dict | None:
    """
    Detect patterns across multiple timeframes.

    Args:
        candles: List of candles
        timeframe: Timeframe string

    Returns:
        Pattern detection result or None
    """
    # Placeholder implementation
    return None


def detect_pattern(candles: list) -> dict | None:
    """
    General pattern detection.

    Args:
        candles: List of candles

    Returns:
        Pattern detection result or None
    """
    # Placeholder implementation
    return None


def detect_all_patterns(candles: list) -> list[dict]:
    """
    Detect all patterns in candles.

    Args:
        candles: List of candles

    Returns:
        List of detected patterns
    """
    # Placeholder implementation
    return []


def detect_rare_patterns(candles: list) -> list | None:
    """
    Detect rare harmonic patterns.

    Args:
        candles: List of candles

    Returns:
        List of rare patterns or None
    """
    # Placeholder implementation
    return None
