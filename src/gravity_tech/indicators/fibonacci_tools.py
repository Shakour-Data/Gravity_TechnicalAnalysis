"""
Fibonacci Tools and Analysis

This module implements comprehensive Fibonacci analysis tools:
- Fibonacci retracements and extensions
- Fibonacci arcs, fans, and time zones
- Golden ratio calculations
- Fibonacci-based support/resistance levels
- Automatic Fibonacci level detection

Author: Gravity Tech Team
Date: December 4, 2025
Version: 1.0.0
License: MIT
"""

from typing import Any

import numpy as np
from gravity_tech.models.schemas import Candle, FibonacciLevel, FibonacciResult, SignalStrength


class FibonacciTools:
    """Comprehensive Fibonacci analysis tools"""

    FIBONACCI_RATIOS = {
        'retracement': [0.236, 0.382, 0.5, 0.618, 0.786],
        'extension': [0.618, 1.0, 1.272, 1.382, 1.618, 2.0, 2.618],
        'arc': [0.382, 0.5, 0.618, 0.786],
        'fan': [0.382, 0.5, 0.618, 0.786],
        'time_zone': [0.618, 1.0, 1.618, 2.618]
    }

    @staticmethod
    def calculate_retracements(high: float, low: float) -> dict[str, float]:
        """
        Calculate Fibonacci retracement levels

        Args:
            high: Highest price point
            low: Lowest price point

        Returns:
            Dictionary of retracement levels
        """
        diff = high - low
        retracements = {}

        for ratio in FibonacciTools.FIBONACCI_RATIOS['retracement']:
            level = high - (diff * ratio)
            retracements[f"{ratio:.3f}"] = level

        return retracements

    @staticmethod
    def calculate_extensions(high: float, low: float, direction: str = "up") -> dict[str, float]:
        """
        Calculate Fibonacci extension levels

        Args:
            high: Highest price point
            low: Lowest price point
            direction: "up" for bullish, "down" for bearish

        Returns:
            Dictionary of extension levels
        """
        diff = high - low
        extensions = {}

        for ratio in FibonacciTools.FIBONACCI_RATIOS['extension']:
            if direction == "up":
                level = high + (diff * ratio)
            else:
                level = low - (diff * ratio)
            extensions[f"{ratio:.3f}"] = level

        return extensions

    @staticmethod
    def calculate_arcs(center: float, radius: float, angle_range: tuple[float, float] = (0, 180)) -> dict[str, float]:
        """
        Calculate Fibonacci arc levels

        Args:
            center: Center point price
            radius: Arc radius
            angle_range: Angle range in degrees

        Returns:
            Dictionary of arc levels
        """
        arcs = {}

        for ratio in FibonacciTools.FIBONACCI_RATIOS['arc']:
            angle = np.arccos(1 - 2 * ratio)  # Convert ratio to angle
            y = center + radius * np.sin(angle)
            arcs[f"arc_{ratio:.3f}"] = y

        return arcs

    @staticmethod
    def calculate_fan_lines(start_point: tuple[float, float], end_point: tuple[float, float]) -> dict[str, tuple[float, float]]:
        """
        Calculate Fibonacci fan lines

        Args:
            start_point: (x, y) start point
            end_point: (x, y) end point

        Returns:
            Dictionary of fan line equations
        """
        x1, y1 = start_point
        x2, y2 = end_point

        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')

        fans = {}

        for ratio in FibonacciTools.FIBONACCI_RATIOS['fan']:
            if slope != float('inf'):
                fan_slope = slope * ratio
                fan_intercept = y1 - fan_slope * x1
                fans[f"fan_{ratio:.3f}"] = (fan_slope, fan_intercept)
            else:
                # Vertical line case
                fans[f"fan_{ratio:.3f}"] = (float('inf'), x1)

        return fans

    @staticmethod
    def find_fibonacci_levels(candles: list[Candle], lookback: int = 50) -> list[FibonacciLevel]:
        """
        Automatically detect significant Fibonacci levels from price action

        Args:
            candles: List of candles
            lookback: Number of candles to analyze

        Returns:
            List of FibonacciLevel objects
        """
        if len(candles) < lookback:
            return []

        recent_candles = candles[-lookback:]
        highs = [c.high for c in recent_candles]
        lows = [c.low for c in recent_candles]

        max_high = max(highs)
        min_low = min(lows)

        # Calculate retracement levels
        retracements = FibonacciTools.calculate_retracements(max_high, min_low)

        levels = []

        for ratio_str, price in retracements.items():
            # Check how many times price touched this level
            touches = 0
            for candle in recent_candles:
                if abs(candle.high - price) / price < 0.001 or abs(candle.low - price) / price < 0.001:
                    touches += 1

            # Calculate strength based on touches and recency
            strength = min(1.0, touches / 10.0)  # Max strength at 10 touches

            # Determine level type
            ratio = float(ratio_str)
            if ratio == 0.382:
                level_type = "MEDIUM"
            elif ratio == 0.618:
                level_type = "STRONG"
            elif ratio == 0.786:
                level_type = "VERY_STRONG"
            else:
                level_type = "WEAK"

            levels.append(FibonacciLevel(
                price=float(price),
                ratio=float(ratio_str),
                level_type=level_type,
                strength=strength,
                touches=touches,
                description=f"Fibonacci {ratio_str} retracement level"
            ))

        # Sort by strength
        levels.sort(key=lambda x: x.strength, reverse=True)

        return levels

    @staticmethod
    def analyze_fibonacci_confluence(candles: list[Candle], current_price: float) -> FibonacciResult | None:
        """
        Analyze Fibonacci confluence zones

        Args:
            candles: List of candles
            current_price: Current market price

        Returns:
            FibonacciResult with confluence analysis
        """
        if len(candles) < 20:
            return None

        # Find recent swing points
        recent_high = max(c.high for c in candles[-20:])
        recent_low = min(c.low for c in candles[-20:])

        # Calculate multiple Fibonacci levels
        retracements = FibonacciTools.calculate_retracements(recent_high, recent_low)

        # Find confluence zones (where multiple levels cluster)
        all_levels = list(retracements.values())
        all_levels.sort()

        confluence_zones = []
        tolerance = (recent_high - recent_low) * 0.005  # 0.5% tolerance

        i = 0
        while i < len(all_levels) - 1:
            zone_levels = [all_levels[i]]
            j = i + 1
            while j < len(all_levels) and all_levels[j] - all_levels[i] <= tolerance:
                zone_levels.append(all_levels[j])
                j += 1

            if len(zone_levels) >= 2:  # At least 2 levels in confluence
                avg_price = sum(zone_levels) / len(zone_levels)
                confluence_zones.append({
                    'price': avg_price,
                    'levels': len(zone_levels),
                    'strength': len(zone_levels) / 3.0  # Max 3 levels
                })

            i = j

        # Find nearest confluence zone to current price
        if confluence_zones:
            nearest_zone = min(confluence_zones, key=lambda x: abs(x['price'] - current_price))

            # Determine signal based on position relative to zone
            if abs(current_price - nearest_zone['price']) / current_price < 0.01:  # Within 1%
                signal = SignalStrength.NEUTRAL
                description = f"Price at Fibonacci confluence zone ({nearest_zone['levels']} levels)"
            elif current_price > nearest_zone['price']:
                signal = SignalStrength.BULLISH
                description = f"Price above Fibonacci confluence zone ({nearest_zone['levels']} levels)"
            else:
                signal = SignalStrength.BEARISH
                description = f"Price below Fibonacci confluence zone ({nearest_zone['levels']} levels)"

            return FibonacciResult(
                retracement_levels=retracements,
                extension_levels=FibonacciTools.calculate_extensions(recent_high, recent_low),
                confluence_zones=[nearest_zone],
                signal=signal,
                confidence=min(0.8, nearest_zone['strength']),
                description=description
            )

        return None

    @staticmethod
    def golden_ratio_analysis(candles: list[Candle]) -> dict[str, Any]:
        """
        Analyze price action using golden ratio relationships

        Args:
            candles: List of candles

        Returns:
            Dictionary with golden ratio analysis
        """
        if len(candles) < 10:
            return {}

        closes = [c.close for c in candles]
        golden_ratio = 1.618

        # Calculate moving averages with golden ratio periods
        periods = [5, 8, 13, 21, 34, 55]  # Fibonacci sequence

        analysis = {}

        for period in periods:
            if len(closes) >= period:
                ma = sum(closes[-period:]) / period
                analysis[f"MA_{period}"] = ma

                # Check if current price is at golden ratio distance from MA
                current_price = closes[-1]
                golden_distance = ma * golden_ratio

                if abs(current_price - golden_distance) / current_price < 0.02:  # Within 2%
                    analysis[f"golden_ratio_signal_{period}"] = "AT_LEVEL"
                elif current_price > golden_distance:
                    analysis[f"golden_ratio_signal_{period}"] = "ABOVE"
                else:
                    analysis[f"golden_ratio_signal_{period}"] = "BELOW"

        return analysis


def analyze_fibonacci_levels(candles: list[Candle], current_price: float | None = None) -> FibonacciResult | None:
    """
    Convenience function for Fibonacci analysis

    Args:
        candles: List of candles
        current_price: Current price (uses last candle if not provided)

    Returns:
        FibonacciResult with analysis
    """
    if not current_price and candles:
        current_price = candles[-1].close

    if current_price is None:
        return None

    return FibonacciTools.analyze_fibonacci_confluence(candles, current_price)


def get_fibonacci_retracements(high: float, low: float) -> dict[str, float]:
    """
    Get Fibonacci retracement levels

    Args:
        high: High price
        low: Low price

    Returns:
        Dictionary of retracement levels
    """
    return FibonacciTools.calculate_retracements(high, low)


def get_fibonacci_extensions(high: float, low: float, direction: str = "up") -> dict[str, float]:
    """
    Get Fibonacci extension levels

    Args:
        high: High price
        low: Low price
        direction: Direction ("up" or "down")

    Returns:
        Dictionary of extension levels
    """
    return FibonacciTools.calculate_extensions(high, low, direction)
