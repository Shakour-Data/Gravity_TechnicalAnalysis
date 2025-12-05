"""
Fibonacci Analysis Result Entity

Clean Architecture - Domain Layer
Immutable entity representing complete Fibonacci analysis results.

Author: Gravity Tech Team
Date: December 4, 2025
Version: 1.0.0
License: MIT
"""

from dataclasses import dataclass
from typing import Any

from .signal_strength import SignalStrength


@dataclass(frozen=True)
class FibonacciResult:
    """
    Immutable entity representing complete Fibonacci analysis results.

    Attributes:
        retracement_levels: Dictionary of retracement levels {ratio: price}
        extension_levels: Dictionary of extension levels {ratio: price}
        confluence_zones: List of confluence zone dictionaries
        signal: Trading signal based on Fibonacci analysis
        confidence: Confidence level (0.0 to 1.0)
        description: Human-readable analysis description
    """
    retracement_levels: dict[str, float]
    extension_levels: dict[str, float]
    confluence_zones: list[dict[str, Any]]
    signal: SignalStrength
    confidence: float
    description: str

    def __post_init__(self):
        """Validate entity invariants"""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        # Validate retracement levels contain valid ratios
        valid_ratios = ['0.236', '0.382', '0.5', '0.618', '0.786']
        for ratio in self.retracement_levels.keys():
            if ratio not in valid_ratios:
                raise ValueError(f"Invalid retracement ratio: {ratio}")

        # Validate extension levels contain valid ratios
        valid_extensions = ['0.618', '1.0', '1.272', '1.382', '1.618', '2.0', '2.618']
        for ratio in self.extension_levels.keys():
            if ratio not in valid_extensions:
                raise ValueError(f"Invalid extension ratio: {ratio}")

    def get_nearest_level(self, current_price: float) -> dict[str, Any] | None:
        """
        Find the nearest Fibonacci level to current price.

        Args:
            current_price: Current market price

        Returns:
            Dictionary with level info or None if no levels found
        """
        all_levels = {**self.retracement_levels, **self.extension_levels}

        if not all_levels:
            return None

        nearest_ratio = min(all_levels.keys(),
                          key=lambda r: abs(all_levels[r] - current_price))

        return {
            'ratio': nearest_ratio,
            'price': all_levels[nearest_ratio],
            'distance': abs(all_levels[nearest_ratio] - current_price),
            'distance_percent': abs(all_levels[nearest_ratio] - current_price) / current_price
        }

    def get_confluence_zones(self, tolerance: float = 0.01) -> list[dict[str, Any]]:
        """
        Get confluence zones where multiple levels cluster.

        Args:
            tolerance: Price tolerance for grouping levels

        Returns:
            List of confluence zones
        """
        all_levels = []
        for ratio, price in self.retracement_levels.items():
            all_levels.append({'ratio': ratio, 'price': price, 'type': 'retracement'})
        for ratio, price in self.extension_levels.items():
            all_levels.append({'ratio': ratio, 'price': price, 'type': 'extension'})

        # Sort by price
        all_levels.sort(key=lambda x: x['price'])

        zones = []
        current_zone = []

        for level in all_levels:
            if not current_zone:
                current_zone.append(level)
            else:
                # Check if level is within tolerance of zone average
                zone_avg = sum(lvl['price'] for lvl in current_zone) / len(current_zone)
                if abs(level['price'] - zone_avg) <= (zone_avg * tolerance):
                    current_zone.append(level)
                else:
                    # Save current zone if it has multiple levels
                    if len(current_zone) > 1:
                        zones.append({
                            'price': zone_avg,
                            'levels': current_zone,
                            'strength': len(current_zone)
                        })
                    current_zone = [level]

        # Don't forget the last zone
        if len(current_zone) > 1:
            zone_avg = sum(lvl['price'] for lvl in current_zone) / len(current_zone)
            zones.append({
                'price': zone_avg,
                'levels': current_zone,
                'strength': len(current_zone)
            })

        return zones

    def get_signal_description(self) -> str:
        """Get human-readable signal description"""
        signal_map = {
            SignalStrength.VERY_BULLISH: "Very Bullish - Strong Fibonacci support",
            SignalStrength.BULLISH: "Bullish - Fibonacci support level",
            SignalStrength.NEUTRAL: "Neutral - At Fibonacci level",
            SignalStrength.BEARISH: "Bearish - Fibonacci resistance level",
            SignalStrength.VERY_BEARISH: "Very Bearish - Strong Fibonacci resistance"
        }
        return signal_map.get(self.signal, "Unknown signal")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'retracement_levels': self.retracement_levels,
            'extension_levels': self.extension_levels,
            'confluence_zones': self.confluence_zones,
            'signal': self.signal.value if hasattr(self.signal, 'value') else str(self.signal),
            'confidence': self.confidence,
            'description': self.description
        }
