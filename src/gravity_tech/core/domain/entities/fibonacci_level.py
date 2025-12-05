"""
Fibonacci Level Entity

Clean Architecture - Domain Layer
Immutable entity representing a single Fibonacci level.

Author: Gravity Tech Team
Date: December 4, 2025
Version: 1.0.0
License: MIT
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FibonacciLevel:
    """
    Immutable entity representing a single Fibonacci level.

    Attributes:
        price: Price level of the Fibonacci retracement/extension
        ratio: Fibonacci ratio (e.g., 0.618, 1.618)
        level_type: Type of level (WEAK, MEDIUM, STRONG, VERY_STRONG)
        strength: Calculated strength (0.0 to 1.0)
        touches: Number of times price touched this level
        description: Human-readable description
    """
    price: float
    ratio: float
    level_type: str  # "WEAK", "MEDIUM", "STRONG", "VERY_STRONG"
    strength: float  # 0.0 to 1.0
    touches: int
    description: str

    def __post_init__(self):
        """Validate entity invariants"""
        if not (0.0 <= self.strength <= 1.0):
            raise ValueError(f"Strength must be between 0.0 and 1.0, got {self.strength}")

        if self.touches < 0:
            raise ValueError(f"Touches must be non-negative, got {self.touches}")

        valid_types = ["WEAK", "MEDIUM", "STRONG", "VERY_STRONG"]
        if self.level_type not in valid_types:
            raise ValueError(f"Level type must be one of {valid_types}, got {self.level_type}")

    def is_support(self, current_price: float, tolerance: float = 0.005) -> bool:
        """
        Check if this level acts as support.

        Args:
            current_price: Current market price
            tolerance: Price tolerance as percentage

        Returns:
            True if level is support
        """
        tolerance_amount = self.price * tolerance
        return abs(current_price - self.price) <= tolerance_amount and current_price >= self.price

    def is_resistance(self, current_price: float, tolerance: float = 0.005) -> bool:
        """
        Check if this level acts as resistance.

        Args:
            current_price: Current market price
            tolerance: Price tolerance as percentage

        Returns:
            True if level is resistance
        """
        tolerance_amount = self.price * tolerance
        return abs(current_price - self.price) <= tolerance_amount and current_price <= self.price

    def get_strength_description(self) -> str:
        """Get human-readable strength description"""
        if self.strength >= 0.8:
            return "Very Strong"
        elif self.strength >= 0.6:
            return "Strong"
        elif self.strength >= 0.4:
            return "Medium"
        elif self.strength >= 0.2:
            return "Weak"
        else:
            return "Very Weak"
