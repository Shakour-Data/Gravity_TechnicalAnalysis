"""
Display Formatters for API Responses

This module provides functions to convert scores from internal range to display range.

Ranges:
Internal:
  - Score: [-1.0, +1.0]
  - Confidence: [0.0, 1.0]
Display:
  - Score: [-100, +100]
  - Confidence: [0, 100]

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT

Usage:
------
```python
from gravity_tech.utils.display_formatters import score_to_display, confidence_to_display

# Internal score
internal_score = 0.75
display_score = score_to_display(internal_score)  # → 75

# Internal confidence
internal_confidence = 0.85
display_confidence = confidence_to_display(internal_confidence)  # → 85
```
"""

# All Union types have been updated to PEP 604 syntax (X | Y)
from datetime import datetime


def score_to_display(score: float) -> int:
    """
    Convert score from range [-1, +1] to [-100, +100].

    Args:
        score: Internal score between -1 and +1

    Returns:
        Display score between -100 and +100 (integer)

    Examples:
        >>> score_to_display(1.0)
        100
        >>> score_to_display(0.75)
        75
        >>> score_to_display(0.0)
        0
        >>> score_to_display(-0.5)
        -50
        >>> score_to_display(-1.0)
        -100
    """
    # Limit to [-1, +1]
    score = max(-1.0, min(1.0, score))

    # Convert to [-100, +100]
    display_score = score * 100

    # Round to integer
    return int(round(display_score))
def confidence_to_display(confidence: float) -> int:
    """
    Convert confidence from range [0, 1] to [0, 100].

    Args:
        confidence: Internal confidence between 0 and 1

    Returns:
        Display confidence between 0 and 100 (integer)

    Examples:
        >>> confidence_to_display(1.0)
        100
        >>> confidence_to_display(0.85)
        85
        >>> confidence_to_display(0.5)
        50
        >>> confidence_to_display(0.0)
        0
    """
    # Limit to [0, 1]
    confidence = max(0.0, min(1.0, confidence))

    # Convert to [0, 100]
    display_confidence = confidence * 100

    # Round to integer
    return int(round(display_confidence))
def display_to_score(display_score: int | float) -> float:
    """
    Convert display score from [-100, +100] to [-1, +1].

    This function is for when we need reverse conversion
    (e.g., receiving input from user).

    Args:
        display_score: Display score between -100 and +100

    Returns:
        Internal score between -1 and +1

    Examples:
        >>> display_to_score(100)
        1.0
        >>> display_to_score(75)
        0.75
        >>> display_to_score(0)
        0.0
        >>> display_to_score(-50)
        -0.5
    """
    # Limit to [-100, +100]
    display_score = max(-100, min(100, display_score))

    # Convert to [-1, +1]
    return display_score / 100.0
def display_to_confidence(display_confidence: int | float) -> float:
    """
    Convert display confidence from [0, 100] to [0, 1].

    Args:
        display_confidence: Display confidence between 0 and 100

    Returns:
        Internal confidence between 0 and 1

    Examples:
        >>> display_to_confidence(100)
        1.0
        >>> display_to_confidence(85)
        0.85
        >>> display_to_confidence(0)
        0.0
    """
    # Limit to [0, 100]
    display_confidence = max(0, min(100, display_confidence))

    # Convert to [0, 1]
    return display_confidence / 100.0
def get_signal_label(score: float, use_persian: bool = False) -> str:
    """
    Convert score to signal label.

    Args:
        score: Score between -1 and +1
        use_persian: Use Persian labels

    Returns:
        Signal label

    Examples:
        >>> get_signal_label(0.95)
        'VERY_BULLISH'
        >>> get_signal_label(0.95, use_persian=True)
        'بسیار صعودی'
    """
    if use_persian:
        if score >= 0.8:
            return "بسیار صعودی"
        elif score >= 0.4:
            return "صعودی"
        elif score >= 0.1:
            return "صعودی ضعیف"
        elif score >= -0.1:
            return "خنثی"
        elif score >= -0.4:
            return "نزولی ضعیف"
        elif score >= -0.8:
            return "نزولی"
        else:
            return "بسیار نزولی"
    else:
        if score >= 0.8:
            return "VERY_BULLISH"
        elif score >= 0.4:
            return "BULLISH"
        elif score >= 0.1:
            return "WEAK_BULLISH"
        elif score >= -0.1:
            return "NEUTRAL"
        elif score >= -0.4:
            return "WEAK_BEARISH"
        elif score >= -0.8:
            return "BEARISH"
        else:
            return "VERY_BEARISH"
def get_confidence_label(confidence: float, use_persian: bool = False) -> str:
    """
    Convert confidence to quality label.

    Args:
        confidence: Confidence between 0 and 1
        use_persian: Use Persian labels

    Returns:
        Quality label

    Examples:
        >>> get_confidence_label(0.95)
        'EXCELLENT'
        >>> get_confidence_label(0.95, use_persian=True)
        'عالی'
    """
    if use_persian:
        if confidence >= 0.9:
            return "عالی"
        elif confidence >= 0.8:
            return "خوب"
        elif confidence >= 0.7:
            return "متوسط به بالا"
        elif confidence >= 0.6:
            return "متوسط"
        elif confidence >= 0.5:
            return "ضعیف"
        else:
            return "بسیار ضعیف"
    else:
        if confidence >= 0.9:
            return "EXCELLENT"
        elif confidence >= 0.8:
            return "HIGH"
        elif confidence >= 0.7:
            return "GOOD"
        elif confidence >= 0.6:
            return "MEDIUM"
        elif confidence >= 0.5:
            return "LOW"
        else:
            return "VERY_LOW"
# ═══════════════════════════════════════════════════════════════════
# Additional Formatting Functions
# ═══════════════════════════════════════════════════════════════════

def format_price(price: float | int, currency: str = "$", decimals: int = 2) -> str:
    """
    Format price with currency and proper decimal places.

    Args:
        price: Price value
        currency: Currency symbol (default: $)
        decimals: Decimal places (default: 2)

    Returns:
        Formatted price string

    Examples:
        >>> format_price(1234.56)
        '$1,234.56'
        >>> format_price(1000000, decimals=0)
        '$1,000,000'
    """
    try:
        # Format with commas for thousands
        formatted = f"{price:,.{decimals}f}"
        return f"{currency}{formatted}"
    except (ValueError, TypeError):
        return f"{currency}{price}"


def format_percentage(value: float, decimals: int = 2, include_symbol: bool = True) -> str:
    """
    Format percentage value.

    Args:
        value: Percentage value (0.1234 = 12.34%)
        decimals: Decimal places
        include_symbol: Include % symbol

    Returns:
        Formatted percentage string

    Examples:
        >>> format_percentage(0.1234)
        '12.34%'
        >>> format_percentage(0.5, decimals=1)
        '50.0%'
    """
    try:
        percentage = value * 100
        formatted = f"{percentage:.{decimals}f}"
        return f"{formatted}%" if include_symbol else formatted
    except (ValueError, TypeError):
        return f"{value}%" if include_symbol else str(value)


def format_volume(volume: int | float, abbreviate: bool = True) -> str:
    """
    Format volume with abbreviations for large numbers.

    Args:
        volume: Volume value
        abbreviate: Use K, M, B abbreviations

    Returns:
        Formatted volume string

    Examples:
        >>> format_volume(1000000)
        '1.00M'
        >>> format_volume(1500)
        '1.50K'
    """
    try:
        if not abbreviate:
            return f"{volume:,.0f}"

        if volume >= 1_000_000_000:
            return f"{volume / 1_000_000_000:.2f}B"
        elif volume >= 1_000_000:
            return f"{volume / 1_000_000:.2f}M"
        elif volume >= 1_000:
            return f"{volume / 1_000:.2f}K"
        else:
            return f"{volume:.0f}"
    except (ValueError, TypeError):
        return str(volume)


def format_timestamp(timestamp: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime timestamp.

    Args:
        timestamp: Datetime object
        format_str: Format string (default: YYYY-MM-DD HH:MM:SS)

    Returns:
        Formatted timestamp string

    Examples:
        >>> from datetime import datetime
        >>> dt = datetime(2025, 12, 5, 14, 30, 45)
        >>> format_timestamp(dt)
        '2025-12-05 14:30:45'
    """
    try:
        return timestamp.strftime(format_str)
    except (AttributeError, ValueError):
        return str(timestamp)


# ═══════════════════════════════════════════════════════════════════
# Usage Examples
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Display Formatters - Test Examples")
    print("=" * 60)
    # Test score conversion
    print("\n📊 Score Conversion [-1, +1] → [-100, +100]:")
    print("-" * 60)
    test_scores = [1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0]
    for score in test_scores:
        display = score_to_display(score)
        label_en = get_signal_label(score)
        label_fa = get_signal_label(score, use_persian=True)
        print(f"  {score:+5.2f} → {display:+4d}  [{label_en:15s}]  [{label_fa}]")
    # Test confidence conversion
    print("\n🎯 Confidence Conversion [0, 1] → [0, 100]:")
    print("-" * 60)
    test_confidences = [1.0, 0.95, 0.85, 0.75, 0.65, 0.55, 0.45]
    for conf in test_confidences:
        display = confidence_to_display(conf)
        label_en = get_confidence_label(conf)
        label_fa = get_confidence_label(conf, use_persian=True)
        print(f"  {conf:4.2f} → {display:3d}%  [{label_en:10s}]  [{label_fa}]")
    # Test reverse conversion
    print("\n🔄 Reverse Conversion:")
    print("-" * 60)
    print(f"  100 → {display_to_score(100):.2f}")
    print(f"   75 → {display_to_score(75):.2f}")
    print(f"    0 → {display_to_score(0):.2f}")
    print(f"  -50 → {display_to_score(-50):.2f}")
    print(f" -100 → {display_to_score(-100):.2f}")
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)
