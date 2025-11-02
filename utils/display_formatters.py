"""
Display Formatters for API Responses
====================================

این ماژول توابعی برای تبدیل امتیازها از محدوده داخلی به محدوده نمایشی فراهم می‌کند.

محدوده‌ها:
-----------
داخلی (Internal):
  - Score: [-1.0, +1.0]
  - Confidence: [0.0, 1.0]

نمایشی برای API (Display):
  - Score: [-100, +100]
  - Confidence: [0, 100]

استفاده:
---------
```python
from utils.display_formatters import score_to_display, confidence_to_display

# امتیاز داخلی
internal_score = 0.75
display_score = score_to_display(internal_score)  # → 75

# اعتماد داخلی
internal_confidence = 0.85
display_confidence = confidence_to_display(internal_confidence)  # → 85
```
"""

from typing import Union


def score_to_display(score: float) -> int:
    """
    تبدیل امتیاز از محدوده [-1, +1] به [-100, +100]
    
    Args:
        score: امتیاز داخلی بین -1 و +1
        
    Returns:
        امتیاز نمایشی بین -100 و +100 (عدد صحیح)
        
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
    # محدود کردن به [-1, +1]
    score = max(-1.0, min(1.0, score))
    
    # تبدیل به [-100, +100]
    display_score = score * 100
    
    # گرد کردن به عدد صحیح
    return int(round(display_score))


def confidence_to_display(confidence: float) -> int:
    """
    تبدیل اعتماد از محدوده [0, 1] به [0, 100]
    
    Args:
        confidence: اعتماد داخلی بین 0 و 1
        
    Returns:
        اعتماد نمایشی بین 0 و 100 (عدد صحیح)
        
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
    # محدود کردن به [0, 1]
    confidence = max(0.0, min(1.0, confidence))
    
    # تبدیل به [0, 100]
    display_confidence = confidence * 100
    
    # گرد کردن به عدد صحیح
    return int(round(display_confidence))


def display_to_score(display_score: Union[int, float]) -> float:
    """
    تبدیل امتیاز نمایشی از [-100, +100] به [-1, +1]
    
    این تابع برای زمانی است که نیاز به تبدیل معکوس داریم
    (مثلاً دریافت ورودی از کاربر)
    
    Args:
        display_score: امتیاز نمایشی بین -100 و +100
        
    Returns:
        امتیاز داخلی بین -1 و +1
        
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
    # محدود کردن به [-100, +100]
    display_score = max(-100, min(100, display_score))
    
    # تبدیل به [-1, +1]
    return display_score / 100.0


def display_to_confidence(display_confidence: Union[int, float]) -> float:
    """
    تبدیل اعتماد نمایشی از [0, 100] به [0, 1]
    
    Args:
        display_confidence: اعتماد نمایشی بین 0 و 100
        
    Returns:
        اعتماد داخلی بین 0 و 1
        
    Examples:
        >>> display_to_confidence(100)
        1.0
        >>> display_to_confidence(85)
        0.85
        >>> display_to_confidence(0)
        0.0
    """
    # محدود کردن به [0, 100]
    display_confidence = max(0, min(100, display_confidence))
    
    # تبدیل به [0, 1]
    return display_confidence / 100.0


def get_signal_label(score: float, use_persian: bool = False) -> str:
    """
    تبدیل امتیاز به برچسب سیگنال
    
    Args:
        score: امتیاز بین -1 و +1
        use_persian: استفاده از برچسب فارسی
        
    Returns:
        برچسب سیگنال
        
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
    تبدیل اعتماد به برچسب کیفیت
    
    Args:
        confidence: اعتماد بین 0 و 1
        use_persian: استفاده از برچسب فارسی
        
    Returns:
        برچسب کیفیت
        
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
# مثال‌های استفاده
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
