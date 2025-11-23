"""
Test Classical Pattern Recognition

این فایل الگوهای کلاسیک را تست می‌کند
"""

import numpy as np
from datetime import datetime, timedelta
from src.core.domain.entities import Candle
from gravity_tech.patterns.classical import ClassicalPatterns


def create_test_candles(pattern_type: str, num_candles: int = 50) -> list:
    """Create synthetic candles for testing patterns"""
    candles = []
    base_time = datetime.now()
    
    if pattern_type == "head_and_shoulders":
        # Create Head and Shoulders pattern
        # Left Shoulder, Head, Right Shoulder
        prices = []
        
        # Uptrend to left shoulder
        for i in range(10):
            prices.append(100 + i * 2)
        
        # Left shoulder peak (120)
        prices.extend([122, 120, 118])
        
        # Dip to trough (115)
        prices.extend([116, 115, 116])
        
        # Head peak (130)
        prices.extend([118, 122, 126, 130, 128, 126])
        
        # Dip to trough (115)
        prices.extend([124, 120, 117, 115, 116])
        
        # Right shoulder peak (122)
        prices.extend([118, 120, 122, 121, 119])
        
        # Break neckline
        prices.extend([117, 114, 112, 110])
        
        for i, price in enumerate(prices):
            candle = Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price - 0.5,
                high=price + 1,
                low=price - 1,
                close=price + 0.5,
                volume=1000000
            )
            candles.append(candle)
    
    elif pattern_type == "inverse_head_and_shoulders":
        # Create Inverse Head and Shoulders
        prices = []
        
        # Downtrend to left shoulder
        for i in range(10):
            prices.append(150 - i * 2)
        
        # Left shoulder trough (130)
        prices.extend([128, 130, 132])
        
        # Rise to peak (135)
        prices.extend([134, 135, 134])
        
        # Head trough (120)
        prices.extend([132, 128, 124, 120, 122, 124])
        
        # Rise to peak (135)
        prices.extend([126, 130, 133, 135, 134])
        
        # Right shoulder trough (128)
        prices.extend([132, 130, 128, 129, 131])
        
        # Break neckline upward
        prices.extend([133, 136, 138, 140])
        
        for i, price in enumerate(prices):
            candle = Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price + 0.5,
                high=price + 1,
                low=price - 1,
                close=price - 0.5,
                volume=1000000
            )
            candles.append(candle)
    
    elif pattern_type == "double_top":
        # Create Double Top pattern
        prices = []
        
        # Uptrend
        for i in range(10):
            prices.append(100 + i * 2)
        
        # First peak (120)
        prices.extend([120, 122, 120, 118])
        
        # Trough (110)
        prices.extend([116, 114, 112, 110, 111])
        
        # Second peak (121)
        prices.extend([113, 116, 118, 121, 120, 118])
        
        # Break support
        prices.extend([115, 112, 109, 107, 105])
        
        for i, price in enumerate(prices):
            candle = Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price - 0.5,
                high=price + 1,
                low=price - 1,
                close=price + 0.5,
                volume=1000000
            )
            candles.append(candle)
    
    elif pattern_type == "double_bottom":
        # Create Double Bottom pattern
        prices = []
        
        # Downtrend
        for i in range(10):
            prices.append(150 - i * 2)
        
        # First trough (130)
        prices.extend([130, 128, 130, 132])
        
        # Peak (140)
        prices.extend([134, 136, 138, 140, 139])
        
        # Second trough (129)
        prices.extend([137, 134, 132, 129, 130, 132])
        
        # Break resistance
        prices.extend([135, 138, 141, 143, 145])
        
        for i, price in enumerate(prices):
            candle = Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price + 0.5,
                high=price + 1,
                low=price - 1,
                close=price - 0.5,
                volume=1000000
            )
            candles.append(candle)
    
    elif pattern_type == "ascending_triangle":
        # Create Ascending Triangle
        prices = []
        
        # Base
        prices.extend([100, 102, 104, 106, 108])
        
        # Touch resistance at 120, fall back
        prices.extend([110, 114, 118, 120, 119, 117])
        
        # Higher low at 112
        prices.extend([114, 112, 113, 115])
        
        # Touch resistance again at 120
        prices.extend([117, 119, 120, 119, 117])
        
        # Even higher low at 115
        prices.extend([116, 115, 116, 117])
        
        # Break resistance
        prices.extend([118, 120, 122, 124, 126])
        
        for i, price in enumerate(prices):
            candle = Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price - 0.5,
                high=price + 1,
                low=price - 1,
                close=price + 0.5,
                volume=1000000
            )
            candles.append(candle)
    
    elif pattern_type == "descending_triangle":
        # Create Descending Triangle
        prices = []
        
        # Start high
        prices.extend([150, 148, 146, 144, 142])
        
        # Touch support at 130, bounce
        prices.extend([138, 134, 130, 131, 133])
        
        # Lower high at 138
        prices.extend([136, 138, 137, 135])
        
        # Touch support again at 130
        prices.extend([133, 131, 130, 131, 133])
        
        # Even lower high at 135
        prices.extend([134, 135, 134, 132])
        
        # Break support
        prices.extend([131, 130, 128, 126, 124])
        
        for i, price in enumerate(prices):
            candle = Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price + 0.5,
                high=price + 1,
                low=price - 1,
                close=price - 0.5,
                volume=1000000
            )
            candles.append(candle)
    
    elif pattern_type == "symmetrical_triangle":
        # Create Symmetrical Triangle (after uptrend)
        prices = []
        
        # Uptrend
        for i in range(8):
            prices.append(100 + i * 3)
        
        # High at 124
        prices.extend([124, 123, 121])
        
        # Low at 115
        prices.extend([118, 115, 116])
        
        # Lower high at 121
        prices.extend([118, 121, 120])
        
        # Higher low at 117
        prices.extend([119, 117, 118])
        
        # Even lower high at 119
        prices.extend([118.5, 119, 118.5])
        
        # Break upward (continuation)
        prices.extend([119, 120, 122, 124, 126])
        
        for i, price in enumerate(prices):
            candle = Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price - 0.5,
                high=price + 1,
                low=price - 1,
                close=price + 0.5,
                volume=1000000
            )
            candles.append(candle)
    
    return candles


def test_head_and_shoulders():
    """Test Head and Shoulders pattern detection"""
    print("\n" + "="*60)
    print("🔴 تست الگوی سر و شانه (Head and Shoulders)")
    print("="*60)
    
    candles = create_test_candles("head_and_shoulders")
    
    result = ClassicalPatterns.detect_head_and_shoulders(candles)
    
    if result:
        print(f"✅ الگو شناسایی شد!")
        print(f"   نام الگو: {result.pattern_name}")
        print(f"   سیگنال: {result.signal.value}")
        print(f"   اطمینان: {result.confidence:.2%}")
        print(f"   هدف قیمتی: ${result.price_target:.2f}")
        print(f"   استاپ لاس: ${result.stop_loss:.2f}")
        print(f"   توضیحات: {result.description}")
    else:
        print("❌ الگو شناسایی نشد")


def test_inverse_head_and_shoulders():
    """Test Inverse Head and Shoulders pattern detection"""
    print("\n" + "="*60)
    print("🟢 تست الگوی سر و شانه معکوس (Inverse H&S)")
    print("="*60)
    
    candles = create_test_candles("inverse_head_and_shoulders")
    
    result = ClassicalPatterns.detect_inverse_head_and_shoulders(candles)
    
    if result:
        print(f"✅ الگو شناسایی شد!")
        print(f"   نام الگو: {result.pattern_name}")
        print(f"   سیگنال: {result.signal.value}")
        print(f"   اطمینان: {result.confidence:.2%}")
        print(f"   هدف قیمتی: ${result.price_target:.2f}")
        print(f"   استاپ لاس: ${result.stop_loss:.2f}")
        print(f"   توضیحات: {result.description}")
    else:
        print("❌ الگو شناسایی نشد")


def test_double_top():
    """Test Double Top pattern detection"""
    print("\n" + "="*60)
    print("🔴 تست الگوی سقف دوقلو (Double Top)")
    print("="*60)
    
    candles = create_test_candles("double_top")
    
    result = ClassicalPatterns.detect_double_top(candles)
    
    if result:
        print(f"✅ الگو شناسایی شد!")
        print(f"   نام الگو: {result.pattern_name}")
        print(f"   سیگنال: {result.signal.value}")
        print(f"   اطمینان: {result.confidence:.2%}")
        print(f"   هدف قیمتی: ${result.price_target:.2f}")
        print(f"   استاپ لاس: ${result.stop_loss:.2f}")
        print(f"   توضیحات: {result.description}")
    else:
        print("❌ الگو شناسایی نشد")


def test_double_bottom():
    """Test Double Bottom pattern detection"""
    print("\n" + "="*60)
    print("🟢 تست الگوی کف دوقلو (Double Bottom)")
    print("="*60)
    
    candles = create_test_candles("double_bottom")
    
    result = ClassicalPatterns.detect_double_bottom(candles)
    
    if result:
        print(f"✅ الگو شناسایی شد!")
        print(f"   نام الگو: {result.pattern_name}")
        print(f"   سیگنال: {result.signal.value}")
        print(f"   اطمینان: {result.confidence:.2%}")
        print(f"   هدف قیمتی: ${result.price_target:.2f}")
        print(f"   استاپ لاس: ${result.stop_loss:.2f}")
        print(f"   توضیحات: {result.description}")
    else:
        print("❌ الگو شناسایی نشد")


def test_ascending_triangle():
    """Test Ascending Triangle pattern detection"""
    print("\n" + "="*60)
    print("🟢 تست الگوی مثلث صعودی (Ascending Triangle)")
    print("="*60)
    
    candles = create_test_candles("ascending_triangle")
    
    result = ClassicalPatterns.detect_ascending_triangle(candles)
    
    if result:
        print(f"✅ الگو شناسایی شد!")
        print(f"   نام الگو: {result.pattern_name}")
        print(f"   سیگنال: {result.signal.value}")
        print(f"   اطمینان: {result.confidence:.2%}")
        print(f"   هدف قیمتی: ${result.price_target:.2f}")
        print(f"   استاپ لاس: ${result.stop_loss:.2f}")
        print(f"   توضیحات: {result.description}")
    else:
        print("❌ الگو شناسایی نشد")


def test_descending_triangle():
    """Test Descending Triangle pattern detection"""
    print("\n" + "="*60)
    print("🔴 تست الگوی مثلث نزولی (Descending Triangle)")
    print("="*60)
    
    candles = create_test_candles("descending_triangle")
    
    result = ClassicalPatterns.detect_descending_triangle(candles)
    
    if result:
        print(f"✅ الگو شناسایی شد!")
        print(f"   نام الگو: {result.pattern_name}")
        print(f"   سیگنال: {result.signal.value}")
        print(f"   اطمینان: {result.confidence:.2%}")
        print(f"   هدف قیمتی: ${result.price_target:.2f}")
        print(f"   استاپ لاس: ${result.stop_loss:.2f}")
        print(f"   توضیحات: {result.description}")
    else:
        print("❌ الگو شناسایی نشد")


def test_symmetrical_triangle():
    """Test Symmetrical Triangle pattern detection"""
    print("\n" + "="*60)
    print("🟡 تست الگوی مثلث متقارن (Symmetrical Triangle)")
    print("="*60)
    
    candles = create_test_candles("symmetrical_triangle")
    
    result = ClassicalPatterns.detect_symmetrical_triangle(candles)
    
    if result:
        print(f"✅ الگو شناسایی شد!")
        print(f"   نام الگو: {result.pattern_name}")
        print(f"   سیگنال: {result.signal.value}")
        print(f"   اطمینان: {result.confidence:.2%}")
        print(f"   هدف قیمتی: ${result.price_target:.2f}")
        print(f"   استاپ لاس: ${result.stop_loss:.2f}")
        print(f"   توضیحات: {result.description}")
    else:
        print("❌ الگو شناسایی نشد")


def test_all_patterns():
    """Test detection of all patterns at once"""
    print("\n" + "="*60)
    print("🎯 تست شناسایی همه الگوها با یک تابع")
    print("="*60)
    
    # Test with Head and Shoulders data
    candles = create_test_candles("head_and_shoulders")
    patterns = ClassicalPatterns.detect_all(candles)
    
    print(f"\n📊 تعداد الگوهای شناسایی شده: {len(patterns)}")
    for i, pattern in enumerate(patterns, 1):
        print(f"\n{i}. {pattern.pattern_name}")
        print(f"   سیگنال: {pattern.signal.value}")
        print(f"   اطمینان: {pattern.confidence:.2%}")
        print(f"   توضیحات: {pattern.description}")


if __name__ == "__main__":
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "تست الگوهای کلاسیک" + " "*23 + "║")
    print("╚" + "="*58 + "╝")
    
    test_head_and_shoulders()
    test_inverse_head_and_shoulders()
    test_double_top()
    test_double_bottom()
    test_ascending_triangle()
    test_descending_triangle()
    test_symmetrical_triangle()
    test_all_patterns()
    
    print("\n" + "="*60)
    print("✅ همه تست‌ها کامل شد")
    print("="*60)
