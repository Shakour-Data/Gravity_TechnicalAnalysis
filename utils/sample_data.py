"""
Sample Data Generation Utilities

تابع‌های کمکی برای تولید داده‌های نمونه برای تست و دمو
"""

from datetime import datetime, timedelta
from typing import List
from models.schemas import Candle


def generate_sample_candles(
    num_candles: int = 100,
    base_price: float = 40000,
    trend: str = "sideways"
) -> List[Candle]:
    """
    تولید کندل‌های نمونه برای تست
    
    Args:
        num_candles: تعداد کندل‌های مورد نیاز
        base_price: قیمت پایه
        trend: نوع روند ("uptrend", "downtrend", "sideways")
    
    Returns:
        لیست کندل‌های نمونه
    
    Example:
        >>> candles = generate_sample_candles(100, 40000, "uptrend")
        >>> print(len(candles))
        100
    """
    candles = []
    current_price = base_price
    base_time = datetime.now() - timedelta(hours=num_candles)
    
    # تنظیم جهت روند
    if trend == "uptrend":
        trend_multiplier = 1
    elif trend == "downtrend":
        trend_multiplier = -1
    else:  # sideways
        trend_multiplier = 0
    
    for i in range(num_candles):
        # شبیه‌سازی حرکت قیمت
        trend_movement = trend_multiplier * (i * 50)
        noise = ((i % 20) - 10) * 100
        
        open_price = current_price + trend_movement
        close_price = open_price + noise
        high_price = max(open_price, close_price) + 200
        low_price = min(open_price, close_price) - 200
        
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=float(open_price),
            high=float(high_price),
            low=float(low_price),
            close=float(close_price),
            volume=float(1000 + (i * 50))
        ))
        
        # به‌روزرسانی قیمت فعلی برای کندل بعدی
        current_price = close_price
    
    return candles


def generate_volatile_candles(num_candles: int = 100) -> List[Candle]:
    """
    تولید کندل‌های نمونه با نوسان بالا
    
    Args:
        num_candles: تعداد کندل‌های مورد نیاز
    
    Returns:
        لیست کندل‌های پرنوسان
    """
    import random
    
    candles = []
    base_price = 40000
    base_time = datetime.now() - timedelta(hours=num_candles)
    
    for i in range(num_candles):
        volatility = random.uniform(500, 2000)
        open_price = base_price + random.uniform(-volatility, volatility)
        close_price = open_price + random.uniform(-volatility, volatility)
        high_price = max(open_price, close_price) + random.uniform(0, volatility)
        low_price = min(open_price, close_price) - random.uniform(0, volatility)
        
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=float(open_price),
            high=float(high_price),
            low=float(low_price),
            close=float(close_price),
            volume=float(random.uniform(500, 5000))
        ))
        
        base_price = close_price
    
    return candles
