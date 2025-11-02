"""
مثال استفاده از API Response Formatters
=========================================

این فایل نمایش می‌دهد چگونه از formatters برای تولید خروجی API استفاده کنیم.

خروجی API با امتیازهای [-100, +100] و اعتماد [0, 100]
"""

import json
from ml.multi_horizon_analysis import HorizonScore
from models.schemas import SignalStrength
from api.response_formatters import (
    format_trend_response,
    format_momentum_response,
    format_combined_response
)


def example_trend_analysis():
    """مثال تحلیل روند"""
    print("=" * 80)
    print("📊 مثال 1: تحلیل روند (TREND ANALYSIS)")
    print("=" * 80)
    
    # فرض کنید این نتایج از MultiHorizonTrendAnalyzer آمده
    trend_results = [
        HorizonScore(
            horizon=3,
            score=0.85,      # داخلی: 0.85
            confidence=0.82,  # داخلی: 0.82
            signal=SignalStrength.VERY_BULLISH
        ),
        HorizonScore(
            horizon=7,
            score=0.75,
            confidence=0.78,
            signal=SignalStrength.BULLISH
        ),
        HorizonScore(
            horizon=30,
            score=0.60,
            confidence=0.75,
            signal=SignalStrength.BULLISH
        )
    ]
    
    # فرمت کردن برای API (انگلیسی)
    api_response_en = format_trend_response(
        trend_results,
        use_persian=False,
        include_raw=True  # برای debugging
    )
    
    print("\n🔹 API Response (English):")
    print(json.dumps(api_response_en, indent=2, ensure_ascii=False))
    
    # فرمت کردن برای API (فارسی)
    api_response_fa = format_trend_response(
        trend_results,
        use_persian=True,
        include_raw=False
    )
    
    print("\n🔹 API Response (Persian):")
    print(json.dumps(api_response_fa, indent=2, ensure_ascii=False))
    
    return api_response_en


def example_momentum_analysis():
    """مثال تحلیل مومنتوم"""
    print("\n\n" + "=" * 80)
    print("📈 مثال 2: تحلیل مومنتوم (MOMENTUM ANALYSIS)")
    print("=" * 80)
    
    # فرض کنید این نتایج از MultiHorizonMomentumAnalyzer آمده
    momentum_results = [
        HorizonScore(
            horizon=3,
            score=-0.20,     # داخلی: -0.20 (نزولی ضعیف)
            confidence=0.70,
            signal=SignalStrength.BEARISH_BROKEN
        ),
        HorizonScore(
            horizon=7,
            score=0.30,      # داخلی: 0.30 (صعودی ضعیف)
            confidence=0.72,
            signal=SignalStrength.BULLISH_BROKEN
        ),
        HorizonScore(
            horizon=30,
            score=0.55,      # داخلی: 0.55 (صعودی)
            confidence=0.68,
            signal=SignalStrength.BULLISH
        )
    ]
    
    # فرمت کردن برای API
    api_response = format_momentum_response(
        momentum_results,
        use_persian=False,
        include_raw=True
    )
    
    print("\n🔹 API Response:")
    print(json.dumps(api_response, indent=2, ensure_ascii=False))
    
    return api_response


def example_combined_analysis():
    """مثال تحلیل ترکیبی"""
    print("\n\n" + "=" * 80)
    print("🔄 مثال 3: تحلیل ترکیبی (COMBINED ANALYSIS)")
    print("=" * 80)
    
    # نتایج روند
    trend_results = [
        HorizonScore(horizon=3, score=0.85, confidence=0.82, signal=SignalStrength.VERY_BULLISH),
        HorizonScore(horizon=7, score=0.75, confidence=0.78, signal=SignalStrength.BULLISH),
        HorizonScore(horizon=30, score=0.60, confidence=0.75, signal=SignalStrength.BULLISH)
    ]
    
    # نتایج مومنتوم
    momentum_results = [
        HorizonScore(horizon=3, score=-0.20, confidence=0.70, signal=SignalStrength.BEARISH_BROKEN),
        HorizonScore(horizon=7, score=0.30, confidence=0.72, signal=SignalStrength.BULLISH_BROKEN),
        HorizonScore(horizon=30, score=0.55, confidence=0.68, signal=SignalStrength.BULLISH)
    ]
    
    # فرمت کردن با وزن‌های مختلف
    print("\n🔹 Scenario 1: وزن یکسان (50-50):")
    api_response_1 = format_combined_response(
        trend_results,
        momentum_results,
        trend_weight=0.5,
        momentum_weight=0.5,
        use_persian=False
    )
    print(json.dumps(api_response_1["combined"], indent=2, ensure_ascii=False))
    
    print("\n🔹 Scenario 2: تاکید بر روند (60-40):")
    api_response_2 = format_combined_response(
        trend_results,
        momentum_results,
        trend_weight=0.6,
        momentum_weight=0.4,
        use_persian=False
    )
    print(json.dumps(api_response_2["combined"], indent=2, ensure_ascii=False))
    
    print("\n🔹 Scenario 3: تاکید بر مومنتوم (40-60):")
    api_response_3 = format_combined_response(
        trend_results,
        momentum_results,
        trend_weight=0.4,
        momentum_weight=0.6,
        use_persian=False
    )
    print(json.dumps(api_response_3["combined"], indent=2, ensure_ascii=False))
    
    return api_response_1


def example_microservice_usage():
    """مثال استفاده در میکروسرویس"""
    print("\n\n" + "=" * 80)
    print("🌐 مثال 4: استفاده در Microservice")
    print("=" * 80)
    
    print("""
نمونه کد Flask/FastAPI:

```python
from flask import Flask, jsonify
from api.response_formatters import format_combined_response

app = Flask(__name__)

@app.route('/api/v1/analysis/<symbol>')
def get_analysis(symbol):
    # 1. دریافت داده‌های بازار
    candles = fetch_market_data(symbol)
    
    # 2. استخراج ویژگی‌ها
    trend_features = trend_extractor.extract(candles)
    momentum_features = momentum_extractor.extract(candles)
    
    # 3. تحلیل (امتیازها داخلی: [-1, +1])
    trend_result = trend_analyzer.analyze(trend_features)
    momentum_result = momentum_analyzer.analyze(momentum_features)
    
    # 4. فرمت کردن برای API (تبدیل به [-100, +100])
    response = format_combined_response(
        trend_result,
        momentum_result,
        trend_weight=0.6,
        momentum_weight=0.4,
        use_persian=False
    )
    
    # 5. برگرداندن JSON
    return jsonify(response)
```

خروجی JSON:
```json
{
  "analysis_type": "COMBINED",
  "trend": {
    "analysis_type": "TREND",
    "horizons": {
      "3d": {"score": 85, "confidence": 82, "signal": "VERY_BULLISH"},
      "7d": {"score": 75, "confidence": 78, "signal": "BULLISH"},
      "30d": {"score": 60, "confidence": 75, "signal": "BULLISH"}
    },
    "overall": {
      "score": 73,
      "confidence": 78,
      "signal": "BULLISH",
      "recommendation": "BUY"
    }
  },
  "momentum": {
    "analysis_type": "MOMENTUM",
    "horizons": {
      "3d": {"score": -20, "confidence": 70, "signal": "WEAK_BEARISH"},
      "7d": {"score": 30, "confidence": 72, "signal": "WEAK_BULLISH"},
      "30d": {"score": 55, "confidence": 68, "signal": "BULLISH"}
    },
    "overall": {
      "score": 22,
      "confidence": 70,
      "signal": "WEAK_BULLISH",
      "recommendation": "PREPARE"
    }
  },
  "combined": {
    "score": 50,
    "confidence": 74,
    "signal": "BULLISH",
    "action": "ACCUMULATE",
    "weights": {"trend": 0.6, "momentum": 0.4}
  }
}
```

نکات مهم:
----------
✅ داخلی همه محاسبات با [-1, +1] انجام می‌شود
✅ فقط در خروجی API به [-100, +100] تبدیل می‌شود
✅ ML models همچنان [-1, +1] استفاده می‌کنند
✅ کاربر عددهای صحیح قابل فهم می‌بیند (85, -20, ...)
✅ توابع include_raw=True برای debugging مقادیر اصلی را هم برمی‌گردانند
    """)


def compare_score_ranges():
    """مقایسه محدوده‌های داخلی و نمایشی"""
    print("\n\n" + "=" * 80)
    print("📊 مثال 5: مقایسه محدوده‌های داخلی vs نمایشی")
    print("=" * 80)
    
    from utils.display_formatters import score_to_display, confidence_to_display
    
    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║              تبدیل امتیازها: [-1, +1] → [-100, +100]          ║")
    print("╠════════════════════════════════════════════════════════════════╣")
    print("║  داخلی (ML)  │  نمایشی (API)  │         توضیح                 ║")
    print("╠══════════════╪═════════════════╪══════════════════════════════╣")
    
    test_cases = [
        (1.0, "حداکثر صعودی"),
        (0.85, "بسیار صعودی"),
        (0.75, "صعودی"),
        (0.5, "صعودی متوسط"),
        (0.25, "صعودی ضعیف"),
        (0.0, "خنثی"),
        (-0.25, "نزولی ضعیف"),
        (-0.5, "نزولی متوسط"),
        (-0.75, "نزولی"),
        (-0.85, "بسیار نزولی"),
        (-1.0, "حداکثر نزولی")
    ]
    
    for internal, description in test_cases:
        display = score_to_display(internal)
        print(f"║   {internal:+6.2f}     │      {display:+4d}       │  {description:30s} ║")
    
    print("╚════════════════════════════════════════════════════════════════╝")
    
    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║              تبدیل اعتماد: [0, 1] → [0, 100]                  ║")
    print("╠════════════════════════════════════════════════════════════════╣")
    print("║  داخلی (ML)  │  نمایشی (API)  │         کیفیت                ║")
    print("╠══════════════╪═════════════════╪══════════════════════════════╣")
    
    conf_cases = [
        (1.0, "عالی (Excellent)"),
        (0.95, "عالی"),
        (0.85, "خوب (High)"),
        (0.75, "متوسط به بالا (Good)"),
        (0.65, "متوسط (Medium)"),
        (0.55, "ضعیف (Low)"),
        (0.45, "بسیار ضعیف (Very Low)"),
        (0.0, "بدون اعتماد")
    ]
    
    for internal, quality in conf_cases:
        display = confidence_to_display(internal)
        print(f"║    {internal:4.2f}      │       {display:3d}        │  {quality:30s} ║")
    
    print("╚════════════════════════════════════════════════════════════════╝")


def main():
    """اجرای همه مثال‌ها"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "API Response Formatters - Examples" + " " * 24 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # اجرای مثال‌ها
    example_trend_analysis()
    example_momentum_analysis()
    example_combined_analysis()
    example_microservice_usage()
    compare_score_ranges()
    
    print("\n\n" + "=" * 80)
    print("✅ همه مثال‌ها با موفقیت اجرا شدند!")
    print("=" * 80)
    print("""
خلاصه:
-------
✅ داخلی: همه محاسبات با [-1, +1] و [0, 1]
✅ API: خروجی با [-100, +100] و [0, 100]
✅ ML Models: تغییری نمی‌کنند، همان [-1, +1]
✅ کاربر: عددهای صحیح قابل فهم می‌بیند

فایل‌های ایجاد شده:
--------------------
1. utils/display_formatters.py - توابع تبدیل
2. api/response_formatters.py - فرمت کننده‌های API
3. example_api_response.py - این فایل (مثال‌ها)

استفاده در میکروسرویس:
-------------------------
from api.response_formatters import format_combined_response

response = format_combined_response(
    trend_result,
    momentum_result,
    use_persian=False
)
return jsonify(response)
    """)


if __name__ == "__main__":
    main()
