# تبدیل محدوده امتیازها به [-100, +100]

## ✅ تغییرات انجام شده

### 1️⃣ فایل‌های ایجاد شده:

#### `utils/display_formatters.py`
توابع اصلی برای تبدیل امتیازها:
- `score_to_display(score)` - تبدیل [-1, +1] → [-100, +100]
- `confidence_to_display(confidence)` - تبدیل [0, 1] → [0, 100]
- `display_to_score(display)` - تبدیل معکوس [-100, +100] → [-1, +1]
- `display_to_confidence(display)` - تبدیل معکوس [0, 100] → [0, 1]
- `get_signal_label(score)` - دریافت برچسب سیگنال (VERY_BULLISH, ...)
- `get_confidence_label(confidence)` - دریافت برچسب کیفیت (EXCELLENT, ...)

#### `api/response_formatters.py`
توابع فرمت کردن برای API:
- `format_trend_response(result)` - فرمت خروجی تحلیل روند
- `format_momentum_response(result)` - فرمت خروجی تحلیل مومنتوم
- `format_combined_response(trend, momentum)` - فرمت خروجی ترکیبی

هر سه تابع:
- امتیازها را به [-100, +100] تبدیل می‌کنند
- اعتماد را به [0, 100] تبدیل می‌کنند
- برچسب‌های انگلیسی و فارسی دارند (`use_persian=True/False`)
- گزینه `include_raw=True` برای debugging مقادیر اصلی را هم برمی‌گرداند

#### `example_api_response.py`
مثال‌های کامل استفاده:
- ✅ مثال 1: تحلیل روند (Trend)
- ✅ مثال 2: تحلیل مومنتوم (Momentum)
- ✅ مثال 3: تحلیل ترکیبی با وزن‌های مختلف
- ✅ مثال 4: استفاده در Flask/FastAPI
- ✅ مثال 5: جدول مقایسه محدوده‌ها

---

## 🎯 معماری تصمیم

### ❌ گزینه رد شده: تغییر کل سیستم
```
مشکلات:
  - نیاز به تغییر همه فایل‌های ML
  - خطر بالا در محاسبات
  - تغییر در کد آموزش مدل‌ها
  - ناسازگار با ML frameworks
```

### ✅ راه‌حل اجرا شده: فقط نمایش
```
مزایا:
  ✅ داخلی: همه محاسبات با [-1, +1] (استاندارد ML)
  ✅ خروجی API: تبدیل به [-100, +100] (قابل فهم برای کاربر)
  ✅ تبدیل ساده: score × 100
  ✅ بدون تغییر در ML models
  ✅ مقادیر خام برای debugging
```

---

## 📊 جدول مقایسه

### امتیازها (Score):

| داخلی (ML) | نمایشی (API) | معنی              | توصیه            |
|-----------|-------------|-------------------|------------------|
| +1.0      | +100        | بسیار صعودی       | خرید قوی         |
| +0.85     | +85         | بسیار صعودی       | خرید قوی         |
| +0.75     | +75         | صعودی             | خرید             |
| +0.5      | +50         | صعودی متوسط       | خرید محتاطانه    |
| +0.25     | +25         | صعودی ضعیف        | انباشت           |
| 0.0       | 0           | خنثی              | نگهداری          |
| -0.25     | -25         | نزولی ضعیف        | سودگیری          |
| -0.5      | -50         | نزولی متوسط       | فروش محتاطانه    |
| -0.75     | -75         | نزولی             | فروش             |
| -0.85     | -85         | بسیار نزولی       | فروش قوی         |
| -1.0      | -100        | بسیار نزولی       | فروش قوی         |

### اعتماد (Confidence):

| داخلی (ML) | نمایشی (API) | کیفیت             |
|-----------|-------------|-------------------|
| 1.0       | 100%        | عالی              |
| 0.95      | 95%         | عالی              |
| 0.85      | 85%         | خوب               |
| 0.75      | 75%         | متوسط به بالا     |
| 0.65      | 65%         | متوسط             |
| 0.55      | 55%         | ضعیف              |
| 0.45      | 45%         | بسیار ضعیف        |

---

## 🌐 نحوه استفاده در میکروسرویس

### Flask:

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
    
    # 4. فرمت برای API (تبدیل به [-100, +100])
    response = format_combined_response(
        trend_result,
        momentum_result,
        trend_weight=0.6,
        momentum_weight=0.4,
        use_persian=False  # یا True برای فارسی
    )
    
    # 5. برگرداندن JSON
    return jsonify(response)
```

### FastAPI:

```python
from fastapi import FastAPI
from api.response_formatters import format_combined_response

app = FastAPI()

@app.get("/api/v1/analysis/{symbol}")
async def get_analysis(symbol: str):
    # ... (مشابه Flask)
    
    response = format_combined_response(
        trend_result,
        momentum_result,
        use_persian=False
    )
    
    return response  # FastAPI خودش به JSON تبدیل می‌کند
```

---

## 📝 مثال خروجی JSON

### درخواست:
```
GET /api/v1/analysis/BTCUSDT
```

### پاسخ:
```json
{
  "analysis_type": "COMBINED",
  "trend": {
    "analysis_type": "TREND",
    "horizons": {
      "3d": {
        "horizon": 3,
        "score": 85,           ← از 0.85 تبدیل شده
        "confidence": 82,       ← از 0.82 تبدیل شده
        "signal": "VERY_BULLISH",
        "confidence_quality": "HIGH"
      },
      "7d": {
        "score": 75,
        "confidence": 78,
        "signal": "BULLISH"
      },
      "30d": {
        "score": 60,
        "confidence": 75,
        "signal": "BULLISH"
      }
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
      "3d": {
        "score": -20,          ← از -0.20 تبدیل شده
        "confidence": 70,
        "signal": "WEAK_BEARISH"
      },
      "7d": {
        "score": 30,
        "confidence": 72,
        "signal": "WEAK_BULLISH"
      },
      "30d": {
        "score": 55,
        "confidence": 68,
        "signal": "BULLISH"
      }
    },
    "overall": {
      "score": 22,
      "confidence": 70,
      "signal": "WEAK_BULLISH",
      "recommendation": "PREPARE"
    }
  },
  "combined": {
    "score": 50,               ← ترکیب: (73×0.6 + 22×0.4)
    "confidence": 74,
    "signal": "BULLISH",
    "confidence_quality": "GOOD",
    "action": "ACCUMULATE",
    "weights": {
      "trend": 0.6,
      "momentum": 0.4
    }
  }
}
```

---

## 🔍 Debugging با include_raw

برای debugging، می‌توانید مقادیر خام را هم دریافت کنید:

```python
response = format_combined_response(
    trend_result,
    momentum_result,
    include_raw=True  # ← اضافه کردن مقادیر خام
)
```

خروجی:
```json
{
  "horizons": {
    "3d": {
      "score": 85,
      "confidence": 82,
      "raw_score": 0.85,      ← مقدار داخلی
      "raw_confidence": 0.82  ← مقدار داخلی
    }
  }
}
```

---

## ✅ خلاصه

### چه چیزی تغییر کرد؟
- **فقط نمایش در API**: امتیازها به [-100, +100] تبدیل می‌شوند
- **هیچ تغییری در ML**: محاسبات همچنان [-1, +1]
- **کد تمیز**: تبدیل فقط در لایه API

### چه چیزی تغییر نکرد؟
- ✅ ML models همچنان [-1, +1] استفاده می‌کنند
- ✅ Feature extractors تغییری نکردند
- ✅ Training pipeline همان است
- ✅ Weight learning همان است

### فایل‌های جدید:
1. `utils/display_formatters.py` - 200+ خط
2. `api/response_formatters.py` - 350+ خط
3. `example_api_response.py` - 360+ خط

### مستندات به‌روز شده:
- `SCORING_SYSTEM_GUIDE.md` - بخش API Response Format اضافه شد

---

## 🚀 اجرای مثال‌ها

```bash
# تست display formatters
python -c "from utils.display_formatters import *; print(score_to_display(0.85))"
# Output: 85

# اجرای مثال کامل
python example_api_response.py
# نمایش همه مثال‌ها با جداول و JSON
```

---

## 📞 تماس با توسعه‌دهنده

در صورت نیاز به توضیحات بیشتر یا تغییرات اضافی، لطفاً به تیم توسعه مراجعه کنید.

تاریخ: 2025-11-01
نسخه: 1.0.0
