# راهنمای سریع - میکروسرویس تحلیل تکنیکال

## ⚠️ نکته بسیار مهم

**تمام قیمت‌ها و حجم‌های ورودی باید تعدیل شده (Adjusted) باشند!**

این میکروسرویس فرض می‌کند داده‌های ورودی برای تقسیم سهام، سود سهام، و افزایش سرمایه تعدیل شده‌اند.

استفاده از داده‌های تعدیل نشده منجر به نتایج غلط می‌شود!

مثال صحیح:
```json
{
  "candles": [
    {
      "timestamp": "2024-01-01T00:00:00",
      "open": 1250.5,      // تعدیل شده ✓
      "high": 1280.0,      // تعدیل شده ✓
      "low": 1240.0,       // تعدیل شده ✓
      "close": 1275.0,     // تعدیل شده ✓
      "volume": 2500000    // تعدیل شده ✓
    }
  ]
}
```

---

## نصب سریع

```bash
# 1. نصب dependencies
pip install -r requirements.txt

# 2. کپی فایل تنظیمات
copy .env.example .env

# 3. اجرای سرویس
python main.py
```

سرویس روی `http://localhost:8000` اجرا می‌شود.

## مستندات API

بعد از اجرا به این آدرس‌ها بروید:
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

## تست سریع

### 1. اجرای مثال

```bash
python example.py
```

### 2. درخواست API با curl

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "candles": [
      {
        "timestamp": "2024-01-01T00:00:00",
        "open": 42000,
        "high": 42500,
        "low": 41800,
        "close": 42300,
        "volume": 1000
      }
    ]
  }'
```

### 3. تست با Python

```python
import requests

url = "http://localhost:8000/api/v1/analyze"
data = {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "candles": [...]  # لیست candles شما
}

response = requests.post(url, json=data)
result = response.json()

print(f"Signal: {result['overall_signal']}")
print(f"Confidence: {result['overall_confidence']}")
```

## ساختار پروژه

```
Gravity_TechAnalysis/
├── main.py                    # نقطه شروع برنامه
├── config.py                  # تنظیمات
├── requirements.txt           # Dependencies
├── models/
│   └── schemas.py            # مدل‌های داده
├── indicators/
│   ├── trend.py              # اندیکاتورهای روند
│   ├── momentum.py           # اندیکاتورهای مومنتوم
│   ├── volume.py             # اندیکاتورهای حجم
│   ├── volatility.py         # اندیکاتورهای نوسان
│   └── support_resistance.py # حمایت/مقاومت
├── patterns/
│   └── candlestick.py        # الگوهای کندل استیک
├── services/
│   └── analysis_service.py   # سرویس اصلی
└── api/v1/
    └── __init__.py           # API endpoints
```

## اندیکاتورهای موجود

### روند (Trend)
- SMA, EMA, WMA, DEMA, TEMA
- MACD
- ADX

### مومنتوم (Momentum)
- RSI
- Stochastic
- CCI
- ROC
- Williams %R
- MFI
- Ultimate Oscillator

### سیکل (Cycle)
- Sine Wave
- Detrended Price Oscillator (DPO)
- Schaff Trend Cycle (STC)
- Dominant Cycle Period
- Market Facilitation Index
- Cycle Phase Index
- Trend vs Cycle Identifier

### حجم (Volume)
- OBV
- CMF
- VWAP
- A/D Line
- PVT
- Volume Oscillator

### نوسان (Volatility)
- Bollinger Bands
- ATR
- Keltner Channel
- Donchian Channel
- Standard Deviation
- Historical Volatility

### حمایت/مقاومت
- Pivot Points
- Fibonacci Retracement
- Camarilla Pivots
- Support/Resistance Levels

### الگوهای کندل استیک
- Doji
- Hammer
- Inverted Hammer
- Bullish/Bearish Engulfing
- Morning/Evening Star
- Bullish/Bearish Harami
- Three White Soldiers/Three Black Crows

### امواج الیوت
- **الگوهای 5 موجی (Impulsive)**: شناسایی موج‌های 1-2-3-4-5
- **الگوهای 3 موجی (Corrective)**: شناسایی موج‌های A-B-C
- **قوانین الیوت**:
  - موج 2 نمی‌تواند بیش از 100% موج 1 اصلاح کند
  - موج 3 نمی‌تواند کوتاه‌ترین موج باشد
  - موج 4 نمی‌تواند با موج 1 همپوشانی داشته باشد
- **پیش‌بینی با فیبوناچی**: اهداف موج‌های بعدی

### تحلیل فاز بازار (نظریه داو)
- **فاز انباشت**: بازار در محدوده، حجم کم، سرمایه‌گذاران آگاه خرید می‌کنند
- **فاز صعود (Markup)**: روند صعودی قوی، Higher Highs و Higher Lows
- **فاز توزیع**: بازار در محدوده، حجم بالا، سرمایه‌گذاران آگاه می‌فروشند
- **فاز نزول (Markdown)**: روند نزولی قوی، Lower Highs و Lower Lows
- **تحلیل ساختار روند**: شناسایی نقاط بالا و پایین
- **تایید حجم**: آیا حجم روند را تایید می‌کند؟
- **توصیه‌های معاملاتی**: بر اساس فاز فعلی بازار

**همه تحلیل‌ها با نظریه داو سازگار است**

## Docker

```bash
# Build
docker build -t technical-analysis .

# Run
docker run -p 8000:8000 technical-analysis

# با docker-compose
docker-compose up -d
```

## تست

```bash
# اجرای تست‌ها
pytest

# با coverage
pytest --cov=. --cov-report=html
```

## سیگنال‌های خروجی

هر اندیکاتور یکی از این سیگنال‌ها را بر می‌گرداند:

- **بسیار صعودی** (VERY_BULLISH): سیگنال خرید قوی
- **صعودی** (BULLISH): سیگنال خرید
- **صعودی شکسته شده** (BULLISH_BROKEN): روند صعودی ضعیف شده
- **خنثی** (NEUTRAL): بدون سیگنال مشخص
- **نزولی شکسته شده** (BEARISH_BROKEN): روند نزولی ضعیف شده
- **نزولی** (BEARISH): سیگنال فروش
- **بسیار نزولی** (VERY_BEARISH): سیگنال فروش قوی

## پشتیبانی

برای سوالات و مشکلات:
- Issues در GitHub
- تماس با تیم GravityTech

## مجوز

MIT License
