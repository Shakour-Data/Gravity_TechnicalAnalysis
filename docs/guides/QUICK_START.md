# راه‌اندازی سریع (Quick Start)

این راهنما کوتاه‌ترین مسیر برای بالا آوردن سرویس و گرفتن اولین خروجی را نشان می‌دهد. پیش‌نیازها: Python 3.12، pip، و در صورت نیاز Redis (اختیاری برای کش).

## ۱) نصب و پیکربندی
1. وابستگی‌ها را نصب کنید:
   ```bash
   pip install -r requirements.txt
   ```
2. فایل محیطی را بسازید و در صورت نیاز مقادیر را تنظیم کنید:
   ```bash
   copy .env.example .env
   ```
   کلیدهای مهم:
   - `CACHE_ENABLED=true` و `REDIS_HOST/REDIS_PORT` برای کش (اختیاری؛ بدون Redis هم اجرا می‌شود).
   - `ENABLE_DATA_INGESTION=true` اگر می‌خواهید نتایج تحلیل در پایگاه داده ذخیره شوند.
   - `DATA_SERVICE_URL` اگر از سرویس داده خارجی استفاده می‌کنید.

## ۲) اجرای سرویس FastAPI
از ریشه مخزن اجرا کنید:
```bash
set PYTHONPATH=src
uvicorn gravity_tech.main:app --host 0.0.0.0 --port 8000 --reload
```
- مستندات تعاملی: `http://localhost:8000/api/docs`
- سلامت: `http://localhost:8000/health`
- متریک‌ها (در صورت فعال بودن): `/metrics`
- برای فعال بودن endpointهای ML/Pattern، فایل مدل‌ها باید در `ml_models/pattern_classifier_*.pkl` موجود باشد؛ در غیر این صورت پاسخ «model missing» دریافت می‌کنید.

## ۳) اولین درخواست تحلیل
کمینه‌ی داده ورودی: ۶۰ کندل. مثال فراخوانی REST:
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "candles": [
      {"timestamp": "2024-01-01T00:00:00Z", "open": 43000, "high": 43500, "low": 42800, "close": 43250, "volume": 120000},
      {"timestamp": "2024-01-01T01:00:00Z", "open": 43250, "high": 43800, "low": 43100, "close": 43720, "volume": 98000}
      // ... حداقل ۵۰ رکورد
    ]
  }'
```
خروجی شامل اندیکاتورهای شش‌گانه، الگوهای شمعی، امواج الیوت، فاز بازار، و سیگنال کلی است.

## ۴) استفاده برنامه‌نویسی (Python)
```python
from gravity_tech.core.contracts.analysis import AnalysisRequest
from gravity_tech.core.domain.entities import Candle
from gravity_tech.services.analysis_service import TechnicalAnalysisService
import asyncio

candles = [Candle(
    symbol="BTCUSDT",
    timeframe="1h",
    timestamp="2024-01-01T00:00:00Z",
    open=43000, high=43500, low=42800, close=43250, volume=120000
) for _ in range(60)]

async def main():
    req = AnalysisRequest(symbol="BTCUSDT", timeframe="1h", candles=candles)
    result = await TechnicalAnalysisService.analyze(req)
    print(result.overall_signal, result.overall_confidence)

asyncio.run(main())
```

## ۵) نکات داده
- حداقل کندل: ۶۰ (برای برخی اندیکاتورها به همین مقدار نیاز است).
- ورودی‌ها باید **adjusted** باشند اگر از سرویس داده خارجی می‌آیند.
- برای endpoint `analyze/historical/{symbol}` باید دیتابیس محلی TSE یا منبع داده معادل از پیش بارگذاری شده باشد (مثلاً با `scripts/populate_last90.py`). در غیاب Data Service می‌توانید داده را مستقیماً در بدنه درخواست `/analyze` ارسال کنید.
- اگر Redis یا Data Service در دسترس نباشد، تحلیل بدون کش/داده خارجی اجرا می‌شود.

## ۶) مسیرهای بعدی
- معماری و جریان درخواست: `docs/architecture/SYSTEM_ARCHITECTURE_DIAGRAMS.md`
- راهنمای API و فهرست endpointها: `docs/guides/API_REFERENCE.md`
- تصمیم‌گیر ۵ بعدی و ماتریس حجم: `docs/guides/FIVE_DIMENSIONAL_DECISION_GUIDE.md`
