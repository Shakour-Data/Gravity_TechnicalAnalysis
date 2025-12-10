# ران‌بوک عملیات

این سند چک‌های رایج و رفع خطاهای معمول سرویس تحلیل تکنیکال را پوشش می‌دهد.

## چک‌های سلامت
- **سرویس اصلی**: `GET /health` و `GET /health/live`
- **آمادگی وابستگی‌ها**: `GET /health/ready` (Redis در صورت فعال بودن)
- **الگوها/ML**: `GET /api/v1/patterns/health`, `GET /api/v1/ml/health`
- **کش**: اگر `cache_enabled=false` است، انتظار `redis: disabled` در پاسخ readiness داشته باشید.

## پاک‌سازی کش Redis
```bash
redis-cli -h <host> -p <port> FLUSHDB
```
یا `cache_enabled=false` در `.env` برای اجرای بدون کش.

## خطاهای رایج و رفع
| علامت | علت محتمل | اقدام |
|-------|-----------|-------|
| `503 Analysis failed` | داده ناکافی (کمتر از ۵۰ کندل) یا ورودی نامعتبر | ورودی را بررسی کنید؛ timestamps مرتب و high>=low باشد. |
| `model missing` در `/ml/*` | فایل مدل در `ml_models/` نیست | مدل `pattern_classifier_advanced_v2.pkl` یا `pattern_classifier_v1.pkl` را اضافه کنید. |
| `redis connection refused` در readiness | Redis در دسترس نیست | Redis را بالا بیاورید یا `CACHE_ENABLED=false` و سرویس را بدون کش اجرا کنید. |
| `Failed to analyze historical` | دیتابیس محلی TSE خالی است | ابتدا داده را با اسکریپت‌های `scripts/` بارگذاری کنید یا از مسیر `/analyze` با داده ورودی استفاده کنید. |
| بک‌تست بدون معامله | داده کافی یا الگوی معتبر نیست | پارامتر `window_size`/`min_confidence` را کاهش دهید یا داده بیشتری بدهید. |

## اینجکشن داده (اختیاری)
- اگر `ENABLE_DATA_INGESTION=true` ولی Kafka/RabbitMQ ندارید، نتیجه به صورت مستقیم با `persist_direct` ذخیره می‌شود.
- برای غیرفعال‌کردن کامل ذخیره‌سازی نتایج، مقدار را `false` کنید.

## امن‌سازی و پرچم‌ها
- سناریو سه‌گانه: فقط در صورت نیاز `ENABLE_SCENARIOS=true` شود.
- DB Explorer: فقط در محیط توسعه `EXPOSE_DB_EXPLORER=true` شود؛ در تولید خاموش بماند.
- CORS و Rate-limit را در لایه لبه (Nginx/Traefik) محدود کنید؛ CORS در کد برای همه Origin باز است.

## بررسی سریع پس از تغییر تنظیمات
1. سرویس را ریستارت کنید.
2. `GET /health/ready` را چک کنید.
3. یک درخواست کوچک `/api/v1/analyze` با داده تست ارسال کنید.
4. در صورت فعال بودن ingestion، دیتابیس `historical_scores` را برای ردیف جدید بررسی کنید.

## لاگ‌ها
- لاگ‌ها با `structlog` ساختاری هستند. کلیدهای مهم: `request_started`, `analysis_completed`, `analysis_failed`, `ml_prediction`, `patterns_detected`.
- در خطاها از مقدار `error_type` (اگر وجود دارد) برای تروبِل‌شوتینگ سریع استفاده کنید.
