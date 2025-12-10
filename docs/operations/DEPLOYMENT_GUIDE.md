# راهنمای استقرار

راهنمای کوتاه برای اجرای محلی و استقرار سبک در تولید بر اساس کد فعلی (FastAPI + Uvicorn).

## پیش‌نیازها
- Python 3.12
- اختیاری: Redis برای کش (اگر `CACHE_ENABLED=true`)
- اختیاری: PostgreSQL/SQLite برای ذخیره نتایج (`DATABASE_URL` یا `sqlite_path`)
- اختیاری: Kafka/RabbitMQ اگر ingestion رویداد محور می‌خواهید
- فایل‌های مدل در `ml_models/` برای endpointهای ML/Pattern

## تنظیمات محیط
فایل `.env` را از `.env.example` بسازید و متغیرهای کلیدی را تنظیم کنید:
- `CACHE_ENABLED`, `REDIS_HOST/PORT/DB` یا `REDIS_URL`
- `ENABLE_DATA_INGESTION`, `DATABASE_URL`, `SQLITE_PATH`
- `DATA_SERVICE_URL` اگر از سرویس داده خارجی استفاده می‌کنید
- `METRICS_ENABLED`, `EUREKA_ENABLED` (در صورت نیاز)
- `ENABLE_SCENARIOS` برای فعال کردن `/api/v1/scenarios/*`
- `EXPOSE_DB_EXPLORER` فقط در توسعه برای `/db/*`

## اجرای محلی (توسعه)
```bash
pip install -r requirements.txt
set PYTHONPATH=src
uvicorn gravity_tech.main:app --host 0.0.0.0 --port 8000 --reload
```
- Swagger: `http://localhost:8000/api/docs`
- سلامت: `/health`, `/health/ready`
- متریک‌ها (در صورت فعال بودن): `/metrics`

## اجرای تولیدی سبک
```bash
set PYTHONPATH=src
uvicorn gravity_tech.main:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info
```
- پشت Nginx/Traefik قرار دهید و CORS/RateLimit را در لایه لبه اعمال کنید.
- اگر Redis ندارید، `CACHE_ENABLED=false` بگذارید تا سرویس بدون کش اجرا شود.
- اگر ingestion و پیام‌رسانی ندارید، `ENABLE_DATA_INGESTION=false` یا Kafka/RabbitMQ را پیکربندی کنید.
- اگر سرویس داده ندارید، داده را مستقیماً در بدنه `/api/v1/analyze` ارسال کنید یا DB محلی را پیش‌بارگذاری کنید.
- DB Explorer را در تولید فعال نکنید (`EXPOSE_DB_EXPLORER=false`).

## صحت‌سنجی پس از استقرار
1. `GET /health` → `{"status":"healthy"}`
2. `GET /health/ready` → چک Redis در صورت فعال بودن
3. `GET /api/v1/indicators/list` → تأیید بارگذاری اندیکاتورها
4. (اختیاری) `GET /patterns/health` و `/ml/health` → بررسی وجود مدل‌ها

## نکات دیباگ
- خطای Redis: `CACHE_ENABLED=false` را تنظیم کنید و سرویس را بدون کش بالا بیاورید.
- مدل‌های ML موجود نیستند: فایل‌های `pattern_classifier_*.pkl` را در `ml_models/` قرار دهید یا endpointهای ML را غیرفعال کنید.
- دیتا سرویس در دسترس نیست: `DATA_SERVICE_URL` را خالی بگذارید و از ورودی دستی کندل استفاده کنید.
