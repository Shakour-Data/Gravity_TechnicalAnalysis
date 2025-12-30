# استقرار و عملیات (Deployment & Operations)

## 1) استقرار با Docker Compose
فایل: `docker-compose.stack.yml`

سرویس‌ها:
- `postgres` (Postgres 16)
- `analysis-api` (FastAPI)
- `ingestion-runner` (اجرای حلقه پایپ‌لاین روزانه)

## 2) اجرای دستی پایپ‌لاین
اسکریپت: `scripts/etl/run_stack_pipeline.py`

### 2.1) init (راه‌اندازی اولیه)
نمونه:
```bash
python scripts/etl/run_stack_pipeline.py --mode init --pg-dsn postgresql://gravity:gravity_db_pass@localhost:5545/tech_analysis
```

### 2.2) daily (آپدیت روزانه)
نمونه:
```bash
python scripts/etl/run_stack_pipeline.py --mode daily --pg-dsn postgresql://gravity:gravity_db_pass@localhost:5545/tech_analysis --analysis-limit 80 --lookback-days 365
```

## 3) مانیتورینگ
- Health: `/health/ready` برای readiness (خصوصاً Redis در صورت فعال بودن)
- Metrics: `/metrics` (در صورت `METRICS_ENABLED=true`)

## 4) نکات بهره‌برداری
- برای production، پیشنهاد می‌شود rate-limit و TLS در لایه reverse proxy (Nginx/Traefik) اعمال شود.
- DB Explorer فقط در شرایط کنترل‌شده و با سیاست دسترسی محدود فعال شود.

