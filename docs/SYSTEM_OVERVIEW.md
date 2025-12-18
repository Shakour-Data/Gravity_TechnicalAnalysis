# Gravity System Overview

این سند نمای کلی از سیستم Gravity Technical Analysis را ارائه می‌دهد، شامل لایه‌ها، جریان داده، سناریوها، یکپارچه‌سازی، اینجکشن، سلامت، امنیت و راه‌اندازی دستی.

## 1) لایه‌های سیستم

این ریپوزیتوری شامل هر دو سرویس است که قبلاً در پروژه‌های جداگانه بودند:

| لایه | مکان | مسئولیت |
|-------|----------|----------------|
| Data Ingestion | `services/data_ingestion` | دریافت داده خام OHLCV، شاخص‌ها، متادیتای شرکت‌ها از TSE، ذخیره در SQLite (`tse_data.db`). |
| Technical Analysis | `src/` (FastAPI service) | مصرف داده خام (از طریق PostgreSQL)، اجرای خط لوله ML، نمایش APIهای REST/WebSocket. |

## 2) جریان داده

```
gravity_tse client ──► data_ingestion (SQLite) ──► migrate_sqlite_to_pg.py ──► PostgreSQL (tse_input)
                                                                               │
                                                                               └──► analysis batch ─► tech_analysis.*
```

1. `services/data_ingestion` از `gravity_tse.py` به علاوه متادیتای JSON برای نگهداری `services/data_ingestion/data/tse_data.db` استفاده می‌کند.
2. `scripts/run_stack_pipeline.py` هماهنگ می‌کند:
   - ingestion CLI (`init-all` یا `load-all-prices`)؛
   - ایجاد شِما در PostgreSQL از طریق `scripts/postgres_schema.sql`؛
   - مهاجرت به جداول `tse_input.*`؛
   - تحلیل تاریخی batch (`scripts/run_full_batch_analysis.py`).
3. اپ FastAPI از جداول `tech_analysis.*` برای سرویس‌دهی APIها می‌خواند.

## 3) سرویس‌های Docker

`docker-compose.stack.yml` فراهم می‌کند:

1. `postgres`: دیتابیس مشترک (`tech_analysis` DB، شِماها `tse_input` + `tech_analysis`).
2. `analysis-api`: سرویس FastAPI (پورت `8000` را نمایش می‌دهد).
3. `ingestion-runner`: `scripts/run_stack_pipeline.py --mode daily` را در یک حلقه اجرا می‌کند (فاصله پیش‌فرض = ۲۴ ساعت) و `services/data_ingestion/data` را به عنوان volume به اشتراک می‌گذارد.

> ⚠️ قبل از ساخت تصویر ingestion، `gravity_tse.py` و فایل‌های متادیتای JSON را در `services/data_ingestion/scripts` و `services/data_ingestion/data/BasicTseInformation/` حذف کنید.

## 4) راه‌اندازی دستی (Manual Runbook)

1. `docker compose -f docker-compose.stack.yml up -d postgres`.
2. `python scripts/run_stack_pipeline.py --mode init --pg-dsn postgresql://gravity:gravity_db_pass@localhost:5544/tech_analysis`.
3. `docker compose -f docker-compose.stack.yml up -d analysis-api` برای نمایش APIها.
4. برنامه‌ریزی `python scripts/run_stack_pipeline.py --mode daily ...` (یا تکیه بر سرویس compose).

از فلگ‌های `--skip-*` در `run_stack_pipeline.py` استفاده کنید وقتی فقط بخشی از جریان را می‌خواهید.

## 5) سناریو سه‌گانه (`/api/v1/scenarios/*` اختیاری)

```mermaid
flowchart LR
    Client --> ScnAPI["/api/v1/scenarios/{symbol}"]
    ScnAPI --> DSC["DataServiceClient
(Adjusted OHLCV)"]
    DSC --> Analyzer["ScenarioAnalyzer
(optimistic/neutral/pessimistic)"]
    Analyzer --> Resp["targets, SL, R/R, expected return"] --> Client
```

- فقط با `ENABLE_SCENARIOS=true` فعال می‌شود. نیازمند داده Adjusted (سرویس داده یا DB محلی).
- ولیدیشن: نماد حداقل ۲ کاراکتر و حروف/عدد/-/./_؛ تایم‌فریم از لیست 1m..1w؛ lookback ≥۳۰ روز؛ حداقل ۱۲۰ کندل معتبر (بدون NaN/Inf، high>=low، حجم>=۰) وگرنه ۴۰۰.
- متریک‌ها: Counter/Histogram برای موفق/خطا/تاخیر اضافه شده است.
- خطاها: ورودی بد => ۴۰۰؛ خطای سرویس داده/داخلی => ۵۰۳.

## 6) یکپارچه‌سازی سرویس داده و کش

```mermaid
flowchart LR
    API --> DSC["DataServiceClient"]
    DSC -->|cache hit| Redis[(Redis)]
    DSC -->|miss| DataSvc["Data Service /api/v1/candles/{symbol}"]
    DataSvc --> Validate["Validate adjusted OHLCV"]
    Validate --> Redis
    Validate --> API
```

- ولیدیشن ورودی: نماد حداقل ۲ کاراکتر، فقط حروف/عدد/-/./_؛ تایم‌فریم 1m..1w؛ start<end و بازه ≤ ۱۰۹۵ روز.
- ولیدیشن خروجی: حداقل ۳۰ کندل، بدون NaN/Inf، high>=low و ترتیب زمانی صعودی؛ نقض => خطا.
- کش: کلید شامل base_url/symbol/timeframe/start/end است؛ miss ثبت می‌شود، TTL پیش‌فرض ۶ ساعت. در نبود Redis یا miss، داده از سرویس گرفته و پس از validate کش می‌شود.

## 7) اینجکشن و ذخیره‌سازی نتایج (اختیاری)

```mermaid
flowchart LR
    Analysis --> Payload["ingestion payload"]
    Payload -->|event broker?| Broker["Kafka/RabbitMQ (اختیاری)"]
    Payload -->|direct| DB[(Historical DB)]
    Broker --> Ingestor["DataIngestorService
(validations + dedup + metrics)"] --> DB
```

- ولیدیشن payload: نماد معتبر، تایم‌فریم 1m..1w، نبود NaN/Inf، حد سقف حجم داده (۱۰MB)، کندل‌های مرتب و مثبت؛ در نقض، event رد می‌شود.
- Dedup ساده: کلید یکتا (symbol/timeframe/timestamp) در حافظهٔ کوتاه‌مدت نگه داشته می‌شود تا تکراری‌ها ذخیره نشوند.
- متریک‌ها: شمارنده/هیستوگرام موفق/خطا/تاخیر، شمارش retries و circuit-breaker.
- FallBack direct: اگر broker نباشد، ذخیره‌سازی مستقیم با ولیدیشن/احراز/تلاش مجدد انجام می‌شود؛ خطاها گزارش می‌شود.

## 8) سلامت و متریک

- `/health`, `/health/live`: وضعیت پایه سرویس.
- `/health/ready`: چک Redis در صورت فعال بودن؛ در نبود Redis می‌توان `CACHE_ENABLED=false` کرد.
- `/metrics`: Prometheus (اگر `METRICS_ENABLED=true`).

## 9) امنیت و دسترسی

- CORS در کد برای همه Origin باز است؛ در تولید باید در لایه لبه محدود شود.
- DB Explorer (`/api/v1/db/*`) فقط با `EXPOSE_DB_EXPLORER=true` فعال می‌شود و صرفاً برای توسعه توصیه می‌شود.
- مدل‌های ML و داده Adjusted باید در مسیر/سرویس مناسب موجود باشند.