# داده‌ذخیره‌سازی در Gravity Technical Analysis

این سند مسیرهای ذخیره داده، شِماها و جریان نوشتن/خواندن را در سرویس Technical Analysis توضیح می‌دهد.

## ورودی بازار بورس تهران (TSE)
- ماژول: `apps/analysis_api/src/database.py` و سینگل‌تن `gravity_tech/database/tse_data_source.py`.
- انتخاب منبع: اگر `TSE_DATABASE_URL` / `TSE_DB_URL` / `TSE_POSTGRES_URL` ست باشد، Postgres استفاده می‌شود؛ در غیر این صورت اولین فایل موجود از مسیرهای فهرست‌شده در `config.py` (پیش‌فرض: `services/data_ingestion/data/tse_data.db`) انتخاب می‌شود.
- جداول مهم: `price_data`, `market_indices`, `sector_indices`, `usd_prices` و جداول مرجع (`sectors`, `companies`, ...). ستون تاریخ در SQLite = `date` و در Postgres = `trading_date`.
- دسترسی: فقط خواندن برای تحلیل/آموزش؛ متدهای کلیدی `fetch_price_data`, `fetch_market_index`, `fetch_sector_index`, `list_symbols`.

## دیتابیس عملیاتی سرویس
- ماژول: `gravity_tech/database/database_manager.py`.
- انتخاب بک‌اند: 1) Postgres بر اساس `DATABASE_URL`/`POSTGRES_URL`/`settings.database_url` (پیش‌فرض `postgresql://gravity:gravity@localhost:5432/tech_analysis`)، 2) در صورت عدم دسترس بودن SQLite از `settings.sqlite_path` (پیش‌فرض `data/TechAnalysis.db`)، 3) در نهایت فایل JSON (`settings.json_storage_path` پیش‌فرض `data/tool_performance.json`).
- شِمای Postgres: از `historical_schemas.sql` و (در صورت وجود) `tool_performance_history.sql` بارگذاری می‌شود و جدول اضافی `pattern_detection_results` نیز ایجاد می‌شود. شِمای SQLite نسخه ساده‌شده همان جداول است و در کد تعبیه شده.
- جداول اصلی و کاربرد:
  - `tool_performance_history` و `tool_performance_stats`: ثبت و تجمع نتایج ابزارها.
  - `ml_weights_history`: اسنپ‌شات آموزش مدل‌ها (`ml/train_pipeline.py`).
  - `tool_recommendations_log`: لاگ پیشنهاد ابزارها (`services/tool_recommendation_service.py`).
  - `market_data_cache`: کش دیتای OHLCV.
  - `pattern_detection_results`: نتایج کشف الگو (API/CLI).
  - `backtest_runs`: خلاصه بک‌تست‌ها (`ml/backtesting.py`).
- مسیر پیش‌فرض فایل SQLite/JSON: `data/TechAnalysis.db` و `data/tool_performance.json` (همچنین فایل‌های واقعی در `data/` نگهداری می‌شوند).

## دیتابیس تاریخچه تحلیل (Historical Scores)
- ماژول: `gravity_tech/database/historical_manager.py` (فقط Postgres، psycopg2).
- شِما: `gravity_tech/database/historical_schemas.sql` (و نسخه قدیمی‌تر `schemas.sql`). جداول کلیدی:
  - `historical_scores` (کلید symbol+ts+timeframe) با trend/momentum/combined score، سیگنال و قیمت.
  - `historical_horizon_scores`, `historical_indicator_scores`, `historical_patterns`, `historical_volume_analysis`, `historical_price_targets`, `analysis_metadata`.
  - جداول روزانه در شِمای `tech_analysis`: `daily_dimension_scores`, `daily_indicator_values`.
- جریان نوشتن:
  1) در `api/v1/analysis.py`، اگر `settings.enable_data_ingestion` فعال باشد، خروجی تحلیل به رویداد یا مستقیماً ارسال می‌شود.
  2) `DataIngestorService.persist_direct` پس از اعتبارسنجی/دی‌دوپ، `HistoricalScoreManager.save_score` را صدا می‌زند و در صورت وجود، الگوها را با `DatabaseManager.save_pattern_detections` ذخیره می‌کند.
  3) در صورت نبود بروکر (Kafka/RabbitMQ) یا خطا، ذخیره همزمان انجام می‌شود.

## کش و فایل‌های جانبی
- کش Redis اختیاری در `services/cache_service.py` و کلاینت‌ها با `settings.redis_host/redis_port/redis_db` یا `REDIS_URL`; در صورت غیرفعال بودن، سیستم بدون کش کار می‌کند.
- مسیر مدل‌های ML: `apps/analysis_api/ml_models`; تنظیم مسیرها در `gravity_tech/config/paths.py`.
- فایل‌های داده موجود در ریشه `data/` (نمونه: `TechAnalysis.db`, `gravity_tech.db`, `tool_performance.json`, `postgres_backup.dump`) برای محیط‌های مختلف استفاده یا پشتیبان‌گیری می‌شوند.

## نکات راه‌اندازی/عیب‌یابی
- فعال‌سازی ذخیره‌سازی نتایج تحلیل: `ENABLE_DATA_INGESTION=true` در `.env` (یا `settings.enable_data_ingestion`).
- اطمینان از Postgres: مقداردهی `DATABASE_URL` و نصب `psycopg2`; در غیر این صورت به SQLite/JSON سقوط می‌کند.
- بررسی وضعیت فعلی دیتابیس: می‌توانید از `DatabaseManager.get_database_info()` یا اندپوینت‌های `db_explorer` (اگر `settings.expose_db_explorer` فعال باشد) استفاده کنید.

## وضعیت فعلی محیط (2025-12-16)
- Postgres هدف (`DATABASE_URL/TSE_DATABASE_URL=postgresql://gravity:gravity@localhost:5544/tech_analysis`) در دسترس نبود: اتصال با خطای `Connection refused` برمی‌گردد. پیامدها:
  - برای TSE، چون env به Postgres اشاره می‌کند، `TSEDatabaseConnector` در زمان اتصال خطا می‌دهد (هیچ fallback روی SQLite ندارد).
  - `DatabaseManager` به دلیل عدم اتصال به Postgres و نامعتبر بودن مسیر SQLite پیش‌فرض (`data/TechAnalysis.db` در حال حاضر یک دایرکتوری است) به fallback JSON (`data/tool_performance.json`) می‌رود.
- منابع موجود فعلی:
  - TSE SQLite: فایل `services/data_ingestion/data/tse_data.db` با جداول `price_data`, `market_indices`, `sector_indices`, `usd_prices`, ...
  - SQLite قدیمی: `data/gravity_tech.db` با جداول پایه (companies/price_data/...)؛ در تنظیمات فعلی استفاده نمی‌شود.
  - JSON fallback: `data/tool_performance.json` موجود است و فعلاً خالی است.

## گپ‌ها و پیشنهادهای عملی
1) بالا آوردن Postgres روی پورت 5544 (یا به‌روزرسانی `DATABASE_URL`/`TSE_DATABASE_URL` به سرور در دسترس) تا مسیر اصلی Postgres فعال شود.
2) در صورت استفاده از SQLite:
   - متغیرهای Postgres را خالی/حذف کنید تا fallback سریع‌تر فعال شود.
   - مقدار `settings.sqlite_path` را به یک فایل واقعی تنظیم کنید (مثلاً `data/gravity_tech.db` یا `data/tool_performance.db`).
   - برای TSE، env را به مسیر SQLite موجود تغییر دهید (مثلاً `TSE_DATABASE_URL=sqlite:///services/data_ingestion/data/tse_data.db`).
3) پایش وضعیت: پس از راه‌اندازی دیتابیس، از `DatabaseManager.get_database_info()` یا اندپوینت‌های `db_explorer` (در حالت dev و با فعال‌سازی `expose_db_explorer`) برای اطمینان از بک‌اند فعال استفاده کنید.

## Focus: PostgreSQL in Docker
- Stack compose (`docker-compose.stack.yml`): Postgres with `user=gravity`, `password=gravity_db_pass`, `db=tech_analysis`, exposed on `STACK_DB_PORT` (default 5545). API/TSE inside containers use host `postgres:5432` with same credentials.
- Data compose (`docker-compose.data.yml`): Postgres with the same user/pass, db `gravity_tech`, exposed on `DATA_DB_PORT` (default 5546).
- Local .env alignment (for host access):
  - `DATABASE_URL=postgresql://gravity:gravity_db_pass@localhost:5545/tech_analysis`
  - `TSE_DATABASE_URL=postgresql://gravity:gravity_db_pass@localhost:5545/tech_analysis`
  - If using the data-only compose, adjust to `localhost:5546/gravity_tech` (or your overrides).
- Ensure Postgres is running before the app; otherwise `DatabaseManager` may fall back to SQLite/JSON unless instantiated with `allow_fallback=False`.
- Storage path on Postgres: `DatabaseManager` applies `historical_schemas.sql` + `tool_performance_history.sql` (if present) and creates `pattern_detection_results`; `HistoricalScoreManager` writes to the same DSN. Both should point to the DSNs above to keep analysis + TSE data in one Postgres instance.

## وضعیت به‌روز (پس از اصلاح env)
- `.env` اکنون با استک داکر هم‌راستا است: پورت 5545 و پسورد `gravity_db_pass` برای هر دو `DATABASE_URL` و `TSE_DATABASE_URL` (db=`tech_analysis`).
- `DatabaseManager` پیش‌فرض را از `settings.allow_db_fallback` می‌گیرد (False)، بنابراین در حالت عادی فقط Postgres استفاده می‌شود و به SQLite/JSON سقوط نمی‌کند.
- برای اطمینان، Postgres را قبل از اجرای سرویس بالا بیاورید (docker-compose.stack.yml -> سرویس `postgres`).
- اگر می‌خواهید منبع TSE حتماً Postgres باشد، از همان DSN استفاده کنید؛ در غیر این صورت می‌توانید `TSE_DATABASE_URL` را به فایل SQLite (`services/data_ingestion/data/tse_data.db`) تغییر دهید.


## Test Plan (PostgreSQL end-to-end)
1) Start Postgres: `docker-compose -f docker-compose.stack.yml up -d postgres`; confirm healthy with `docker-compose -f docker-compose.stack.yml ps`.
2) Align env: set `DATABASE_URL`/`TSE_DATABASE_URL` to `postgresql://gravity:gravity_db_pass@localhost:5545/tech_analysis`, `ALLOW_DB_FALLBACK=false`.
3) Connectivity check:
```
PYTHONPATH=apps/analysis_api/src python - <<'PY'
from gravity_tech.database.database_manager import DatabaseManager
m=DatabaseManager(auto_setup=False, allow_fallback=False)
print(m.get_database_info())
PY
```
Expected: db_type=postgresql.
4) Schema check: in psql `\dt` should list tool_performance_* , ml_weights_history, backtest_runs, pattern_detection_results, historical_* tables.
5) TSE source: if using Postgres, `SELECT count(*) FROM tse_input.price_data;`; if SQLite, run query on `services/data_ingestion/data/tse_data.db` to verify rows.
6) Insert historical sample:
```
PYTHONPATH=apps/analysis_api/src python - <<'PY'
from gravity_tech.database.historical_manager import HistoricalScoreManager, HistoricalScoreEntry
from datetime import datetime
m=HistoricalScoreManager('postgresql://gravity:gravity_db_pass@localhost:5545/tech_analysis')
e=HistoricalScoreEntry(symbol='TEST', timestamp=datetime.utcnow(), timeframe='1h',
    trend_score=0.1, trend_confidence=0.9, momentum_score=0.2, momentum_confidence=0.8,
    combined_score=0.15, combined_confidence=0.85, trend_weight=0.5, momentum_weight=0.5,
    trend_signal='NEUTRAL', momentum_signal='NEUTRAL', combined_signal='NEUTRAL',
    recommendation='HOLD', action='HOLD', price_at_analysis=100)
with m: m.save_score(e)
print('ok')
PY
```
Expected: row appears in historical_scores.
7) Insert backtest summary:
```
PYTHONPATH=apps/analysis_api/src python - <<'PY'
from gravity_tech.database.database_manager import DatabaseManager
from datetime import datetime
m=DatabaseManager(auto_setup=True, allow_fallback=False)
id=m.save_backtest_run(symbol='TEST', source='db', interval='1d', params={'k':1}, metrics={'ok':True},
    period_start=datetime.utcnow(), period_end=datetime.utcnow(), model_version='v1')
print('id', id)
PY
```
Expected: id returned; table backtest_runs has row.
8) Negative fallback test: stop Postgres and rerun step 3 with allow_fallback=False; should raise error (confirms no SQLite/JSON).
9) API/ingestion e2e: set `enable_data_ingestion=true`, call analysis API, then check historical_scores and pattern_detection_results in Postgres for inserted data.

## نتایج تست‌های اجرا‌شده (2025-12-16)
- Postgres از طریق docker-compose.stack.yml روی پورت 5545 بالا و healthy شد.
- auto_setup با `DatabaseManager` موفق اجرا شد (table_count=10 پس از ایجاد جداول).
- درج نمونه HistoricalScore + Indicator توسط `HistoricalScoreManager.save_score` موفق (رکورد در historical_scores و historical_indicator_scores).
- درج خلاصه بک‌تست با `save_backtest_run` موفق (رکورد در backtest_runs).
- تست منفی fallback: با خاموش کردن Postgres و allow_fallback=False خطای اتصال رخ داد (سقوط به SQLite/JSON نشد).
- شمارش فعلی پس از تست: historical_scores=1، historical_indicator_scores=1، backtest_runs=1.

## Test Results (2025-12-16)
- Postgres up via `docker-compose.stack.yml` on port 5545 (healthy).
- `DatabaseManager` auto_setup succeeded; tables created (table_count=15 after adding tool performance tables).
- `HistoricalScoreManager.save_score` inserted rows into `historical_scores` and `historical_indicator_scores`.
- `save_backtest_run` inserted a row into `backtest_runs`.
- Negative fallback: with Postgres stopped and `allow_fallback=False` a connection error is raised (no SQLite/JSON fallback).
- Current row counts: `historical_scores=1`, `historical_indicator_scores=1`, `backtest_runs=1`; tool_performance_* tables exist and are empty (ready for writes).

## HTTP Ingestion Verification (2025-12-16)
- FastAPI app اجرا شد و `enable_data_ingestion` روی True قرار گرفت؛ درخواست POST به `/api/v1/analyze` با 60 کندل ساخته شد.
- مسیر ingestion در لایه سرویس (DataIngestorService.persist_direct) با DSN Postgres `postgresql://gravity:gravity_db_pass@localhost:5545/tech_analysis` تست شد و رکوردهای historical/indicator اضافه شد.
- شمارش پس از HTTP/ingestion: `historical_scores=3`, `historical_indicator_scores=3`, `pattern_detection_results=1` (و جداول ابزار با داده تستی پر شدند).
- نکته اجرا: در حالت ران‌تایم عادی (با لایف‌اسپن)، `data_ingestor` از `settings.database_url` استفاده می‌کند؛ کافی است `enable_data_ingestion=true` و DSN درست در `.env` باشد تا درج از مسیر HTTP هم انجام شود.
