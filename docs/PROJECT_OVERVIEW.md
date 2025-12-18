# معرفی جامع پروژه Gravity Technical Analysis

این سند توضیح می‌دهد سرویس دقیقاً چه می‌کند، چه جدول‌ها و مدل‌هایی دارد، امتیازدهی چگونه انجام می‌شود، چه اندپوینت‌ها و خط لوله‌هایی وجود دارد و چطور می‌توانید در n8n یک Agent بسازید. (نسخه کد: 1.0.0)

## 1) مأموریت سرویس
- یک میکروسرویس FastAPI برای تحلیل تکنیکال چندبُعدی با بیش از ۶۰ اندیکاتور و تشخیص الگوهای کندلی/کلاسیک/هارمونیک.
- خروجی شامل امتیازدهی چند اُفقه (۳/۷/۳۰ روزه)، ماتریس ۵ بعدی تصمیم، ماتریس تعامل حجم، و امتیاز ML برای الگوهای هارمونیک.
- قابلیت کار با دیتابیس TSE (SQLite/PostgreSQL)، کش Redis، انتشار رویداد در Kafka/RabbitMQ، و بک‌تست.

## 2) ساختار پروژه

**Version:** 1.0.0  
**Last Updated:** 2025-12-18

### Directory Map
```
Gravity_TechnicalAnalysis/
├─ apps/analysis_api/
│  ├─ src/gravity_tech/       # FastAPI routers, services, core domain, ML, database helpers
│  ├─ ml_models/             # Saved ML artifacts (indicator/dimension weights, classifiers)
│  └─ tests/                 # Unit/integration/slow suites
├─ scripts/                  # Utilities: populate_last90.py, run_full_pipeline.py, migrations/maintenance
├─ services/data_ingestion/  # Fetch raw OHLCV, indices, company metadata from TSE, persist into SQLite
├─ docs/                     # Project documentation
├─ data/                     # SQLite databases (gitignored)
└─ docker-compose.stack.yml  # Docker services
```

### Key Components
- **API (`apps/analysis_api/src/gravity_tech/api/v1`)**: `analysis.py`, `patterns.py`, `ml.py`, `tools.py`, `backtest.py`, `db_explorer.py`.
- **Services (`apps/analysis_api/src/gravity_tech/services`)**: `analysis_service.py`, `cache_service.py`, `data_ingestor_service.py`, `tool_recommendation_service.py`, `fast_indicators.py`, `signal_engine.py`.
- **Indicators (`apps/analysis_api/src/gravity_tech/core/indicators`)**: Trend, Momentum, Volume, Volatility, Cycle, Support/Resistance calculators with confidence scores.
- **ML (`apps/analysis_api/src/gravity_tech/ml`)**: `complete_analysis_pipeline.py`, `five_dimensional_decision_matrix.py`, `volume_dimension_matrix.py`, multi-horizon analyzers/feature extractors, `pattern_classifier.py`, `backtesting.py`.
- **Patterns (`apps/analysis_api/src/gravity_tech/patterns`)**: Harmonic, classical, Elliott, candlestick detection utilities.
- **Database (`apps/analysis_api/src/gravity_tech/database/`)**: Canonical schema files plus `DatabaseManager`/`HistoricalScoreManager`.
- **Scripts (`scripts/`)**: Operational helpers. `run_full_pipeline.py` runs the full TSE→analysis→TechAnalysis.db flow; other populate/maintenance scripts live here.
- **Clients**: `clients/data_service_client.py` for adjusted OHLCV retrieval via HTTP + Redis cache.
- **Middleware**: CORS, discovery, security, tracing, and metrics helpers.
- **Feature Flags (see `settings`)**: `enable_scenarios` toggles `/api/v1/scenarios/*`; `expose_db_explorer` toggles `/api/v1/db/*`.

### Notes
- API/tests assume a SQLite backend by default; configure environment variables for PostgreSQL if needed.
- Redis and ingestion flags live in `.env`; defaults are safe for local development.
- ML artifacts expected in `ml_models/` for ML/Pattern endpoints.

## 3) اجزای اصلی کد
- Entry-point: `apps/analysis_api/src/gravity_tech/main.py`
- Routerها:
  - `api/v1/analysis.py` (تحلیل کامل، اندیکاتور خاص، داده تاریخی)
  - `api/v1/patterns.py` (تشخیص هارمونیک + امتیاز ML)
  - `api/v1/ml.py` (کلاسیفای الگو + متادیتا مدل + batch)
  - `api/v1/backtest.py` (بک‌تست روی دادهٔ واقعی یا آرایهٔ OHLCV)
  - `api/v1/tools.py`, `api/v1/scenarios.py`, `api/v1/db_explorer.py`, `api/v1/auth.py`
- مدل‌ها و کانتراکت‌ها: `gravity_tech/core/contracts/analysis.py`, `core/domain/entities.py`
- سرویس‌ها: `gravity_tech/services/*` (analysis_service, data_ingestor_service, signal_engine, cache_service)
- الگوها و ML: `gravity_tech/patterns/*`, `gravity_tech/ml/*`, مدل‌ها در `apps/analysis_api/ml_models/`
- خط لوله آفلاین: `scripts/etl/run_full_pipeline.py`
- شِمای PostgreSQL: `scripts/schema/postgres_schema.sql`

## 4) جریان کلی داده و امتیازدهی
1. دریافت OHLCV (از API یا از DB).
2. محاسبهٔ اندیکاتورهای شش بُعدی:
   - Trend, Momentum, Volatility, Cycle, Volume, Support-Resistance.
3. تشخیص الگوهای کندلی/کلاسیک/Elliott و الگوهای هارمونیک.
4. محاسبهٔ ماتریس حجم (Volume Interaction Matrix) برای تأیید سیگنال‌های قیمتی.
5. محاسبهٔ ماتریس ۵بُعدی تصمیم (Trend/Momentum/Volatility/Cycle/SR) و امتیاز نهایی `decision_matrix_score`.
6. امتیازدهی ML برای الگوهای هارمونیک (XGBoost، ۲۱ Feature).
7. تجمیع نهایی:
   - `overall_signal` و `overall_confidence`
   - امتیازهای بعدی: `trend_score`, `momentum_score`, `volatility_score`, `cycle_score`, `sr_score`
   - `volume_interaction_score` میانگین تعامل حجم
   - `final_signal` و `final_confidence` از تصمیم‌ساز
8. ذخیره در DB (`analysis_results`) و/یا انتشار رویداد `ANALYSIS_COMPLETED` اگر ingestion فعال باشد.

## 5) ماتریس ۵بُعدی تصمیم (Decision Matrix)
- ورودی: امتیازهای هر بُعد (نرمال‌شده ۰..۱)، وزن هر بُعد (پیش‌فرض یا **ML-based**).
- روند تصمیم:
  1. نرمال‌سازی امتیازها و اعمال وزن‌ها.
  2. آستانه‌ها:
     - > 0.65 → سیگنال BULLISH/BULLISH_STRONG
     - 0.35..0.65 → NEUTRAL / CAUTION
     - < 0.35 → BEARISH/BEARISH_STRONG
  3. اطمینان (confidence) بر اساس پراکندگی امتیازها و توافق بین ابعاد.
  4. خروجی: `final_signal` (enum) و `final_confidence` (0..1).
- **وزن‌دهی**:
  - پیش‌فرض از `ml_models/multi_horizon/indicator_weights_btcusdt.json`.
  - **توجه: وزن‌دهی ML-based یک ویژگی کلیدی برای دقت بالاتر سیگنال‌ها است و باید در مدل‌های تولیدی استفاده شود.**

## 6) نکات عملیاتی و ترفندها
- حداقل ۶۰ کندل بدهید تا اندیکاتورهای بلندمدت کار کنند.
- برای دادهٔ زیاد، Redis را فعال کنید تا زمان پاسخ کاهش یابد.
- اگر مدل هارمونیک موجود نیست، مسیر `apps/analysis_api/ml_models/` را بررسی کنید؛ fallback فقط احتمال یکنواخت می‌دهد.
- برای PostgreSQL، قبل از اجرا شِما را با `scripts/schema/postgres_schema.sql` ایجاد کنید.
- `EXPOSE_DB_EXPLORER=true` را فقط در محیط امن فعال کنید (روتر `db_explorer` کوئری می‌گیرد).
- اگر ingestion فعال است و Kafka/RabbitMQ ندارید، سرویس به‌صورت مستقیم persist می‌کند (`data_ingestor.persist_direct`).

## 7) عیب‌یابی سریع
- خطای مدل: اگر لاگ `ml_model_fallback_used` دیدید، فایل مدل موجود نیست یا hash نمی‌خواند.
- خطای اندیکاتور: اگر تعداد کندل < 60 باشد، 400 برمی‌گردد.
- کندی پاسخ: Redis را فعال کنید و تعداد کندل را محدود کنید؛ در n8n از اندپوینت سبک‌تر استفاده کنید.
- خطای بک‌تست: بررسی کنید طول آرایه‌ها برابر و بدون NaN/Inf باشد؛ حداقل `max(window_size+step_size, 300)` میله لازم است.

## 8) مسیرهای مهم فایل‌ها (برای توسعه)
- API و روترها: `apps/analysis_api/src/gravity_tech/api/v1/*.py`
- سرویس تحلیل: `gravity_tech/services/analysis_service.py`
- تصمیم‌ساز: `gravity_tech/services/signal_engine.py`
- کش: `gravity_tech/services/cache_service.py`
- الگوهای هارمونیک: `gravity_tech/patterns/harmonic.py`
- مدل و ویژگی‌ها: `gravity_tech/ml/pattern_features.py`, `ml/pipeline_factory.py`
- خط لوله آفلاین: `scripts/etl/run_full_pipeline.py`
- شِمای DB: `scripts/schema/postgres_schema.sql`
- اسکریپت‌های کمکی: `scripts/exports`, `scripts/data_population`, `scripts/backtesting`, `scripts/maintenance`

## 9) خلاصه کوتاه
این سرویس یک موتور تحلیل تکنیکال چندبُعدی + تشخیص هارمونیک با امتیاز ML و بک‌تست است. ورودی OHLCV (از API یا DB)، خروجی سیگنال نهایی با اطمینان، ذخیره در DB و امکان انتشار رویداد. می‌توانید در n8n با چند نود ساده آن را به Agent تبدیل کنید و بر اساس `overall_signal/decision.final_signal` هشدار بفرستید یا اتوماسیون اجرا کنید.