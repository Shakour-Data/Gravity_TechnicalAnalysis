# ساختار پروژه

**نسخه:** 1.0.0  
**به‌روز شده:** 2025-12-10

## نمای کلی درخت
```
Gravity_TechnicalAnalysis/
├─ src/
│  └─ gravity_tech/
│     ├─ api/v1/          # مسیرهای FastAPI: analysis, patterns, ml, tools, backtest, db
│     ├─ services/        # منطق کسب‌وکار: TechnicalAnalysisService، cache، ingestion، توصیه ابزار
│     ├─ core/            # مدل‌ها و اندیکاتورها (trend/momentum/volume/volatility/cycle/SR)
│     ├─ ml/              # خط لوله ۵بعدی، ماتریس حجم، multi-horizon، backtesting، مدل‌ها
│     ├─ patterns/        # الگوهای شمعی/هارمونیک و امواج الیوت
│     ├─ clients/         # DataServiceClient (HTTP + Redis cache)
│     ├─ database/        # مدیریت اتصال، اکتشاف DB، schema helpers
│     └─ middleware/      # لاگ ساختاری، discovery، رویدادها، CORS
├─ database/              # اسکریپت‌های SQL و DatabaseManager
├─ ml_models/             # فایل‌های مدل ML (مثلاً pattern_classifier_*.pkl)
├─ scripts/               # ابزارهایی مثل populate_last90.py، به‌روزرسانی/تأیید migration
├─ tests/                 # واحد/یکپارچه (پوشش اندیکاتورها و سرویس‌ها)
├─ docs/                  # همین مستندات
└─ data/                  # داده نمونه/SQLite (در .gitignore)
```

## لایه‌ها و پوشه‌های کلیدی
- **API (`src/gravity_tech/api/v1`)**:  
  - `analysis.py` (تحلیل کامل + اندیکاتورهای انتخابی + لیست اندیکاتورها)  
  - `patterns.py` (تشخیص هارمونیک + ML اختیاری)  
  - `ml.py` (پیش‌بینی الگو، batch، اطلاعات مدل)  
  - `tools.py` (توصیه ابزار، تحلیل سفارشی)  
  - `backtest.py` (بک‌تست تشخیص الگو)  
  - `db_explorer.py` (خواندن شِما/کوئری محدود)  

- **Services (`src/gravity_tech/services`)**:  
  - `analysis_service.py` (اورکستریتور اندیکاتورها + الگوهای شمعی + الیوت + فاز بازار)  
  - `cache_service.py` (Redis async)  
  - `data_ingestor_service.py` (ذخیره نتایج تحلیل؛ Kafka/RabbitMQ اختیاری)  
  - `tool_recommendation_service.py`، `fast_indicators.py`, `signal_engine.py`

- **Indicators (`src/gravity_tech/core/indicators`)**: ۶ دسته اندیکاتور پیاده‌سازی‌شده با سیگنال و confidence استاندارد (Trend، Momentum، Cycle، Volume، Volatility، Support/Resistance).

- **ML (`src/gravity_tech/ml`)**:  
  - `complete_analysis_pipeline.py`، `five_dimensional_decision_matrix.py`، `volume_dimension_matrix.py`  
  - multi-horizon analyzers/feature extractors برای trend/momentum/volatility/cycle/SR  
  - `pattern_classifier.py` و `backtesting.py`

- **Patterns (`src/gravity_tech/patterns`)**: شمعی، کلاسیک، هارمونیک، تحلیل موج الیوت.

- **Database (`database/` + `src/gravity_tech/database/`)**:  
  - SQL schemaهای تاریخی، `DatabaseManager`، `HistoricalScoreManager`  
  - پشتیبانی SQLite/PostgreSQL (وابسته به تنظیمات)

- **Clients**: `clients/data_service_client.py` برای واکشی داده‌ی Adjusted با httpx + Redis cache.

- **Middleware**: لاگ ساختاری، CORS، discovery اختیاری، انتشار رویداد.

- **Feature Flags (در settings)**:
  - `enable_scenarios`: فعال‌سازی Router سناریو سه‌گانه (`/api/v1/scenarios/*`)
  - `expose_db_explorer`: فعال‌سازی مسیرهای `/db/*` (فقط برای توسعه توصیه می‌شود)

## یادداشت‌ها
- حداقل ورودی تحلیل: ۵۰ کندل؛ برخی اندیکاتورها به ۶۰ نیاز دارند.  
- کش Redis و ingestion رویدادها اختیاری و با تنظیمات `.env` فعال می‌شوند.  
- سناریوهای سه‌گانه در `api/v1/scenarios.py` موجود است ولی به‌صورت پیش‌فرض mount نشده است.  
- مدل‌های ML باید در `ml_models/` موجود باشند تا endpointهای ML/Pattern با موفقیت پاسخ دهند.
