# معرفی جامع پروژه Gravity Technical Analysis

این سند توضیح می‌دهد سرویس دقیقاً چه می‌کند، چه جدول‌ها و مدل‌هایی دارد، امتیازدهی چگونه انجام می‌شود، چه اندپوینت‌ها و خط لوله‌هایی وجود دارد و چطور می‌توانید در n8n یک Agent بسازید. (نسخه کد: 1.0.0)

## 1) مأموریت سرویس
- یک میکروسرویس FastAPI برای تحلیل تکنیکال چندبُعدی با بیش از ۶۰ اندیکاتور و تشخیص الگوهای کندلی/کلاسیک/هارمونیک.
- خروجی شامل امتیازدهی چند اُفقه (۳/۷/۳۰ روزه)، ماتریس ۵ بعدی تصمیم، ماتریس تعامل حجم، و امتیاز ML برای الگوهای هارمونیک.
- قابلیت کار با دیتابیس TSE (SQLite/PostgreSQL)، کش Redis، انتشار رویداد در Kafka/RabbitMQ، و بک‌تست.

## 2) اجزای اصلی کد
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

## 3) جریان کلی داده و امتیازدهی
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

## 4) ماتریس ۵بُعدی تصمیم (Decision Matrix)
- ورودی: امتیازهای هر بُعد (نرمال‌شده ۰..۱)، وزن هر بُعد (پیش‌فرض یا ML-based).
- روند تصمیم:
  1. نرمال‌سازی امتیازها و اعمال وزن‌ها.
  2. آستانه‌ها:
     - > 0.65 → سیگنال BULLISH/BULLISH_STRONG
     - 0.35..0.65 → NEUTRAL / CAUTION
     - < 0.35 → BEARISH/BEARISH_STRONG
  3. اطمینان (confidence) بر اساس پراکندگی امتیازها و توافق بین ابعاد.
  4. خروجی: `final_signal` (enum) و `final_confidence` (0..1).
- وزن‌دهی:
  - پیش‌فرض از `ml_models/multi_horizon/indicator_weights_btcusdt.json`.
  - حالت adaptive/ML: اگر مدل پیکل موجود باشد (`indicator_weights_btcusdt.pkl`) و در سرویس بارگذاری شود.

## 5) ماتریس تعامل حجم (Volume Interaction Matrix)
- هدف: تأیید یا رد سیگنال‌های قیمتی با حجم.
- محاسبه:
  - حجم نسبی، انحراف از میانگین، اسپایک حجم، و هم‌راستایی با حرکت قیمت.
  - خروجی: `volume_interaction_score` (میانگین تعاملات) و جزئیات در `volume_interactions`.
- کاربرد: اگر سیگنال قیمتی قوی ولی حجم ضعیف باشد، اطمینان کاهش می‌یابد.

## 6) فهرست اندیکاتورها (نمونه)
- Trend: SMA, EMA, WMA, DEMA, TEMA, MACD, ADX, Parabolic SAR, Supertrend, Ichimoku, Donchian, Aroon, Vortex, McGinley
- Momentum: RSI, Stochastic, CCI, ROC, Williams %R, MFI, Ultimate Oscillator, TSI, KST, PMO
- Volatility: Bollinger Bands, ATR, Keltner, Donchian, StdDev, HistVol, Chandelier Exit, Mass Index, Ulcer Index, RVI
- Cycle: Sine Wave, Hilbert Dominant Cycle, DPO, Schaff Trend Cycle, Market Facilitation Index, Cycle Period, Phase Change Index, Autocorrelation Periodogram, Cycle Phase Index, Trend-vs-Cycle
- Volume: OBV, CMF, VWAP, A/D Line, Volume Profile, PVT, EMV, VPT, Volume Oscillator, VWMA
- Support/Resistance: Pivot Points, Fib Retracement/Extension, Camarilla, Woodie, DeMark, S/R Levels, Floor Pivots, Psychological Levels, Previous High/Low

## 7) الگوهای پوشش‌داده‌شده
- هارمونیک: gartley, butterfly, bat, crab (ML + قواعد فیبوناچی با tolerance پیش‌فرض 0.05–0.15).
- کندلی: Doji, Hammer, Engulfing, Morning/Evening Star, Harami, Three White Soldiers, Three Black Crows, Marubozu, ...
- کلاسیک: مثلث، پرچم، کنج (در `gravity_tech.patterns`)، و تحلیل موج Elliott (IMP/ABC + نسبت‌های فیبو).
- نمودارهای جایگزین: Renko، Three Line Break، Point & Figure.

## 8) مدل‌های ML
- Pattern Classifier (XGBoost)، ورژن‌ها `v1`, `v2` در `apps/analysis_api/ml_models/pattern_classifier_v*.pkl`.
- ویژگی‌ها (۲۱ بعد): نسبت‌ها (XAB/ABC/BCD/XAD)، هندسه (symmetry/slope/angles)، طول و اندازهٔ موج‌ها، حجم (volume_at_d/trend/confirmation)، اندیکاتورها (RSI/MACD/momentum_divergence).
- متریک‌های سرویس: `ml_prediction_requests_total`, `ml_prediction_latency_seconds`, cache hits/loads.
- fallback: اگر مدل پیدا نشود، مدل dummy با احتمال یکنواخت برمی‌گردد (`model_version=fallback`).

## 9) پایگاه داده و جدول‌ها (مرکز تحلیل)
### SQLite پیش‌فرض (`data/TechAnalysis.db`)
- `analysis_results`:
  - کلید: `(symbol, analysis_date)`
  - ستون‌ها: `final_signal`, `confidence`, `trend_score`, `momentum_score`, `volatility_score`, `cycle_score`, `sr_score`, `volume_interaction_score`, `decision_matrix_score`, `created_at`
- `backtest_runs` (در صورت فعال‌سازی persist): خلاصهٔ بک‌تست شامل متریک‌ها و timestamp.

### PostgreSQL (شِما `tech_analysis`) در `scripts/schema/postgres_schema.sql`
- `tech_analysis.symbols`: نگاشت نماد به شرکت/سکتور/مارکت.
- `tech_analysis.market_data_cache`: کش دادهٔ بازار (symbol, timeframe, ts, OHLCV).
- `tech_analysis.historical_scores`: امتیازهای تاریخی چند اُفقه.
- `tech_analysis.analysis_results`: همان ستون‌های بالا با نوع `DOUBLE PRECISION` و کلید یونیک `(symbol, analysis_date)`.
- `tech_analysis.backtest_runs`: نتیجهٔ بک‌تست (metrics JSON، دوره زمانی، مدل، timestamp).
- شِمای ورودی TSE (`tse_input.*`):
  - `price_data` (OHLCV تعدیل‌شده)، `companies`, `sectors`, `panels`, `markets`, `indices_info`, `market_indices`, `sector_indices`, `last_updates`.

## 10) خط لوله آفلاین TSE → تحلیل → خروجی
- اسکریپت: `scripts/etl/run_full_pipeline.py`
- ورودی: `--source-db` (SQLite TSE)، `--symbols` یا `--max-symbols`، `--limit` (تعداد کندل)، `--timeframe` (برچسب منطقی)، وزن‌ها `--weights-json` و `--weights-model`.
- خروجی: جدول `analysis_results` در SQLite یا PostgreSQL (با `--target-db`).
- مراحل:
  1. خواندن OHLCV از `tse_data.db` (یا منبع تعیین‌شده).
  2. ساخت `Candle` ها و اجرای `CompleteAnalysisPipeline`.
  3. محاسبهٔ امتیازها + ماتریس حجم + تصمیم نهایی.
  4. upsert در `analysis_results`.
  5. (اختیاری) ثبت لاگ و سطح verbose.

## 11) اندپوینت‌های کلیدی (خلاصه عملکرد)
- `POST /api/v1/analyze` → تحلیل کامل؛ حداقل ۶۰ کندل؛ خروجی `TechnicalAnalysisResult`.
- `POST /api/v1/analyze/indicators` → محاسبهٔ لیستی از اندیکاتورها؛ خروجی `IndicatorResult[]`.
- `GET /api/v1/analyze/historical/{symbol}` → خواندن از DB TSE و تحلیل.
- `GET /api/v1/indicators/list` → فهرست اندیکاتورها و الگوها.
- `POST /api/v1/patterns/detect` → تشخیص هارمونیک + امتیاز ML اختیاری.
- `POST /api/v1/ml/predict` و `/predict/batch` → کلاسیفای الگو.
- `POST /api/v1/backtest` → بک‌تست روی آرایه یا دادهٔ واقعی؛ امکان `persist=true`.
- سلامت و مانیتورینگ: `/health`, `/health/ready`, `/metrics`.
- Auth دمو: `POST /api/auth/login` (توکن JWT ساده برای تست).

## 12) نمونهٔ درخواست‌ها
### تحلیل کامل
```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "candles": [
    {"timestamp": "2024-01-01T00:00:00Z", "open": 43000, "high": 43500, "low": 42800, "close": 43250, "volume": 120000},
    {"timestamp": "2024-01-01T01:00:00Z", "open": 43250, "high": 43800, "low": 43100, "close": 43720, "volume": 98000}
  ]
}
```
پاسخ: `overall_signal`, `overall_confidence`, امتیازهای ۶ بعد، الگوهای کشف‌شده، موج Elliott، Renko/ThreeLineBreak/PointFigure (اگر محاسبه شده)، `decision.final_signal`.

### تشخیص هارمونیک
```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "candles": [ ... حداقل 60 کندل با timestamp میلی‌ثانیه ... ],
  "use_ml": true,
  "min_confidence": 0.6,
  "tolerance": 0.05
}
```
پاسخ: `patterns_found`, لیست الگوها با `confidence` ML و اهداف قیمت/استاپ.

### ML Predict (تک)
```json
{
  "features": {
    "xab_ratio_accuracy": 0.95,
    "abc_ratio_accuracy": 0.87,
    "bcd_ratio_accuracy": 0.82,
    "xad_ratio_accuracy": 0.91,
    "pattern_symmetry": 0.7,
    "pattern_slope": 0.1,
    "xa_angle": 35,
    "ab_angle": -20,
    "bc_angle": 15,
    "cd_angle": -30,
    "pattern_duration": 120,
    "xa_magnitude": 1.2,
    "ab_magnitude": 0.8,
    "bc_magnitude": 0.6,
    "cd_magnitude": 1.0,
    "volume_at_d": 1.1,
    "volume_trend": 0.5,
    "volume_confirmation": 0.8,
    "rsi_at_d": 55,
    "macd_at_d": 0.12,
    "momentum_divergence": 0.05
  },
  "timeout_seconds": 2.0
}
```

## 13) تنظیمات محیطی مهم (.env)
- `CACHE_ENABLED`, `REDIS_HOST`, `REDIS_PORT`
- `ENABLE_DATA_INGESTION` (فعال‌سازی رویداد و persist مستقیم)
- `KAFKA_ENABLED` یا `RABBITMQ_ENABLED` (انتشار رویداد)
- `EXPOSE_DB_EXPLORER`, `ENABLE_SCENARIOS`
- `DATABASE_URL` (اگر از PostgreSQL استفاده می‌کنید)
- `METRICS_ENABLED` (Prometheus)
- `JWT_SECRET_KEY`, `JWT_EXPIRATION_MINUTES` (برای auth دمو)

## 14) کش و پیام‌رسان
- اگر Redis فعال باشد، نتایج و داده‌های خوانده‌شده cache می‌شود (کاهش زمان پاسخ).
- اگر Kafka/RabbitMQ فعال باشد، رویداد `ANALYSIS_COMPLETED` منتشر می‌شود و ingestor می‌تواند آن را ذخیره کند.

## 15) بک‌تست
- اندپوینت: `POST /api/v1/backtest`
- ورودی: آرایه‌های `highs/lows/closes/volumes/dates` یا `symbol` برای دادهٔ واقعی TSE.
- کنترل کیفیت ورودی: چک طول، NaN/Inf، ترتیب زمانی، تطابق طول آرایه‌ها، حداقل میله (`max(window_size+step_size, 300)`).
- خروجی: متریک‌ها (win_rate, total_pnl, sharpe_ratio, drawdown، hit targets)، زمان تحلیل، نسخه مدل.
- `persist=true` → ذخیره در `backtest_runs` (در PostgreSQL یا SQLite بسته به config).

## 16) Health, Metrics, Observability
- `/health` و `/health/ready` (بررسی Redis و Service Discovery اگر فعال باشد).
- `/metrics` (Prometheus): شامل متریک‌های API، ML، بک‌تست.
- لاگ ساختاریافته با `structlog`.

## 17) ساخت Agent در n8n (گام‌به‌گام)
1. **Trigger**: Webhook (برای دریافت symbol/timeframe/candles) یا Schedule.
2. **Function** (تبدیل ورودی به بدنهٔ تحلیل):
   ```javascript
   const candles = $json.candles ?? [
     { timestamp: "2024-01-01T00:00:00Z", open: 43000, high: 43500, low: 42800, close: 43250, volume: 120000 },
     { timestamp: "2024-01-01T01:00:00Z", open: 43250, high: 43800, low: 43100, close: 43720, volume: 98000 }
   ];
   return [{ json: { symbol: $json.symbol ?? "BTCUSDT", timeframe: $json.timeframe ?? "1h", candles } }];
   ```
3. **HTTP Request**:
   - Method: `POST`
   - URL: `http://<host>:8000/api/v1/analyze`
   - Body: JSON = `{{$json}}`
4. **Switch / IF** روی `overall_signal` یا `decision.final_signal` برای شاخه‌های BUY/SELL/NEUTRAL.
5. **Set / Function** برای خلاصه‌سازی:
   ```javascript
   const r = $json;
   return [{
     json: {
       symbol: r.symbol,
       signal: r.overall_signal,
       confidence: r.overall_confidence,
       trend: r.trend_score?.score ?? null,
       momentum: r.momentum_score?.score ?? null,
       volume_interaction: r.volume_interaction_score ?? null
     }
   }];
   ```
6. **Action**: ارسال به Slack/Telegram/Email یا ذخیره در DB (مثلاً PostgreSQL نود).
7. **Respond to Webhook** (اگر Trigger وب‌هوک بود) با خلاصهٔ سیگنال.

### Agent پیشرفته در n8n (کارایی بالاتر)
- استفاده از `/api/v1/analyze/indicators` اگر فقط چند اندیکاتور خاص می‌خواهید (کاهش هزینه محاسبه).
- اسکن چند نماد: نود Split In Batches برای لیست symbols و اجرای موازی HTTP Request.
- هشدار حجم: اگر `volume_interaction_score < 0.3` آلارم احتیاط بدهید.
- هشدار تضاد: اگر `overall_signal` ≠ `trend_score.signal` → برچسب “دوراهی”.
- کش سمت n8n: اگر خروجی تغییر نکرد، پیام نفرستید (compare with last state in a KV).

## 18) راه‌اندازی سریع سرور (محلی)
```bash
pip install -r requirements.txt
copy .env.example .env
set PYTHONPATH=apps/analysis_api/src
uvicorn gravity_tech.main:app --host 0.0.0.0 --port 8000 --reload
```
- مستندات: `http://localhost:8000/api/docs`
- سلامت: `/health`, آمادگی: `/health/ready`
- متریک: `/metrics` (اگر فعال)

## 19) نکات عملیاتی و ترفندها
- حداقل ۶۰ کندل بدهید تا اندیکاتورهای بلندمدت کار کنند.
- برای دادهٔ زیاد، Redis را فعال کنید تا زمان پاسخ کاهش یابد.
- اگر مدل هارمونیک موجود نیست، مسیر `apps/analysis_api/ml_models/` را بررسی کنید؛ fallback فقط احتمال یکنواخت می‌دهد.
- برای PostgreSQL، قبل از اجرا شِما را با `scripts/schema/postgres_schema.sql` ایجاد کنید.
- `EXPOSE_DB_EXPLORER=true` را فقط در محیط امن فعال کنید (روتر `db_explorer` کوئری می‌گیرد).
- اگر ingestion فعال است و Kafka/RabbitMQ ندارید، سرویس به‌صورت مستقیم persist می‌کند (`data_ingestor.persist_direct`).

## 20) عیب‌یابی سریع
- خطای مدل: اگر لاگ `ml_model_fallback_used` دیدید، فایل مدل موجود نیست یا hash نمی‌خواند.
- خطای اندیکاتور: اگر تعداد کندل < 60 باشد، 400 برمی‌گردد.
- کندی پاسخ: Redis را فعال کنید و تعداد کندل را محدود کنید؛ در n8n از اندپوینت سبک‌تر استفاده کنید.
- خطای بک‌تست: بررسی کنید طول آرایه‌ها برابر و بدون NaN/Inf باشد؛ حداقل `max(window_size+step_size, 300)` میله لازم است.

## 21) مسیرهای مهم فایل‌ها (برای توسعه)
- API و روترها: `apps/analysis_api/src/gravity_tech/api/v1/*.py`
- سرویس تحلیل: `gravity_tech/services/analysis_service.py`
- تصمیم‌ساز: `gravity_tech/services/signal_engine.py`
- کش: `gravity_tech/services/cache_service.py`
- الگوهای هارمونیک: `gravity_tech/patterns/harmonic.py`
- مدل و ویژگی‌ها: `gravity_tech/ml/pattern_features.py`, `ml/pipeline_factory.py`
- خط لوله آفلاین: `scripts/etl/run_full_pipeline.py`
- شِمای DB: `scripts/schema/postgres_schema.sql`
- اسکریپت‌های کمکی: `scripts/exports`, `scripts/data_population`, `scripts/backtesting`, `scripts/maintenance`

## 22) خلاصه کوتاه
این سرویس یک موتور تحلیل تکنیکال چندبُعدی + تشخیص هارمونیک با امتیاز ML و بک‌تست است. ورودی OHLCV (از API یا DB)، خروجی سیگنال نهایی با اطمینان، ذخیره در DB و امکان انتشار رویداد. می‌توانید در n8n با چند نود ساده آن را به Agent تبدیل کنید و بر اساس `overall_signal/decision.final_signal` هشدار بفرستید یا اتوماسیون اجرا کنید.
