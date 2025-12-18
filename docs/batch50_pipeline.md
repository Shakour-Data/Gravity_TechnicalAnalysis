# پایپ‌لاین بچ ۵۰تایی (TSETMC ← فچ ← تحلیل ← ذخیره در Postgres)

این سند تنها اسکریپت پیشنهادی را توضیح می‌دهد و می‌گوید هر بچ دقیقاً چه داده‌هایی را به‌صورت سری زمانی در دیتابیس ذخیره می‌کند.

## ورودی‌ها
- **منبع زنده (TSETMC)** از طریق `gravity_tse.py` و `finpy_tse`.
- **کش سورس (SQLite)**: ترجیحاً `E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db` (~3.2M رکورد). جایگزین: `temp_gravity_tse/data/tse_data.db`.
- **هدف (Postgres/Docker)**: DSN توصیه‌شده `postgresql://gravity:gravity_db_pass@127.0.0.1:5545/tech_analysis` (داخل کانتینر: `postgresql://gravity:gravity_db_pass@postgres:5432/tech_analysis`). همه اسکریپت‌ها را روی این DSN تنظیم کنید.
- **پارامترها**: `--batch-size` (پیش‌فرض 50)، `--min-candles` (پیش‌فرض 400)، `--limit` (محدودیت کندل برای تحلیل)، `--ingest-limit` (محدودیت کندل برای اینجست سری زمانی؛ 0 یعنی کل تاریخچه)، `--no-indices`, `--no-usd`, `--loop`.

## تنها اسکریپت پیشنهادی (یک‌مرحله‌ای: فچ + تحلیل + اینجست کامل)
```bash
python scripts/etl/run_batch50_full_ingest.py ^
  --source-db "E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db" ^
  --target-db "postgresql://gravity:gravity_db_pass@127.0.0.1:5545/tech_analysis" ^
  --batch-size 50 ^
  --min-candles 120 ^
  --limit 500 ^
  --ingest-limit 0 ^
  --trend-window 30
```
- نمادهایی که در `analysis_results` یا `historical_scores` مقصد هستند، رد می‌شوند؛ هر بار که اجرا کنید به ۵۰ نماد بعدی می‌رود.
- سری زمانی برای همه جداول در یک اجرا ذخیره می‌شود (per-symbol, per-day): `analysis_results`, `historical_scores`, `historical_indicator_scores`, `tool_performance_history`, `backtest_runs`, `pattern_detection_results`, `ml_weights_history`.
- محاسبه trend/volatility بر اساس پنجره‌ی `--trend-window` (پیش‌فرض 30 روز) روی همان کندل‌ها انجام می‌شود.
- در شروع اجرا ایندکس/قیود کمکی ساخته می‌شود:  
  - `historical_scores(symbol, ts, timeframe)`  
  - `historical_indicator_scores(symbol, ts, timeframe)`  
  - `ml_weights_history` یکتا `(symbol, ts, model_name, timeframe)`  
  - `tool_performance_history` یکتا `(symbol, timeframe, prediction_timestamp, tool_name)`  
  - `backtest_runs` یکتا `(symbol, interval, period_start, period_end, model_version)`  
  - `pattern_detection_results` یکتا `(symbol, timeframe, timestamp, pattern_type, pattern_name)`

## مراحل هر بچ
1) انتخاب نمادها: قدیمی‌ترین `last_date` در سورس، فیلتر `min_candles`، حذف نمادهای پردازش‌شده در مقصد.
2) فچ TSETMC: فقط همان نمادها + USD + شاخص‌ها (مگر `--no-usd`/`--no-indices`).
3) تحلیل: اجرای `run_full_pipeline.py` برای همان نمادها.
4) اینجست سری زمانی: برای هر روز، برای هر نماد همه جداول بالا پر می‌شود؛ وزن‌ها (ml_weights_history) هم per-symbol، per-day هستند.

## گزارش/اعتبارسنجی پس از هر بچ
1) لیست نمادهای بچ را در یک فایل (هر خط یک نماد) ذخیره کنید، مثال: `batch1_symbols.txt`.
2) دستور چک خودکار:
```bash
python scripts/etl/validate_batch.py ^
  --target-db "postgresql://gravity:gravity_db_pass@127.0.0.1:5545/tech_analysis" ^
  --source-db "E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db" ^
  --symbols-file batch1_symbols.txt ^
  --limit 300
```
- خروجی: شمارش ردیف/نماد در هر جدول، بازه زمانی، و پوشش کندل‌های سورس (تا `limit`). اگر نمادی در جدولی غایب باشد، گزارش می‌شود.

## نکات
- برای پوشش کامل تاریخچه، `--ingest-limit` را 0 بگذارید (پیش‌فرض جدید).
- DSN قدیمی (postgres/Bedaan4D@5432) را استفاده نکنید؛ همه جا DSN بالا (gravity_db_pass@5545) را ست کنید.
- اگر می‌خواهید سرعت بیشتر باشد، می‌توانید `--ingest-limit` را عددی بگذارید؛ اما تاریخچه کوتاه‌تر خواهد شد. پس از آن می‌توانید دوباره با limit بالاتر اینجست کنید؛ قیود یکتا مانع داده تکراری می‌شود.
