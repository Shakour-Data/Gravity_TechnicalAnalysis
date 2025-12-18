# Batch-50 Pipeline (TSETMC -> Fetch -> Analysis -> Postgres)

این سند توضیح می‌دهد در هر بچ چه اتفاقی می‌افتد، چطور اجرا کنید، و چطور بعد از هر بچ گزارش بگیرید.

## ورودی‌ها
- **منبع زنده (TSETMC)** از طریق `gravity_tse.py` + `finpy_tse`.
- **کش سورس (SQLite)**: ترجیحاً `E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db` (~3.2M رکورد). جایگزین: `temp_gravity_tse/data/tse_data.db`.
- **هدف (Postgres/Docker)**: DSN پروژه `postgresql://gravity:gravity@127.0.0.1:5544/tech_analysis` (داخل کانتینر: `postgresql://gravity:gravity@db:5432/tech_analysis`). اگر تنظیم نشده باشد، کد پیش‌فرض را `postgresql://postgres:Bedaan4D@127.0.0.1:5432/bedaan4d_db` استفاده می‌کند. می‌توانید با `--target-db` یا envهای `ANALYSIS_TARGET_DB` / `DATABASE_URL` عوض کنید.
- **پارامترها**: `--batch-size` (پیش‌فرض 50)، `--min-candles` (پیش‌فرض 400)، `--limit` (تعداد کندل تحلیل؛ 0 = کل تاریخچه)، `--ingest-limit` (برای اینجست جداول تاریخی)، `--no-indices`, `--no-usd`, `--loop`.

## تنها اسکریپت پیشنهادی (یک‌مرحله‌ای، فچ + تحلیل + پر شدن همه جداول)
فقط از `scripts/etl/run_batch50_full_ingest.py` استفاده کنید؛ بقیه اسکریپت‌ها (مثلاً `run_batch50.py`) را اجرا نکنید.
```bash
python scripts/etl/run_batch50_full_ingest.py ^
  --source-db "E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db" ^
  --target-db "postgresql://gravity:gravity@127.0.0.1:5544/tech_analysis" ^
  --batch-size 50 ^
  --min-candles 120 ^
  --limit 500 ^
  --ingest-limit 500
```
- اول، نمادهایی که قبلاً در `analysis_results` یا `historical_scores` هستند را رد می‌کند تا هر بار به بچ بعدی بروید.
- سپس فچ از TSETMC، اجرای تحلیل، و پر کردن تمام جداول کلیدی در یک اجرا: `analysis_results`, `historical_scores`, `historical_indicator_scores`, `tool_performance_history`, `backtest_runs`, `pattern_detection_results`, `ml_weights_history`.

## اجرای کلاسیک (Deprecated)
`run_batch50.py` دیگر توصیه نمی‌شود؛ فقط برای سازگاری نگه داشته شده است و نباید اجرا شود.

## مراحل هر بچ
1) انتخاب نمادها: بر اساس `last_updates` و تعداد کندل (حداقل `min_candles`)، قدیمی‌ترین‌ها اول. در اسکریپت full-ingest، نمادهای موجود در Postgres رد می‌شوند.
2) فچ TSETMC: فقط نمادهای همان بچ در `price_data`/`last_updates` ثبت می‌شوند. USD و شاخص‌های بازار/صنعت پیش‌فرض فعال هستند مگر `--no-usd`/`--no-indices`.
3) تحلیل: اجرای `scripts/etl/run_full_pipeline.py --symbols ...` و نوشتن خروجی در دیتابیس هدف.
4) اینجست تاریخی (فقط در full-ingest): پر کردن جداول سری‌زمانی برای همان نمادها.

## گزارش بعد از هر بچ
1) لیست نمادهای بچ را در یک فایل (هر خط یک نماد) ذخیره کنید، مثال: `batch1_symbols.txt`.
2) دستور گزارش‌گیری:
```bash
python scripts/etl/report_batch.py ^
  --target-db "postgresql://gravity:gravity@127.0.0.1:5544/tech_analysis" ^
  --symbols-file batch1_symbols.txt ^
  --outfile batch1_report.txt
```
- خروجی شامل تعداد ردیف/نماد در هر جدول، پوشش ۹۰ روزه `analysis_results`، و نمادهای ناقص (اگر باشند) است. بعد از هر بچ این فایل گزارش را برای مستندسازی نگه دارید.

## چک‌های سریع
- اگر نمادی غایب است، بررسی کنید حداقل `min_candles` داده در سورس داشته باشد.
- اگر Postgres خالی است، DSN را مطابق کانتینر (`gravity:gravity@127.0.0.1:5544/tech_analysis`) تنظیم کنید.
- برای ادامه بچ‌ها، همان دستور full-ingest را تکرار کنید؛ نمادهای قبلی را رد می‌کند و سراغ ۵۰ نماد بعدی می‌رود.
