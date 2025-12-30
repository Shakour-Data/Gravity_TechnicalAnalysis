# پایپ‌لاین داده و ذخیره‌سازی (Data Pipeline & Storage)

## 1) اهداف لایه داده
- دریافت داده بازار (TSE) و نگهداری محلی/کش.
- انتقال داده به Postgres به‌عنوان مخزن مشترک (تحلیل + داده خام).
- محاسبه Batch امتیازها/اندیکاتورها در بازه زمانی و ذخیره برای گزارش‌گیری/ML.

## 2) پایپ‌لاین یکپارچه (Ingestion → Schema → Migration → Analysis)
اسکریپت مرجع: `scripts/etl/run_stack_pipeline.py`

```mermaid
flowchart LR
  A[services/data_ingestion CLI] --> B[(SQLite: services/data_ingestion/data/tse_data.db)]
  B --> C[SQLite→Postgres Migration]
  C --> D[(Postgres: tse_input.*)]
  D --> E[Batch Analysis (compute_daily_scores)]
  E --> F[(Postgres: tech_analysis.*)]
```

### 2.1) ایجاد/به‌روزرسانی اسکیمای Postgres
فایل اسکیمای مرجع: `scripts/schema/postgres_schema.sql`
- اسکیمای `tse_input` برای داده خام (شرکت‌ها، قیمت‌ها، شاخص‌ها)
- اسکیمای `tech_analysis` برای خروجی تحلیل و artifactهای ML

### 2.2) جداول مهم در `tse_input`
- `tse_input.companies` (متادیتای نماد)
- `tse_input.price_data` (Adjusted OHLCV روزانه) با کلید یکتا `(symbol, trading_date)`

### 2.3) جداول مهم در `tech_analysis`
- `tech_analysis.historical_scores`: امتیازهای تجمیعی (trend/momentum/volume/volatility/cycle/…)
- `tech_analysis.historical_indicator_scores`: ذخیره جزئیات اندیکاتورها (نام/پارامتر/مقدار)
- `tech_analysis.pattern_detection_results`: نتایج Pattern با متادیتا/هدف/حدضرر
- `tech_analysis.ml_weights_history`: تاریخچه وزن‌ها/دقت مدل/پیکربندی
- `tech_analysis.backtest_runs`: نتایج اجرای بک‌تست

## 3) سیاست‌های کیفیت داده (Data Quality)
در کل پروژه چند اصل رعایت می‌شود:
- اعتبارسنجی OHLCV: NaN/Inf، `high>=low`، ترتیب زمانی، حداقل تعداد کندل
- جلوگیری از تکرار (Dedup): کلیدهای یکتا روی `(symbol,timeframe,ts)` یا ترکیب‌های مشابه
- ثبت متادیتا برای ردیابی: `created_at`, `updated_at`, `metadata JSONB`

## 4) گزارش‌گیری/اعتبارسنجی Batch
اسکریپت‌های گزارش (خروجی در `docs/reports/`):
- `scripts/etl/auto_validate_report.py` → گزارش اعتبارسنجی جداول/رنج زمانی/duplicate
- `scripts/etl/recompute_validation.py` → مقایسه باز-محاسبه trend/vol با دیتابیس

