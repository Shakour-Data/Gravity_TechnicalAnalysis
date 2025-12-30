# تست و اعتبارسنجی (Testing & Validation)

## 1) تست‌های سرویس
راهنمای تست در پروژه:
- `apps/analysis_api/tests/PHASE_4_TESTING_GUIDE.md`

## 2) تست‌های اسکریپت‌های ETL
مسیر: `scripts/etl/tests/*`

## 3) گزارش‌های اعتبارسنجی
گزارش‌ها در `docs/reports/` تولید می‌شوند.

- گزارش صحت جداول/رنج زمانی/duplicate:
  - اسکریپت: `scripts/etl/auto_validate_report.py`
  - خروجی: `docs/reports/validation_report.md`

- گزارش باز-محاسبه و مقایسه trend/vol:
  - اسکریپت: `scripts/etl/recompute_validation.py`
  - خروجی: `docs/reports/recompute_report.md`

## 4) پیشنهادات تکمیلی برای بسته ارائه
- افزودن نمودار Coverage/کیفیت (از CI یا خروجی pytest)
- افزودن نمونه خروجی JSON از `/api/v1/analyze` و `/api/v1/patterns/detect` برای پیوست

