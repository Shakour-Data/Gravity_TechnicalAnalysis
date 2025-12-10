# نمایه مستندات

این فهرست فقط مستندات منطبق با کد فعلی را پوشش می‌دهد. نسخه سرویس: **1.0.0**.

## شروع سریع و API
- `docs/guides/QUICK_START.md` — نصب، پیکربندی و اولین درخواست.
- `docs/guides/API_REFERENCE.md` — فهرست endpointها و ورودی/خروجی‌های کلیدی.

## معماری و طراحی
- `docs/architecture/SYSTEM_ARCHITECTURE_DIAGRAMS.md` — نمای کلی معماری، وابستگی‌ها و جریان درخواست (Mermaid).
- `docs/architecture/DATA_SERVICE_INTEGRATION.md` — جزئیات ارتباط با Data Service و کش Redis.
- `docs/architecture/SIGNAL_CALCULATION.md` — منطق وزن‌دهی و محاسبه سیگنال کلی.

## تحلیل و مدل تصمیم
- `docs/guides/FIVE_DIMENSIONAL_DECISION_GUIDE.md` — خط لوله کامل، ۵ بعد پایه، ماتریس حجم و خروجی ۵بعدی.
- `docs/guides/VOLUME_MATRIX_GUIDE.md` — نحوه تعامل حجم با Trend/Momentum/Volatility/Cycle/SR و انواع تأیید/واگرایی.
- `docs/guides/TREND_ANALYSIS_SUMMARY.md` — خلاصه اندیکاتورهای پیاده‌سازی‌شده و دسته‌بندی‌ها.

## عملیات
- `docs/operations/DEPLOYMENT_GUIDE.md` — اجرای محلی/تولید، متغیرهای محیطی مهم.
- `docs/operations/RUNBOOK.md` — چک‌های سلامت، پاک‌سازی کش، خطاهای رایج و راه‌حل سریع.
- `docs/PROCESS_OVERVIEW.md` — خلاصه همه فرایندها با دیاگرام (تحلیل، الگو، ML، ابزار، بک‌تست، سناریو، کش/داده، ingestion).
- `docs/processes/TECHNICAL_ANALYSIS.md` — فرایند کامل /api/v1/analyze با دیاگرام و وابستگی‌ها.
- `docs/processes/PATTERN_DETECTION.md` — تشخیص هارمونیک با ML اختیاری.
- `docs/processes/ML_PREDICTION.md` — پیش‌بینی ML، کش مدل، خطاها.
- `docs/processes/TOOL_RECOMMENDATION.md` — توصیه ابزار و تحلیل سفارشی.
- `docs/processes/BACKTEST.md` — بک‌تست تشخیص الگو.
- `docs/processes/FIVE_DIMENSIONAL_DECISION.md` — تصمیم‌گیر ۵بعدی و ماتریس حجم.
- `docs/processes/SCENARIO_ANALYSIS.md` — سناریو سه‌گانه (اختیاری).
- `docs/processes/DATA_CACHE.md` — کش/سرویس داده.
- `docs/processes/INGESTION_METRICS.md` — ingestion و متریک‌ها.

## ساختار و تغییرات
- `docs/PROJECT_STRUCTURE.md` — درخت پوشه‌ها و اجزای اصلی کد.
- `docs/changelog/CHANGELOG.md` — تغییرات واقعی نسخه 1.0.0 و پاک‌سازی مستندات.

## پوشش قابلیت‌های موجود در کد
- FastAPI با مسیرهای تحلیل تکنیکال، تشخیص الگو، ML، توصیه ابزار، بک‌تست و اکسپلورر دیتابیس.
- موتور تحلیل کلاسیک (`TechnicalAnalysisService`) با ۶ دسته اندیکاتور + الگوهای شمعی + الیوت + فاز بازار و سیگنال نهایی.
- تشخیص الگوهای هارمونیک (۴ الگو) با امتیازدهی ML اختیاری و مدل cache شده.
- توصیه ابزار و تحلیل سفارشی روی کاتالوگ ۹۵+ ابزار با وزن‌دهی ML.
- بک‌تست سبک و ذخیره نتایج در DB در صورت فعال‌سازی ingestion.
- کش Redis و یکپارچه‌سازی اختیاری با سرویس داده.
- سناریو سه‌گانه و DB Explorer اختیاری هستند و با پرچم‌های `ENABLE_SCENARIOS` و `EXPOSE_DB_EXPLORER` فعال می‌شوند.
