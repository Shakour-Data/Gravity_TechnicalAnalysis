# تعریف محصول و دامنه (Product & Scope)

## 1) مسئله و نیاز
در تصمیم‌گیری‌های بازار، تحلیل‌گر نیاز دارد:
- وضعیت روند، مومنتوم، نوسان، حجم و سطوح کلیدی را همزمان ببیند.
- خروجی قابل اتکا (Confidence) دریافت کند.
- بتواند تحلیل را برای تعداد زیادی نماد، به‌صورت Batch و قابل ذخیره‌سازی تولید کند.

این پروژه با یک API استاندارد، ورودی OHLCV را گرفته و تحلیل چندلایه تولید می‌کند.

## 2) ورودی‌ها
### ورودی اصلی تحلیل
- `symbol` (نماد)
- `timeframe` (۱m … ۱w)
- `candles[]`: هر آیتم شامل `timestamp/open/high/low/close/volume`

### ورودی جایگزین (historical)
برخی endpointها داده تاریخی را از دیتابیس محلی/پس‌زمینه می‌خوانند (TSE).

## 3) خروجی‌ها
- سیگنال نهایی: `overall_signal` (از نوع `SignalStrength`) + `overall_confidence` (۰..۱)
- خروجی‌های لایه‌ای: Trend/Momentum/Cycle/Volume/Volatility/Support-Resistance
- خروجی‌های تحلیلی تکمیلی: Candlestick Patterns، Elliott Wave Analysis، Market Phase
- خروجی ذخیره‌سازی/گزارش: جداول `tech_analysis.*` در Postgres

## 4) محدودیت‌ها و فرض‌ها
- برای تحلیل قابل‌اتکا، حداقل ۶۰ کندل توصیه/اجبار می‌شود (در API enforce شده است).
- در حالت بدون Data Service، تحلیل روی همان کندل‌های ارسال‌شده انجام می‌شود.
- برخی اجزا (Events/Kafka/RabbitMQ/Scenarios) قابل فعال‌سازی با تنظیمات هستند.

