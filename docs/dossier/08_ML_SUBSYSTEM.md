# زیرسیستم یادگیری ماشین (ML Subsystem)

## 1) هدف ML در این پروژه
در نسخه فعلی، ML عمدتاً برای **امتیازدهی/طبقه‌بندی الگوهای هارمونیک** و ارائه confidence استفاده می‌شود.

## 2) API و مدل‌ها
- API: `apps/analysis_api/src/gravity_tech/api/v1/ml.py`
- مدل‌ها: `ml_models/pattern_classifier_advanced_v2.pkl` و `ml_models/pattern_classifier_v1.pkl`
- استراتژی بارگذاری:
  - کش در حافظه + Hash فایل برای تشخیص تغییر
  - fallback از v2 به v1 در صورت نبودن v2

## 3) ویژگی‌ها (Features) برای طبقه‌بندی الگو
کلاس ورودی: `PatternFeatures` در `api/v1/ml.py` شامل ۲۱ ویژگی از جمله:
- دقت نسبت‌های XAB/ABC/BCD/XAD
- symmetry/slope/angles/duration/magnitudes
- ویژگی‌های حجمی (volume_at_d, volume_trend, volume_confirmation)
- ویژگی‌های تکنیکال (rsi_at_d, macd_at_d, momentum_divergence)

## 4) کنترل کیفیت ورودی ML
در `PatternFeatures` اعتبارسنجی‌هایی وجود دارد:
- محدودیت بازه (۰..۱ برای برخی ویژگی‌ها)
- finite بودن مقادیر (عدم NaN/Inf)
- محدودیت تعداد درخواست batch و timeouts

## 5) متریک‌ها و مشاهده‌پذیری
در صورت فعال بودن Prometheus:
- `ml_model_cache_hits_total`
- `ml_model_loads_total`
- `ml_prediction_requests_total`
- `ml_prediction_latency_seconds`
- `ml_backtest_requests_total`
- `ml_backtest_latency_seconds`

## 6) رفتار در نبود مدل
اگر فایل مدل موجود نباشد:
- endpointهای ML با 503 پاسخ می‌دهند.
- endpoint الگوها می‌تواند `ml_status` را `not_available` گزارش کند و بدون ML خروجی بدهد (بسته به مسیر اجرا).

