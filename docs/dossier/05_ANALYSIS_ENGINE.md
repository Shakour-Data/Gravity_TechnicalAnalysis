# موتور تحلیل و منطق محاسبه سیگنال (Analysis Engine)

## 1) نقطه ورود تحلیل
سرویس تحلیل: `apps/analysis_api/src/gravity_tech/services/analysis_service.py`

Pipeline تحلیل کامل در سطح سرویس:
1. (اختیاری) مسیر سریع (Fast Indicators) برای چند اندیکاتور پرتکرار
2. محاسبه اندیکاتورهای Trend
3. محاسبه اندیکاتورهای Momentum
4. محاسبه اندیکاتورهای Cycle
5. محاسبه اندیکاتورهای Volume
6. محاسبه اندیکاتورهای Volatility
7. محاسبه Support/Resistance
8. تشخیص الگوهای کندلی
9. تحلیل Elliott Wave
10. تحلیل Market Phase (Dow Theory)
11. تجمیع سیگنال‌ها و تولید `overall_signal` و `overall_confidence`

```mermaid
flowchart TB
  Req[AnalysisRequest: candles>=60] --> Fast[FastBatchAnalyzer (اختیاری)]
  Req --> Trend[TrendIndicators.calculate_all]
  Req --> Mom[MomentumIndicators.calculate_all]
  Req --> Cyc[CycleIndicators.calculate_all]
  Req --> Vol[VolumeIndicators.calculate_all]
  Req --> Vola[VolatilityIndicators.calculate_all]
  Req --> SR[SupportResistanceIndicators.calculate_all]
  Req --> CandleP[CandlestickPatterns.detect_patterns]
  Req --> EW[analyze_elliott_waves]
  Req --> Phase[analyze_market_phase]
  Trend --> Agg[compute_overall_signals]
  Mom --> Agg
  Cyc --> Agg
  Vol --> Agg
  Agg --> Out[TechnicalAnalysisResult]
```

## 2) قرارداد خروجی اندیکاتورها (IndicatorResult)
کلاس‌های دامنه: `apps/analysis_api/src/gravity_tech/core/domain/entities/*`

الگوی خروجی هر اندیکاتور:
- `indicator_name` (مثلاً `SMA(20)`)
- `category` (Trend/Momentum/…)
- `signal` (مقادیر استاندارد `SignalStrength`)
- `value` (مقدار خام اندیکاتور)
- `confidence` (۰..۱)
- `description` (توضیح انسانی)
- (اختیاری) `additional_values` برای اجزای داخلی (مثل bandها/VI+/VI-)

## 3) موتور تجمیع سیگنال (Signal Engine)
پیاده‌سازی مرجع: `apps/analysis_api/src/gravity_tech/services/signal_engine.py`

### 3.1) امتیاز هر دسته (Category Score)
برای هر دسته، میانگین وزنی امتیاز سیگنال‌ها محاسبه می‌شود:
- `signal_score = signal.get_score()` (نگاشت سیگنال به عدد)
- `category_score = Σ(signal_score * confidence) / Σ(confidence)`

همچنین «دقت/کیفیت» دسته به‌صورت میانگین confidence محاسبه می‌شود:
- `category_accuracy = Σ(confidence) / N`

### 3.2) وزن‌های پایه و تطبیق وزن‌ها
وزن پایه:
- Trend: `0.30`
- Momentum: `0.25`
- Cycle: `0.25`
- Volume: `0.20`

سپس وزن‌ها بر اساس `category_accuracy` نرمال می‌شوند تا دسته‌هایی که خروجی مطمئن‌تری دارند سهم بیشتری بگیرند.

### 3.3) ترکیب نهایی و نقش Volume
ابتدا امتیاز کلی از Trend/Momentum/Cycle ساخته می‌شود و سپس Volume نقش «تأیید/واگرایی» بازی می‌کند:
- اگر جهت Volume هم‌جهت باشد → تقویت امتیاز
- اگر خلاف جهت باشد → تضعیف امتیاز

در نهایت امتیاز به بازه `[-2,+2]` clip و سپس به `[-1,+1]` نرمال می‌شود.

### 3.4) اعتماد نهایی (overall_confidence)
اعتماد از دو مؤلفه ساخته می‌شود:
- **Agreement**: هرچه پراکندگی سیگنال‌ها کمتر باشد (stddev)، توافق بیشتر است.
- **Mean confidence**: میانگین confidence همه اندیکاتورها.

## 4) مسیر «اندیکاتورهای انتخابی»
Endpoint: `POST /api/v1/analyze/indicators`
- اجازه می‌دهد فقط چند اندیکاتور مشخص محاسبه شود.
- از مسیر سریع (Fast) هم در صورت فعال بودن استفاده می‌کند.

