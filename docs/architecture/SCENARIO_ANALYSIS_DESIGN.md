# Three-Scenario Analysis Design (تحلیل سه‌سناریویی)

**Document Version:** 1.0  
**Created:** November 14, 2025  
**Author:** Dr. James Richardson (Chief Quantitative Analyst)  
**Reviewed By:** Shakour Alishahi (CTO)

---

## 🎯 Overview

این میکروسرویس باید برای هر نماد در **سه سناریو** تحلیل ارائه دهد:

1. **خوشبینانه (Optimistic)** - بهترین حالت ممکن
2. **خنثی (Neutral)** - حالت متعادل و واقع‌بینانه
3. **بدبینانه (Pessimistic)** - بدترین حالت ممکن

---

## 📊 Scenario Analysis Framework

### **1. Optimistic Scenario (سناریو خوشبینانه)**

**فرضیات:**
- همه سیگنال‌های مثبت تأیید می‌شوند
- حجم معاملات افزایش می‌یابد
- روند صعودی قوی ادامه دارد
- شکست مقاومت‌ها موفقیت‌آمیز است
- نسبت ریسک به ریوارد بهینه است (1:3+)

**محاسبات:**
```python
optimistic_score = (
    trend_score * 1.2 +          # وزن بیشتر به روند
    momentum_score * 1.3 +       # وزن بیشتر به مومنتوم
    volume_score * 1.1 +         # تأیید حجمی
    pattern_score * 1.2 +        # الگوهای صعودی
    support_resistance_score * 0.9
) / 5.7

target_price_optimistic = current_price * (1 + (atr_percentage * 3))
stop_loss_optimistic = current_price * (1 - (atr_percentage * 0.5))
```

**سیگنال‌های کلیدی:**
- ✅ Golden Cross (SMA50 > SMA200)
- ✅ RSI بین 50-70 (صعودی اما نه overbought)
- ✅ MACD Bullish Crossover
- ✅ Volume > Average Volume * 1.5
- ✅ Breakout از مقاومت با حجم بالا
- ✅ Elliott Wave موج 3 یا 5
- ✅ Bullish Candlestick Patterns

**احتمال موفقیت:** 65-75%  
**Risk/Reward:** 1:3 یا بهتر

---

### **2. Neutral Scenario (سناریو خنثی)**

**فرضیات:**
- سیگنال‌های مختلط (مثبت و منفی)
- حجم معاملات متعادل
- روند نامشخص یا رنج
- احتمال موفقیت متوسط
- نسبت ریسک به ریوارد متعادل (1:1.5)

**محاسبات:**
```python
neutral_score = (
    trend_score * 1.0 +
    momentum_score * 1.0 +
    volume_score * 1.0 +
    pattern_score * 1.0 +
    support_resistance_score * 1.0
) / 5.0

target_price_neutral = current_price * (1 + (atr_percentage * 1.5))
stop_loss_neutral = current_price * (1 - (atr_percentage * 1.0))
```

**سیگنال‌های کلیدی:**
- ⚠️ روند نامشخص یا sideways
- ⚠️ RSI بین 40-60 (خنثی)
- ⚠️ MACD نزدیک به خط صفر
- ⚠️ Volume معمولی
- ⚠️ قیمت در محدوده support-resistance
- ⚠️ سیگنال‌های متناقض از اندیکاتورها

**احتمال موفقیت:** 45-55%  
**Risk/Reward:** 1:1.5

---

### **3. Pessimistic Scenario (سناریو بدبینانه)**

**فرضیات:**
- سیگنال‌های منفی غالب هستند
- حجم معاملات کاهشی
- روند نزولی قوی
- شکست حمایت‌ها
- نسبت ریسک به ریوارد نامطلوب

**محاسبات:**
```python
pessimistic_score = (
    trend_score * 0.8 +          # وزن کمتر به روند ضعیف
    momentum_score * 0.7 +       # مومنتوم منفی
    volume_score * 0.9 +         # حجم کاهشی
    pattern_score * 0.8 +        # الگوهای نزولی
    support_resistance_score * 1.1  # اهمیت بیشتر به حمایت‌ها
) / 4.3

target_price_pessimistic = current_price * (1 + (atr_percentage * 0.5))
stop_loss_pessimistic = current_price * (1 - (atr_percentage * 1.5))
```

**سیگنال‌های کلیدی:**
- ❌ Death Cross (SMA50 < SMA200)
- ❌ RSI < 30 (oversold شدید)
- ❌ MACD Bearish Crossover
- ❌ Volume کاهشی در صعودها
- ❌ Breakdown از حمایت
- ❌ Elliott Wave موج A-B-C نزولی
- ❌ Bearish Candlestick Patterns

**احتمال موفقیت:** 25-35%  
**Risk/Reward:** 1:0.5 (نامطلوب)

---

## 🎲 Probability Weighting

هر سناریو یک **احتمال وقوع** دارد:

```python
probabilities = {
    "optimistic": calculate_optimistic_probability(),  # 0-100%
    "neutral": calculate_neutral_probability(),         # 0-100%
    "pessimistic": calculate_pessimistic_probability()  # 0-100%
}

# مجموع احتمالات = 100%
total = sum(probabilities.values())
normalized_probabilities = {k: (v/total)*100 for k, v in probabilities.items()}
```

**محاسبه احتمالات بر اساس:**
- تعداد سیگنال‌های مثبت/منفی
- قدرت روند
- تأیید حجمی
- کیفیت الگوها
- موقعیت قیمت نسبت به support/resistance

---

## 📈 Expected Value Calculation

```python
expected_return = (
    (optimistic_return * prob_optimistic) +
    (neutral_return * prob_neutral) +
    (pessimistic_return * prob_pessimistic)
)

expected_risk = (
    (optimistic_risk * prob_optimistic) +
    (neutral_risk * prob_neutral) +
    (pessimistic_risk * prob_pessimistic)
)

risk_adjusted_score = expected_return / expected_risk
```

---

## 🔧 Implementation

### **File Structure:**
```
src/gravity_tech/
├── analysis/
│   ├── scenario_analysis.py       # NEW - اصلی
│   ├── optimistic_analyzer.py     # NEW
│   ├── neutral_analyzer.py        # NEW
│   ├── pessimistic_analyzer.py    # NEW
│   └── probability_calculator.py  # NEW
```

### **API Endpoint:**
```python
POST /api/v1/analysis/scenarios
{
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "candles": [...],
    "include_probabilities": true
}

Response:
{
    "symbol": "BTCUSDT",
    "timestamp": "2025-11-14T12:00:00Z",
    "scenarios": {
        "optimistic": {
            "score": 85.5,
            "probability": 35.0,
            "target_price": 45000,
            "stop_loss": 42000,
            "risk_reward": 3.0,
            "key_signals": ["golden_cross", "bullish_macd", "high_volume"],
            "recommendation": "STRONG_BUY"
        },
        "neutral": {
            "score": 55.0,
            "probability": 45.0,
            "target_price": 43500,
            "stop_loss": 41500,
            "risk_reward": 1.5,
            "key_signals": ["sideways_trend", "neutral_rsi"],
            "recommendation": "HOLD"
        },
        "pessimistic": {
            "score": 25.5,
            "probability": 20.0,
            "target_price": 41000,
            "stop_loss": 38500,
            "risk_reward": 0.5,
            "key_signals": ["bearish_divergence", "low_volume"],
            "recommendation": "AVOID"
        }
    },
    "expected_value": {
        "return": 4.5,  # درصد بازدهی مورد انتظار
        "risk": 2.8,    # درصد ریسک مورد انتظار
        "sharpe_ratio": 1.61
    },
    "recommended_scenario": "optimistic",
    "confidence_level": "MEDIUM-HIGH"
}
```

---

## ✅ Success Criteria

1. **Coverage:** همه نمادها در هر 3 سناریو تحلیل شوند
2. **Accuracy:** احتمالات با واقعیت بازار تطابق داشته باشد (backtesting)
3. **Performance:** محاسبه هر سناریو < 5ms
4. **Interpretability:** توضیحات واضح برای هر سناریو

---

**Team Assignment:**
- **Dr. Richardson:** طراحی ریاضی و فرمول‌ها
- **Dr. Patel:** ML برای محاسبه احتمالات
- **Prof. Dubois:** تعریف سیگنال‌های تکنیکال
- **Shakour:** تأیید نهایی از منظر trading

---

**Status:** 🔴 در حال طراحی  
**Priority:** 🔥 CRITICAL  
**ETA:** 3-5 روز کاری
