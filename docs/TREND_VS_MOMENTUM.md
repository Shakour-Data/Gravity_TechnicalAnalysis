# سیستم Multi-Horizon: Trend vs Momentum

## 📊 دو سیستم تحلیل مستقل

### 1️⃣ تحلیل روند (Trend Analysis)

**هدف**: تشخیص جهت کلی بازار

**اندیکاتورها** (10 عدد):
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- WMA (Weighted Moving Average)
- DEMA (Double Exponential Moving Average)
- TEMA (Triple Exponential Moving Average)
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)
- Parabolic SAR
- Supertrend
- Ichimoku Cloud

**فایل‌های مرتبط**:
- `indicators/trend.py` - محاسبه اندیکاتورها
- `ml/multi_horizon_feature_extraction.py` - استخراج ویژگی
- `ml/multi_horizon_analysis.py` - تحلیل و امتیازدهی
- `ml/train_multi_horizon.py` - آموزش مدل

**خروجی**:
```python
{
  '3d': {'score': 0.75, 'confidence': 0.85, 'trend': 'BULLISH'},
  '7d': {'score': 0.82, 'confidence': 0.88, 'trend': 'STRONG_BULLISH'},
  '30d': {'score': 0.65, 'confidence': 0.80, 'trend': 'BULLISH'}
}
```

---

### 2️⃣ تحلیل مومنتوم (Momentum Analysis)

**هدف**: تشخیص قدرت و سرعت حرکت

**اندیکاتورها** (8 عدد):
- RSI (Relative Strength Index)
- Stochastic Oscillator
- CCI (Commodity Channel Index)
- Williams %R
- ROC (Rate of Change)
- Momentum
- OBV (On-Balance Volume)
- CMF (Chaikin Money Flow)

**تحلیل Divergence**:
- Regular Bullish Divergence (برگشت صعودی)
- Regular Bearish Divergence (برگشت نزولی)
- Hidden Bullish Divergence (ادامه صعود)
- Hidden Bearish Divergence (ادامه نزول)

**فایل‌های مرتبط**:
- `indicators/momentum.py` - محاسبه اندیکاتورها
- `patterns/divergence.py` - تشخیص Divergence
- `ml/multi_horizon_momentum_features.py` - استخراج ویژگی
- `ml/multi_horizon_momentum_analysis.py` - تحلیل و امتیازدهی
- `ml/train_multi_horizon_momentum.py` - آموزش مدل

**خروجی**:
```python
{
  '3d': {'score': -0.25, 'confidence': 0.70, 'signal': 'WEAK_BEARISH'},
  '7d': {'score': 0.15, 'confidence': 0.65, 'signal': 'WEAK_BULLISH'},
  '30d': {'score': 0.55, 'confidence': 0.75, 'signal': 'BULLISH'}
}
```

---

## 🔄 تفاوت اساسی

| ویژگی | Trend Analysis | Momentum Analysis |
|-------|----------------|-------------------|
| **هدف** | جهت کلی بازار | قدرت و سرعت حرکت |
| **اندیکاتورها** | 10 اندیکاتور روند | 8 اندیکاتور مومنتوم |
| **تحلیل اضافی** | الگوها، امواج الیوت | Divergence Detection |
| **کاربرد** | روند بلندمدت | نقاط ورود/خروج |
| **افق زمانی** | میان‌مدت تا بلند | کوتاه‌مدت تا میان |

---

## 🎯 سناریوهای استفاده

### سناریو 1: فقط تحلیل روند
```python
from ml.multi_horizon_feature_extraction import MultiHorizonFeatureExtractor
from ml.multi_horizon_analysis import MultiHorizonAnalyzer

# استخراج ویژگی
extractor = MultiHorizonFeatureExtractor()
X, Y = extractor.extract_training_dataset(candles)

# آموزش
learner = MultiHorizonWeightLearner()
learner.train(X, Y)

# تحلیل
analyzer = MultiHorizonAnalyzer(learner)
trend_analysis = analyzer.analyze(features)

print(f"روند 7d: {trend_analysis.trend_7d.score}")
```

### سناریو 2: فقط تحلیل مومنتوم
```python
from ml.multi_horizon_momentum_features import MultiHorizonMomentumFeatureExtractor
from ml.multi_horizon_momentum_analysis import MultiHorizonMomentumAnalyzer

# استخراج ویژگی
extractor = MultiHorizonMomentumFeatureExtractor()
X, Y = extractor.extract_training_dataset(candles)

# آموزش
learner = MultiHorizonWeightLearner()
learner.train(X, Y)

# تحلیل
analyzer = MultiHorizonMomentumAnalyzer(learner)
momentum_analysis = analyzer.analyze(features)

print(f"مومنتوم 3d: {momentum_analysis.momentum_3d.score}")
```

### سناریو 3: ترکیب هوشمند
```python
from ml.combined_trend_momentum_analysis import CombinedTrendMomentumAnalyzer

# هر دو تحلیلگر آماده
trend_analyzer = MultiHorizonAnalyzer(trend_learner)
momentum_analyzer = MultiHorizonMomentumAnalyzer(momentum_learner)

# ترکیب
combined = CombinedTrendMomentumAnalyzer(
    trend_analyzer,
    momentum_analyzer,
    trend_weight=0.6,      # 60% وزن به روند
    momentum_weight=0.4    # 40% وزن به مومنتوم
)

# تحلیل
analysis = combined.analyze(trend_features, momentum_features)

print(f"توصیه نهایی: {analysis.final_action.value}")
print(f"اعتماد: {analysis.final_confidence:.0%}")
```

---

## 📈 مثال کاربردی

### وضعیت بازار: Bitcoin
```
قیمت فعلی: $50,000
```

#### نتایج تحلیل روند:
```
3d:  امتیاز = +0.85 (صعودی قوی)
7d:  امتیاز = +0.78 (صعودی)
30d: امتیاز = +0.65 (صعودی)

تفسیر: روند کلی صعودی است
```

#### نتایج تحلیل مومنتوم:
```
3d:  امتیاز = -0.15 (نزولی ضعیف)
7d:  امتیاز = +0.25 (صعودی ضعیف)
30d: امتیاز = +0.50 (صعودی)

تفسیر: مومنتوم کوتاه‌مدت ضعیف، اما بلندمدت مثبت
```

#### تفسیر ترکیبی:
```
✅ روند: صعودی قوی
⚠️ مومنتوم: ضعیف در کوتاه‌مدت

💡 توصیه:
   - Day Trading (3d): احتیاط → اصلاح کوتاه‌مدت محتمل
   - Swing Trading (7d): خرید در اصلاح
   - Position Trading (30d): نگهداری → روند صعودی ادامه دارد
```

---

## 🧠 تفسیر ترکیبی

| Trend | Momentum | تفسیر | اقدام |
|-------|----------|-------|-------|
| ✅ صعودی قوی | ✅ صعودی قوی | **بازار داغ** | STRONG BUY |
| ✅ صعودی قوی | ⚠️ ضعیف | **اصلاح کوتاه** | HOLD / BUY DIP |
| ✅ صعودی | ❌ نزولی | **واگرایی** | TAKE PROFIT |
| ⚠️ خنثی | ✅ صعودی | **شروع روند** | BUY |
| ⚠️ خنثی | ❌ نزولی | **شروع نزول** | SELL |
| ❌ نزولی | ❌ نزولی قوی | **بازار خرسی** | STRONG SELL |
| ❌ نزولی | ✅ صعودی | **برگشت احتمالی** | WAIT / SMALL BUY |

---

## 📂 ساختار فایل‌ها

```
Gravity_TechAnalysis/
│
├── indicators/
│   ├── trend.py              # اندیکاتورهای روند
│   └── momentum.py           # اندیکاتورهای مومنتوم
│
├── patterns/
│   └── divergence.py         # تشخیص Divergence
│
├── ml/
│   ├── multi_horizon_weights.py                    # کلاس اصلی ML
│   │
│   ├── multi_horizon_feature_extraction.py         # ویژگی روند
│   ├── multi_horizon_analysis.py                   # تحلیل روند
│   ├── train_multi_horizon.py                      # آموزش روند
│   │
│   ├── multi_horizon_momentum_features.py          # ویژگی مومنتوم
│   ├── multi_horizon_momentum_analysis.py          # تحلیل مومنتوم
│   ├── train_multi_horizon_momentum.py             # آموزش مومنتوم
│   │
│   └── combined_trend_momentum_analysis.py         # ترکیب هوشمند
│
├── test_multi_horizon.py                           # تست روند
├── test_combined_system.py                         # تست کامل
└── example_separate_analysis.py                    # مثال جداگانه
```

---

## 🚀 نحوه اجرا

### 1. آموزش مدل روند
```bash
python ml/train_multi_horizon.py
```

### 2. آموزش مدل مومنتوم
```bash
python ml/train_multi_horizon_momentum.py
```

### 3. تست سیستم کامل
```bash
python test_combined_system.py
```

### 4. مثال استفاده جداگانه
```bash
python example_separate_analysis.py
```

---

## ✅ نکات کلیدی

1. **دو سیستم مستقل**: Trend و Momentum هر کدام جداگانه کار می‌کنند
2. **اندیکاتورهای متفاوت**: 10 روند vs 8 مومنتوم
3. **افق‌های مختلف**: 3d (day), 7d (swing), 30d (position)
4. **ترکیب اختیاری**: می‌توانید هر کدام را جداگانه یا ترکیبی استفاده کنید
5. **ML مستقل**: هر سیستم مدل ML خودش را دارد

---

## 📚 مراجع

- راهنمای روند: `TREND_ANALYSIS_GUIDE.md`
- راهنمای مومنتوم: `MOMENTUM_ANALYSIS_PLAN.md`
- کد اصلی: `ml/*.py`
