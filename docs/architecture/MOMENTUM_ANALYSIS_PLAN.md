# تحلیل مومنتوم (Momentum Analysis)

## 📊 نقشه راه پیاده‌سازی

### فاز 1: اندیکاتورهای مومنتوم (Momentum Indicators)

#### 1️⃣ Oscillators (نوسانگرها)

**RSI (Relative Strength Index)** ✅
```python
RSI = 100 - (100 / (1 + RS))
RS = Average Gain / Average Loss

سطوح کلیدی:
  - RSI > 70 → Overbought (اشباع خرید)
  - RSI < 30 → Oversold (اشباع فروش)
  - RSI 50 → خط میانی

سیگنال:
  - RSI > 70 → نزولی (احتمال اصلاح)
  - RSI < 30 → صعودی (احتمال بازگشت)
  - واگرایی: RSI صعودی + قیمت نزولی → صعودی قوی

دقت: 0.75-0.85
```

**Stochastic Oscillator** ✅
```python
%K = ((Close - Low14) / (High14 - Low14)) × 100
%D = SMA(%K, 3)

سطوح:
  - %K > 80 → Overbought
  - %K < 20 → Oversold

سیگنال:
  - %K crosses above %D در oversold → صعودی
  - %K crosses below %D در overbought → نزولی

دقت: 0.7-0.8
```

**CCI (Commodity Channel Index)** ⏳
```python
CCI = (Typical Price - SMA) / (0.015 × Mean Deviation)
Typical Price = (High + Low + Close) / 3

سطوح:
  - CCI > +100 → Overbought
  - CCI < -100 → Oversold

سیگنال:
  - CCI از -100 به بالا → صعودی
  - CCI از +100 به پایین → نزولی

دقت: 0.75
```

**Williams %R** ⏳
```python
%R = (Highest High - Close) / (Highest High - Lowest Low) × -100

سطوح:
  - %R > -20 → Overbought
  - %R < -80 → Oversold

معکوس Stochastic (محدوده -100 تا 0)

دقت: 0.7-0.75
```

#### 2️⃣ Rate of Change (نرخ تغییر)

**ROC (Rate of Change)** ⏳
```python
ROC = ((Close - Close_n) / Close_n) × 100

سیگنال:
  - ROC > 0 → مومنتوم مثبت
  - ROC < 0 → مومنتوم منفی
  - ROC عبور از صفر → تغییر مومنتوم

دقت: 0.7
```

**Momentum** ⏳
```python
Momentum = Close - Close_n

ساده‌ترین اندیکاتور مومنتوم

دقت: 0.65
```

#### 3️⃣ Volume-Based Momentum ⏳

**OBV (On-Balance Volume)** ⏳
```python
if Close > Close_prev:
    OBV = OBV_prev + Volume
elif Close < Close_prev:
    OBV = OBV_prev - Volume
else:
    OBV = OBV_prev

سیگنال:
  - OBV صعودی + قیمت صعودی → تایید
  - واگرایی: OBV نزولی + قیمت صعودی → ضعف

دقت: 0.8 (با حجم)
```

**CMF (Chaikin Money Flow)** ⏳
```python
Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
Money Flow Volume = Money Flow Multiplier × Volume
CMF = Sum(Money Flow Volume, 20) / Sum(Volume, 20)

سطوح:
  - CMF > 0 → فشار خرید
  - CMF < 0 → فشار فروش

دقت: 0.75-0.8
```

---

### فاز 2: ساختار یادگیری ماشین برای مومنتوم

#### معماری ML برای Momentum:

```python
# مشابه Trend Analysis، اما با ویژگی‌های مومنتوم

class MomentumMLSystem:
    """
    سیستم یادگیری ماشین برای مومنتوم
    """
    
    # سطح 1: اندیکاتورهای مومنتوم
    momentum_indicators = [
        'rsi',           # RSI
        'stochastic',    # Stochastic %K, %D
        'cci',           # CCI
        'williams_r',    # Williams %R
        'roc',           # Rate of Change
        'momentum',      # Simple Momentum
        'obv',           # On-Balance Volume
        'cmf'            # Chaikin Money Flow
    ]
    
    # ویژگی‌ها برای هر اندیکاتور:
    features_per_indicator = [
        'signal',        # [-2, 2] سیگنال نرمال شده
        'confidence',    # [0, 1] دقت
        'weighted'       # signal × confidence
    ]
    
    # سطح 2: دسته مومنتوم (Momentum Category)
    # ترکیب وزن‌دار همه اندیکاتورها
    
    # Multi-Horizon Learning:
    horizons = [3, 7, 30]  # روز
    
    # هدف:
    target = future_return  # بازدهی آینده
```

#### ویژگی‌های استخراج شده:

```python
# مثال برای RSI:
features = {
    'rsi_signal': 0.6,      # RSI در ناحیه صعودی
    'rsi_confidence': 0.8,   # دقت بالا
    'rsi_weighted': 0.48,    # 0.6 × 0.8
    
    'rsi_divergence': 1.0,   # واگرایی مثبت تشخیص داده شد
    'rsi_overbought': 0.0,   # خیر
    'rsi_oversold': 0.0      # خیر
}

# در مجموع:
# 8 اندیکاتور × 3 ویژگی اصلی = 24 ویژگی
# + ویژگی‌های اضافی (واگرایی، overbought/oversold)
# = ~30-35 ویژگی برای سطح 1
```

---

### فاز 3: تشخیص واگرایی (Divergence Detection)

**واگرایی = قدرتمندترین سیگنال مومنتوم**

#### انواع واگرایی:

**1. واگرایی معمولی (Regular Divergence)**

```
واگرایی صعودی (Bullish):
  قیمت: Lower Low
  اندیکاتور: Higher Low
  → احتمال برگشت صعودی

واگرایی نزولی (Bearish):
  قیمت: Higher High
  اندیکاتور: Lower High
  → احتمال برگشت نزولی
```

**2. واگرایی پنهان (Hidden Divergence)**

```
واگرایی پنهان صعودی:
  قیمت: Higher Low
  اندیکاتور: Lower Low
  → ادامه روند صعودی

واگرایی پنهان نزولی:
  قیمت: Lower High
  اندیکاتور: Higher High
  → ادامه روند نزولی
```

#### پیاده‌سازی:

```python
class DivergenceDetector:
    """
    تشخیص واگرایی در اندیکاتورهای مومنتوم
    """
    
    def detect_divergence(
        self,
        prices: List[float],
        indicator_values: List[float],
        lookback: int = 20
    ) -> DivergenceResult:
        """
        تشخیص واگرایی
        
        Returns:
            DivergenceResult(
                type="regular_bullish" | "regular_bearish" | 
                     "hidden_bullish" | "hidden_bearish" | None,
                strength=0.0-1.0,
                description="..."
            )
        """
        # 1. یافتن swing points در قیمت
        price_swings = self._find_swing_points(prices)
        
        # 2. یافتن swing points در اندیکاتور
        indicator_swings = self._find_swing_points(indicator_values)
        
        # 3. مقایسه و تشخیص واگرایی
        divergence = self._compare_swings(price_swings, indicator_swings)
        
        return divergence
```

---

### فاز 4: ترکیب مومنتوم با روند (Trend + Momentum)

#### ماتریس تصمیم‌گیری:

| Trend | Momentum | Divergence | نتیجه | اطمینان | اقدام |
|-------|----------|------------|-------|---------|--------|
| ✅ صعودی قوی | ✅ RSI 50-70 | - | بسیار صعودی | 90% | **خرید قوی** |
| ✅ صعودی | ⚠️ RSI > 70 | ❌ واگرایی نزولی | مشکوک | 60% | **خروج جزئی** |
| ❌ نزولی | ⚠️ RSI < 30 | ✅ واگرایی صعودی | برگشت احتمالی | 75% | **خرید محتاطانه** |
| ⚠️ خنثی | ✅ Stoch کراس صعودی | - | شروع روند؟ | 65% | **ورود کوچک** |
| ✅ صعودی | ❌ نزولی | ✅ واگرایی نزولی | تضاد! | 50% | **صبر و انتظار** |

---

### فاز 5: Multi-Horizon ML برای مومنتوم

```python
# مشابه train_multi_horizon.py برای روند

ml/
├── multi_horizon_momentum_features.py    # استخراج ویژگی‌های مومنتوم
├── multi_horizon_momentum_weights.py     # یادگیری وزن اندیکاتورها
├── multi_horizon_momentum_analysis.py    # تحلیل چند افقی مومنتوم
├── train_multi_horizon_momentum.py       # Pipeline آموزش
└── test_multi_horizon_momentum.py        # تست سیستم
```

#### Pipeline آموزش:

```python
def train_momentum_ml_system():
    """
    آموزش سیستم ML برای مومنتوم
    """
    
    # 1. استخراج ویژگی‌ها
    extractor = MultiHorizonMomentumFeatureExtractor(
        indicators=[
            'rsi', 'stochastic', 'cci', 'williams_r',
            'roc', 'momentum', 'obv', 'cmf'
        ],
        horizons=[3, 7, 30]
    )
    
    X, Y = extractor.extract_training_dataset(candles)
    
    # 2. آموزش مدل
    learner = MultiHorizonMomentumWeightLearner()
    learner.train(X, Y)
    
    # 3. ذخیره وزن‌ها
    learner.save_weights('ml_models/momentum_weights.json')
    
    # 4. ارزیابی
    # - R² برای هر افق
    # - MAE
    # - Confidence
```

---

## 📋 پلان اجرایی (Action Plan)

### مرحله 1: اندیکاتورهای پایه ⏳
- [ ] پیاده‌سازی 8 اندیکاتور مومنتوم در `indicators/momentum.py`
- [ ] تست هر اندیکاتور
- [ ] محاسبه دقت (confidence) هر اندیکاتور

### مرحله 2: تشخیص واگرایی ⏳
- [ ] کلاس `DivergenceDetector`
- [ ] تشخیص واگرایی معمولی
- [ ] تشخیص واگرایی پنهان
- [ ] امتیازدهی قدرت واگرایی

### مرحله 3: ترکیب مومنتوم (سطح دسته) ⏳
- [ ] محاسبه `Momentum Score` از همه اندیکاتورها
- [ ] محاسبه `Momentum Accuracy`
- [ ] ترکیب با تشخیص واگرایی

### مرحله 4: یادگیری ماشین ⏳
- [ ] `multi_horizon_momentum_features.py`
- [ ] `multi_horizon_momentum_weights.py`
- [ ] `multi_horizon_momentum_analysis.py`
- [ ] `train_multi_horizon_momentum.py`
- [ ] تست و ارزیابی

### مرحله 5: ترکیب با روند ⏳
- [ ] ماتریس تصمیم‌گیری Trend + Momentum
- [ ] تحلیل جامع با دو بُعد
- [ ] تنظیم وزن‌های نهایی

---

## 🎯 خروجی نهایی مومنتوم

```python
# مثال خروجی تحلیل مومنتوم:

momentum_analysis = {
    'momentum_score': 0.65,        # [−1, 1]
    'momentum_confidence': 0.78,    # [0, 1]
    
    'indicators': {
        'rsi': {
            'value': 58.3,
            'signal': 'neutral',
            'confidence': 0.75
        },
        'stochastic': {
            'k': 65.2,
            'd': 58.1,
            'signal': 'bullish',  # %K > %D
            'confidence': 0.8
        },
        # ... سایر اندیکاتورها
    },
    
    'divergence': {
        'detected': True,
        'type': 'regular_bullish',
        'strength': 0.85,
        'description': 'قیمت Lower Low اما RSI Higher Low'
    },
    
    'multi_horizon': {
        '3d': {
            'score': 0.45,
            'confidence': 0.72,
            'recommendation': '📈 مومنتوم مثبت کوتاه‌مدت'
        },
        '7d': {
            'score': 0.68,
            'confidence': 0.78,
            'recommendation': '🚀 مومنتوم قوی میان‌مدت'
        },
        '30d': {
            'score': 0.58,
            'confidence': 0.75,
            'recommendation': '📈 مومنتوم مثبت بلندمدت'
        }
    },
    
    'combined_with_trend': {
        'trend_score': 0.72,
        'momentum_score': 0.65,
        'overall': 0.69,
        'confidence': 0.82,
        'recommendation': '✅ خرید - روند و مومنتوم هم‌جهت'
    }
}
```

---

## 🚀 شروع کار

آیا آماده‌ای شروع کنیم؟

1️⃣ **گزینه 1**: شروع با اندیکاتورهای پایه
   - پیاده‌سازی RSI, Stochastic, CCI, Williams %R
   - در فایل `indicators/momentum.py`

2️⃣ **گزینه 2**: شروع با تشخیص واگرایی
   - کلاس `DivergenceDetector`
   - تشخیص واگرایی‌های معمولی و پنهان

3️⃣ **گزینه 3**: شروع با ML Multi-Horizon
   - مستقیم به سراغ یادگیری ماشین برای مومنتوم

**توصیه من: شروع با گزینه 1** (اندیکاتورها) تا پایه محکم باشد، بعد ML 🎯
