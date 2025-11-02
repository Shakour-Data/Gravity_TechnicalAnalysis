# محاسبه سیگنال نهایی

## فرمول محاسبه امتیاز کلی

سیگنال نهایی با ترکیب وزن‌دار اندیکاتورهای مختلف محاسبه می‌شود، **با در نظر گرفتن دقت (Accuracy) هر دسته**.

### وزن‌های پایه:
- **Trend (روند)**: 30%
- **Momentum (مومنتوم)**: 25%
- **Cycle (سیکل)**: 25%
- **Volume (حجم)**: 20% (به عنوان تاییدکننده)

### 🎯 تعدیل وزن‌ها بر اساس دقت

وزن‌های بالا **پایه** هستند. وزن‌های واقعی بر اساس **دقت (Accuracy)** هر دسته تعدیل می‌شوند:

```
Adjusted Weight[category] = (Base Weight × Accuracy[category]) / Σ(Base Weight × Accuracy)
```

**مثال:**
اگر روند دقت 0.9، مومنتوم 0.6، سیکل 0.6، و حجم 0.6 داشته باشند:
```
Sum = (0.30 × 0.9) + (0.25 × 0.6) + (0.25 × 0.6) + (0.20 × 0.6)
    = 0.27 + 0.15 + 0.15 + 0.12
    = 0.69

Adjusted Weights:
  - Trend: (0.30 × 0.9) / 0.69 = 0.27 / 0.69 = 39.1% ⬆️ (+9.1%)
  - Momentum: (0.25 × 0.6) / 0.69 = 0.15 / 0.69 = 21.7% ⬇️ (-3.3%)
  - Cycle: (0.25 × 0.6) / 0.69 = 0.15 / 0.69 = 21.7% ⬇️ (-3.3%)
  - Volume: (0.20 × 0.6) / 0.69 = 0.12 / 0.69 = 17.4% ⬇️ (-2.6%)
```

**نتیجه:** دسته‌های با دقت بالاتر، وزن بیشتری دریافت می‌کنند! 📊

### فرمول نهایی:

```
Overall Score = (Trend × Weight_trend) + (Momentum × Weight_momentum) + (Cycle × Weight_cycle)

سپس تایید با Volume:
if (Overall Score و Volume هم‌جهت):
    Overall Score × (1 + |Volume Score| × Weight_volume)
else:
    Overall Score × (1 - |Volume Score| × Weight_volume)
```

### محاسبه امتیاز هر دسته:

برای هر دسته از اندیکاتورها (Trend, Momentum, Cycle):

```
Category Score = Σ(Signal Score × Confidence) / Σ(Confidence)
Category Accuracy = Σ(Confidence) / Count(Indicators)
```

که در آن:
- **Signal Score**: عدد بین -2 تا +2
  - Very Bullish = +2
  - Bullish = +1
  - Bullish Broken = +0.5
  - Neutral = 0
  - Bearish Broken = -0.5
  - Bearish = -1
  - Very Bearish = -2

- **Confidence**: عدد بین 0 تا 1 (دقت هر اندیکاتور)
- **Category Accuracy**: میانگین دقت همه اندیکاتورهای آن دسته

### محاسبه اعتماد کلی (Overall Confidence):

```
Agreement Factor = 1 - (StdDev(All Scores) / 4)
Accuracy Factor = Mean(All Confidences)

Overall Confidence = (Agreement Factor × 0.6) + (Accuracy Factor × 0.4)
```

- **Agreement Factor (60%)**: اندیکاتورها چقدر با هم هماهنگ هستند؟
- **Accuracy Factor (40%)**: میانگین دقت همه اندیکاتورها چقدر است؟

هر چه اندیکاتورها بیشتر هماهنگ باشند و دقت بالاتری داشته باشند، اعتماد کلی بیشتر است.

### مثال محاسبه با دقت:

فرض کنید:
```
Trend Indicators:
  - SMA(20): Bullish (1.0) × 0.8 confidence
  - EMA(20): Bullish (1.0) × 0.85 confidence
  - MACD: Bullish Broken (0.5) × 0.7 confidence

Trend Score = (1.0×0.8 + 1.0×0.85 + 0.5×0.7) / (0.8+0.85+0.7)
            = (0.8 + 0.85 + 0.35) / 2.35
            = 2.0 / 2.35
            = 0.85

Trend Accuracy = (0.8 + 0.85 + 0.7) / 3 = 0.783

Momentum Score = 0.6 (فرض)
Momentum Accuracy = 0.7 (فرض)

Cycle Score = 0.7 (فرض)  
Cycle Accuracy = 0.65 (فرض)

Volume Score = 0.4 (فرض)
Volume Accuracy = 0.75 (فرض)

تعدیل وزن‌ها بر اساس دقت:
Sum = (0.30×0.783) + (0.25×0.7) + (0.25×0.65) + (0.20×0.75)
    = 0.235 + 0.175 + 0.1625 + 0.15
    = 0.7225

Adjusted Weights:
  - Trend: 0.235 / 0.7225 = 32.5% (بجای 30%)
  - Momentum: 0.175 / 0.7225 = 24.2% (بجای 25%)
  - Cycle: 0.1625 / 0.7225 = 22.5% (بجای 25%)
  - Volume: 0.15 / 0.7225 = 20.8% (بجای 20%)

Overall = (0.85 × 0.325) + (0.6 × 0.242) + (0.7 × 0.225)
        = 0.276 + 0.145 + 0.158
        = 0.579

تایید با Volume (هم‌جهت):
Overall = 0.579 × (1 + 0.4 × 0.208)
        = 0.579 × 1.083
        = 0.627

Normalized = 0.627 / 2 = 0.314
Signal = Bullish Broken (چون 0.1 < 0.314 < 0.4)

Overall Confidence:
  - Agreement: 1 - (StdDev / 4) = فرض 0.85
  - Accuracy: (0.783 + 0.7 + 0.65 + 0.75) / 4 = 0.721
  - Overall: (0.85 × 0.6) + (0.721 × 0.4) = 0.51 + 0.288 = 0.798
```

## کد پیاده‌سازی (با دقت)

```python
def calculate_overall_signal(self):
    """Calculate overall signals based on all indicators with accuracy weighting"""
    
    def calc_category_score_and_accuracy(indicators: List[IndicatorResult]) -> tuple[float, float]:
        """
        Calculate category score and average accuracy
        Returns: (weighted_score, average_accuracy)
        """
        if not indicators:
            return 0.0, 0.0
        
        weighted_sum = sum(
            ind.signal.get_score() * ind.confidence 
            for ind in indicators
        )
        total_weight = sum(ind.confidence for ind in indicators)
        
        # Calculate average accuracy (confidence) for this category
        avg_accuracy = total_weight / len(indicators) if indicators else 0.0
        
        score = weighted_sum / total_weight if total_weight > 0 else 0.0
        return score, avg_accuracy
    
    # Calculate category scores and accuracies
    trend_score, trend_accuracy = calc_category_score_and_accuracy(self.trend_indicators)
    momentum_score, momentum_accuracy = calc_category_score_and_accuracy(self.momentum_indicators)
    cycle_score, cycle_accuracy = calc_category_score_and_accuracy(self.cycle_indicators)
    volume_score, volume_accuracy = calc_category_score_and_accuracy(self.volume_indicators)
    
    # Base weights
    base_weights = {
        'trend': 0.30,
        'momentum': 0.25,
        'cycle': 0.25,
        'volume': 0.20
    }
    
    # Apply accuracy to weights
    accuracies = {
        'trend': trend_accuracy,
        'momentum': momentum_accuracy,
        'cycle': cycle_accuracy,
        'volume': volume_accuracy
    }
    
    # Calculate accuracy-adjusted weights
    total_weighted_accuracy = sum(
        base_weights[cat] * accuracies[cat] 
        for cat in base_weights.keys()
    )
    
    if total_weighted_accuracy > 0:
        adjusted_weights = {
            cat: (base_weights[cat] * accuracies[cat]) / total_weighted_accuracy
            for cat in base_weights.keys()
        }
    else:
        adjusted_weights = base_weights
    
    # Calculate overall score with accuracy-adjusted weights
    overall_score = (
        (trend_score * adjusted_weights['trend']) + 
        (momentum_score * adjusted_weights['momentum']) + 
        (cycle_score * adjusted_weights['cycle'])
    )
    
    # Volume confirms or weakens the signal using adjusted weight
    volume_weight = adjusted_weights['volume']
    volume_confirmation = abs(volume_score) * volume_weight
    
    if overall_score * volume_score > 0:  # Same direction
        overall_score *= (1 + volume_confirmation)
    else:  # Different direction
        overall_score *= (1 - volume_confirmation)
    
    # Clamp to [-2, 2] range
    overall_score = max(-2.0, min(2.0, overall_score))
    
    # Calculate overall confidence based on:
    # 1. Agreement between indicators (lower std dev = higher confidence)
    # 2. Average accuracy of all categories
    all_scores = []
    all_confidences = []
    
    for indicators in [self.trend_indicators, self.momentum_indicators,
                      self.cycle_indicators, self.volume_indicators]:
        all_scores.extend([ind.signal.get_score() for ind in indicators])
        all_confidences.extend([ind.confidence for ind in indicators])
    
    if all_scores and all_confidences:
        import numpy as np
        
        # Agreement factor: Lower standard deviation = higher confidence
        std_dev = np.std(all_scores)
        agreement_confidence = max(0.0, min(1.0, 1.0 - (std_dev / 4.0)))
        
        # Accuracy factor: Average accuracy of all indicators
        accuracy_confidence = np.mean(all_confidences)
        
        # Combined confidence: 60% agreement + 40% accuracy
        self.overall_confidence = (agreement_confidence * 0.6) + (accuracy_confidence * 0.4)
    else:
        self.overall_confidence = 0.5
```
    
    # Volume confirms or weakens the signal (20% weight)
    volume_confirmation = abs(volume_score) * 0.2
    if overall_score * volume_score > 0:  # Same direction
        overall_score *= (1 + volume_confirmation)
    else:  # Different direction
        overall_score *= (1 - volume_confirmation)
    
    # Clamp to [-2, 2] range
    overall_score = max(-2.0, min(2.0, overall_score))
    
    # Normalize to [-1, 1] for SignalStrength conversion
    normalized_score = overall_score / 2.0
    
    self.overall_trend_signal = SignalStrength.from_value(trend_score / 2.0)
    self.overall_momentum_signal = SignalStrength.from_value(momentum_score / 2.0)
    self.overall_cycle_signal = SignalStrength.from_value(cycle_score / 2.0)
    self.overall_signal = SignalStrength.from_value(normalized_score)
```

## چرا این وزن‌ها؟

### Trend (30%): وزن بالا
- روند اصلی بازار مهم‌ترین عامل است
- بر اساس نظریه داو، روند تا سیگنال قطعی ادامه دارد
- اندیکاتورهای روند (SMA, EMA, MACD) پایه‌ای هستند

### Momentum (25%): وزن بالا
- سرعت تغییرات قیمت
- تشخیص اشباع خرید/فروش
- پیش‌بینی تغییرات روند

### Cycle (25%): وزن بالا
- شناسایی چرخه‌های بازار
- زمان‌بندی ورود و خروج
- تکمیل‌کننده روند و مومنتوم

### Volume (20%): تاییدکننده
- تایید قدرت روند
- بر اساس نظریه داو: حجم باید روند را تایید کند
- عامل کمکی نه اصلی

## خروجی

```json
{
  "overall_trend_signal": "صعودی",
  "overall_momentum_signal": "صعودی شکسته شده",
  "overall_cycle_signal": "صعودی",
  "overall_signal": "صعودی شکسته شده",
  "overall_confidence": 0.75
}
```

توضیح: سیگنال نهایی "صعودی شکسته شده" نشان می‌دهد که روند کلی صعودی است اما ممکن است در حال ضعیف شدن باشد.
