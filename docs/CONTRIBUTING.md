# 🤝 راهنمای مشارکت در پروژه
## Contributing to Gravity Technical Analysis

<div dir="rtl">

خوشحالیم که می‌خواهید در پروژه مشارکت کنید! این راهنما به شما کمک می‌کند تا مشارکت موثری داشته باشید.

---

## 📋 فهرست

1. [نحوه مشارکت](#نحوه-مشارکت)
2. [استانداردهای کد](#استانداردهای-کد)
3. [ساختار Branch](#ساختار-branch)
4. [فرآیند Pull Request](#فرآیند-pull-request)
5. [نوشتن تست](#نوشتن-تست)
6. [مستندسازی](#مستندسازی)
7. [کد رفتار](#کد-رفتار)

---

## 🎯 نحوه مشارکت

### انواع مشارکت

✅ **گزارش باگ** - اگر مشکلی پیدا کردید  
✅ **پیشنهاد ویژگی** - ایده جدید دارید؟  
✅ **توسعه کد** - پیاده‌سازی ویژگی یا رفع باگ  
✅ **بهبود مستندات** - کمک به مستندسازی  
✅ **بهینه‌سازی** - بهبود عملکرد  

### گام‌های اولیه

1. **Fork کردن پروژه**
   ```bash
   # Fork در GitHub
   # سپس clone کنید:
   git clone https://github.com/YOUR_USERNAME/Gravity_TechAnalysis.git
   cd Gravity_TechAnalysis
   ```

2. **نصب Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # برای توسعه
   ```

3. **ایجاد محیط مجازی**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

4. **اجرای تست‌ها**
   ```bash
   pytest tests/
   ```

---

## 📝 استانداردهای کد

### 1. Python Style Guide

ما از **PEP 8** استفاده می‌کنیم با تنظیمات زیر:

```python
# خوب ✅
def calculate_trend_score(
    candles: List[Candle],
    period: int = 20,
    use_volume: bool = True
) -> TrendScore:
    """
    محاسبه امتیاز روند
    
    Args:
        candles: لیست کندل‌ها
        period: دوره محاسبه
        use_volume: استفاده از حجم
    
    Returns:
        TrendScore: امتیاز محاسبه شده
    """
    pass

# بد ❌
def calc(c, p=20):
    pass
```

### 2. نامگذاری

```python
# متغیرها و توابع: snake_case
trend_score = 0.85
def calculate_momentum(): pass

# کلاس‌ها: PascalCase
class TrendAnalysis: pass
class VolumeMatrix: pass

# ثابت‌ها: UPPER_CASE
MAX_CANDLES = 1000
DEFAULT_TIMEFRAME = "1h"

# فایل‌ها: snake_case
trend_analysis.py
volume_matrix.py
```

### 3. Type Hints

**همیشه** از Type Hints استفاده کنید:

```python
# خوب ✅
from typing import List, Optional, Dict

def analyze(
    candles: List[Candle],
    weights: Optional[Dict[str, float]] = None
) -> FiveDimensionalDecision:
    pass

# بد ❌
def analyze(candles, weights=None):
    pass
```

### 4. Docstrings

از **Google Style** استفاده کنید:

```python
def volume_interaction(
    volume_score: float,
    dimension_score: float,
    threshold: float = 0.5
) -> VolumeInteraction:
    """
    محاسبه تعامل حجم با یک بُعد
    
    این تابع تعامل بین حجم و یک بُعد تحلیلی را محاسبه می‌کند.
    تعامل می‌تواند تایید، هشدار، یا واگرایی باشد.
    
    Args:
        volume_score: امتیاز حجم [-1, +1]
        dimension_score: امتیاز بُعد [-1, +1]
        threshold: آستانه تشخیص (پیش‌فرض: 0.5)
    
    Returns:
        VolumeInteraction: شی حاوی نوع تعامل و امتیاز
    
    Raises:
        ValueError: اگر امتیازها خارج از بازه [-1, +1] باشند
    
    Example:
        >>> interaction = volume_interaction(0.7, 0.8)
        >>> print(interaction.type)
        VolumeInteractionType.STRONG_CONFIRM
    """
    if not -1 <= volume_score <= 1:
        raise ValueError("volume_score must be in [-1, +1]")
    
    # محاسبات...
    pass
```

### 5. کد تمیز

```python
# خوب ✅ - خوانا و ساده
def is_bullish_trend(score: float) -> bool:
    """بررسی صعودی بودن روند"""
    return score > 0.3

if is_bullish_trend(trend_score):
    execute_buy_signal()

# بد ❌ - پیچیده و غیرخوانا
if not (trend_score <= 0.3 or not trend_score > -1):
    execute_buy_signal()
```

### 6. استفاده از Enums

```python
# خوب ✅
class SignalStrength(Enum):
    VERY_BULLISH = "VERY_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    VERY_BEARISH = "VERY_BEARISH"

signal = SignalStrength.BULLISH

# بد ❌
signal = "bullish"  # رشته ساده
```

---

## 🌳 ساختار Branch

### نامگذاری Branch‌ها

```
feature/add-rsi-indicator
bugfix/fix-macd-calculation
docs/update-trend-guide
refactor/optimize-volume-matrix
test/add-momentum-tests
```

### الگوی Git Flow

```
main (production-ready)
  ├── develop (توسعه فعال)
  │   ├── feature/new-indicator
  │   ├── feature/improve-ml
  │   └── bugfix/fix-error
  └── release/v1.1.0 (آماده انتشار)
```

### دستورات Git

```bash
# ایجاد feature branch
git checkout -b feature/add-rsi-indicator

# کار روی feature
git add .
git commit -m "feat: add RSI indicator with confidence calculation"

# push کردن
git push origin feature/add-rsi-indicator

# ایجاد Pull Request در GitHub
```

---

## 🔄 فرآیند Pull Request

### 1. قبل از ایجاد PR

✅ تست‌ها را اجرا کنید:
```bash
pytest tests/
```

✅ کد را format کنید:
```bash
black .
isort .
```

✅ بررسی linting:
```bash
flake8 .
mypy .
```

### 2. ایجاد PR

**عنوان PR:**
```
feat: add RSI indicator with dynamic confidence
fix: correct MACD histogram calculation
docs: update 5D decision matrix guide
refactor: optimize volume matrix calculations
```

**توضیحات PR:**
```markdown
## 📋 تغییرات

### چه چیزی اضافه شد؟
- اضافه کردن اندیکاتور RSI با محاسبه confidence دینامیک
- تست‌های واحد برای RSI

### چرا این تغییر لازم بود؟
RSI یکی از مهم‌ترین اندیکاتورهای مومنتوم است و نبود آن در سیستم احساس می‌شد.

### نحوه تست
```python
from indicators.momentum import calculate_rsi
result = calculate_rsi(candles, period=14)
assert result.value >= 0 and result.value <= 100
```

## ✅ Checklist

- [x] تست‌های جدید اضافه شده
- [x] مستندات به‌روز شده
- [x] کد format شده (black + isort)
- [x] همه تست‌ها پاس می‌کنند
- [x] Type hints اضافه شده

## 📸 Screenshots (در صورت نیاز)

(اگر تغییرات UI/UX دارد)
```

### 3. بررسی PR

PR شما توسط maintainer‌ها بررسی می‌شود:

✅ **کد خوانا است؟**  
✅ **تست‌ها کافی است؟**  
✅ **مستندات کامل است؟**  
✅ **استانداردها رعایت شده؟**  

### 4. بعد از تایید

```bash
# merge می‌شود
# شما contributor می‌شوید! 🎉
```

---

## 🧪 نوشتن تست

### ساختار تست

```python
# tests/unit/test_indicators/test_trend.py

import pytest
from indicators.trend import calculate_sma
from models.schemas import Candle

class TestSMA:
    """تست‌های اندیکاتور SMA"""
    
    @pytest.fixture
    def sample_candles(self):
        """کندل‌های نمونه برای تست"""
        return [
            Candle(open=100, high=105, low=95, close=102, volume=1000),
            Candle(open=102, high=107, low=100, close=105, volume=1100),
            # ...
        ]
    
    def test_sma_calculation(self, sample_candles):
        """تست محاسبه صحیح SMA"""
        result = calculate_sma(sample_candles, period=5)
        
        assert result is not None
        assert result.value > 0
        assert 0 <= result.confidence <= 1
    
    def test_sma_with_insufficient_data(self):
        """تست با داده کافی"""
        candles = []  # خالی
        
        with pytest.raises(ValueError):
            calculate_sma(candles, period=20)
    
    def test_sma_signal_strength(self, sample_candles):
        """تست قدرت سیگنال"""
        result = calculate_sma(sample_candles, period=5)
        
        assert result.signal in [
            SignalStrength.VERY_BULLISH,
            SignalStrength.BULLISH,
            SignalStrength.NEUTRAL,
            SignalStrength.BEARISH,
            SignalStrength.VERY_BEARISH
        ]
```

### اجرای تست‌ها

```bash
# همه تست‌ها
pytest tests/

# یک فایل خاص
pytest tests/unit/test_indicators/test_trend.py

# یک تست خاص
pytest tests/unit/test_indicators/test_trend.py::TestSMA::test_sma_calculation

# با coverage
pytest --cov=indicators tests/

# verbose
pytest -v tests/
```

### Coverage

**هدف**: حداقل 80% coverage

```bash
pytest --cov=indicators --cov-report=html tests/
# گزارش در htmlcov/index.html
```

---

## 📚 مستندسازی

### 1. Code Documentation

```python
# خوب ✅ - Docstring کامل
def calculate_volume_interaction(
    volume: VolumeScore,
    dimension: DimensionScore
) -> VolumeInteraction:
    """
    محاسبه تعامل بین حجم و یک بُعد تحلیلی
    
    این تابع تعامل بین حجم معاملات و یک بُعد تحلیلی (مثل روند یا مومنتوم)
    را محاسبه می‌کند. تعامل می‌تواند تایید، هشدار، یا واگرایی باشد.
    
    Args:
        volume: امتیاز حجم
        dimension: امتیاز بُعد تحلیلی
    
    Returns:
        VolumeInteraction: شی حاوی:
            - type: نوع تعامل (CONFIRM, WARN, DIVERGENCE, ...)
            - score: امتیاز تعامل [-0.35, +0.35]
            - confidence_multiplier: ضریب اطمینان [0.6, 1.15]
    
    Raises:
        ValueError: اگر امتیازها خارج از بازه مجاز باشند
    
    Example:
        >>> vol = VolumeScore(score=0.7, confidence=0.8)
        >>> dim = DimensionScore(score=0.8, confidence=0.85)
        >>> interaction = calculate_volume_interaction(vol, dim)
        >>> print(interaction.type)
        VolumeInteractionType.STRONG_CONFIRM
    """
    pass
```

### 2. User Documentation

برای اضافه کردن به مستندات:

```markdown
# در docs/guides/NEW_FEATURE.md

# 📊 راهنمای ویژگی جدید

## معرفی

توضیح مختصر ویژگی

## نحوه استفاده

مثال‌های کاربردی

## مثال‌ها

کدهای عملی

## نکات مهم

نکاتی که کاربر باید بداند
```

### 3. API Documentation

برای endpoint‌های جدید:

```python
@router.get("/analyze/{symbol}")
async def analyze_symbol(
    symbol: str,
    timeframe: str = "1h",
    use_volume_matrix: bool = True
):
    """
    تحلیل کامل یک سمبول
    
    این endpoint تحلیل جامع یک سمبول را با استفاده از
    سیستم 5D Decision Matrix انجام می‌دهد.
    
    Args:
        symbol: نماد (مثلاً BTC/USDT)
        timeframe: تایم‌فریم (1m, 5m, 15m, 1h, 4h, 1d)
        use_volume_matrix: استفاده از Volume Matrix
    
    Returns:
        AnalysisResponse: شامل:
            - signal: سیگنال نهایی (9 سطح)
            - confidence: اطمینان [0, 1]
            - risk_level: سطح ریسک (5 سطح)
            - recommendations: توصیه‌های معاملاتی
    
    Example:
        GET /api/v1/analyze/BTC/USDT?timeframe=1h&use_volume_matrix=true
    """
    pass
```

---

## 🐛 گزارش باگ

### Template گزارش باگ

```markdown
## 🐛 توضیح باگ

توضیح واضح و مختصر باگ

## 🔄 مراحل بازتولید

1. انجام عمل X
2. کلیک روی Y
3. دیدن خطا Z

## ✅ رفتار مورد انتظار

توضیح دهید که چه اتفاقی باید می‌افتاد

## ❌ رفتار واقعی

توضیح دهید چه اتفاقی افتاد

## 📸 Screenshots

در صورت امکان، screenshot اضافه کنید

## 🖥️ محیط

- OS: Windows 11
- Python: 3.10.5
- نسخه پروژه: 1.0.0

## ➕ اطلاعات اضافی

هر اطلاعات دیگری که مفید است
```

---

## ✨ پیشنهاد ویژگی

### Template پیشنهاد

```markdown
## 💡 ایده

توضیح ایده خود

## 🎯 مشکلی که حل می‌کند

چه مشکلی را حل می‌کند؟

## 💭 راه‌حل پیشنهادی

توضیح راه‌حل

## 🔄 جایگزین‌ها

راه‌حل‌های دیگر که در نظر گرفته‌اید

## ➕ Context اضافی

اطلاعات اضافی، screenshots، لینک‌ها، ...
```

---

## 📊 استانداردهای Commit

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

```
feat:     ویژگی جدید
fix:      رفع باگ
docs:     تغییر مستندات
style:    فرمت کد (بدون تغییر منطق)
refactor: بازنویسی کد (بدون تغییر رفتار)
test:     اضافه/تغییر تست
chore:    کارهای نگهداری (build, dependencies)
perf:     بهبود عملکرد
```

### مثال‌ها

```bash
# ویژگی جدید
git commit -m "feat(indicators): add RSI indicator with dynamic confidence"

# رفع باگ
git commit -m "fix(volume): correct volume adjustment calculation"

# مستندات
git commit -m "docs(guides): update 5D decision matrix examples"

# refactor
git commit -m "refactor(ml): optimize volume matrix performance"
```

---

## ⚡ بهینه‌سازی

### قوانین بهینه‌سازی

1. **اول کار کن، بعد بهینه کن**
   - Premature optimization is the root of all evil
   
2. **Profile قبل از بهینه‌سازی**
   ```python
   import cProfile
   cProfile.run('analyze_function()')
   ```

3. **benchmark بنویسید**
   ```python
   import timeit
   time = timeit.timeit('function()', number=1000)
   ```

4. **مستند کنید**
   - قبل از بهینه‌سازی: X ms
   - بعد از بهینه‌سازی: Y ms
   - بهبود: Z%

---

## 🏆 کد رفتار

### اصول ما

✅ **محترمانه** - با همه با احترام رفتار کنید  
✅ **سازنده** - نقد سازنده، نه تخریب  
✅ **مشارکتی** - کمک به یکدیگر  
✅ **شفاف** - ارتباط واضح و صریح  
✅ **فراگیر** - همه خوش‌آمدید  

### رفتارهای غیرقابل قبول

❌ زبان توهین‌آمیز یا تحقیرآمیز  
❌ حملات شخصی  
❌ هراساندن (harassment)  
❌ تبعیض به هر شکل  

---

## 📞 تماس

- **Issues**: [GitHub Issues](https://github.com/YOUR_REPO/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_REPO/discussions)
- **Email**: your.email@example.com

---

## 🎉 تشکر

از مشارکت شما متشکریم! 🙏

هر مشارکتی، کوچک یا بزرگ، برای ما ارزشمند است.

**Contributors خواهید شد** در:
- README.md
- صفحه Contributors در GitHub
- Release Notes

---

**موفق باشید!** 🚀

</div>
