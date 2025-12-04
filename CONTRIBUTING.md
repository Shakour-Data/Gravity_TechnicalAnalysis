# 🤝 راهنمای مشارکت در پروژه

<div dir="rtl">

خوش آمدید! از علاقه شما برای مشارکت در پروژه **Gravity Technical Analysis** سپاسگزاریم. این راهنما به شما کمک می‌کند تا به بهترین شکل در پروژه مشارکت کنید.

</div>

## 📋 فهرست مطالب

- [کد رفتار](#-کد-رفتار)
- [چگونه می‌توانم کمک کنم؟](#-چگونه-میتوانم-کمک-کنم)
- [راه‌اندازی محیط توسعه](#️-راهاندازی-محیط-توسعه)
- [فرآیند توسعه](#-فرآیند-توسعه)
- [استانداردهای کد](#-استانداردهای-کد)
- [نوشتن تست](#-نوشتن-تست)
- [ارسال Pull Request](#-ارسال-pull-request)
- [گزارش باگ](#-گزارش-باگ)
- [پیشنهاد ویژگی جدید](#-پیشنهاد-ویژگی-جدید)

---

## 📜 کد رفتار

<div dir="rtl">

### اصول کلی:
- ✅ با احترام و محترمانه رفتار کنید
- ✅ نظرات سازنده بدهید
- ✅ از زبان توهین‌آمیز خودداری کنید
- ✅ به تنوع و شمول احترام بگذارید

</div>

---

## 🎯 چگونه می‌توانم کمک کنم؟

<div dir="rtl">

### روش‌های مشارکت:

#### 1. 🐛 گزارش و رفع باگ
- باگ‌های موجود را بررسی کنید
- باگ‌های جدید گزارش دهید
- رفع باگ‌های ساده را شروع کنید

#### 2. ✨ افزودن ویژگی جدید
- اندیکاتورهای تکنیکال جدید
- بهبود الگوریتم‌های ML
- افزودن الگوهای جدید

#### 3. 📝 بهبود مستندات
- ترجمه مستندات
- افزودن مثال‌ها
- اصلاح اشتباهات

#### 4. 🧪 نوشتن تست
- افزایش پوشش تست
- تست‌های performance
- تست‌های integration

#### 5. 🎨 بهبود کد
- Refactoring
- بهینه‌سازی performance
- رفع Code smells

</div>

---

## 🛠️ راه‌اندازی محیط توسعه

### پیش‌نیازها

```bash
# Python 3.12+
python --version

# Git
git --version

# Make (optional)
make --version
```

### نصب و راه‌اندازی

```bash
# 1. Fork و Clone کردن
git clone https://github.com/YOUR-USERNAME/Gravity_TechnicalAnalysis.git
cd Gravity_TechnicalAnalysis

# 2. ایجاد Virtual Environment
python -m venv venv

# فعال‌سازی (Windows)
venv\Scripts\activate

# فعال‌سازی (Linux/Mac)
source venv/bin/activate

# 3. نصب Dependencies
pip install -e ".[dev]"

# یا با Make
make install-dev

# 4. راه‌اندازی دیتابیس
python setup_database.py

# یا با Make
make setup-db

# 5. اجرای تست‌ها برای اطمینان
pytest tests/ -v

# یا با Make
make test
```

### تنظیم Git

```bash
# تنظیم remote upstream
git remote add upstream https://github.com/Shakour-Data/Gravity_TechnicalAnalysis.git

# بررسی remotes
git remote -v
```

---

## 🔄 فرآیند توسعه

### 1. ایجاد Branch جدید

```bash
# همیشه از آخرین نسخه main شروع کنید
git checkout main
git pull upstream main

# ایجاد branch با نام مناسب
git checkout -b feature/add-new-indicator
# یا
git checkout -b fix/resolve-calculation-bug
# یا
git checkout -b docs/improve-api-documentation
```

### 2. نام‌گذاری Branch

```
feature/    - برای ویژگی‌های جدید
fix/        - برای رفع باگ
docs/       - برای تغییرات مستندات
refactor/   - برای refactoring
test/       - برای افزودن تست
chore/      - برای کارهای عمومی
```

### 3. توسعه

```bash
# کد خود را بنویسید
# تست بنویسید
# مستندات را آپدیت کنید

# بررسی تغییرات
git status
git diff
```

### 4. Commit کردن

```bash
# فرمت commit message:
# <type>: <subject>
#
# <body>
#
# <footer>

git add .
git commit -m "feat: add RSI divergence detection

- Implement bullish and bearish divergence detection
- Add tests for edge cases
- Update documentation

Closes #123"
```

#### انواع Commit:

- `feat`: ویژگی جدید
- `fix`: رفع باگ
- `docs`: تغییرات مستندات
- `style`: فرمت کد (بدون تغییر منطق)
- `refactor`: refactoring کد
- `test`: افزودن یا تغییر تست
- `chore`: کارهای عمومی
- `perf`: بهبود performance

---

## 📏 استانداردهای کد

### Style Guide

<div dir="rtl">

پروژه از استانداردهای زیر پیروی می‌کند:

- **PEP 8**: Python style guide
- **Black**: Code formatting (line length: 100)
- **Ruff**: Fast linting
- **isort**: Import sorting
- **Type Hints**: برای همه توابع

</div>

### بررسی کد قبل از Commit

```bash
# فرمت کردن کد
make format

# بررسی lint
make lint

# بررسی type hints
make type-check

# یا همه با هم
make quality
```

### نمونه کد خوب

```python
from typing import List, Optional
import numpy as np
import pandas as pd


def calculate_rsi(
    prices: pd.Series,
    period: int = 14,
    smoothing: Optional[str] = "ema"
) -> pd.Series:
    """
    محاسبه Relative Strength Index (RSI).

    Args:
        prices: سری قیمت‌ها
        period: دوره محاسبه (پیش‌فرض: 14)
        smoothing: نوع smoothing (ema یا sma)

    Returns:
        سری مقادیر RSI

    Raises:
        ValueError: اگر period کمتر از 2 باشد

    Examples:
        >>> prices = pd.Series([100, 102, 101, 103, 105])
        >>> rsi = calculate_rsi(prices, period=3)
        >>> print(rsi)
    """
    if period < 2:
        raise ValueError("Period must be at least 2")

    # محاسبات
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi
```

### نکات مهم:

- ✅ Type hints برای همه توابع
- ✅ Docstring با فرمت Google/NumPy
- ✅ نام‌های واضح و meaningful
- ✅ توابع کوچک و focused
- ✅ Error handling مناسب
- ✅ مثال‌های کاربردی

---

## 🧪 نوشتن تست

### ساختار تست

```python
import pytest
import pandas as pd
from gravity_tech.core.indicators.momentum import calculate_rsi


class TestRSI:
    """تست‌های RSI indicator."""

    @pytest.fixture
    def sample_prices(self) -> pd.Series:
        """داده نمونه برای تست."""
        return pd.Series([100, 102, 101, 103, 105, 104, 106, 108])

    def test_rsi_basic_calculation(self, sample_prices):
        """تست محاسبه پایه RSI."""
        rsi = calculate_rsi(sample_prices, period=3)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_prices)
        assert rsi.iloc[-1] >= 0
        assert rsi.iloc[-1] <= 100

    def test_rsi_invalid_period(self, sample_prices):
        """تست با period نامعتبر."""
        with pytest.raises(ValueError, match="Period must be at least 2"):
            calculate_rsi(sample_prices, period=1)

    def test_rsi_empty_series(self):
        """تست با سری خالی."""
        empty = pd.Series([])
        rsi = calculate_rsi(empty, period=3)
        assert len(rsi) == 0

    @pytest.mark.parametrize("period,expected_range", [
        (7, (40, 60)),
        (14, (45, 55)),
        (21, (48, 52)),
    ])
    def test_rsi_different_periods(self, sample_prices, period, expected_range):
        """تست با دوره‌های مختلف."""
        rsi = calculate_rsi(sample_prices, period=period)
        last_value = rsi.iloc[-1]
        assert expected_range[0] <= last_value <= expected_range[1]
```

### اجرای تست‌ها

```bash
# تمام تست‌ها
make test

# فقط unit tests
make test-unit

# با coverage
make test-cov

# تست خاص
pytest tests/unit/test_momentum.py -v

# تست با marker
pytest -m "not slow"
```

---

## 📤 ارسال Pull Request

### قبل از ارسال PR

```bash
# 1. همگام‌سازی با upstream
git checkout main
git pull upstream main
git checkout your-branch
git rebase main

# 2. بررسی کیفیت کد
make quality

# 3. اجرای تمام تست‌ها
make test

# 4. Push به fork خود
git push origin your-branch
```

### ایجاد Pull Request

1. به صفحه fork خود در GitHub بروید
2. روی "Compare & pull request" کلیک کنید
3. عنوان واضح و توضیحات کامل بنویسید

#### Template PR:

```markdown
## 📝 توضیحات

توضیح واضح از تغییراتی که اعمال کرده‌اید.

## 🎯 نوع تغییر

- [ ] رفع باگ (fix)
- [ ] ویژگی جدید (feature)
- [ ] تغییر شکسته (breaking change)
- [ ] بهبود مستندات (docs)
- [ ] بهبود کد (refactor)

## ✅ Checklist

- [ ] کد از استانداردهای پروژه پیروی می‌کند
- [ ] تست‌های مربوطه اضافه شده
- [ ] مستندات آپدیت شده
- [ ] تمام تست‌ها پاس می‌شوند
- [ ] CHANGELOG.md آپدیت شده

## 📊 تست‌ها

نحوه تست تغییرات را توضیح دهید.

## 🔗 مسائل مرتبط

Closes #123
Fixes #456
```

---

## 🐛 گزارش باگ

### قبل از گزارش

- جستجو کنید که باگ قبلاً گزارش نشده باشد
- آخرین نسخه را امتحان کنید
- مشخص کنید باگ قابل تکرار است

### Template گزارش باگ

```markdown
## 🐛 توضیحات باگ

توضیح واضح و مختصر از باگ.

## 🔄 مراحل بازتولید

1. برو به '...'
2. کلیک روی '...'
3. مشاهده خطا

## ✅ رفتار مورد انتظار

توضیح دهید چه انتظاری داشتید.

## ❌ رفتار واقعی

توضیح دهید چه اتفاقی افتاد.

## 📸 Screenshots

در صورت امکان تصویر اضافه کنید.

## 💻 محیط

- OS: [e.g. Windows 11]
- Python: [e.g. 3.12.0]
- Version: [e.g. 1.2.0]

## 📝 اطلاعات اضافی

هر اطلاعات دیگری که مفید باشد.
```

---

## ✨ پیشنهاد ویژگی جدید

### Template پیشنهاد

```markdown
## 🎯 مشکل یا نیاز

توضیح دهید چه مشکلی حل می‌شود.

## 💡 راه‌حل پیشنهادی

توضیحات واضح از راه‌حل.

## 🔀 جایگزین‌های بررسی شده

چه راه‌حل‌های دیگری بررسی کرده‌اید؟

## 📚 منابع

لینک به مقالات، الگوریتم‌ها و...

## ✅ پیاده‌سازی

آیا می‌خواهید خودتان پیاده کنید؟
```

---

## 📚 منابع مفید

<div dir="rtl">

### مستندات پروژه:
- [راهنمای شروع سریع](docs/QUICKSTART.md)
- [ساختار پروژه](docs/PROJECT_STRUCTURE.md)
- [راهنماهای جامع](docs/guides/)

### منابع خارجی:
- [PEP 8 Style Guide](https://pep8.org/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [pytest Documentation](https://docs.pytest.org/)
- [Git Best Practices](https://git-scm.com/book/en/v2)

</div>

---

## 🙏 سپاسگزاری

<div dir="rtl">

از وقت و تلاشی که برای بهبود این پروژه می‌گذارید سپاسگزاریم! 🚀

هر سوالی داشتید، از طریق Issues یا Discussions بپرسید.

</div>

---

**آخرین بروزرسانی:** 2025-12-03  
**نسخه:** 1.0
