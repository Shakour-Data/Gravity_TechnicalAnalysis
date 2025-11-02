# 📁 ساختار پروژه Gravity Technical Analysis

<div dir="rtl">

## 📊 نمای کلی ساختار

```
Gravity_TechAnalysis/
├── 📁 api/                          # API و Endpoints
│   ├── routes/                      # مسیرهای API
│   ├── schemas/                     # Validation schemas
│   └── __init__.py
│
├── 📁 analysis/                     # هسته تحلیل تکنیکال
│   ├── market_phase.py             # تحلیل فاز بازار
│   ├── trend_structure.py          # ساختار روند (Dow Theory)
│   └── __init__.py
│
├── 📁 config/                       # تنظیمات پروژه
│   ├── settings.py                 # تنظیمات اصلی
│   └── __init__.py
│
├── 📁 database/                     # مدیریت پایگاه داده
│   ├── connection.py               # اتصال به DB
│   ├── models/                     # مدل‌های ORM
│   └── __init__.py
│
├── 📁 docs/                         # 📚 مستندات کامل پروژه
│   ├── 📄 PROJECT_SUMMARY.md       # خلاصه کامل پروژه
│   ├── 📄 QUICKSTART.md            # شروع سریع
│   ├── 📄 README.md                # معرفی اصلی
│   │
│   ├── 📁 guides/                  # راهنماهای جامع (7 راهنما)
│   │   ├── TREND_ANALYSIS_GUIDE.md             # روند (10 اندیکاتور)
│   │   ├── MOMENTUM_ANALYSIS_GUIDE.md          # مومنتوم (8 اندیکاتور)
│   │   ├── VOLATILITY_ANALYSIS_GUIDE.md        # نوسان (8 اندیکاتور)
│   │   ├── CYCLE_ANALYSIS_GUIDE.md             # چرخه (7 اندیکاتور)
│   │   ├── SUPPORT_RESISTANCE_GUIDE.md         # حمایت/مقاومت (6 روش)
│   │   ├── VOLUME_MATRIX_GUIDE.md              # ماتریس حجم-ابعاد
│   │   ├── FIVE_DIMENSIONAL_DECISION_GUIDE.md  # 5D Decision Matrix
│   │   ├── TREND_ANALYSIS_SUMMARY.md           # خلاصه روند
│   │   ├── DOW_THEORY.md                       # نظریه داو
│   │   └── HISTORICAL_SYSTEM_GUIDE.md          # سیستم تاریخی
│   │
│   ├── 📁 architecture/            # معماری و طراحی
│   │   ├── SIGNAL_CALCULATION.md           # نحوه محاسبه سیگنال
│   │   ├── ML_WEIGHTS.md                   # وزن‌دهی ML
│   │   ├── ML_FEATURES_GUIDE.md            # ویژگی‌های ML
│   │   ├── MOMENTUM_ANALYSIS_PLAN.md       # طرح مومنتوم
│   │   └── SCORING_SYSTEM_GUIDE.md         # سیستم امتیازدهی
│   │
│   ├── 📁 api/                     # مستندات API
│   │   └── (فایل‌های مستندات API)
│   │
│   └── 📁 changelogs/              # تاریخچه تغییرات
│       ├── CHANGELOG_ACCURACY.md
│       ├── CHANGELOG_CLASSICAL_PATTERNS.md
│       ├── API_SCORE_RANGE_CHANGE.md
│       └── TREND_VS_MOMENTUM.md
│
├── 📁 examples/                     # 💡 مثال‌های کاربردی
│   ├── 📁 basic/                   # مثال‌های ساده
│   │   ├── example.py                          # مثال اصلی
│   │   ├── example_api_response.py             # مثال API
│   │   ├── example_separate_analysis.py        # تحلیل جداگانه
│   │   ├── example_trend_vs_momentum.py        # مقایسه
│   │   └── example_volatility_analysis.py      # نوسان
│   │
│   ├── 📁 advanced/                # مثال‌های پیشرفته
│   │   ├── example_5d_decision_matrix.py       # 5D Matrix
│   │   ├── example_comprehensive_analysis.py   # تحلیل جامع
│   │   ├── example_scoring_system.py           # سیستم امتیاز
│   │   ├── example_complete_analysis.py        # تحلیل کامل
│   │   └── example_historical_system.py        # سیستم تاریخی
│   │
│   └── 📁 ml/                      # مثال‌های ML
│       └── (مثال‌های یادگیری ماشین)
│
├── 📁 indicators/                   # 🔢 اندیکاتورهای تکنیکال
│   ├── trend.py                    # 10 اندیکاتور روند
│   ├── momentum.py                 # 8 اندیکاتور مومنتوم
│   ├── volatility.py               # 8 اندیکاتور نوسان
│   ├── cycle.py                    # 7 اندیکاتور چرخه
│   ├── support_resistance.py       # 6 روش S/R
│   ├── volume.py                   # اندیکاتورهای حجم
│   └── __init__.py
│
├── 📁 middleware/                   # Middleware و Utilities
│   ├── error_handler.py            # مدیریت خطا
│   ├── logger.py                   # Logging
│   └── __init__.py
│
├── 📁 ml/                          # 🤖 Machine Learning & AI
│   ├── 🔵 Base Analysis
│   │   ├── combined_trend_momentum_analysis.py
│   │   ├── multi_horizon_analysis.py
│   │   ├── multi_horizon_momentum_analysis.py
│   │   ├── multi_horizon_volatility_analysis.py
│   │   ├── multi_horizon_cycle_analysis.py
│   │   └── multi_horizon_support_resistance_analysis.py
│   │
│   ├── 🟢 Feature Extraction
│   │   ├── feature_extraction.py
│   │   ├── multi_horizon_feature_extraction.py
│   │   ├── multi_horizon_momentum_features.py
│   │   ├── multi_horizon_volatility_features.py
│   │   ├── multi_horizon_cycle_features.py
│   │   └── multi_horizon_support_resistance_features.py
│   │
│   ├── 🔴 Core Systems (3 Layers)
│   │   ├── volume_dimension_matrix.py          # Layer 2
│   │   ├── five_dimensional_decision_matrix.py # Layer 3
│   │   ├── integrated_multi_horizon_analysis.py
│   │   └── complete_analysis_pipeline.py       # Orchestrator
│   │
│   ├── 🟡 ML Models & Training
│   │   ├── ml_indicator_weights.py
│   │   ├── ml_dimension_weights.py
│   │   ├── multi_horizon_weights.py
│   │   ├── weight_optimizer.py
│   │   └── train_pipeline.py
│   │
│   ├── 🟣 Data & Utils
│   │   └── data_connector.py
│   │
│   └── __init__.py
│
├── 📁 ml_models/                    # مدل‌های ذخیره‌شده
│   ├── weights/                    # وزن‌های آموزش‌دیده
│   ├── checkpoints/                # Checkpoints
│   └── __init__.py
│
├── 📁 models/                       # 📦 Data Models & Schemas
│   ├── schemas.py                  # مدل‌های داده
│   ├── enums.py                    # Enumerations
│   └── __init__.py
│
├── 📁 patterns/                     # 🕯️ Pattern Recognition
│   ├── candlestick.py              # الگوهای شمعی
│   ├── classical.py                # الگوهای کلاسیک
│   ├── elliott.py                  # امواج الیوت
│   └── __init__.py
│
├── 📁 scripts/                      # 🔧 ابزارها و اسکریپت‌ها
│   ├── 📁 training/                # اسکریپت‌های آموزش
│   │   ├── train_ml.py
│   │   └── (بقیه train_*.py)
│   │
│   └── 📁 visualization/           # ابزارهای Visualization
│       └── visualize_trend_analysis.py
│
├── 📁 services/                     # سرویس‌های Business Logic
│   ├── analysis_service.py         # سرویس تحلیل
│   ├── data_service.py             # سرویس داده
│   └── __init__.py
│
├── 📁 tests/                        # 🧪 تست‌های واحد
│   ├── 📁 unit/                    # تست‌های واحد
│   │   ├── test_indicators/
│   │   ├── test_patterns/
│   │   └── test_ml/
│   │
│   ├── 📁 integration/             # تست‌های یکپارچه
│   │   ├── test_complete_analysis.py
│   │   ├── test_combined_system.py
│   │   └── test_multi_horizon.py
│   │
│   └── 📁 accuracy/                # تست‌های دقت
│       ├── test_accuracy_weighting.py
│       ├── test_comprehensive_accuracy.py
│       └── test_confidence_metrics.py
│
├── 📁 utils/                        # ⚙️ ابزارهای کمکی
│   ├── calculations.py             # محاسبات عمومی
│   ├── validators.py               # اعتبارسنجی
│   └── __init__.py
│
├── 📄 .env.example                 # نمونه تنظیمات محیطی
├── 📄 .gitignore                   # Git ignore
├── 📄 docker-compose.yml           # Docker Compose
├── 📄 Dockerfile                   # Docker configuration
├── 📄 LICENSE                      # مجوز (MIT)
├── 📄 main.py                      # نقطه ورود اصلی
├── 📄 README.md                    # معرفی پروژه
├── 📄 requirements.txt             # وابستگی‌های Python
├── 📄 STRUCTURE.md                 # این فایل
└── 📄 CONTRIBUTING.md              # راهنمای مشارکت
```

---

## 📚 توضیح فولدرها

### 🔵 `api/` - لایه API
```
وظیفه: ارائه RESTful API برای سرویس‌ها
محتوا:
  - routes/: تعریف endpoint‌ها
  - schemas/: مدل‌های validation
  - middleware/: میان‌افزارها
```

### 🟢 `analysis/` - هسته تحلیل
```
وظیفه: منطق تحلیلی پایه
محتوا:
  - market_phase.py: شناسایی فاز بازار
  - trend_structure.py: تحلیل ساختار روند
```

### 🔴 `docs/` - مستندات
```
وظیفه: تمام مستندات پروژه
ساختار:
  docs/
  ├── guides/          راهنماهای جامع (7 راهنما)
  ├── architecture/    معماری و طراحی
  ├── api/            مستندات API
  └── changelogs/      تاریخچه تغییرات
```

### 🟡 `examples/` - مثال‌ها
```
وظیفه: مثال‌های کاربردی
دسته‌بندی:
  - basic/: مثال‌های ساده برای شروع
  - advanced/: مثال‌های پیشرفته
  - ml/: مثال‌های یادگیری ماشین
```

### 🟣 `indicators/` - اندیکاتورها
```
وظیفه: پیاده‌سازی اندیکاتورهای تکنیکال
محتوا:
  - trend.py: 10 اندیکاتور روند
  - momentum.py: 8 اندیکاتور مومنتوم
  - volatility.py: 8 اندیکاتور نوسان
  - cycle.py: 7 اندیکاتور چرخه
  - support_resistance.py: 6 روش
  - volume.py: اندیکاتورهای حجم
```

### 🔵 `ml/` - Machine Learning
```
وظیفه: هوش مصنوعی و یادگیری ماشین
معماری 3 لایه:
  Layer 1: Base Dimensions (5 بُعد)
  Layer 2: Volume-Dimension Matrix
  Layer 3: 5D Decision Matrix

دسته‌بندی:
  - Base Analysis: تحلیل‌های پایه
  - Feature Extraction: استخراج ویژگی
  - Core Systems: سیستم‌های اصلی
  - ML Models: مدل‌ها و آموزش
  - Data: اتصال به داده
```

### 🟢 `patterns/` - الگوها
```
وظیفه: تشخیص الگوهای تحلیلی
محتوا:
  - candlestick.py: الگوهای شمعی
  - classical.py: الگوهای کلاسیک
  - elliott.py: امواج الیوت
```

### 🟡 `scripts/` - اسکریپت‌ها
```
وظیفه: ابزارها و اسکریپت‌های کمکی
دسته‌بندی:
  - training/: آموزش مدل‌های ML
  - visualization/: تصویرسازی داده
```

### 🔴 `tests/` - تست‌ها
```
وظیفه: تست‌های خودکار
ساختار:
  tests/
  ├── unit/           تست‌های واحد
  ├── integration/    تست‌های یکپارچه
  └── accuracy/       تست‌های دقت
```

---

## 🎯 نقاط ورود اصلی

### 1. استفاده از CLI
```bash
python main.py --symbol BTC/USDT --timeframe 1h
```

### 2. استفاده از API
```python
# شروع سرور
uvicorn api.main:app --reload

# فراخوانی
curl http://localhost:8000/api/v1/analyze/BTC/USDT?timeframe=1h
```

### 3. استفاده به صورت کتابخانه
```python
from ml.complete_analysis_pipeline import quick_analyze

result = quick_analyze(candles)
print(result.decision.final_signal)
```

### 4. مثال‌های آماده
```bash
# مثال ساده
python examples/basic/example.py

# مثال پیشرفته
python examples/advanced/example_5d_decision_matrix.py

# مثال ML
python examples/ml/example_ml_training.py
```

---

## 🔄 جریان داده در سیستم

```
1. ورودی داده (Candles)
         ↓
2. Layer 1: Base Dimensions
   ├─ Trend Analysis
   ├─ Momentum Analysis
   ├─ Volatility Analysis
   ├─ Cycle Analysis
   └─ S/R Analysis
         ↓
3. Layer 2: Volume Matrix
   └─ 5 تعامل حجم × هر بُعد
         ↓
4. Layer 3: 5D Decision Matrix
   ├─ وزن‌دهی دینامیک
   ├─ تحلیل توافق
   ├─ ارزیابی ریسک
   └─ تصمیم نهایی
         ↓
5. خروجی:
   ├─ سیگنال (9 سطح)
   ├─ ریسک (5 سطح)
   ├─ توصیه‌ها
   └─ استراتژی معاملاتی
```

---

## 📊 آمار پروژه

- **کل خطوط کد**: ~15,000 خط
- **فایل‌های Python**: 25+ فایل
- **مستندات**: 6,500+ خط (فارسی)
- **اندیکاتورها**: 39 اندیکاتور
- **تعاملات حجم**: 5 تعامل
- **سطوح تصمیم**: 9 سطح سیگنال + 5 سطح ریسک
- **لایه‌های معماری**: 3 لایه

---

## 🚀 شروع سریع

### 1. نصب
```bash
pip install -r requirements.txt
```

### 2. تنظیمات
```bash
cp .env.example .env
# ویرایش .env
```

### 3. اجرا
```bash
# CLI
python main.py

# API
uvicorn api.main:app --reload

# مثال
python examples/basic/example.py
```

### 4. مستندات
```bash
# راهنمای شروع
docs/QUICKSTART.md

# راهنمای کامل
docs/guides/
```

---

## 📚 مستندات مرتبط

- **شروع سریع**: [`docs/QUICKSTART.md`](docs/QUICKSTART.md)
- **خلاصه پروژه**: [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md)
- **راهنماهای جامع**: [`docs/guides/`](docs/guides/)
- **معماری**: [`docs/architecture/`](docs/architecture/)
- **مشارکت**: [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## 🤝 مشارکت

برای مشارکت در پروژه:
1. فایل [`CONTRIBUTING.md`](CONTRIBUTING.md) را مطالعه کنید
2. استانداردهای کد را رعایت کنید
3. تست‌های لازم را اضافه کنید
4. Pull Request ایجاد کنید

---

## 📄 لایسنس

MIT License - استفاده آزاد در پروژه‌های شخصی و تجاری

---

**نسخه**: 1.0.0  
**تاریخ به‌روزرسانی**: فروردین 1403  
**وضعیت**: ✅ Production Ready

</div>
