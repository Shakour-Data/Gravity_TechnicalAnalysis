# 🏗️ طرح سازماندهی استاندارد پروژه Gravity Technical Analysis

## 🔍 مشکلات ساختار فعلی

### ❌ مشکلات شناسایی شده:

1. **دوگانگی کد** (Code Duplication)
   - فایل‌های مشابه در `src/gravity_tech/` و `src/core/`
   - تکرار indicators در چند مکان مختلف
   - پوشه‌های `ml/` و `src/gravity_tech/ml/` جدا از هم

2. **ساختار نامشخص** (Unclear Structure)
   - نبود نقطه ورود اصلی در root (main.py یا app.py)
   - فایل‌های پیکربندی پراکنده (database در root و src)
   - مدل‌ها در چند مکان (`models/`, `src/gravity_tech/models/`)

3. **مستندات نامرتب** (Disorganized Documentation)
   - فایل‌های markdown بیش از حد در root
   - مستندات در `docs/` نیاز به دسته‌بندی بهتر
   - راهنماهای فارسی و انگلیسی مخلوط

4. **Tests نامنظم** (Unorganized Tests)
   - تست‌های مختلف در یک سطح
   - نبود ساختار مشخص unit/integration/e2e
   - فایل‌های benchmark پراکنده

5. **فایل‌های اضافی** (Extra Files)
   - فایل‌های cache (`__pycache__/`, `.pytest_cache/`)
   - htmlcov در root
   - venv در git (باید در .gitignore باشد)

---

## ✅ ساختار پیشنهادی استاندارد

```
Gravity_TechnicalAnalysis/
│
├── .github/                          # GitHub workflows, templates
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── cd.yml
│   │   └── release.yml
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
│
├── docs/                             # 📚 تمام مستندات
│   ├── README.md                     # Index مستندات
│   ├── en/                          # 🇬🇧 مستندات انگلیسی
│   │   ├── getting-started/
│   │   ├── api/
│   │   ├── architecture/
│   │   └── deployment/
│   ├── fa/                          # 🇮🇷 مستندات فارسی
│   │   ├── getting-started/
│   │   ├── guides/
│   │   ├── api/
│   │   └── tutorials/
│   ├── changelog/                   # تاریخچه تغییرات
│   │   ├── CHANGELOG.md
│   │   └── releases/
│   └── diagrams/                    # نمودارها و تصاویر
│
├── src/                             # 📦 کد اصلی برنامه
│   └── gravity_tech/
│       ├── __init__.py
│       ├── __version__.py           # مدیریت ورژن
│       │
│       ├── api/                     # 🌐 API Layer
│       │   ├── __init__.py
│       │   ├── main.py             # FastAPI app
│       │   ├── dependencies.py     # DI container
│       │   ├── routers/
│       │   │   ├── __init__.py
│       │   │   ├── health.py
│       │   │   ├── analysis.py
│       │   │   ├── patterns.py
│       │   │   └── ml.py
│       │   └── middleware/
│       │       ├── __init__.py
│       │       ├── logging.py
│       │       ├── cors.py
│       │       └── auth.py
│       │
│       ├── core/                    # 💎 Core Business Logic
│       │   ├── __init__.py
│       │   ├── domain/             # Domain models
│       │   │   ├── __init__.py
│       │   │   ├── candle.py
│       │   │   ├── analysis_result.py
│       │   │   └── pattern.py
│       │   ├── indicators/         # Technical indicators
│       │   │   ├── __init__.py
│       │   │   ├── base.py
│       │   │   ├── trend/
│       │   │   ├── momentum/
│       │   │   ├── volatility/
│       │   │   ├── volume/
│       │   │   └── cycle/
│       │   ├── patterns/           # Pattern recognition
│       │   │   ├── __init__.py
│       │   │   ├── candlestick/
│       │   │   ├── chart/
│       │   │   └── harmonic/
│       │   └── analysis/           # Analysis engines
│       │       ├── __init__.py
│       │       ├── trend.py
│       │       ├── momentum.py
│       │       ├── support_resistance.py
│       │       └── multi_dimensional.py
│       │
│       ├── ml/                      # 🤖 Machine Learning
│       │   ├── __init__.py
│       │   ├── models/             # ML models
│       │   │   ├── __init__.py
│       │   │   ├── base.py
│       │   │   ├── pattern_classifier.py
│       │   │   └── weight_optimizer.py
│       │   ├── features/           # Feature engineering
│       │   │   ├── __init__.py
│       │   │   ├── extractors.py
│       │   │   └── transformers.py
│       │   ├── training/           # Training pipelines
│       │   │   ├── __init__.py
│       │   │   └── trainer.py
│       │   └── inference/          # Inference
│       │       ├── __init__.py
│       │       └── predictor.py
│       │
│       ├── data/                    # 📊 Data Layer
│       │   ├── __init__.py
│       │   ├── database/           # Database access
│       │   │   ├── __init__.py
│       │   │   ├── connection.py
│       │   │   ├── repositories/
│       │   │   └── migrations/     # Alembic migrations
│       │   ├── cache/              # Caching
│       │   │   ├── __init__.py
│       │   │   └── redis_cache.py
│       │   └── connectors/         # External data sources
│       │       ├── __init__.py
│       │       └── market_data.py
│       │
│       ├── services/                # 🔧 Application Services
│       │   ├── __init__.py
│       │   ├── analysis_service.py
│       │   ├── pattern_service.py
│       │   └── ml_service.py
│       │
│       ├── config/                  # ⚙️ Configuration
│       │   ├── __init__.py
│       │   ├── settings.py         # Pydantic settings
│       │   └── constants.py
│       │
│       └── utils/                   # 🛠️ Utilities
│           ├── __init__.py
│           ├── logging.py
│           ├── validators.py
│           └── helpers.py
│
├── tests/                           # ✅ Tests
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures
│   │
│   ├── unit/                       # Unit tests
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── test_indicators.py
│   │   │   ├── test_patterns.py
│   │   │   └── test_analysis.py
│   │   ├── ml/
│   │   │   └── test_models.py
│   │   └── utils/
│   │       └── test_helpers.py
│   │
│   ├── integration/                # Integration tests
│   │   ├── __init__.py
│   │   ├── test_api.py
│   │   ├── test_database.py
│   │   └── test_ml_pipeline.py
│   │
│   ├── e2e/                        # End-to-end tests
│   │   ├── __init__.py
│   │   └── test_complete_flow.py
│   │
│   ├── performance/                # Performance tests
│   │   ├── __init__.py
│   │   ├── benchmark_indicators.py
│   │   └── load_test.py
│   │
│   └── accuracy/                   # Accuracy validation
│       ├── __init__.py
│       └── test_indicator_accuracy.py
│
├── scripts/                         # 🔨 Scripts & Tools
│   ├── setup/
│   │   ├── init_database.py
│   │   └── seed_data.py
│   ├── migration/
│   │   └── migrate_old_structure.py
│   ├── deployment/
│   │   ├── deploy.sh
│   │   └── rollback.sh
│   └── maintenance/
│       ├── backup.py
│       └── optimize_db.py
│
├── deployment/                      # 🚀 Deployment configs
│   ├── docker/
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   └── docker-compose.dev.yml
│   ├── kubernetes/
│   │   ├── base/
│   │   ├── overlays/
│   │   │   ├── dev/
│   │   │   ├── staging/
│   │   │   └── production/
│   │   └── helm/
│   └── terraform/
│
├── examples/                        # 📝 Example usage
│   ├── basic_usage.py
│   ├── advanced_analysis.py
│   └── ml_training.py
│
├── data/                           # 📂 Data files (not in git)
│   ├── raw/
│   ├── processed/
│   └── models/                     # Trained ML models
│
├── configs/                        # 📋 Config files
│   ├── .env.example
│   ├── settings.yaml
│   └── logging.yaml
│
├── .github/                        # GitHub specific
├── .gitignore
├── .dockerignore
├── .editorconfig
│
├── pyproject.toml                  # Project metadata
├── setup.py                        # Setup script
├── requirements/                   # Requirements files
│   ├── base.txt
│   ├── dev.txt
│   ├── prod.txt
│   └── ml.txt
│
├── Makefile                        # Common commands
├── README.md                       # Main readme
├── LICENSE
├── CONTRIBUTING.md
└── CHANGELOG.md
```

---

## 🚀 مراحل اجرای سازماندهی

### Phase 1: آماده‌سازی (Preparation)
1. ✅ Backup کامل پروژه
2. ✅ ایجاد branch جدید: `restructure/standard-layout`
3. ✅ آپدیت `.gitignore`

### Phase 2: ساختار جدید (New Structure)
1. ✅ ایجاد ساختار فولدرها
2. ✅ انتقال فایل‌ها به مکان صحیح
3. ✅ آپدیت import paths
4. ✅ حذف duplic duplicates

### Phase 3: تنظیمات (Configuration)
1. ✅ تنظیم `pyproject.toml`
2. ✅ ایجاد `setup.py`
3. ✅ تنظیم `Makefile`
4. ✅ بهبود Docker configs

### Phase 4: مستندات (Documentation)
1. ✅ سازماندهی docs
2. ✅ آپدیت README
3. ✅ ایجاد CONTRIBUTING.md
4. ✅ بهبود API docs

### Phase 5: تست و اعتبارسنجی (Testing)
1. ✅ اجرای تمام تست‌ها
2. ✅ رفع خطاها
3. ✅ آپدیت CI/CD
4. ✅ Merge به main

---

## 📝 فایل‌های کلیدی جدید

### 1. `Makefile` - دستورات سریع

```makefile
.PHONY: help install test lint format clean run docker-build

help:
	@echo "Available commands:"
	@echo "  make install       Install dependencies"
	@echo "  make test          Run tests"
	@echo "  make lint          Run linters"
	@echo "  make format        Format code"
	@echo "  make run           Run development server"

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src/gravity_tech

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage

run:
	uvicorn src.gravity_tech.api.main:app --reload
```

### 2. `setup.py` - برای backward compatibility

```python
from setuptools import setup

setup()
```

### 3. `.gitignore` بهبود یافته

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
coverage.xml
*.cover

# ML Models
data/models/*.pkl
data/models/*.joblib
ml_models/

# Logs
*.log
logs/

# Environment
.env
.env.local
.env.*.local

# OS
.DS_Store
Thumbs.db
```

---

## 🎯 مزایای ساختار جدید

### ✅ مزایا:

1. **وضوح بیشتر** (Better Clarity)
   - ساختار مشخص و استاندارد
   - جداسازی واضح concerns
   - مسیر یادگیری ساده‌تر

2. **قابلیت نگهداری** (Maintainability)
   - کد مرتب و قابل پیدا کردن
   - کاهش duplication
   - مدیریت آسان‌تر dependencies

3. **مقیاس‌پذیری** (Scalability)
   - آماده برای رشد
   - افزودن features جدید آسان
   - تست‌پذیری بهتر

4. **استانداردسازی** (Standardization)
   - مطابق با PEP 517/518
   - سازگار با Python packaging
   - قابل نصب با pip

5. **همکاری بهتر** (Better Collaboration)
   - راحت برای contributors جدید
   - کد review آسان‌تر
   - مستندات منظم

---

## ⚠️ نکات مهم

### هشدارها:
- ⚠️ تمام import ها باید آپدیت شوند
- ⚠️ فایل‌های environment variables را جابجا نکنید
- ⚠️ مدل‌های ML trained را backup بگیرید
- ⚠️ دیتابیس را backup بگیرید

### Best Practices:
- ✅ یک branch جداگانه برای restructure
- ✅ تست کامل قبل از merge
- ✅ آپدیت تدریجی، نه یکباره
- ✅ مستندات را همزمان آپدیت کنید

---

## 📊 تایم‌لاین پیشنهادی

| فاز | مدت زمان | وضعیت |
|-----|----------|-------|
| آماده‌سازی | 1 روز | ⏳ Pending |
| ساختار جدید | 2-3 روز | ⏳ Pending |
| تنظیمات | 1 روز | ⏳ Pending |
| مستندات | 1-2 روز | ⏳ Pending |
| تست و اعتبارسنجی | 2 روز | ⏳ Pending |
| **جمع** | **7-9 روز** | ⏳ Pending |

---

## 🤝 مشارکت

برای کمک به سازماندهی:

1. Issue باز کنید با label `restructure`
2. PR خود را به branch `restructure/standard-layout` بزنید
3. تست‌های خود را اضافه کنید
4. مستندات را آپدیت کنید

---

**تاریخ ایجاد:** 2025-12-03  
**نسخه:** 1.0  
**وضعیت:** 📋 Proposed
