# 🔄 راهنمای Migration به ساختار استاندارد

<div dir="rtl">

این راهنما مراحل کامل انتقال پروژه به ساختار استاندارد را شرح می‌دهد.

</div>

## 📋 پیش‌نیازها

```bash
# 1. Backup کامل پروژه
cp -r Gravity_TechnicalAnalysis Gravity_TechnicalAnalysis_backup

# 2. Commit تمام تغییرات فعلی
git add .
git commit -m "chore: prepare for structure migration"

# 3. ایجاد branch جدید
git checkout -b refactor/standard-structure

# 4. اطمینان از نصب dependencies
pip install -e ".[dev]"
```

---

## 🚀 مرحله 1: اجرای Dry Run

```bash
# مشاهده تغییرات بدون اعمال
python scripts/migration/migrate_to_standard_structure.py --dry-run
```

این دستور:
- ✅ ساختار جدید را نمایش می‌دهد
- ✅ فایل‌هایی که جابجا می‌شوند را لیست می‌کند
- ✅ گزارش کامل ایجاد می‌کند
- ❌ هیچ تغییری اعمال نمی‌کند

---

## 🔧 مرحله 2: اجرای Migration

```bash
# اعمال تغییرات
python scripts/migration/migrate_to_standard_structure.py --execute
```

این مرحله:
- ✅ ساختار فولدرهای جدید را ایجاد می‌کند
- ✅ فایل‌های مستندات را جابجا می‌کند
- ✅ فایل‌های deployment را منتقل می‌کند
- ✅ اسکریپت‌ها را سازماندهی می‌کند
- ⚠️ **توجه:** برخی کارها نیاز به دستی انجام دارند

---

## 📝 مرحله 3: کارهای دستی

### 3.1. ادغام کد منبع

<div dir="rtl">

**مشکل:** کد در چند مکان مختلف است:
- `src/gravity_tech/`
- `src/core/`
- `ml/` در root

**راه‌حل:**

</div>

```bash
# 1. Merge src/core/ into src/gravity_tech/core/
# بررسی تفاوت‌ها
diff -r src/core/ src/gravity_tech/core/

# کپی فایل‌های منحصر به فرد
# (این کار نیاز به بررسی دستی دارد)

# 2. Merge ml/ into src/gravity_tech/ml/
# بررسی تفاوت‌ها
diff -r ml/ src/gravity_tech/ml/

# کپی فایل‌های لازم
# (احتیاط: از duplicate جلوگیری کنید)

# 3. حذف فولدرهای قدیمی (بعد از اطمینان)
# rm -rf src/core/
# rm -rf ml/
```

### 3.2. آپدیت Import Statements

```bash
# یافتن تمام importها
grep -r "from ml\." src/ tests/
grep -r "from core\." src/ tests/

# جایگزینی خودکار (با احتیاط!)
find src -name "*.py" -type f -exec sed -i 's/from ml\./from gravity_tech.ml./g' {} +
find src -name "*.py" -type f -exec sed -i 's/from core\./from gravity_tech.core./g' {} +
find tests -name "*.py" -type f -exec sed -i 's/from ml\./from gravity_tech.ml./g' {} +
find tests -name "*.py" -type f -exec sed -i 's/from core\./from gravity_tech.core./g' {} +

# بررسی تغییرات
git diff src/ tests/
```

### 3.3. سازماندهی Tests

```bash
# فعلاً tests در یک سطح هستند
# باید به unit/integration/e2e تقسیم شوند

# ایجاد ساختار
mkdir -p tests/{unit,integration,e2e,performance,accuracy}

# جابجایی دستی بر اساس نوع تست
# - test_*.py → tests/unit/
# - test_integration_*.py → tests/integration/
# - benchmark_*.py → tests/performance/
# - validate_*.py → tests/accuracy/
```

### 3.4. آپدیت Configuration Files

#### pyproject.toml

بررسی و تایید:
```toml
[tool.setuptools]
packages = {find = {where = ["src"]}}

[tool.setuptools.package-dir]
"" = "src"
```

#### docker-compose.yml

اگر مسیرها در آن استفاده شده:
```yaml
# قبل
volumes:
  - ./ml:/app/ml

# بعد
volumes:
  - ./src/gravity_tech/ml:/app/gravity_tech/ml
```

#### Kubernetes manifests

بررسی `k8s/` یا `deployment/kubernetes/`:
- مسیرهای ConfigMap
- مسیرهای Volume
- Environment variables

---

## ✅ مرحله 4: اعتبارسنجی

### 4.1. بررسی Import ها

```bash
# اجرای Python و import کردن
python -c "from gravity_tech.core.indicators import trend"
python -c "from gravity_tech.ml.models import base"
python -c "from gravity_tech.api.main import app"
```

### 4.2. اجرای Linters

```bash
# Format check
make format-check

# Linting
make lint

# Type checking
make type-check
```

### 4.3. اجرای Tests

```bash
# تمام تست‌ها
make test

# با coverage
make test-cov

# فقط unit tests
make test-unit

# فقط integration tests
make test-integration
```

### 4.4. تست Application

```bash
# اجرای development server
make run

# تست endpoints
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

---

## 🐳 مرحله 5: Docker & Deployment

### 5.1. Docker Build

```bash
# Build image
docker build -t gravity-tech-analysis:restructured -f deployment/docker/Dockerfile .

# Test run
docker run -p 8000:8000 gravity-tech-analysis:restructured
```

### 5.2. Docker Compose

```bash
# با فایل جدید
docker-compose -f deployment/docker/docker-compose.yml up -d

# بررسی logs
docker-compose logs -f
```

### 5.3. Kubernetes (اگر استفاده می‌کنید)

```bash
# Dry run
kubectl apply -f deployment/kubernetes/base/ --dry-run=client

# Apply
kubectl apply -f deployment/kubernetes/base/
```

---

## 📊 مرحله 6: مستندسازی تغییرات

### 6.1. آپدیت README.md

```markdown
# تغییرات ساختار پروژه

نسخه جدید دارای ساختار استاندارد Python package است:

```
src/
  gravity_tech/
    api/
    core/
    ml/
    ...
```

برای جزئیات بیشتر: [RESTRUCTURE_PLAN.md](RESTRUCTURE_PLAN.md)
```

### 6.2. آپدیت CHANGELOG.md

```markdown
## [2.0.0] - 2025-12-03

### Changed
- 🏗️ **BREAKING:** Restructured project to standard Python package layout
- 📦 Consolidated source code to `src/gravity_tech/`
- 📚 Reorganized documentation by language (en/fa)
- 🧪 Restructured tests by type (unit/integration/e2e)
- 🚀 Moved deployment configs to `deployment/`
- 🔧 Improved tooling with Makefile

### Migration Guide
See [MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md)
```

---

## 🔄 مرحله 7: CI/CD Updates

### 7.1. GitHub Actions

آپدیت `.github/workflows/ci.yml`:

```yaml
# قبل
- name: Run tests
  run: pytest tests/

# بعد
- name: Run tests
  run: make test
```

### 7.2. Pre-commit Hooks

ایجاد `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        args: [--line-length=100]
  
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix]
```

نصب:
```bash
pip install pre-commit
pre-commit install
```

---

## ✨ مرحله 8: Finalization

### 8.1. پاکسازی

```bash
# حذف فایل‌های قدیمی
make clean

# حذف فولدرهای خالی
find . -type d -empty -delete

# حذف فایل‌های backup
rm -rf *_backup/
```

### 8.2. Git Operations

```bash
# Review تمام تغییرات
git status
git diff

# Commit مرحله‌ای
git add src/
git commit -m "refactor: consolidate source code"

git add tests/
git commit -m "refactor: reorganize tests structure"

git add deployment/
git commit -m "refactor: move deployment configs"

git add docs/
git commit -m "docs: reorganize documentation"

git add Makefile setup.py .editorconfig .dockerignore
git commit -m "chore: add standard project files"

# نهایی
git add .
git commit -m "refactor: complete migration to standard structure"
```

### 8.3. Testing & Review

```bash
# تست کامل
make check

# اجرای application
make run

# بررسی Docker
make docker-build
make docker-run
```

### 8.4. Merge

```bash
# Push branch
git push origin refactor/standard-structure

# ایجاد Pull Request در GitHub
# Review توسط تیم
# Merge به main
```

---

## 📋 Checklist نهایی

- [ ] Backup کامل گرفته شده
- [ ] Migration script اجرا شده
- [ ] کد منبع ادغام شده
- [ ] Import statements آپدیت شده
- [ ] Tests سازماندهی شده
- [ ] تمام تست‌ها pass می‌شوند
- [ ] Linters و type checkers موفق
- [ ] Docker image ساخته می‌شود
- [ ] Application اجرا می‌شود
- [ ] مستندات آپدیت شده
- [ ] CHANGELOG.md بروز شده
- [ ] CI/CD workflows آپدیت شده
- [ ] Git commits منظم و واضح
- [ ] Pull Request ایجاد شده
- [ ] Review انجام شده
- [ ] Merge به main

---

## ⚠️ مشکلات احتمالی و راه‌حل

### مشکل 1: Import Errors

```python
# خطا
ModuleNotFoundError: No module named 'ml'

# راه‌حل
# 1. بررسی PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

# 2. نصب مجدد package
pip install -e .
```

### مشکل 2: Tests Fail

```bash
# خطا
ImportError: cannot import name 'X' from 'gravity_tech.core'

# راه‌حل
# بررسی __init__.py files
# اطمینان از export صحیح در __init__.py
```

### مشکل 3: Docker Build Fails

```bash
# خطا
COPY failed: file not found

# راه‌حل
# آپدیت Dockerfile paths
# بررسی .dockerignore
```

---

## 📞 کمک و پشتیبانی

اگر با مشکلی مواجه شدید:

1. مستندات را دوباره بخوانید
2. MIGRATION_REPORT.md را بررسی کنید
3. Issue در GitHub باز کنید
4. با تیم تماس بگیرید

---

## 🎉 پس از Migration

بعد از migration موفق:

1. ✅ ساختار استاندارد Python package
2. ✅ Codebase تمیز و منظم
3. ✅ مستندات سازماندهی شده
4. ✅ Tests دسته‌بندی شده
5. ✅ Deployment ساده‌تر
6. ✅ Onboarding راحت‌تر برای developers جدید
7. ✅ Scalability بهتر
8. ✅ Maintainability بالاتر

**تبریک! 🎊 پروژه شما حالا ساختار استاندارد دارد.**

---

**تاریخ ایجاد:** 2025-12-03  
**نسخه:** 1.0  
**نویسنده:** Gravity Team
