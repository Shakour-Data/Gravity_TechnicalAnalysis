خطوات إطلاق الإصدار - Release Process Steps

# گام‌های ریلیز v1.3.3

**تاریخ**: ۵ دسامبر ۲۰۲۵  
**نسخه فعلی**: ۱.۳.۲  
**نسخه جدید**: ۱.۳.۳  
**وضعیت**: آماده‌سازی برای انتشار  

---

## 📋 خلاصه موارد بحرانی

### 1️⃣ کیفیت و تست‌ها (CRITICAL)

**الزامات:**
- ✅ **پوشش تست**: هدف ۹۵٪+
- ✅ **تست‌های واحد**: تمام تست‌ها موفق
- ✅ **تست‌های یکپارچه**: بدون خرابی
- ✅ **تست‌های ML**: بدون ناپایداری (flaky)
- ✅ **smoke tests**: تمام endpoints پاسخ می‌دهند

**دستورات:**
```bash
# اجرای کامل تست‌ها با coverage
python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# تست‌های واحد
python -m pytest tests/unit/ -v --tb=short

# تست‌های یکپارچه
python -m pytest tests/integration/ -v --tb=short

# تست‌های ML
python -m pytest tests/ml/ -v --tb=short
```

**وضعیت فعلی:**
- README: ۱۱.۷۱٪ ❌ (نیاز به بروزرسانی)
- اهداف: ۹۵٪+ ✅ (بررسی لازم)

---

### 2️⃣ همگام‌سازی نسخه

**وضعیت فعلی:**

| فایل | نسخه فعلی | نسخه جدید | وضعیت |
|------|---------|---------|-------|
| `pyproject.toml` | 1.3.2 | 1.3.3 | ✅ بررسی شود |
| `README.md` | 1.2.0 | 1.3.3 | ❌ بروزرسانی لازم |
| `configs/VERSION` | 1.3.2 | 1.3.3 | ✅ بررسی شود |

**اقدامات:**
- [ ] بروزرسانی نشان (badge) نسخه در README (ردیف ۳)
- [ ] بروزرسانی لینک GitHub release در README (ردیف ۳)
- [ ] بروزرسانی پوشش تست در README (ردیف ۹)
- [ ] تایید هماهنگی تمام منابع نسخه

---

### 3️⃣ مستندات ریلیز

**وضعیت فعلی:**

| موضوع | وضعیت | موقعیت |
|------|------|--------|
| CHANGELOG.md | بخش [Unreleased] موجود | ردیف ۵۶۲ |
| Release Notes v1.3.0 | موجود | `docs/releases/` |
| Release Notes v1.3.1 | موجود | `docs/releases/` |
| Release Notes v1.3.3 | ❌ نیاز دارد | `docs/releases/` |
| 10-Day Checklist | "Create release notes" ناتمام | ردیف ۱۲۶ |

**اقدامات:**
- [ ] منتقل کردن [Unreleased] به [1.3.3] در CHANGELOG.md
- [ ] ایجاد `docs/releases/RELEASE_NOTES_v1.3.3.md` (مرجع: v1.3.0 و v1.3.1)
- [ ] ایجاد `docs/releases/RELEASE_SUMMARY_v1.3.3_FA.md` (خلاصه فارسی)
- [ ] بروزرسانی QUICK_START_10_DAYS.md (ردیف ۱۲۶)

**محتوای Release Notes:**
```
- بررسی اجمالی و تاریخ انتشار
- تغییرات کلیدی (۱۴۳ فایل بهبود یافت)
- رفع اشکالات (whitespace، imports، type hints)
- مراحل آزمون و استقرار
- راهنمای مهاجرت (در صورت لزوم)
```

---

### 4️⃣ استقرار و تأیید سلامت

**مرجع**: docs/releases/RELEASE_NOTES_v1.3.0.md (ردیف‌های ۶۶۱-۶۹۱)

**اقدامات:**
- [ ] دریافت کد جدید: `git pull origin main`
- [ ] تایید: تمام تست‌ها موفق
- [ ] استقرار Kubernetes: اعمال manifests
- [ ] بررسی health endpoint: `GET /health` → 200 OK
- [ ] بررسی version endpoint: `GET /version` → {"version": "1.3.3"}
- [ ] smoke tests: SMA، RSI، MACD endpoints

**معیارهای موفقیت:**
- ✅ HTTP 200 برای تمام endpoints
- ✅ زمان پاسخ < ۱۰۰ میلی‌ثانیه
- ✅ بدون خرابی در logs

---

### 5️⃣ نهایی‌سازی انتشار

**اقدامات:**
- [ ] ایجاد تگ Git: `git tag -a v1.3.3 -m "Release v1.3.3"`
- [ ] push تگ: `git push origin v1.3.3`
- [ ] ایجاد GitHub Release
  - عنوان: "v1.3.3 - Code Quality & Type Safety Improvements"
  - توضیحات: محتوای Release Notes
  - نشان as latest: بله
  - Pre-release: خیر
- [ ] اطلاع‌رسانی تیم (Slack/Teams)

---

## 🎯 ترتیب اجرایی توصیه‌شده

### **مرحله ۱: آزمون و کیفیت (اولویت بحرانی)**

```bash
# ۱. اجرای تمام تست‌ها با coverage
python -m pytest tests/ -v --cov=src --cov-report=html

# ۲. بررسی نتایج coverage
open htmlcov/index.html  # یا مرور درصد کل

# ۳. حداقل اهداف:
# - Overall: 95%+
# - unit/: 100%
# - integration/: 95%
# - ml/: 90%
```

⏸️ **توقف** اگر coverage < 95%
- ✅ لازم است تست‌های اضافی اضافه شوند (CRITICAL_PRIORITY_ANALYSIS.md)

---

### **مرحله ۲: همگام‌سازی نسخه**

**فایل‌های برای بروزرسانی:**

1. **README.md** (ردیف ۳، ۹)
   ```markdown
   Before: [![Version](https://img.shields.io/badge/version-1.2.0-blue.svg)]
   After:  [![Version](https://img.shields.io/badge/version-1.3.3-blue.svg)]
   
   Before: ![Test Coverage](https://img.shields.io/badge/coverage-11.71%25-red)
   After:  ![Test Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)
   ```

2. **pyproject.toml** (ردیف ۷)
   ```toml
   version = "1.3.3"
   ```

3. **configs/VERSION**
   ```
   1.3.3
   ```

---

### **مرحله ۳: مستندات**

**۱. بروزرسانی CHANGELOG.md:**
```markdown
## [Unreleased] → ## [1.3.3] - 2025-12-05

### Changed
- Fixed 143 files with code quality improvements
- Updated type hints to Python 3.9+ standards
- Added matplotlib lazy-loading for optional dependencies

### Fixed
- W293: Removed blank line whitespace
- I001: Fixed import sorting and organization
- B007: Resolved unused loop variables
- F841: Removed unused variables
- UP006/UP035: Modernized deprecated type imports
```

**۲. ایجاد Release Notes v1.3.3:**
- قالب: مرجع v1.3.0 (docs/releases/RELEASE_NOTES_v1.3.0.md)
- شامل:
  - Overview & Key Changes
  - Testing Summary
  - Deployment Steps
  - Health Checks
  - Version Info

**۳. ایجاد Release Summary (فارسی):**
- عنوان: "خلاصه انتشار v1.3.3"
- شامل: تغییرات کلیدی، اطلاعات استقرار، اطلاع‌رسانی تیم

---

### **مرحله ۴: استقرار**

```bash
# ۱. تایید کد جدید
git pull origin main

# ۲. تایید تست‌ها
python -m pytest tests/ --tb=short

# ۳. استقرار
kubectl apply -f deployment/kubernetes/overlays/prod/

# ۴. بررسی health
curl http://service/health
curl http://service/version

# ۵. Smoke tests
# - Test SMA endpoint
# - Test RSI endpoint
# - Test MACD endpoint
```

---

### **مرحله ۵: آخرین تایید‌ها**

**چک‌لیست نهایی:**

- [ ] تمام تست‌ها: ✅ PASS
- [ ] Coverage: ✅ 95%+
- [ ] نسخه‌ها: ✅ همگام
- [ ] CHANGELOG: ✅ بروزرسانی شد
- [ ] Release Notes: ✅ ایجاد شد
- [ ] Health endpoint: ✅ 200 OK
- [ ] Version endpoint: ✅ 1.3.3 را برمی‌گرداند

---

### **مرحله ۶: GitHub Release**

```bash
# ۱. ایجاد tag
git tag -a v1.3.3 -m "Release v1.3.3: Code quality improvements"

# ۲. Push tag
git push origin v1.3.3

# ۳. ایجاد GitHub Release (UI):
# - Title: "v1.3.3 - Code Quality & Type Safety Improvements"
# - Description: Content from RELEASE_NOTES_v1.3.3.md
# - Mark as latest: YES
# - Pre-release: NO
```

---

### **مرحله ۷: اطلاع‌رسانی**

**پیام تیم:**
```
🚀 Release v1.3.3 is Live!

📊 Changes:
- Code quality improvements (143 files)
- Type hints modernized to Python 3.9+
- Test coverage: 95%+

📍 Links:
- Release: https://github.com/Shakour-Data/Gravity_TechnicalAnalysis/releases/tag/v1.3.3
- Release Notes: docs/releases/RELEASE_NOTES_v1.3.3.md

✅ Status: All health checks passing
🔗 Version endpoint: 1.3.3
```

---

## 📌 نکات مهم

### ⚠️ موارد بحرانی (DO NOT SKIP)

1. **Coverage < 95%**: قطع فرآیند تا رسیدن به هدف
2. **نسخه‌های ناهماهنگ**: تمام منابع نسخه باید یکسان باشند
3. **CHANGELOG‌ بروزرسانی نشده**: [Unreleased] باید [1.3.3] شود
4. **Health checks ناموفق**: تمام endpoints باید 200 OK بدهند

### ✅ معیارهای موفقیت

- تمام ۱۲۰۰+ تست: ✅ PASS
- Coverage: ✅ 95%+
- نسخه‌ها: ✅ 1.3.3 همه جا
- مستندات: ✅ کامل و بروزرسانی شده
- Endpoints: ✅ پاسخ‌گو و سالم

---

## 📞 تماس و مسئولان

- **مدیر ریلیز**: @Shakour-Data
- **تیم QA**: بررسی coverage و تست‌ها
- **DevOps**: استقرار و health checks
- **تیم ML**: تایید ML تست‌ها

---

**زمان انتظار کل**: ۲-۴ ساعت  
**وضعیت**: آماده شروع ✅  
**آخرین به‌روزرسانی**: ۵ دسامبر ۲۰۲۵

---

## 🔗 منابع و مراجع

- CRITICAL_PRIORITY_ANALYSIS.md (ردیف‌های 15-67, 346)
- docs/releases/RELEASE_NOTES_v1.3.0.md (ردیف‌های 661-707)
- QUICK_START_10_DAYS.md (ردیف 126)
- docs/changelog/CHANGELOG.md (ردیف 562)
