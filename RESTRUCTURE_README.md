# 🏗️ سازماندهی استاندارد پروژه

## 🎯 هدف

تبدیل پروژه Gravity Technical Analysis به ساختار استاندارد Python Package.

## 📚 فایل‌های کلیدی

| فایل | توضیحات |
|------|---------|
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | **شروع از اینجا** - خلاصه کامل کارها |
| [RESTRUCTURE_PLAN.md](RESTRUCTURE_PLAN.md) | طرح کامل سازماندهی |
| [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) | راهنمای گام به گام migration |
| [CONTRIBUTING.md](CONTRIBUTING.md) | راهنمای مشارکت در پروژه |

## 🚀 شروع سریع

### گام 1: مطالعه
```bash
# خواندن خلاصه
cat IMPLEMENTATION_SUMMARY.md

# مطالعه جزئیات
cat RESTRUCTURE_PLAN.md
```

### گام 2: Backup
```bash
# کپی پروژه
cp -r Gravity_TechnicalAnalysis Gravity_TechnicalAnalysis_backup

# ایجاد branch
git checkout -b refactor/standard-structure
```

### گام 3: Dry Run
```bash
# مشاهده تغییرات (بدون اعمال)
python scripts/migration/migrate_to_standard_structure.py --dry-run
```

### گام 4: Migration
```bash
# اعمال تغییرات
python scripts/migration/migrate_to_standard_structure.py --execute

# ادامه مطابق MIGRATION_GUIDE.md
```

## 📋 ابزارهای جدید

### Makefile
```bash
make help           # نمایش تمام دستورات
make install        # نصب dependencies
make test           # اجرای تست‌ها
make lint           # بررسی کد
make format         # فرمت کردن کد
make run            # اجرای server
```

### اسکریپت Migration
```bash
# Dry run
python scripts/migration/migrate_to_standard_structure.py --dry-run

# Execute
python scripts/migration/migrate_to_standard_structure.py --execute
```

## 🎯 ساختار نهایی

```
src/gravity_tech/       # تمام کد اصلی
tests/                  # Tests مرتب (unit/integration/e2e)
docs/                   # مستندات (en/fa)
deployment/             # Docker/K8s configs
scripts/                # ابزارها
examples/               # مثال‌ها
```

## ✅ چک‌لیست

- [ ] مطالعه IMPLEMENTATION_SUMMARY.md
- [ ] بررسی RESTRUCTURE_PLAN.md
- [ ] Backup گرفتن
- [ ] اجرای dry run
- [ ] مطالعه MIGRATION_GUIDE.md
- [ ] اجرای migration
- [ ] انجام کارهای دستی
- [ ] تست و اعتبارسنجی

## 📞 پشتیبانی

سوال دارید؟
1. مستندات را مطالعه کنید
2. GitHub Issue باز کنید
3. با تیم تماس بگیرید

---

**وضعیت:** ✅ آماده برای اجرا  
**تاریخ:** 2025-12-03
