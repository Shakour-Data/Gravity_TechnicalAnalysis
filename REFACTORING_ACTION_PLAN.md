# Gravity Technical Analysis - Refactoring Action Plan

## Executive Summary

هدف این سند تعریف یک برنامه مرحله‌ای برای بهبود معماری پروژه Gravity Technical Analysis است. این برنامه بر اساس بررسی دقیق ریپو و شناسایی مشکلات معماری و الگوریتمی تهیه‌شده است.

## مشکلات شناسایی‌شده

### 1. معماری و جداسازی لایه‌ها

**مشکل**: جداسازی ناقص بین لایه‌های مختلف
- منطق دامین، API، و پایپ‌لاین داده به شکل غیرمتناسب ترکیب شده‌اند
- وابستگی‌های دوطرفه (circular dependencies) احتمالی
- لایه core جزئی است و هنوز کاملاً از API جدا نشده است

**تأثیر**: تست‌پذیری پایین، نگه‌داری سخت، مقیاس‌پذیری محدود

### 2. فایل‌ها و پوشه‌های موقتی

**مشکل**: موارد زیر در ریشه ریپو موجودند:
- `temp_gravity_base/` - موقتی
- `temp_gravity_tse/` - موقتی
- `batch1_report.txt`, `batch1_symbols.txt` - آرتیفکت‌های گزارش
- `database_connection_info.txt` - اطلاعات حساس!
- `gravity_full_prompt.txt` - موقتی
- `status.txt` - موقتی

**تأثیر**: مرز production/experimental مبهم، ریسک اشتباه، نشت اطلاعات

### 3. هم‌پوشانی پایپ‌لاین داده

**مشکل**: اسکریپت‌های ingestion در دو جای مختلف:
- `services/data_ingestion/scripts/`
- `scripts/` (ریشه)

دقیق نیست که کدام مسیر مرجع است و فرایند loading داده غیرشفاف است.

**تأثیر**: نگه‌داری دشوار، مستندات ناکافی

### 4. تنظیمات چندگانه

**مشکل**:
- متغیرهای محیطی در `.env.example`
- کانفیگ در `configs/tools/catalog.json`
- Docker configs متعدد
- Feature flags مثل `ENABLE_SCENARIOS` و `EXPOSE_DB_EXPLORER` فقط در مستندات هستند

**تأثیر**: ریسک ناسازگاری بین محیط‌ها

### 5. تست و وابستگی‌های خارجی

**مشکل**:
- وابستگی‌های خارجی (Redis، PostgreSQL، مدل‌های ML، سرویس‌های TSE) بدون abstraction مناسب
- تست‌ها نیاز به غیرفعال کردن features دارند به جای mocking

**تأثیر**: تست واقعی دشوار، وابستگی سختی

## راهکار - برنامه مرحله‌ای

### فاز ۱: تمیزکاری و سازمان‌دهی ریپو (1-2 هفته)

#### مرحله ۱.۱: تمیزکردن فایل‌های موقتی
- حذف/منتقل کردن تمام موارد موقتی به `experiments/`
- اضافه کردن `.gitignore` برای جلوگیری از تکرار
- **Status**: پوشه `experiments/` ایجاد شد ✓

#### مرحله ۱.۲: مستندسازی ساختار ریپو
- به‌روزرسانی README با بخش "Repository Structure"
- مشخص کردن production code vs experimental
- اضافه کردن راهنمای Contributing

### فاز ۲: جداسازی Core/Domain (2-3 هفته)

#### مرحله ۲.۱: تقویت پکیج gravity_tech/core
- اندیکاتورها: هم‌اکنون در `core/indicators/` ✓
- الگوهای شمعی و هارمونیک: جابه‌جایی به `core/patterns/`
- اندیکاتورهای امواج الیوت: جابه‌جایی به `core/elliott/`
- Signal Engine: جابه‌جایی به `core/signal_engine/`
- Multi-Horizon: جابه‌جایی به `core/analysis/`

#### مرحله ۲.۲: تعریف اینترفیس‌های مشترک
- ایجاد `core/domain/entities.py` برای مدل‌های مشترک
- تعریف `core/domain/contracts.py` برای اینترفیس‌های خدمات

### فاز ۳: یکپارچه‌سازی پایپ‌لاین داده (2-3 هفته)

#### مرحله ۳.۱: ایجاد ماژول gravity_pipeline
- ایجاد `gravity_pipeline/` در سطح apps
- انتقال منطق ingestion از `scripts/` و `services/`
- تعریف مراحل: Extract → Transform → Load

#### مرحله ۳.۲: منطق‌پذیر کردن کانفیگ‌ها
- ایجاد `gravity_pipeline/config.py` برای تمام تنظیمات
- حذف مسیرهای هاردکد

### فاز ۴: بهبود تست و Mocking (1-2 هفته)

#### مرحله ۴.۱: ایجاد Abstraction برای وابستگی‌های خارجی
- ایجاد `core/infrastructure/` برای adapter patterns
- تعریف interfaces برای Redis، Database، external services

#### مرحله ۴.۲: افزایش تست‌های واحد
- هر اندیکاتور: unit test ✓ (98% coverage)
- الگوها: unit test
- Signal Engine: integration test

### فاز ۵: مستندات و استقرار (1 هفته)

#### مرحله ۵.۱: به‌روزرسانی مستندات
- معماری: بالا‌روز کردن `docs/architecture/`
- API: تکمیل `docs/guides/API_REFERENCE.md`
- Deployment: بالا‌روز کردن `docs/operations/`

## اولویت‌بندی

### Priority 1 (باید انجام شود)
1. ✓ ایجاد `experiments/` پوشه
2. حذف فایل‌های موقتی
3. تقویت `core/` package
4. یکپارچه‌سازی پایپ‌لاین داده

### Priority 2 (مهم)
5. بهبود تست و mocking
6. منطق‌پذیر کردن کانفیگ‌ها
7. مستندسازی کامل

### Priority 3 (خوب است)
8. بهبود performance profiling
9. اضافه کردن metrics و monitoring
10. صفحه‌بندی log‌ها

## Timeline

**کل مدت**: 8-10 هفته

```
هفته 1-2:   تمیزکاری ریپو + مستندات
هفته 3-5:   جداسازی Core/Domain
هفته 6-8:   پایپ‌لاین داده + تست
هفته 9-10:  مستندات نهایی + استقرار
```

## معیارهای موفقیت

✓ تمام فایل‌های موقتی حذف/منظم‌شده
✓ Core package کاملاً جدا‌شده از API
✓ Pipeline داده transparent و قابل تست
✓ 90%+ test coverage برای core
✓ مستندات معماری کامل و آپ‌تودیت
✓ صفر circular dependencies

## نکات اضافی

- هر فاز می‌تواند یک PR جداگانه باشد
- کمیت‌های کوچک و متعادل برای سهولت review
- هر مرحله باید backward compatible باشد
- تست‌ها باید PASS کنند قبل از merge

---

**آخرین به‌روزرسانی**: 2025-12-18  
**وضعیت**: مرحله ۱ شروع شده
