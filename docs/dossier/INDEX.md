# پرونده فنی پروژه «Gravity Technical Analysis» (نسخه ارائه به کارفرما)

این پوشه یک «پرونده مستندات کامل» برای توضیح دقیق اینکه پروژه چه می‌کند، چگونه پیاده‌سازی شده، چه الگوریتم‌ها/اندیکاتورهایی دارد، چه داده‌هایی مصرف/تولید می‌کند و چگونه استقرار و بهره‌برداری می‌شود ارائه می‌دهد.

> هدف: ارائه به کارفرما و استفاده در مستندات اخذ/تکمیل پرونده دانش‌بنیان (شرح محصول، نوآوری فنی، معماری، الگوریتم‌ها، داده، امنیت و کیفیت).

## 0) راهنمای مطالعه
- اگر می‌خواهید سریع متوجه شوید پروژه چه می‌کند: `docs/dossier/01_EXECUTIVE_SUMMARY.md`
- اگر می‌خواهید معماری و اجزای سیستم را ببینید: `docs/dossier/03_SYSTEM_ARCHITECTURE.md`
- اگر می‌خواهید دقیقاً موتور تحلیل چگونه خروجی می‌دهد: `docs/dossier/05_ANALYSIS_ENGINE.md`
- اگر می‌خواهید لیست اندیکاتورها/فرمول‌ها/پارامترها را ببینید: `docs/dossier/06_INDICATORS_CATALOG.md`
- اگر می‌خواهید Pattern/ML را ببینید: `docs/dossier/07_PATTERN_DETECTION.md` و `docs/dossier/08_ML_SUBSYSTEM.md`
- اگر می‌خواهید استقرار و عملیات و امنیت را ببینید: `docs/dossier/10_DEPLOYMENT_AND_OPERATIONS.md` و `docs/dossier/11_SECURITY_AND_COMPLIANCE.md`

## 1) فهرست مستندات
- `docs/dossier/01_EXECUTIVE_SUMMARY.md` — خلاصه مدیریتی، دامنه، خروجی‌ها، نقاط تمایز
- `docs/dossier/02_PRODUCT_AND_SCOPE.md` — تعریف محصول/مسئله، پرسونای کاربر، ورودی/خروجی‌ها، سناریوهای استفاده
- `docs/dossier/03_SYSTEM_ARCHITECTURE.md` — معماری (C4 ساده)، دیاگرام‌ها، اجزا و مسئولیت‌ها
- `docs/dossier/04_DATA_PIPELINE_AND_STORAGE.md` — جریان داده، لایه ذخیره‌سازی، اسکیمای Postgres/SQLite
- `docs/dossier/05_ANALYSIS_ENGINE.md` — Pipeline تحلیل، محاسبه سیگنال نهایی، شبه‌کد و منطق امتیازدهی
- `docs/dossier/06_INDICATORS_CATALOG.md` — کاتالوگ اندیکاتورها: فرمول/پارامتر/سیگنال/اعتماد
- `docs/dossier/07_PATTERN_DETECTION.md` — تشخیص الگو: کندلی، هارمونیک، الیوت، فاز بازار
- `docs/dossier/08_ML_SUBSYSTEM.md` — زیرسیستم ML (مدل‌ها، ویژگی‌ها، مدیریت مدل، fallback)
- `docs/dossier/09_API_SPEC.md` — مشخصات API و قراردادها (REST/Health/Metrics)
- `docs/dossier/10_DEPLOYMENT_AND_OPERATIONS.md` — استقرار (Docker)، Runbook، پایپ‌لاین روزانه
- `docs/dossier/11_SECURITY_AND_COMPLIANCE.md` — امنیت، لاگینگ، کنترل دسترسی، ملاحظات ارائه سازمانی
- `docs/dossier/12_TESTING_AND_VALIDATION.md` — تست/کیفیت، گزارش‌گیری و اعتبارسنجی داده/نتیجه

## 2) ارجاعات به مستندات موجود پروژه
این پرونده جایگزین مستندات فنی داخل `docs/` نیست؛ بلکه یک بسته یکپارچه و ارائه‌محور است. برای مسیرهای قدیمی/پراکنده، از `docs/INDEX.md` استفاده کنید.

