# راهنمای دستورات CLI مدیریت دیتابیس

این فایل راهنمای کامل استفاده از دستورات CLI برای مدیریت دیتابیس پروژه Gravity Tech را ارائه می‌دهد.

## نصب وابستگی‌ها

ابتدا کتابخانه `click` را نصب کنید:

```bash
pip install click>=8.1.7
```

یا تمام وابستگی‌ها را نصب کنید:

```bash
pip install -r requirements.txt
```

## دستورات موجود

### 1. راه‌اندازی دیتابیس (init)

ایجاد دیتابیس و جداول:

```bash
# تشخیص خودکار نوع دیتابیس
python -m gravity_tech.cli.db_commands init

# استفاده از PostgreSQL
python -m gravity_tech.cli.db_commands init --type postgresql --connection "postgresql://user:pass@localhost/gravity"

# استفاده از SQLite
python -m gravity_tech.cli.db_commands init --type sqlite --sqlite-path data/gravity_tech.db

# بازنویسی دیتابیس موجود
python -m gravity_tech.cli.db_commands init --force
```

### 2. بررسی وضعیت دیتابیس (status)

نمایش وضعیت و آمار دیتابیس:

```bash
python -m gravity_tech.cli.db_commands status
```

خروجی نمونه:
```
📊 وضعیت دیتابیس:
   نوع: sqlite
   وضعیت: ✅ فعال
   مسیر: data/gravity_tech.db
   تعداد جداول: 4

📈 آمار:
   historical_scores: 1,250 رکورد
   tool_performance_history: 3,456 رکورد
   tool_performance_stats: 89 رکورد
   ml_weights_history: 23 رکورد
```

### 3. لیست جداول (tables)

نمایش تمام جداول و تعداد رکوردها:

```bash
python -m gravity_tech.cli.db_commands tables
```

### 4. نمایش schema جدول (schema)

مشاهده ساختار یک جدول:

```bash
python -m gravity_tech.cli.db_commands schema historical_scores
```

### 5. بازنشانی جدول (reset-table)

حذف تمام داده‌های یک جدول:

```bash
# با تأیید کاربر
python -m gravity_tech.cli.db_commands reset-table historical_scores

# بدون تأیید (احتیاط!)
python -m gravity_tech.cli.db_commands reset-table historical_scores --force
```

### 6. بازنشانی تمام جداول (reset-all)

حذف تمام داده‌ها از همه جداول:

```bash
# با تأیید دوبار
python -m gravity_tech.cli.db_commands reset-all

# بدون تأیید (خطرناک!)
python -m gravity_tech.cli.db_commands reset-all --force
```

### 7. به‌روزرسانی دیتابیس (migrate)

اعمال تغییرات schema:

```bash
python -m gravity_tech.cli.db_commands migrate
```

### 8. پشتیبان‌گیری (backup)

ایجاد backup از دیتابیس:

```bash
# Backup کامل
python -m gravity_tech.cli.db_commands backup

# Backup با نام مشخص
python -m gravity_tech.cli.db_commands backup --output my_backup.json

# Backup جداول خاص
python -m gravity_tech.cli.db_commands backup --tables historical_scores,tool_performance_history

# Backup با مسیر SQLite مشخص
python -m gravity_tech.cli.db_commands backup --sqlite-path data/custom.db
```

### 9. بازیابی از backup (restore)

بازگردانی داده‌ها از فایل backup:

```bash
# بازیابی با تأیید
python -m gravity_tech.cli.db_commands restore backup_20251205_120000.json

# بازیابی بدون تأیید
python -m gravity_tech.cli.db_commands restore backup.json --force
```

### 10. import داده‌ها (import-data)

وارد کردن داده‌ها از فایل JSON:

```bash
# Import ساده
python -m gravity_tech.cli.db_commands import-data data.json --table historical_scores

# Import با batch size مشخص
python -m gravity_tech.cli.db_commands import-data large_data.json --table historical_scores --batch-size 500
```

فرمت فایل JSON:

```json
[
  {
    "symbol": "BTCUSDT",
    "timestamp": "2025-12-05T10:00:00",
    "timeframe": "1h",
    "trend_score": 0.75,
    "combined_score": 0.82
  },
  ...
]
```

یا:

```json
{
  "historical_scores": [
    {...},
    {...}
  ]
}
```

### 11. export جدول (export-table)

خروجی گرفتن از یک جدول:

```bash
# Export کامل
python -m gravity_tech.cli.db_commands export-table historical_scores

# Export با محدودیت تعداد
python -m gravity_tech.cli.db_commands export-table historical_scores --limit 100

# Export با فیلتر
python -m gravity_tech.cli.db_commands export-table historical_scores --where "symbol='BTCUSDT'"

# Export با نام فایل مشخص
python -m gravity_tech.cli.db_commands export-table historical_scores --output my_export.json
```

### 12. اجرای Query (query)

اجرای یک query SQL دلخواه:

```bash
# Query ساده
python -m gravity_tech.cli.db_commands query "SELECT COUNT(*) FROM historical_scores"

# Query پیچیده
python -m gravity_tech.cli.db_commands query "SELECT symbol, AVG(combined_score) as avg_score FROM historical_scores GROUP BY symbol"

# ذخیره نتایج در فایل
python -m gravity_tech.cli.db_commands query "SELECT * FROM historical_scores LIMIT 100" --output results.json
```

## سناریوهای کاربردی

### راه‌اندازی اولیه پروژه

```bash
# 1. ایجاد دیتابیس
python -m gravity_tech.cli.db_commands init

# 2. بررسی وضعیت
python -m gravity_tech.cli.db_commands status

# 3. مشاهده جداول
python -m gravity_tech.cli.db_commands tables
```

### Backup روزانه

```bash
# ایجاد backup با تاریخ
python -m gravity_tech.cli.db_commands backup --output "backup_$(date +%Y%m%d).json"
```

در Windows (PowerShell):
```powershell
python -m gravity_tech.cli.db_commands backup --output "backup_$(Get-Date -Format 'yyyyMMdd').json"
```

### انتقال داده‌ها بین دیتابیس‌ها

```bash
# 1. Export از دیتابیس قدیم
python -m gravity_tech.cli.db_commands export-table historical_scores --sqlite-path data/old.db --output old_data.json

# 2. Import به دیتابیس جدید
python -m gravity_tech.cli.db_commands import-data old_data.json --table historical_scores --sqlite-path data/new.db
```

### پاکسازی داده‌های قدیمی

```bash
# حذف داده‌های قدیمی‌تر از 30 روز
python -m gravity_tech.cli.db_commands query "DELETE FROM historical_scores WHERE created_at < datetime('now', '-30 days')"
```

### تحلیل داده‌ها

```bash
# آمار به تفکیک symbol
python -m gravity_tech.cli.db_commands query "SELECT symbol, COUNT(*) as count, AVG(combined_score) as avg_score FROM historical_scores GROUP BY symbol ORDER BY count DESC"

# بهترین سیگنال‌ها
python -m gravity_tech.cli.db_commands query "SELECT * FROM historical_scores WHERE combined_confidence > 0.9 ORDER BY combined_score DESC LIMIT 10"
```

## گزینه‌های عمومی

تمام دستورات از گزینه‌های زیر پشتیبانی می‌کنند:

- `--sqlite-path`: مسیر فایل SQLite (پیش‌فرض: `data/gravity_tech.db`)
- `--help`: نمایش راهنما

مثال:

```bash
python -m gravity_tech.cli.db_commands status --help
```

## نکات مهم

### 🔒 امنیت

- قبل از عملیات بازنشانی، حتماً backup بگیرید
- از دستورات `--force` با احتیاط استفاده کنید
- فایل‌های backup را در مکان امن نگهداری کنید

### 🚀 عملکرد

- برای import داده‌های زیاد، از `--batch-size` استفاده کنید
- export جداول بزرگ را با `--limit` محدود کنید
- از index‌ها برای بهبود سرعت query استفاده کنید

### 🐛 عیب‌یابی

اگر با خطا مواجه شدید:

1. بررسی کنید دیتابیس وجود دارد:
   ```bash
   python -m gravity_tech.cli.db_commands status
   ```

2. بررسی schema:
   ```bash
   python -m gravity_tech.cli.db_commands schema TABLE_NAME
   ```

3. بررسی logs در فایل `logs/database.log`

## پشتیبانی از PostgreSQL

برای استفاده از PostgreSQL:

```bash
# نصب driver
pip install psycopg2-binary

# راه‌اندازی
python -m gravity_tech.cli.db_commands init --type postgresql --connection "postgresql://user:password@localhost:5432/gravity_tech"
```

## استفاده در اسکریپت‌ها

می‌توانید CLI را در اسکریپت‌های خود استفاده کنید:

```python
import subprocess

# اجرای دستور
result = subprocess.run(
    ["python", "-m", "gravity_tech.cli.db_commands", "status"],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print("Success:", result.stdout)
else:
    print("Error:", result.stderr)
```

## توسعه بیشتر

برای افزودن دستورات جدید، فایل `src/gravity_tech/cli/db_commands.py` را ویرایش کنید.

## لایسنس

MIT License - Gravity Tech Team
