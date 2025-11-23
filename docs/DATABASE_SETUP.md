# راهنمای راه‌اندازی دیتابیس

## 🎯 خلاصه

این سیستم **خودکار** دیتابیس را راه‌اندازی می‌کند و **بدون دیتابیس** هم کار می‌کند!

## 🚀 یک دستور - همه چیز آماده!

```bash
python setup_database.py
```

**همین!** سیستم خودش همه چیز را تشخیص و راه‌اندازی می‌کند.

---

## 📋 استراتژی Auto-Detection

سیستم به ترتیب اولویت این گزینه‌ها را امتحان می‌کند:

### 1️⃣ **PostgreSQL** (اولویت اول)
```bash
# اگر psycopg2 نصب باشد و connection string موجود باشد
export DATABASE_URL="postgresql://user:pass@localhost:5432/gravity_tech"
python setup_database.py
```

**مزایا:**
- ✅ عملکرد بالا
- ✅ مقیاس‌پذیری
- ✅ Transaction قوی
- ✅ Functions و Triggers

### 2️⃣ **SQLite** (Fallback اول)
```bash
# اگر PostgreSQL موجود نباشد، خودکار به SQLite می‌رود
python setup_database.py
```

**مزایا:**
- ✅ بدون نیاز به سرور
- ✅ فایل محلی (`data/tool_performance.db`)
- ✅ سریع برای تست و توسعه
- ✅ Schema کامل

### 3️⃣ **JSON File** (Fallback نهایی)
```bash
# اگر هیچ دیتابیس‌ای موجود نباشد
python setup_database.py
```

**مزایا:**
- ✅ هیچ dependency نیاز ندارد
- ✅ فایل JSON ساده (`data/tool_performance.json`)
- ✅ قابل خواندن توسط انسان
- ✅ همیشه کار می‌کند

---

## 🔧 استفاده در کد

### استفاده خودکار (پیشنهادی)

```python
from database.database_manager import DatabaseManager

# Auto-detect و auto-setup
db = DatabaseManager(auto_setup=True)

# استفاده
record_id = db.record_tool_performance(
    tool_name="MACD",
    tool_category="trend_indicators",
    symbol="BTCUSDT",
    timeframe="1d",
    market_regime="trending_bullish",
    prediction_type="bullish",
    confidence_score=0.85
)

# دریافت آمار
stats = db.get_tool_accuracy(
    tool_name="MACD",
    market_regime="trending_bullish",
    days=30
)

print(f"Accuracy: {stats['accuracy']:.1%}")

# بستن اتصال
db.close()
```

### استفاده با Context Manager

```python
from database.database_manager import DatabaseManager

with DatabaseManager(auto_setup=True) as db:
    # اتصال خودکار باز می‌شود
    record_id = db.record_tool_performance(...)
    
    # استفاده
    stats = db.get_tool_accuracy("RSI")
    
# اتصال خودکار بسته می‌شود
```

### مشخص کردن نوع دیتابیس

```python
from database.database_manager import DatabaseManager, DatabaseType

# Force PostgreSQL
db = DatabaseManager(
    db_type=DatabaseType.POSTGRESQL,
    connection_string="postgresql://user:pass@localhost/db",
    auto_setup=True
)

# Force SQLite
db = DatabaseManager(
    db_type=DatabaseType.SQLITE,
    sqlite_path="data/my_custom.db",
    auto_setup=True
)

# Force JSON
db = DatabaseManager(
    db_type=DatabaseType.JSON_FILE,
    json_path="data/my_custom.json",
    auto_setup=True
)
```

---

## 🗄️ ساختار دیتابیس

### جداول ایجاد شده:

#### 1. `tool_performance_history`
ذخیره عملکرد تاریخی هر ابزار

```sql
- tool_name, tool_category
- symbol, timeframe, market_regime
- prediction_type, confidence_score
- actual_result, success, accuracy
- timestamps, metadata
```

#### 2. `tool_performance_stats`
آمار تجمیعی برای سرعت بیشتر

```sql
- tool_name, market_regime
- total_predictions, correct_predictions
- accuracy, avg_confidence
- success rates
```

#### 3. `ml_weights_history`
وزن‌های یادگرفته شده ML در طول زمان

```sql
- model_name, model_version
- weights (JSON)
- training_accuracy, validation_accuracy
- training_date
```

#### 4. `tool_recommendations_log`
لاگ پیشنهادات داده شده

```sql
- request_id, user_id
- symbol, timeframe, analysis_goal
- recommended_tools (JSON)
- user_feedback, trade_result
```

---

## 📦 نصب Dependencies

### برای PostgreSQL:
```bash
pip install psycopg2-binary
```

### برای SQLite:
```bash
# SQLite built-in است - نیازی به نصب نیست
```

### برای JSON:
```bash
# JSON built-in است - نیازی به نصب نیست
```

---

## 🐳 Docker Setup

### با PostgreSQL:

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: gravity_tech
      POSTGRES_USER: gravity
      POSTGRES_PASSWORD: your_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  app:
    build: .
    environment:
      DATABASE_URL: postgresql://gravity:your_password@postgres:5432/gravity_tech
    depends_on:
      - postgres

volumes:
  postgres_data:
```

```bash
docker-compose up -d
python setup_database.py
```

---

## 🧪 تست

### تست اتصال:

```bash
python -c "from database.database_manager import DatabaseManager; db = DatabaseManager(); print(f'✅ {db.db_type.value}')"
```

### تست نوشتن/خواندن:

```python
from database.database_manager import DatabaseManager

with DatabaseManager() as db:
    # نوشتن
    id = db.record_tool_performance(
        tool_name="TEST",
        tool_category="test",
        symbol="TEST",
        timeframe="1d",
        market_regime="test",
        prediction_type="test",
        confidence_score=0.5
    )
    print(f"✅ Written: ID={id}")
    
    # خواندن
    stats = db.get_tool_accuracy("TEST")
    print(f"✅ Read: {stats}")
```

---

## ⚠️ عیب‌یابی

### مشکل: PostgreSQL متصل نمی‌شود

```bash
# بررسی connection string
echo $DATABASE_URL

# تست دستی
psql $DATABASE_URL

# اگر مشکل دارد، سیستم خودکار به SQLite می‌رود
```

### مشکل: SQLite permission error

```bash
# بررسی مجوزها
ls -la data/

# ساخت directory
mkdir -p data
chmod 755 data
```

### مشکل: JSON file corrupted

```bash
# پاک کردن و ساخت مجدد
rm data/tool_performance.json
python setup_database.py
```

---

## 🔄 Migration

### به‌روزرسانی schema:

```bash
# سیستم خودکار schema را بررسی و به‌روزرسانی می‌کند
python setup_database.py
```

### Backup:

```bash
# PostgreSQL
pg_dump $DATABASE_URL > backup.sql

# SQLite
cp data/tool_performance.db backup.db

# JSON
cp data/tool_performance.json backup.json
```

### Restore:

```bash
# PostgreSQL
psql $DATABASE_URL < backup.sql

# SQLite
cp backup.db data/tool_performance.db

# JSON
cp backup.json data/tool_performance.json
```

---

## 📊 مانیتورینگ

### بررسی تعداد رکوردها:

```python
from database.database_manager import DatabaseManager

with DatabaseManager() as db:
    if db.db_type.value == "json_file":
        count = len(db.json_data["tool_performance_history"])
    else:
        result = db.execute_query(
            "SELECT COUNT(*) FROM tool_performance_history",
            fetch=True
        )
        count = result[0][0]
    
    print(f"📊 Total records: {count}")
```

---

## 🎓 Best Practices

### 1. همیشه از Context Manager استفاده کنید
```python
with DatabaseManager() as db:
    # code
```

### 2. Error handling
```python
try:
    with DatabaseManager() as db:
        db.record_tool_performance(...)
except Exception as e:
    logger.error(f"Database error: {e}")
    # Fallback logic
```

### 3. Connection pooling
```python
# PostgreSQL خودکار از connection pool استفاده می‌کند
# SQLite و JSON نیازی به pool ندارند
```

### 4. Regular backups
```bash
# Cron job for daily backup
0 2 * * * python backup_database.py
```

---

## 🚀 Production Checklist

- [ ] PostgreSQL نصب و راه‌اندازی شده
- [ ] `DATABASE_URL` در environment variables تنظیم شده
- [ ] `python setup_database.py` اجرا شده
- [ ] Schema ایجاد شده (4 جدول)
- [ ] Test record نوشته و خوانده شده
- [ ] Backup strategy تنظیم شده
- [ ] Monitoring فعال است

---

## 📞 پشتیبانی

اگر مشکلی پیش آمد:

1. **Log بررسی کنید**: سیستم همه چیز را log می‌کند
2. **Fallback امتحان کنید**: `DatabaseType.SQLITE` یا `DatabaseType.JSON_FILE`
3. **سیستم همیشه کار می‌کند**: حتی بدون دیتابیس!

---

**✅ سیستم آماده است! یک دستور کافی است:**

```bash
python setup_database.py
```
