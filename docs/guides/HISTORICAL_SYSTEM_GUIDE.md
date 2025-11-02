# راهنمای سیستم تاریخی امتیازدهی

## 🎯 هدف

این سیستم **تمام امتیازها، اندیکاتورها، و ضرایب را به صورت تاریخی ذخیره** می‌کند تا:

1. ✅ کاربر بتواند امتیاز **هر تاریخی** را ببیند
2. ✅ نمودارهای تاریخی ترسیم شود
3. ✅ Backtesting انجام شود
4. ✅ عملکرد اندیکاتورها تحلیل شود
5. ✅ الگوهای موفق شناسایی شوند

---

## 📁 فایل‌های ایجاد شده

### 1. `database/schemas.sql`
Schema کامل PostgreSQL شامل:
- **8 جدول اصلی**:
  - `historical_scores` - امتیازهای کلی
  - `historical_horizon_scores` - امتیازهای 3d/7d/30d
  - `historical_indicator_scores` - امتیازهای تک تک اندیکاتورها
  - `historical_patterns` - الگوهای تشخیص داده شده
  - `historical_ml_weights` - وزن‌های یادگیری شده ML
  - `historical_price_targets` - اهداف قیمتی
  - `historical_volume_analysis` - تحلیل حجم
  - `analysis_metadata` - متادیتا برای کش
  
- **Views و Functions**:
  - `v_complete_scores` - نمای کامل با horizons
  - `v_latest_scores` - آخرین تحلیل هر symbol
  - `get_score_at_date()` - دریافت امتیاز در تاریخ خاص
  - `get_score_timeseries()` - سری زمانی برای نمودار
  - `cleanup_old_scores()` - حذف داده‌های قدیمی

### 2. `database/historical_manager.py`
کلاس Python برای مدیریت دیتابیس:
- `HistoricalScoreManager` - مدیر اصلی
- `save_score()` - ذخیره کامل یک تحلیل
- `get_latest_score()` - دریافت آخرین امتیاز
- `get_score_at_date()` - دریافت امتیاز در تاریخ خاص
- `get_score_timeseries()` - دریافت سری زمانی
- `get_indicator_performance()` - عملکرد اندیکاتورها
- `get_pattern_success_rate()` - نرخ موفقیت الگوها

### 3. `example_historical_system.py`
مثال کامل استفاده:
- ذخیره خودکار در هنگام تحلیل
- نمایش امتیازهای تاریخی
- رسم نمودارها
- تحلیل عملکرد

---

## 🚀 راه‌اندازی (Setup)

### مرحله 1: نصب PostgreSQL

#### Windows:
```bash
# دانلود از: https://www.postgresql.org/download/windows/
# نصب با installer
# یا با Chocolatey:
choco install postgresql
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

#### macOS:
```bash
brew install postgresql
brew services start postgresql
```

### مرحله 2: ایجاد دیتابیس

```bash
# ورود به PostgreSQL
psql -U postgres

# ایجاد دیتابیس
CREATE DATABASE trading_db;

# ایجاد کاربر
CREATE USER trading_user WITH PASSWORD 'your_secure_password';

# دادن دسترسی
GRANT ALL PRIVILEGES ON DATABASE trading_db TO trading_user;

# خروج
\q
```

### مرحله 3: اجرای Schema

```bash
# اجرای فایل SQL
psql -U trading_user -d trading_db -f database/schemas.sql

# یا از داخل psql:
psql -U trading_user -d trading_db
\i database/schemas.sql
```

### مرحله 4: نصب پکیج‌های Python

```bash
pip install psycopg2-binary
pip install pandas
pip install matplotlib
```

### مرحله 5: تنظیم Connection String

در فایل‌های Python، این خط را تنظیم کنید:

```python
DATABASE_URL = "postgresql://trading_user:your_password@localhost:5432/trading_db"
```

یا از متغیر محیطی:

```bash
export DATABASE_URL="postgresql://trading_user:your_password@localhost:5432/trading_db"
```

```python
import os
DATABASE_URL = os.getenv("DATABASE_URL")
```

---

## 💻 نحوه استفاده

### 1. ذخیره خودکار در هنگام تحلیل

```python
from database.historical_manager import HistoricalScoreManager, HistoricalScoreEntry
from ml.multi_horizon_analysis import MultiHorizonTrendAnalyzer
from ml.multi_horizon_momentum_analysis import MultiHorizonMomentumAnalyzer
from datetime import datetime

# اتصال به دیتابیس
manager = HistoricalScoreManager(DATABASE_URL)

# تحلیل
trend_result = trend_analyzer.analyze(trend_features)
momentum_result = momentum_analyzer.analyze(momentum_features)

# محاسبه امتیازهای کلی
trend_overall = calculate_overall(trend_result)
momentum_overall = calculate_overall(momentum_result)
combined_score = (trend_overall * 0.6) + (momentum_overall * 0.4)

# ساخت Entry
score_entry = HistoricalScoreEntry(
    symbol="BTCUSDT",
    timestamp=datetime.now(),
    timeframe="1h",
    trend_score=trend_overall,
    trend_confidence=0.82,
    momentum_score=momentum_overall,
    momentum_confidence=0.70,
    combined_score=combined_score,
    combined_confidence=0.76,
    trend_weight=0.6,
    momentum_weight=0.4,
    trend_signal="VERY_BULLISH",
    momentum_signal="BULLISH",
    combined_signal="BULLISH",
    recommendation="BUY",
    action="ACCUMULATE",
    price_at_analysis=50000.00
)

# ذخیره
with manager:
    score_id = manager.save_score(
        score_entry,
        horizon_scores=[...],  # لیست امتیازهای 3d, 7d, 30d
        indicator_scores=[...],  # لیست امتیازهای هر اندیکاتور
        patterns=[...]  # لیست الگوهای تشخیص داده شده
    )
    print(f"✅ Saved with ID: {score_id}")
```

### 2. دریافت آخرین امتیاز

```python
with HistoricalScoreManager(DATABASE_URL) as manager:
    latest = manager.get_latest_score("BTCUSDT", "1h")
    print(f"Latest score: {latest['combined_score']:.3f}")
    print(f"Recommendation: {latest['recommendation']}")
```

### 3. دریافت امتیاز در تاریخ خاص

```python
from datetime import datetime

target_date = datetime(2024, 1, 15, 10, 0, 0)

with HistoricalScoreManager(DATABASE_URL) as manager:
    score = manager.get_score_at_date("BTCUSDT", target_date, "1h")
    print(f"Score at {target_date}: {score['combined_score']:.3f}")
```

### 4. دریافت سری زمانی (برای نمودار)

```python
from datetime import datetime, timedelta

to_date = datetime.now()
from_date = to_date - timedelta(days=30)

with HistoricalScoreManager(DATABASE_URL) as manager:
    timeseries = manager.get_score_timeseries(
        "BTCUSDT", 
        from_date, 
        to_date, 
        "1h"
    )
    
    # تبدیل به DataFrame
    df = pd.DataFrame(timeseries)
    
    # رسم نمودار
    plt.plot(df['timestamp'], df['combined_score'])
    plt.show()
```

### 5. تحلیل عملکرد اندیکاتورها

```python
with HistoricalScoreManager(DATABASE_URL) as manager:
    performance = manager.get_indicator_performance("BTCUSDT", days=30)
    
    for ind in performance:
        print(f"{ind['indicator_name']}: "
              f"avg_confidence={ind['avg_confidence']:.3f}, "
              f"usage={ind['usage_count']}")
```

### 6. نرخ موفقیت الگوها

```python
with HistoricalScoreManager(DATABASE_URL) as manager:
    success_rates = manager.get_pattern_success_rate(days=90)
    
    for pattern in success_rates:
        print(f"{pattern['pattern_name']}: "
              f"success_rate={pattern['success_rate']:.1%}, "
              f"detected={pattern['detected_count']}")
```

---

## 🌐 API Endpoints (برای میکروسرویس)

### 1. دریافت آخرین امتیاز

```
GET /api/v1/analysis/{symbol}/latest
GET /api/v1/analysis/BTCUSDT/latest?timeframe=1h
```

Response:
```json
{
  "symbol": "BTCUSDT",
  "timestamp": "2024-01-20T10:00:00Z",
  "price": 50000.00,
  "trend_score": 85,
  "momentum_score": 55,
  "combined_score": 72,
  "recommendation": "BUY"
}
```

### 2. دریافت سری زمانی

```
GET /api/v1/history/{symbol}
GET /api/v1/history/BTCUSDT?from=2024-01-01&to=2024-01-31&timeframe=1h
```

Response:
```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "count": 720,
  "data": [
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "trend_score": 75,
      "momentum_score": 50,
      "combined_score": 65,
      "price": 48000.00
    },
    ...
  ]
}
```

### 3. دریافت امتیاز در تاریخ خاص

```
GET /api/v1/history/{symbol}/at/{datetime}
GET /api/v1/history/BTCUSDT/at/2024-01-15T10:00:00Z
```

### 4. عملکرد اندیکاتورها

```
GET /api/v1/indicators/performance?symbol=BTCUSDT&days=30
```

### 5. نرخ موفقیت الگوها

```
GET /api/v1/patterns/success-rate?days=90
```

---

## 📊 نمودارها و Visualization

### نمودار سری زمانی امتیازها

```python
import matplotlib.pyplot as plt
import pandas as pd

# دریافت داده
df = pd.DataFrame(timeseries)

# رسم
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# امتیازها
axes[0].plot(df['timestamp'], df['trend_score'], label='Trend')
axes[0].plot(df['timestamp'], df['momentum_score'], label='Momentum')
axes[0].plot(df['timestamp'], df['combined_score'], label='Combined')
axes[0].legend()
axes[0].set_ylabel('Score')

# قیمت
axes[1].plot(df['timestamp'], df['price'])
axes[1].set_ylabel('Price ($)')
axes[1].set_xlabel('Date')

plt.show()
```

### نمودار عملکرد اندیکاتورها

```python
df_perf = pd.DataFrame(performance)
df_perf = df_perf.sort_values('avg_confidence', ascending=False).head(20)

plt.barh(df_perf['indicator_name'], df_perf['avg_confidence'])
plt.xlabel('Average Confidence')
plt.title('Top 20 Indicators')
plt.show()
```

---

## 🔧 Maintenance

### 1. حذف داده‌های قدیمی

```python
with HistoricalScoreManager(DATABASE_URL) as manager:
    deleted = manager.cleanup_old_data(days_to_keep=365)
    print(f"Deleted {deleted} old records")
```

یا از psql:
```sql
SELECT cleanup_old_scores(365);  -- حذف بیشتر از 1 سال
```

### 2. بررسی حجم دیتا

```sql
-- حجم هر جدول
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- تعداد رکوردها
SELECT 
    'historical_scores' as table_name,
    COUNT(*) as count,
    MIN(timestamp) as oldest,
    MAX(timestamp) as newest
FROM historical_scores;
```

### 3. Backup

```bash
# Backup کامل
pg_dump -U trading_user trading_db > backup_$(date +%Y%m%d).sql

# Backup فقط Schema
pg_dump -U trading_user -s trading_db > schema_backup.sql

# Restore
psql -U trading_user trading_db < backup_20240120.sql
```

---

## 🎯 Use Cases

### 1. Backtesting استراتژی

```python
# دریافت امتیازهای 90 روز گذشته
historical_data = manager.get_score_timeseries(
    "BTCUSDT",
    datetime.now() - timedelta(days=90),
    datetime.now(),
    "1d"
)

# شبیه‌سازی معاملات
for row in historical_data:
    if row['combined_score'] > 0.7:
        # خرید
        ...
    elif row['combined_score'] < -0.7:
        # فروش
        ...
```

### 2. مقایسه عملکرد Timeframe های مختلف

```sql
SELECT 
    timeframe,
    AVG(combined_score) as avg_score,
    STDDEV(combined_score) as volatility,
    COUNT(*) as analyses_count
FROM historical_scores
WHERE symbol = 'BTCUSDT'
  AND timestamp > NOW() - INTERVAL '30 days'
GROUP BY timeframe
ORDER BY timeframe;
```

### 3. شناسایی الگوهای موفق

```sql
SELECT 
    pattern_name,
    COUNT(*) as total_detected,
    AVG(confidence) as avg_confidence,
    COUNT(CASE WHEN target_reached THEN 1 END) as success_count
FROM historical_patterns hp
JOIN historical_price_targets hpt ON hp.score_id = hpt.score_id
GROUP BY pattern_name
HAVING COUNT(*) > 10
ORDER BY success_count DESC;
```

---

## 🔒 Security & Performance

### Security:
- ✅ استفاده از parameterized queries (جلوگیری از SQL injection)
- ✅ محدود کردن دسترسی کاربر دیتابیس
- ✅ رمزنگاری connection string
- ✅ استفاده از SSL برای اتصال production

### Performance:
- ✅ Indexes روی timestamp, symbol, timeframe
- ✅ استفاده از Views برای queries پرکاربرد
- ✅ Partitioning جداول برای دیتای خیلی زیاد
- ✅ Connection pooling برای concurrent requests

---

## 📚 مراجع

- Schema: `database/schemas.sql`
- Manager: `database/historical_manager.py`
- Example: `example_historical_system.py`
- PostgreSQL Docs: https://www.postgresql.org/docs/

---

## ✅ Checklist راه‌اندازی

- [ ] PostgreSQL نصب شده
- [ ] دیتابیس `trading_db` ایجاد شده
- [ ] Schema اجرا شده (`schemas.sql`)
- [ ] پکیج‌های Python نصب شده (`psycopg2`, `pandas`, `matplotlib`)
- [ ] Connection string تنظیم شده
- [ ] تست اتصال موفق
- [ ] اولین تحلیل ذخیره شده
- [ ] نمودارها نمایش داده می‌شوند
- [ ] API endpoints پیاده‌سازی شده (اختیاری)

---

**🎉 با راه‌اندازی این سیستم، تمام امتیازها، اندیکاتورها، و ضرایب به صورت تاریخی ذخیره می‌شوند و کاربر می‌تواند امتیاز هر تاریخی را بازیابی کند!**
