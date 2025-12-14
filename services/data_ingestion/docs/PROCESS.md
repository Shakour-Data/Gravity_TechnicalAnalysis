# راهنمای فرآیندها (GravityTseHisPrice)

این راهنما مراحل اجرایی، به‌روزرسانی و عیب‌یابی را بر اساس پیاده‌سازی فعلی پوشش می‌دهد. همه مسیرها و فایل‌ها در `src/config.py` نگه‌داری می‌شوند.

## پیش‌نیاز سریع
- نصب وابستگی‌ها: `pip install -r requirements.txt`
- پایگاه داده: `data/tse_data.db` (SQLite)
- داده‌های مرجع اولیه: `data/BasicTseInformation/*.json`

## راه‌اندازی اولیه
- ایجاد جدول‌های پایه:  
  `python main.py create-db`
- ایجاد جدول‌های شاخص و USD:  
  `python main.py create-indices-tables`
- بارگذاری داده‌های مرجع (سکتور، بازار، تابلو، شرکت):  
  `python main.py load-initial`
- واکشی و درج قیمت‌ها، USD و شاخص‌ها (افزایشی):  
  `python main.py load-all-prices`
- مسیر سریع همه مراحل با یک دستور:  
  `python main.py init-all`

## به‌روزرسانی دوره‌ای
- اجرای مجدد لود افزایشی (از `last_updates`) برای قیمت سهام، USD، شاخص‌های بازار و صنعت:  
  `python main.py load-all-prices`
- همان فرآیند از داخل داشبورد (دکمه «به‌روزرسانی داده‌ها») نیز `DataFetcher.run()` را صدا می‌زند و رکوردهای جدید را درج می‌کند.
- پس از درج، `update_price_data_sectors` وابستگی سکتور برای رکوردهای جدید را تکمیل می‌کند.

## اجرای داشبورد
- دستور: `python run_dashboard.py`
- پورت پیش‌فرض: `http://127.0.0.1:8050/`
- امکانات: وضعیت پایگاه داده، نمودار قیمت/حجم، فهرست شرکت‌ها، اجرای همه دستورات CLI (create/load/drop/update) از UI و مشاهده لاگ‌ها.

## اجرای API
- دستور: `python web/api.py`
- پورت پیش‌فرض: `http://127.0.0.1:5000/`
- Endpointها:
  - `GET /api/summary` خلاصه بازار
  - `GET /api/companies?sector_id=&limit=` فهرست شرکت‌ها
  - `GET /api/price-data/<ticker>?limit=` تاریخچه قیمت
  - `GET /api/sectors` فهرست سکتورها + تعداد شرکت‌ها
  - `GET /api/market-indices?limit=` شاخص‌های بازار (OHLC)
  - `GET /health` بررسی سلامتی

## بازسازی و عیب‌یابی
- اسکریپت‌ها در `scripts/`:
  - `init_db.py` ساخت دوباره جداول و بارگذاری اولیه
  - `fix_db.py`, `migrate_remove_unused_columns.py` تعمیر ساختار و ستون‌های اضافی
  - `check_db_status.py`, `check_tables.py`, `find_empty_columns.py` بررسی سلامت و ستون‌های خالی
  - `reload_all_indices.py`, `clear_market_indices.py`, `check_indices_direct.py` کار با داده‌های شاخص‌ها
- نسخه‌های پشتیبان DB در `data/*.db` نگه‌داری شده‌اند؛ قبل از عملیات تخریبی از DB پشتیبان بگیرید.

## جریان‌های تصویری
### راه‌اندازی سرتاسری (init-all)
```mermaid
flowchart LR
    Start[شروع] --> Create[create-db\ncreate-indices-tables]
    Create --> Seed[load-initial\n(sectors/markets/panels/companies)]
    Seed --> Fetch[load-all-prices\n(DataFetcher.run)]
    Fetch --> Done[DB آماده]
```

## اسناد تفصیلی فرآیندها
- `docs/processes/01_create_db.md` ایجاد DB و جداول پایه/شاخص
- `docs/processes/02_load_initial.md` بارگذاری داده‌های مرجع
- `docs/processes/03_incremental_load.md` بارگذاری افزایشی قیمت/شاخص
- `docs/processes/04_api_service.md` راه‌اندازی و مصرف API
- `docs/processes/05_dashboard.md` اجرای داشبورد وب

### پاسخ‌گویی به درخواست‌های UI/API
```mermaid
flowchart LR
    User[کاربر\nCLI/Dashboard/API Client] --> CLI[main.py CLI\nیا دکمه‌های داشبورد]
    CLI --> Fetcher[DataFetcher.run\n(در صورت به‌روزرسانی)]
    Fetcher --> DB[(SQLite\ndata/tse_data.db)]
    CLI --> DB
    API[web/api.py\nFlask] --> DB
    Dashboard[web/dashboard.py\nDash] --> DB
    DB --> Dashboard
    DB --> API
    DB --> CLI
```
