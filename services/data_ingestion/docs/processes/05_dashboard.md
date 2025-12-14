# فرآیند ۵: داشبورد وب (Dash)

این فرآیند اجرای داشبورد تحلیلی و دسترسی تعاملی به داده‌ها و دستورات CLI را شرح می‌دهد.

## هدف
- ارائه UI برای خلاصه بازار، نمودار قیمت/حجم، فهرست شرکت‌ها و جدول‌های DB
- امکان اجرای دستورات CLI از UI (create/load/drop/update)

## پیش‌نیازها
- پایگاه داده موجود (`data/tse_data.db`)
- وابستگی‌ها: `dash`, `plotly`, `dash-bootstrap-components`, `pandas`

## دستور اجرا
```bash
python run_dashboard.py
# پیش‌فرض: http://127.0.0.1:8050/
```

## گام‌ها (کد)
1) `run_dashboard.py` بررسی وابستگی‌ها و وجود DB → سپس `web/dashboard.py`
2) داشبورد:
   - بارگذاری خلاصه بازار (`get_market_summary`)
   - نمودار قیمت/حجم (`get_recent_price_data`)
   - جدول سکتورها (`get_sectors_data`)
   - اجرای `DataFetcher.run()` از دکمه «به‌روزرسانی داده‌ها»
   - اجرای CLI از UI (`run_cli_command`) با لاگ

## دیاگرام معماری
```mermaid
flowchart LR
    User[کاربر مرورگر] --> Dash[Dash App\n(web/dashboard.py)]
    Dash --> DB[(SQLite DB)]
    Dash -. optional update .-> Fetcher[DataFetcher.run()]
    Fetcher --> DB
    Dash -. CLI bridge .-> CLI[main.py commands]
```

## خطاهای رایج
- نبود DB: پیام راهنما برای اجرای create/load نمایش داده می‌شود.
- وابستگی نصب نشده: در شروع گزارش می‌شود (`pip install -r requirements.txt`).
- تایم‌اوت در به‌روزرسانی داده‌ها: پیام هشدار با پیشنهاد بررسی شبکه/VPN.
