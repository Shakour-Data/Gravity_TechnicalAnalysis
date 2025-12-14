# فرآیند ۳: بارگذاری افزایشی قیمت‌ها و شاخص‌ها (DataFetcher.run)

این فرآیند با تکیه بر `last_updates` داده‌ها را افزایشی از `gravity_tse` واکشی و در DB درج می‌کند.

## هدف
- واکشی و درج قیمت سهام، USD، شاخص‌های بازار و صنعت به‌صورت افزایشی
- به‌روزرسانی متادیتا (`last_updates`) و نگاشت سکتور در `price_data`

## پیش‌نیازها
- فرآیندهای ۱ و ۲ اجرا شده باشند و جداول مرجع پر شده باشند.
- دسترسی به ماژول `scripts/gravity_tse.py` و اینترنت/منبع داده.

## دستورات اجرا
```bash
python main.py load-all-prices   # یا از داشبورد: دکمه «به‌روزرسانی داده‌ها»
# در init-all نیز این مرحله در انتها صدا زده می‌شود.
```

## گام‌ها (کد `DataFetcher.run`)
1) `fetch_all_prices_to_json`
   - برای هر شرکت: خواندن `last_updates[ticker]`
   - تبدیل تاریخ میلادی به جلالی؛ اگر sentinel (`2011-03-21` یا `1390-01-01`) → شروع جلالی 1390-01-01
   - واکشی `Get_Price_History`، محاسبه `AdjVolume`، درج در `price_data`
   - نوشتن `last_updates[ticker]` با جدیدترین تاریخ میلادی
2) `fetch_usd_price_history`
   - مشابه بالا برای USD؛ درج در `usd_prices` و به‌روزرسانی `last_updates['USD']`
3) `fetch_all_market_indices_to_json`
   - برای هر شاخص بازار (CWI/EWI/…): خواندن `last_updates[code]`، واکشی `Get_*_History`، درج در `market_indices`
4) `fetch_all_sector_indices_to_json`
   - برای هر سکتور: خواندن `last_updates['SECTOR_<code>']`، واکشی `Get_SectorIndex_History` (با نگاشت نام)، درج در `sector_indices`
5) `update_price_data_sectors`
   - تکمیل `sector_id` در `price_data` از جدول `companies`

## ورودی/خروجی
- ورودی: DB موجود + JSON مرجع شرکت/سکتور
- خروجی: رکوردهای جدید در `price_data`, `usd_prices`, `market_indices`, `sector_indices` و به‌روزرسانی `last_updates`

## خطاهای رایج
- تاریخ sentinel اشتباه در `last_updates` → موجب پرش دامنه واکشی می‌شود (اکنون sentinel‌ها سخت‌گیرانه تنظیم شده‌اند).
- قطعی شبکه → برخی رکوردها درج نمی‌شوند؛ می‌توان دوباره اجرا کرد (Idempotent به‌لطف `INSERT OR REPLACE`).

## دیاگرام جریان
```mermaid
flowchart LR
    Start[load-all-prices/DataFetcher.run] --> Prices[fetch_all_prices_to_json<br>per ticker]
    Prices --> PriceInsert[insert_price_data]
    PriceInsert --> LU1[update_last_update(ticker)]
    Start --> USD[fetch_usd_price_history]
    USD --> USDInsert[insert_usd_data] --> LU2[update_last_update(USD)]
    Start --> MktIdx[fetch_all_market_indices_to_json]
    MktIdx --> MktInsert[insert_market_indices] --> LU3[update_last_update(code)]
    Start --> SecIdx[fetch_all_sector_indices_to_json]
    SecIdx --> SecInsert[insert_sector_indices] --> LU4[update_last_update(SECTOR_code)]
    PriceInsert --> MapSector[update_price_data_sectors]
    LU1 --> DB[(SQLite DB)]
    LU2 --> DB
    LU3 --> DB
    LU4 --> DB
    MapSector --> DB
```
