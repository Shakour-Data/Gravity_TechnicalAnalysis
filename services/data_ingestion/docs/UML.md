# UML و معماری (GravityTseHisPrice)

دیاگرام‌ها با Mermaid نگارش شده‌اند و اجزای اصلی، توالی به‌روزرسانی و کلاس‌های کلیدی کد فعلی را نمایش می‌دهند.

## معماری سطح بالا
```mermaid
flowchart LR
    subgraph Clients["کاربران و ورودی‌ها"]
        CLI["CLI\n(main.py / src/cli.py)"]
        Dashboard["Dash UI\n(web/dashboard.py)"]
        APIClient["مشتری API\n(HTTP)"]
    end

    Fetcher["DataFetcher\n(src/fetcher.py)"]
    DBLayer["init_price_data\n(src/database.py)"]
    DB["SQLite\n(data/tse_data.db)"]
    Gravity["gravity_tse\n(scripts/gravity_tse.py)"]
    API["Flask API\n(web/api.py)"]

    CLI -->|create/load/init| Fetcher
    CLI -->|دستورات مستقیم| DBLayer
    Dashboard -->|خواندن گزارش| DB
    Dashboard -. trigger updates .-> Fetcher
    APIClient -->|HTTP| API --> DB
    Fetcher -->|CRUD| DBLayer --> DB
    Fetcher --> Gravity
```

## Sequence: به‌روزرسانی کامل داده‌ها (`load-all-prices`/`init-all`)
```mermaid
sequenceDiagram
    participant User
    participant CLI as CLI / Dashboard
    participant Fetcher as DataFetcher
    participant DBLayer as init_price_data
    participant DB as SQLite
    participant Gravity as gravity_tse

    User->>CLI: اجرای load-all-prices
    CLI->>Fetcher: DataFetcher.run()
    Fetcher->>DBLayer: get_last_update(symbol)
    loop برای هر شرکت
        Fetcher->>Gravity: Get_Price_History(...)
        Gravity-->>Fetcher: DataFrame قیمت
        Fetcher->>DBLayer: insert_price_data(records)
        DBLayer->>DB: INSERT/REPLACE price_data
        Fetcher->>DBLayer: update_last_update(ticker, max_date)
    end
    Fetcher->>Gravity: Get_USD_RIAL(...)
    Fetcher->>DBLayer: insert_usd_data(records)
    loop شاخص‌های بازار و صنعت
        Fetcher->>Gravity: Get_*_History(...)
        Fetcher->>DBLayer: insert_market_indices/insert_sector_indices
        DBLayer->>DB: INSERT/REPLACE
        Fetcher->>DBLayer: update_last_update(symbol, max_date)
    end
    Fetcher->>DBLayer: update_price_data_sectors()
    DBLayer->>DB: UPDATE price_data.sector_id
    CLI-->>User: پایان به‌روزرسانی
```

## کلاس‌ها و نقش‌ها
```mermaid
classDiagram
    class DataFetcher {
        +fetch_all_prices_to_json()
        +fetch_all_market_indices_to_json()
        +fetch_all_sector_indices_to_json()
        +fetch_company_price_history(...)
        +fetch_sector_index_history(...)
        +fetch_index_history(...)
        +fetch_usd_price_history(...)
        +run()
    }

    class init_price_data {
        +create_tables()
        +create_indices_tables()
        +insert_price_data(records)
        +insert_usd_data(records)
        +insert_market_indices(code,name,df)
        +insert_sector_indices(code,name,df)
        +insert_companies(list)
        +insert_sectors(list)
        +insert_markets(list)
        +insert_panels(list)
        +insert_last_updates(map)
        +get_last_update(symbol)
        +update_last_update(symbol,date)
        +update_price_data_sectors()
        +get_connection()
    }

    class DashboardApp {
        +layout
        +callbacks(update_summary, update_price_chart, render_table_viewer, handle_actions, handle_cli_commands)
        -run_cli_command(...)
    }

    class API {
        +get_summary()
        +get_companies()
        +get_price_data(ticker)
        +get_sectors()
        +get_market_indices()
        +health_check()
    }

    DataFetcher --> init_price_data : uses
    DashboardApp --> init_price_data : read via SQL
    DashboardApp --> DataFetcher : trigger updates
    API --> init_price_data : read via SQL
```
