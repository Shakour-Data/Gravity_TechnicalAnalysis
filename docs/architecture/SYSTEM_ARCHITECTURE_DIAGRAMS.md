# معماری سامانه

این سند نمای کلی اجزا و نحوه عبور درخواست‌ها در نسخه 1.0.0 را توضیح می‌دهد.

## اجزا
- **FastAPI (`gravity_tech.main`)**: ثبت Routerهای `analysis`, `patterns`, `ml`, `tools`, `backtest`, `db`.
- **Core Services**: `TechnicalAnalysisService` برای اندیکاتورها و سیگنال نهایی؛ `ToolRecommendationService` برای توصیه ابزار؛ `PatternBacktester`.
- **کش اختیاری**: `cache_service` با Redis (TTL پیش‌فرض ۵ دقیقه).
- **دیتابیس**: SQLite/PostgreSQL از طریق `DatabaseManager` و `HistoricalScoreManager` (برای ذخیره نتایج در صورت فعال‌سازی ingestion).
- **سرویس داده اختیاری**: `DataServiceClient` جهت واکشی کندل Adjusted؛ Redis-cache داخلی دارد.
- **مدل‌های ML**: فایل‌های `ml_models/pattern_classifier_*.pkl` برای طبقه‌بندی الگو.
- **رویدادها (اختیاری)**: Kafka/RabbitMQ از طریق `middleware/events` و `data_ingestor_service`.

## نمودار اجزا
```mermaid
flowchart LR
    Client["Client / SDK / cURL"] --> API["FastAPI\nRouters v1"]
    API -->|/analyze| Analysis["TechnicalAnalysisService\n(indicators, patterns, market phase)"]
    API -->|/patterns| Pattern["Harmonic Detector\n+ ML scorer"]
    API -->|/ml| ML["ML Inference\n(model cache)"]
    API -->|/tools| Tools["ToolRecommendationService"]
    API -->|/backtest| Backtest["PatternBacktester"]
    API -->|/db| DBExplorer["DB Explorer (read-only)"]

    Analysis --> Cache[(Redis Cache?)]
    Pattern --> Cache
    Tools --> Cache

    Analysis --> Indicators["Indicators (Trend/Momentum/Volume/Volatility/Cycle/SR)"]
    Analysis --> PatternsCandles["Candlestick/Elliott/Market Phase"]

    Cache -->|miss| DataSvc["DataServiceClient\n(Adjusted OHLCV, optional)"]
    DataSvc --> Redis[(Redis inner cache)]

    subgraph Persistence
        DB[(SQLite/PostgreSQL)]
        Ingestor["DataIngestorService\n(store analysis results)"]
    end

    Analysis --> Ingestor
    Pattern --> Ingestor
    Tools --> Ingestor
    Ingestor --> DB
```

## جریان نمونه درخواست /api/v1/analyze
```mermaid
sequenceDiagram
    participant C as Client
    participant A as API (FastAPI)
    participant S as TechnicalAnalysisService
    participant R as Redis (اختیاری)
    participant D as DataIngestor/DB (اختیاری)

    C->>A: POST /api/v1/analyze (candles>=50)
    A->>S: فراخوانی analyze(request)
    S->>S: محاسبه اندیکاتورها و الگوهای شمعی/الیوت
    S->>S: محاسبه فاز بازار + سیگنال کلی
    S-->>A: TechnicalAnalysisResult
    A-->>C: پاسخ JSON
    alt ingestion فعال
        A->>D: ذخیره نتیجه (event یا مستقیم)
    end
```

## نکات کلیدی
- CORS برای همه originها باز است؛ در محیط تولید محدود کنید.
- اگر Redis یا سرویس داده تنظیم نشود، مسیرها همچنان اجرا می‌شوند ولی کش/داده خارجی غیرفعال است.
- متریک‌های Prometheus با `settings.metrics_enabled` فعال می‌شوند و روی `/metrics` mount می‌شوند.
