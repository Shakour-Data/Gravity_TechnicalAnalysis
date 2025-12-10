# فرآیند توصیه ابزار و تحلیل سفارشی (`/api/v1/tools/*`)

## جریان کلی
```mermaid
flowchart LR
    Client --> List["/tools/"]
    Client --> Rec["/tools/recommend"]
    Client --> Custom["/tools/analyze/custom"]
    Rec --> Service["ToolRecommendationService
(کاتالوگ + وزن ML)"]
    Custom --> Service
    Service --> Resp["recommendations / custom analysis"]
    Resp --> Client
```

## ورودی/خروجی
- `/tools/`: فهرست ابزارها با فیلتر `category|timeframe|min_accuracy` و `limit` (۱..۲۰۰).
- `/recommend`: ورودی `symbol|timeframe|analysis_goal|trading_style|limit_candles|top_n`. خروجی: گروه‌بندی must_use/recommended/optional/avoid + استراتژی پویا + متادیتای ML.
- `/analyze/custom`: ورودی `selected_tools` (۱..۳۰ نام معتبر)، `include_ml_scoring`، `include_patterns`، `limit_candles` (۶۰..۱۰۰۰)، `timeframe` معتبر. خروجی: نتایج هر ابزار + امتیاز ML + الگوهای کشف‌شده + خلاصه.

## کنترل‌ها و تضمین‌ها
- `timeframe` در هر دو درخواست روی لیست مجاز `1m..1w` اعتبارسنجی می‌شود.
- نام ابزارهای ورودی در `/analyze/custom` با کاتالوگ چک می‌شود؛ ابزار نامعتبر => HTTP 400.
- `limit_candles` علاوه بر Pydantic، در سطح سرویس ۶۰..۱۰۰۰ سخت‌گیرانه می‌شود؛ عدد نامعتبر => HTTP 400.
- خطاهای کاربری به ۴۰۰ و خطاهای داخلی به ۵۰۰ تفکیک شده‌اند.

## دیاگرام اعتبارسنجی تحلیل سفارشی
```mermaid
flowchart TD
    A[Request /tools/analyze/custom] --> B{timeframe معتبر؟}
    B -- خیر --> E[400 invalid timeframe]
    B -- بله --> C{selected_tools همه معتبر؟}
    C -- خیر --> F[400 invalid tool names]
    C -- بله --> D{limit_candles 60..1000؟}
    D -- خیر --> G[400 invalid limit]
    D -- بله --> H[ToolRecommendationService.analyze_custom_tools]
    H --> I[نتیجه تحلیل + امتیاز ML + الگوها]
```

## محدودیت‌ها
- صفحه‌بندی فهرست ابزارها فقط با `limit` انجام می‌شود (offset/pagination کامل وجود ندارد).
- Rate limiting / timeout لایه بالادستی نیاز است؛ درون سرویس کنترل فشار سنگین انجام نمی‌شود.
- دقت/وزن‌دهی ML وابسته به کیفیت مدل و داده ورودی است؛ نیازمند مانیتورینگ.
