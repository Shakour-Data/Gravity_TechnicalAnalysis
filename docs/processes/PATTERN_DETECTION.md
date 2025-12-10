# فرایند تشخیص الگوهای هارمونیک (`POST /api/v1/patterns/detect`)

## جریان کلی
```mermaid
flowchart LR
    Client --> API["/api/v1/patterns/detect\n(api/v1/patterns.py)"]
    API --> Detector["HarmonicPatternDetector"]
    Detector -->|optional| ML["Pattern Classifier (pkl, cached)\npattern_classifier_*.pkl"]
    ML --> Filter["min_confidence filter"]
    Detector --> Filter --> Resp["patterns + dynamic targets/SL + confidence"]
    Resp --> Client
```

## ورودی/خروجی
- ورودی: `PatternDetectionRequest` با حداقل ۶۰ و حداکثر 5000 کندل، `pattern_types` اختیاری، `use_ml`, `min_confidence`, `tolerance`.
- `timeframe` با regex محدود است؛ `candles` باید از نظر زمانی strictly increasing باشند.
- خروجی: الگوها با نقاط X/A/B/C/D، اهداف/حدضرر داینامیک (ATR-like)، مدت تحلیل (ms)، `ml_enabled` و `ml_status` (enabled/disabled/not_available/failed).

## سکانس
```mermaid
sequenceDiagram
    participant C as Client
    participant A as /patterns/detect
    participant D as HarmonicPatternDetector
    participant M as Pattern Classifier (optional)

    C->>A: POST patterns/detect (candles>=60, <=5000)
    A->>A: validate timestamps increasing
    A->>D: detect_patterns(highs,lows,closes)
    alt use_ml == true
        A->>M: load model (v2/v1 cached)
        M-->>A: confidence per pattern
        A->>A: filter < min_confidence
    else
        A->>A: ml_enabled=false in response
    end
    A-->>C: patterns_found + patterns[] + ml_status
```

## تنظیمات/وابستگی‌ها
- مدل در `ml_models/pattern_classifier_*.pkl` برای امتیاز ML؛ در نبود مدل `ml_enabled=false` و `ml_status` منعکس می‌شود.
- `tolerance` پیش‌فرض 0.05 (۵٪).
- حداقل کندل ۶۰؛ حداکثر 5000.

## خطاها
- ترتیب زمانی نادرست: 400 با پیام "Candles must be strictly increasing in time".
- تعداد کندل بیش از حد: 400 با پیام "Too many candles".
- نبود/خطای مدل: هشدار log و `ml_enabled=false`/`ml_status` متناسب.
- خطای داخلی: 500 با پیام «Pattern detection failed: ...».

## مشکلات/ریسک‌های باقی‌مانده
- وابستگی به فایل مدل روی دیسک (عدم حضور → فقط تشخیص هندسی).
- سقف 5000 کندل ممکن است برای ورودی بسیار بزرگ همچنان زمان‌بر باشد.
- اهداف/حدضرر ATR-like ساده هستند؛ در صورت نیاز باید منطق دقیق‌تری اعمال شود.
