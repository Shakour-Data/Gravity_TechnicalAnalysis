# راهنمای API (نسخه v1)

پایگاه همه endpointهایی که در `gravity_tech.main` قابل استفاده‌اند. تمام مسیرها زیر `/api/v1` هستند مگر سلامت/متریک.

## پایه دسترسی
- مستندات تعاملی: `/api/docs` (Swagger)
- OpenAPI JSON: `/api/openapi.json`
- سلامت: `/health`, `/health/ready`, `/health/live`
- متریک‌ها (در صورت فعال بودن): `/metrics`

## تحلیل تکنیکال
| متد | مسیر | توضیح |
|-----|------|-------|
| `POST` | `/api/v1/analyze` | تحلیل کامل روی کندل‌های ورودی (حداقل ۶۰ کندل الزامی). |
| `GET` | `/api/v1/analyze/historical/{symbol}` | واکشی داده از دیتابیس محلی (TSE) و اجرای تحلیل (حداقل ۶۰ کندل). |
| `POST` | `/api/v1/analyze/indicators` | محاسبه انتخابی چند اندیکاتور مشخص (حداقل ۶۰ کندل). |
| `GET` | `/api/v1/indicators/list` | فهرست اندیکاتورهای موجود به تفکیک دسته. |

### نمونه درخواست `POST /analyze`
```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "candles": [
    {"timestamp": "2024-01-01T00:00:00Z", "open": 43000, "high": 43500, "low": 42800, "close": 43250, "volume": 120000}
  ]
}
```

## تشخیص الگو
| متد | مسیر | توضیح |
|-----|------|-------|
| `POST` | `/api/v1/patterns/detect` | تشخیص الگوهای هارمونیک (Gartley, Butterfly, Bat, Crab) + امتیاز ML اختیاری (حداقل ۶۰ کندل، حداکثر 5000، timestamps صعودی). |
| `GET` | `/api/v1/patterns/types` | فهرست انواع الگوها و نسبت‌های فیبوناچی. |
| `GET` | `/api/v1/patterns/health` | سلامت سرویس تشخیص الگو (وجود مدل ML). |

## یادگیری ماشین
| متد | مسیر | توضیح |
|-----|------|-------|
| `POST` | `/api/v1/ml/predict` | پیش‌بینی نوع الگو بر اساس ۲۱ ویژگی. |
| `POST` | `/api/v1/ml/predict/batch` | پیش‌بینی دسته‌ای برای چند ورودی (حداکثر 256 رکورد؛ خطای تک‌رکورد باقی را متوقف نمی‌کند). |
| `GET` | `/api/v1/ml/model/info` | اطلاعات مدل فعال (نسخه، دقت، ویژگی‌ها). |
| `GET` | `/api/v1/ml/health` | سلامت لایه ML و وضعیت بارگذاری مدل. |

## توصیه ابزار و تحلیل سفارشی
| متد | مسیر | توضیح |
|-----|------|-------|
| `GET` | `/api/v1/tools/` | فهرست ۹۵+ ابزار به‌همراه دسته‌بندی و دقت تاریخی. |
| `POST` | `/api/v1/tools/recommend` | توصیه ابزار بر اساس استایل ترید، هدف تحلیل و ML (حداقل ۶۰ کندل). |
| `POST` | `/api/v1/tools/analyze/custom` | تحلیل فقط با ابزارهای انتخابی + امتیاز ML اختیاری (حداقل ۶۰ کندل). |
| `GET` | `/api/v1/tools/categories` | تعداد و مثال‌های هر دسته ابزار. |
| `GET` | `/api/v1/tools/tool/{name}` | جزئیات یک ابزار خاص. |
| `GET` | `/api/v1/tools/health` | سلامت سرویس توصیه ابزار. |

- اعتبارسنجی ابزارها: در `/api/v1/tools/analyze/custom` همه نام‌ها با کاتالوگ چک می‌شوند؛ نام نامعتبر => HTTP 400.
- اعتبارسنجی بازه: `timeframe` در `/api/v1/tools/recommend` و `/api/v1/tools/analyze/custom` فقط از لیست `1m,5m,15m,30m,1h,2h,4h,6h,12h,1d,1w` پذیرفته می‌شود؛ خلاف آن => 400.
- محدودیت کندل: `limit_candles` باید 60..1000 باشد؛ مقدار نامعتبر => 400.

## بک‌تست
| متد | مسیر | توضیح |
|-----|------|-------|
| `POST` | `/api/v1/backtest` | بک‌تست تشخیص الگو با دادهٔ OHLCV ورودی یا داده واقعی TSE. خروجی شامل آمار معاملات (win rate، Sharpe، drawdown). |
- ورودی‌ها: یا آرایه‌های `highs/lows/closes/volumes` + `dates` اختیاری، یا `symbol` برای بارگذاری دیتای واقعی (حداقل max(window_size+step_size,400) کندل). پارامترها: `min_confidence`=0.6، `window_size`=200، `step_size`=50، `persist` اختیاری.
- صحت‌سنجی: آرایه‌ها هم‌طول و حداقل max(window+step,300)؛ NaN/Inf یا `high<low` یا `close` خارج از بازه => 400؛ `dates` باید non-decreasing باشد.
- خروجی: `metrics` شامل total_trades/winning_trades/losing_trades/win_rate/total_pnl/average_pnl/profit_factor/sharpe_ratio/max_drawdown/target1_hits/target2_hits، به‌همراه `trade_count`، `backtest_period.start/end`، `analysis_time_ms`، `model_version`، `data_source` (provided/tse_db/synthetic)، `warnings`.
- Persist: اگر `persist=true` و `data_source != synthetic` خلاصه در جدول `backtest_runs` ذخیره می‌شود؛ در حالت synthetic فقط پاسخ برمی‌گردد.

## سناریوهای سه‌گانه (اختیاری)
| متد | مسیر | توضیح |
|-----|------|-------|
| `GET` | `/api/v1/scenarios/{symbol}` | تحلیل سناریوی خوش‌بینانه/خنثی/بدبینانه با داده Adjusted؛ فقط وقتی `ENABLE_SCENARIOS=true` باشد mount می‌شود. |

## اکسپلورر دیتابیس (پشتیبانی/داخلی - اختیاری)
| متد | مسیر | توضیح |
|------|------|-------|
| `GET` | `/api/v1/db/tables`, `/api/v1/db/info`, `/api/v1/db/schema` | مشاهده جدول‌ها و شِما (فقط اگر `EXPOSE_DB_EXPLORER=true`). |
| `GET` | `/api/v1/db/backup` | دانلود پشتیبان SQLite. |
| `GET` | `/api/v1/db/query` | اجرای کوئری خواندنی محدود (برای پشتیبانی). |
| `GET` | `/api/v1/db/ui`, `/api/v1/db/home` | رابط HTML ساده برای مرور دیتابیس. |

## قرارداد داده‌ها (خلاصه)
- Candle: `timestamp`, `open`, `high`, `low`, `close`, `volume` به همراه `symbol`, `timeframe`.
- حداقل کندل: ۶۰ (اعتبارسنجی API) برای پوشش اندیکاتورهایی مثل DEMA/TEMA/ADX.
- مدل ML: فایل‌های `ml_models/pattern_classifier_advanced_v2.pkl` یا `pattern_classifier_v1.pkl` باید وجود داشته باشند؛ در غیر این صورت endpointهای ML/Pattern پیام «model missing» می‌دهند.
- پیش‌بینی ML: در نبود مدل، پاسخ fallback با احتمال برابر و `model_version=fallback` برمی‌گردد؛ timeout پیش‌فرض 2s (predict) و 5s (batch) است و قابل override در بدنه درخواست.

## قوانین نرخ و کش
- CORS برای همه originها فعال است (در تولید محدود کنید).
- کش Redis در صورت تنظیم `CACHE_ENABLED=true` و `REDIS_URL`; TTL پیش‌فرض ۵ دقیقه.
- اگر Redis یا سرویس داده در دسترس نباشد، تحلیل همچنان اجرا می‌شود ولی کش/داده خارجی استفاده نمی‌شود.

## نکات امنیت و فعال‌سازی
- سناریو سه‌گانه: با `ENABLE_SCENARIOS=true` فعال می‌شود.
- DB Explorer: فقط در توسعه توصیه می‌شود و با `EXPOSE_DB_EXPLORER=true` mount می‌شود.
- برای تولید، CORS و Rate-limit را در لایه لبه (Nginx/Traefik) محدود کنید.
