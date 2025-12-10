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
| `POST` | `/analyze` | تحلیل کامل روی کندل‌های ورودی (حداقل ۶۰ کندل الزامی). |
| `GET` | `/analyze/historical/{symbol}` | واکشی داده از دیتابیس محلی (TSE) و اجرای تحلیل (حداقل ۶۰ کندل). |
| `POST` | `/analyze/indicators` | محاسبه انتخابی چند اندیکاتور مشخص (حداقل ۶۰ کندل). |
| `GET` | `/indicators/list` | فهرست اندیکاتورهای موجود به تفکیک دسته. |

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
| `POST` | `/patterns/detect` | تشخیص الگوهای هارمونیک (Gartley, Butterfly, Bat, Crab) + امتیاز ML اختیاری (حداقل ۶۰ کندل، حداکثر 5000، timestamps صعودی). |
| `GET` | `/patterns/types` | فهرست انواع الگوها و نسبت‌های فیبوناچی. |
| `GET` | `/patterns/health` | سلامت سرویس تشخیص الگو (وجود مدل ML). |

## یادگیری ماشین
| متد | مسیر | توضیح |
|-----|------|-------|
| `POST` | `/ml/predict` | پیش‌بینی نوع الگو بر اساس ۲۱ ویژگی. |
| `POST` | `/ml/predict/batch` | پیش‌بینی دسته‌ای برای چند ورودی (حداکثر 256 رکورد؛ خطای تک‌رکورد باقی را متوقف نمی‌کند). |
| `GET` | `/ml/model/info` | اطلاعات مدل فعال (نسخه، دقت، ویژگی‌ها). |
| `GET` | `/ml/health` | سلامت لایه ML و وضعیت بارگذاری مدل. |

## توصیه ابزار و تحلیل سفارشی
| متد | مسیر | توضیح |
|-----|------|-------|
| `GET` | `/tools/` | فهرست ۹۵+ ابزار به‌همراه دسته‌بندی و دقت تاریخی. |
| `POST` | `/tools/recommend` | توصیه ابزار بر اساس استایل ترید، هدف تحلیل و ML (حداقل ۶۰ کندل). |
| `POST` | `/tools/analyze/custom` | تحلیل فقط با ابزارهای انتخابی + امتیاز ML اختیاری (حداقل ۶۰ کندل). |
| `GET` | `/tools/categories` | تعداد و مثال‌های هر دسته ابزار. |
| `GET` | `/tools/tool/{name}` | جزئیات یک ابزار خاص. |
| `GET` | `/tools/health` | سلامت سرویس توصیه ابزار. |

- اعتبارسنجی ابزارها: در `/tools/analyze/custom` همه نام‌ها با کاتالوگ چک می‌شوند؛ نام نامعتبر => HTTP 400.
- اعتبارسنجی بازه: `timeframe` در `/tools/recommend` و `/tools/analyze/custom` فقط از لیست `1m..1w` پذیرفته می‌شود؛ خلاف آن => 400.
- محدودیت کندل: `limit_candles` باید 60..1000 باشد؛ مقدار نامعتبر => 400.

## بک‌تست
| متد | مسیر | توضیح |
|-----|------|-------|
| `POST` | `/backtest` | بک‌تست تشخیص الگو با دادهٔ OHLCV ورودی یا داده واقعی TSE. خروجی شامل آمار معاملات (win rate، Sharpe، drawdown). |
- ورودی بک‌تست: همه آرایه‌های OHLCV باید هم‌طول و حداقل max(window+step,300) باشند؛ NaN/Inf یا high<low => 400.
- پاسخ بک‌تست: فیلدهای `data_source` (provided/tse_db/synthetic) و `warnings` اضافه شده؛ در حالت synthetic یا خطای داده واقعی، persist غیرفعال می‌شود.

## سناریوهای سه‌گانه (اختیاری)
| متد | مسیر | توضیح |
|-----|------|-------|
| `GET` | `/scenarios/{symbol}` | تحلیل سناریوی خوش‌بینانه/خنثی/بدبینانه با داده Adjusted؛ فقط وقتی `ENABLE_SCENARIOS=true` باشد mount می‌شود. |

## اکسپلورر دیتابیس (پشتیبانی/داخلی - اختیاری)
| متد | مسیر | توضیح |
|------|------|-------|
| `GET` | `/db/tables`, `/db/info`, `/db/schema` | مشاهده جدول‌ها و شِما (فقط اگر `EXPOSE_DB_EXPLORER=true`). |
| `GET` | `/db/backup` | دانلود پشتیبان SQLite. |
| `GET` | `/db/query` | اجرای کوئری خواندنی محدود (برای پشتیبانی). |
| `GET` | `/db/ui`, `/db/home` | رابط HTML ساده برای مرور دیتابیس. |

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
