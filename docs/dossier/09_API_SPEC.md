# مشخصات API (API Specification)

## 1) درگاه‌ها
- Swagger UI: `/api/docs`
- OpenAPI JSON: `/api/openapi.json`
- Health:
  - `/health`
  - `/health/ready`
  - `/health/live`
- Metrics (اختیاری): `/metrics`

## 2) مسیرهای اصلی (v1)
مرجع کد:
- `apps/analysis_api/src/gravity_tech/api/v1/analysis.py`
- `apps/analysis_api/src/gravity_tech/api/v1/patterns.py`
- `apps/analysis_api/src/gravity_tech/api/v1/ml.py`
- `apps/analysis_api/src/gravity_tech/api/v1/tools.py`
- `apps/analysis_api/src/gravity_tech/api/v1/backtest.py`
- `apps/analysis_api/src/gravity_tech/api/v1/db_explorer.py`

### 2.1) تحلیل تکنیکال
- `POST /api/v1/analyze` — تحلیل کامل (candles>=60)
- `GET /api/v1/analyze/historical/{symbol}` — خواندن داده از DB و تحلیل
- `POST /api/v1/analyze/indicators` — محاسبه اندیکاتورهای انتخابی
- `GET /api/v1/indicators/list` — فهرست اندیکاتورها/الگوها برای کلاینت

### 2.2) الگوها
- `POST /api/v1/patterns/detect`
- `GET /api/v1/patterns/types`
- `GET /api/v1/patterns/health`

### 2.3) ML
- `POST /api/v1/ml/predict`
- `POST /api/v1/ml/predict/batch`
- `GET /api/v1/ml/model/info`
- `GET /api/v1/ml/health`

## 3) قراردادهای داده (Contracts)
تعریف‌های Pydantic:
- `apps/analysis_api/src/gravity_tech/core/contracts/analysis.py`

## 4) کدهای خطا و قواعد ورودی
الگوی کلی:
- `400` برای ورودی نامعتبر (کمبود کندل، timeframe نامعتبر، NaN/Inf و …)
- `404` برای نبود داده historical
- `503` برای نبود مدل ML یا وابستگی‌های ضروری
- `500` برای خطاهای غیرمنتظره

