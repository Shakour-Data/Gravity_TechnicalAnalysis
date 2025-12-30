# کاتالوگ اندیکاتورها (Indicators Catalog)

این سند «منبع واحد» برای لیست اندیکاتورها، فرمول کلی، پارامترهای اصلی و تفسیر سیگنال است.

> مرجع پیاده‌سازی: `apps/analysis_api/src/gravity_tech/core/indicators/*`

## 0) استاندارد سیگنال و Confidence
اکثر اندیکاتورها یک سیگنال گسسته (Bullish/Bearish/Neutral و درجات آن) و یک confidence در بازه ۰..۱ تولید می‌کنند. آستانه‌ها و heuristics در هر اندیکاتور ممکن است متفاوت باشد اما همگی با قرارداد `IndicatorResult` برگردانده می‌شوند.

## 1) Trend Indicators
فایل: `apps/analysis_api/src/gravity_tech/core/indicators/trend.py`

اندیکاتورهای پیاده‌سازی شده (نمونه‌های شاخص):
- SMA, EMA, WMA, DEMA, TEMA
- MACD
- ADX
- Parabolic SAR
- Supertrend
- Ichimoku
- Donchian Channels
- Aroon
- Vortex Indicator
- McGinley Dynamic

### 1.1) SMA (Simple Moving Average)
- فرمول: `SMA_t = mean(close_{t-period+1..t})`
- تفسیر سیگنال: نسبت قیمت فعلی به SMA (درصد اختلاف) با چند آستانه
- پارامتر اصلی: `period` (پیش‌فرض 20)

### 1.2) EMA (Exponential Moving Average)
- فرمول: `EMA_t = α*close_t + (1-α)*EMA_{t-1}` ، `α = 2/(period+1)`
- تفسیر: مشابه SMA (فاصله قیمت تا EMA + شیب برای confidence)

### 1.3) MACD
- فرمول: `MACD = EMA_fast - EMA_slow` و `SignalLine = EMA(MACD, signal_period)`
- خروجی: مقدار MACD و (در صورت وجود) خط سیگنال و هیستوگرام

### 1.4) ADX
- هدف: اندازه قدرت روند مستقل از جهت
- خروجی: ADX و (در صورت وجود) DI+/DI-

### 1.5) Supertrend
- مبتنی بر ATR و باندهای بالا/پایین (Trailing)
- خروجی: وضعیت روند و سطح supertrend

### 1.6) Ichimoku
- اجزاء: Tenkan/Kijun/SenkouSpanA/SenkouSpanB/Chikou
- خروجی: سیگنال بر اساس موقعیت قیمت نسبت به ابر و کراس‌ها

## 2) Momentum Indicators
فایل: `apps/analysis_api/src/gravity_tech/core/indicators/momentum.py`

اندیکاتورهای کلیدی:
- RSI, Stochastic, CCI, ROC, Williams %R, MFI, Ultimate Oscillator

### 2.1) RSI
- فرمول: `RSI = 100 - 100/(1+RS)` ، `RS = avg_gain/avg_loss`
- تفسیر: نواحی Overbought/Oversold و بازگشت به میانه

### 2.2) Stochastic
- فرمول کلی: `%K = 100*(close - low_n)/(high_n - low_n)` و `%D = SMA(%K)`

### 2.3) MFI
- مبتنی بر Typical Price و Money Flow برای تشخیص فشار خرید/فروش

## 3) Cycle Indicators
فایل: `apps/analysis_api/src/gravity_tech/core/indicators/cycle.py`

اندیکاتورهای کلیدی:
- DPO، Ehlers Cycle Period، Dominant Cycle، Hilbert Transform Phase، Market Cycle Model، Sine Wave، STC

### 3.1) DPO (Detrended Price Oscillator)
- فرمول: `DPO = close - SMA(close, period) shifted by (period/2 + 1)`
- خروجی: مقدار DPO + برآورد فاز/دوره چرخه

## 4) Volatility Indicators
فایل: `apps/analysis_api/src/gravity_tech/core/indicators/volatility.py`

اندیکاتورهای کلیدی:
- ATR، Bollinger Bands، Keltner Channel، Donchian Channel، StdDev، Historical Volatility، ATR%، Chaikin Volatility

### 4.1) ATR
- TR: `max(H-L, |H-prevClose|, |L-prevClose|)`
- ATR: EMA(TR, period)
- خروجی: مقدار ATR + percentile + normalized

### 4.2) Bollinger Bands
- `Middle = SMA(close, n)`
- `Upper/Lower = Middle ± k*StdDev(close, n)`
- خروجی: باندها + موقعیت قیمت نسبت به باند

## 5) Volume Indicators
فایل: `apps/analysis_api/src/gravity_tech/core/indicators/volume.py`

اندیکاتورهای کلیدی:
- OBV، CMF، VWAP، A/D Line، Volume Profile، PVT، Volume Oscillator، VWMA

### 5.1) OBV
- `OBV_t = OBV_{t-1} ± volume_t` (بر اساس جهت تغییر قیمت)

### 5.2) VWAP
- `VWAP = Σ(price_i * volume_i) / Σ(volume_i)`

## 6) Support/Resistance
فایل: `apps/analysis_api/src/gravity_tech/core/indicators/support_resistance.py`

روش‌های کلیدی:
- Pivot Points (standard/woodie/camarilla/fibonacci/demark/floor)
- Fibonacci Retracement
- Zone Detection / Support-Resistance levels (window)

## 7) لیست رسمی (برای UI/Client)
Endpoint: `GET /api/v1/indicators/list`
این endpoint لیست نام‌های قابل نمایش را به تفکیک دسته برمی‌گرداند (برای UI/فرانت/کلاینت).

