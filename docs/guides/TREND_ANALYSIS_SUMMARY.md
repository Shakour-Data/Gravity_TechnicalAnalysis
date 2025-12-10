# خلاصه اندیکاتورها و دسته‌بندی‌ها

این سند تنها اندیکاتورهای پیاده‌سازی‌شده در کد را فهرست می‌کند. هر اندیکاتور خروجی `IndicatorResult` شامل `signal`, `confidence`, `value`, `description` دارد و در `TechnicalAnalysisService` مصرف می‌شود.

## دسته‌ها و اندیکاتورها
| دسته | اندیکاتورها |
|------|-------------|
| **Trend** | SMA(20/50), EMA(12/26), WMA, DEMA, TEMA, MACD, ADX(14), Donchian, Aroon, Vortex, McGinley Dynamic, Parabolic SAR, Supertrend, Ichimoku |
| **Momentum** | RSI, Stochastic, CCI, ROC, Williams %R, MFI, Ultimate Oscillator, TSI, KST, PMO |
| **Cycle** | Sine Wave، Hilbert Dominant Cycle، Detrended Price Oscillator، Schaff Trend Cycle، Market Facilitation Index، Cycle Period، Phase Change Index، Trend vs Cycle Identifier، Autocorrelation Periodogram، Cycle Phase Index |
| **Volume** | OBV، CMF، VWAP، A/D Line، Volume Profile، PVT، EMV، VPT، Volume Oscillator، VWMA |
| **Volatility** | Bollinger Bands، ATR، Keltner Channel، Donchian Channel، Std Dev، Historical Volatility، Chandelier Exit، Mass Index، Ulcer Index، RVI |
| **Support/Resistance** | Pivot Points، Fibonacci Retracement/Extension، Camarilla، Woodie، DeMark، Floor Pivots، Psychological Levels، Previous High/Low |

## الگوها و تحلیل‌های مکمل
- **Candlestick**: Doji، Hammer، Engulfing، Morning/Evening Star، Harami، Three Soldiers/Crows، Piercing، Dark Cloud، Tweezer، Marubozu و …  
- **Elliott Wave**: تشخیص موج‌های ۱-۵ و ABC به‌همراه نسبت‌های فیبوناچی.  
- **Market Phase (Dow Theory)**: Accumulation, Markup, Distribution, Markdown + تأیید حجمی.

## حداقل داده
- برای بیشتر اندیکاتورها حداقل ۲۰ تا ۶۰ کندل لازم است؛ درخواست API کمتر از ۵۰ کندل را رد می‌کند.
- داده باید به‌ترتیب زمانی و بدون گپ ساختگی باشد؛ در غیر این صورت اعتبارسنجی Candle خطا می‌دهد.
