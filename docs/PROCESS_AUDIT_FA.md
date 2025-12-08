# گزارش جامع فرآیندها و مشکلات Gravity Technical Analysis

این سند جریان‌های اصلی پروژه، ورودی/خروجی هر فرآیند و مشکلات فعلی آن‌ها را خلاصه می‌کند تا مسیر بهبود مشخص باشد.

## نمای کلی فرآیندها

| # | فرآیند | هدف | ماژول‌های کلیدی | خروجی‌های اصلی |
|---|--------|-----|-----------------|-----------------|
| 1 | گردآوری و همگام‌سازی داده | دریافت کندل‌های تاریخی و آماده‌سازی اولیه | `src/gravity_tech/ml/data_connector.py`, `src/gravity_tech/data/*` | لیست `Candle`، داده‌ی تنظیم‌شده |
| 2 | استخراج ویژگی چند‌افقی | ساخت بردارهای ویژگی و لیبل‌های چند‌افقی | `MultiHorizonFeatureExtractor`, `MultiHorizonMomentumFeatureExtractor`, `FeatureExtractor` | ماتریس ویژگی/هدف برای 3d/7d/30d |
| 3 | آموزش وزن‌ها و مدل‌ها | یادگیری وزن شاخص‌ها و ابعاد، ذخیره مدل‌ها | `IndicatorWeightLearner`, `DimensionWeightLearner`, `MultiHorizonWeightLearner`, `ml/train_pipeline.py` | وزن‌های آموزش‌دیده و مدل‌های ML |
| 4 | تحلیل چند‌افقی و ترکیبی | تولید سیگنال‌های روند/مومنتوم و جمع‌بندی | `MultiHorizonAnalyzer`, `MultiHorizonMomentumAnalyzer`, `CombinedTrendMomentumAnalyzer`, `CompleteAnalysisPipeline` | امتیاز هر افق، اکشن نهایی |
| 5 | ماتریس حجم-بعد و تصمیم 5بعدی | اعتبارسنجی سیگنال با حجم و صدور توصیه | `VolumeDimensionMatrix`, `FiveDimensionalDecisionMatrix` | برآیند 5 بعد، ریسک، توصیه معاملاتی |
| 6 | تشخیص الگو و API ML | سرویس‌دهی REST برای مدل‌های الگو | `api/v1/ml.py`, `pattern_classifier`, `ml/models` | پیش‌بینی الگو، اطلاعات مدل، بک‌تست |
| 7 | سرویس تحلیل جامع | اجرای هم‌زمان اندیکاتورها و Market Phase | `services/analysis_service.py`, `services/fast_indicators.py` | آبجکت `TechnicalAnalysisResult` |
| 8 | پیشنهادگر ابزار و سناریو | پیشنهاد ابزارهای تحلیلی و وزن سناریو | `ml/ml_tool_recommender.py`, `api/v1/tools.py`, `scenario_weight_optimizer.py` | لیست ابزار، استراتژی پیشنهادی |
| 9 | بک‌تست و بررسی سناریو | اعتبارسنجی استراتژی‌ها و وزن‌دهی دستی | `ml/backtesting.py`, `ml/scenario_weight_optimizer.py` | متریک‌های PnL، وزن سناریو |
|10 | استقرار و عملیات | آماده‌سازی محیط تولید و مانیتورینگ | `docs/operations/DEPLOYMENT_GUIDE.md`, `deployment/*` | چک‌لیست استقرار، Runbook |
|11 | تضمین کیفیت و تست | پوشش تست واحد/یکپارچه و گزارش‌دهی | `tests/*`, گزارش‌های README | اطمینان از صحت خروجی‌ها |

---

## 1. گردآوری و همگام‌سازی داده
**جریان:** `DataConnector.fetch_daily_candles` داده را از میکروسرویس تاریخچه (REST) می‌خواند، در غیبت سرویس داده‌ی ساختگی تولید می‌شود و `fetch_multiple_symbols` برای چند نماد اجرای سریالی دارد.

- **ورودی:** نماد، بازه تاریخ، `limit`.
- **خروجی:** لیست `Candle`، یا داده‌ی mock.
- **وابستگی‌ها:** میکروسرویس REST روی پورت 8000، `requests`, `pandas`.

**مشکلات و ریسک‌ها**
- فراخوانی HTTP کاملاً همگام است و هیچ Retry/Backoff یا متریک ساختاری ثبت نمی‌شود؛ هر خطای شبکه مستقیماً به تولید دیتای ساختگی ختم می‌شود و مانیتورینگ را دور می‌زند (`src/gravity_tech/ml/data_connector.py:35-103`).
- متد `_generate_mock_data` در همان خطای اول داده‌ی تصنعی را بازمی‌گرداند و هشدار تنها با `print` صادر می‌شود؛ pipeline نمی‌تواند بین داده واقعی و تستی تمایز بگذارد (`src/gravity_tech/ml/data_connector.py:100-169`).
- `fetch_multiple_symbols` حلقه‌ی سریالی با `print` است و هیچ Parallelism یا محدودکننده‌ی نرخ ندارد؛ در بارهای بالا به راحتی فرایند را کند می‌کند (`src/gravity_tech/ml/data_connector.py:176-210`).

---

## 2. استخراج ویژگی چند‌افقی
**جریان:** Sliding-window بر روی `candles`، محاسبه‌ی ده‌ها اندیکاتور، تشخیص الگو و برچسب‌گذاری بازده‌های آینده برای افق‌های 3/7/30 روز.

- **ورودی:** حداقل 100 کندل (قابل پیکربندی).
- **خروجی:** `X (features)` و `Y (targets)` برای هر سطح (اندیکاتوری یا ابعادی).
- **وابستگی‌ها:** `TrendIndicators`, `MomentumIndicators`, `ElliottWaveAnalyzer`, `ClassicalPatterns`.

**مشکلات و ریسک‌ها**
- در `extract_training_dataset` همه‌ی اندیکاتورها برای هر پنجره مجدداً محاسبه می‌شوند؛ پیچیدگی زمانی `O(n * lookback)` است و هیچ لایه‌ی کش یا اشتراک محاسبه وجود ندارد (`src/gravity_tech/ml/multi_horizon_feature_extraction.py:244-329`).
- در `MultiHorizonMomentumFeatureExtractor` چهار حلقه‌ی تو در تو برای محاسبه‌ی RSI/CCI/… تعریف شده که برای هر کندل تمام سری‌ها را مجدداً می‌سازد (`src/gravity_tech/ml/multi_horizon_momentum_features.py:90-147`)، بنابراین هزینه‌ی زمانی `O(n^2)` شده و برای دیتاست‌های بزرگ عملاً غیرقابل استفاده است.
- سامانه به جای logging از `print` استفاده می‌کند و سطح هشدار/خطا استاندارد نشده است؛ در محیط‌های چند-ریسمانی خوانایی لاگ از بین می‌رود (`src/gravity_tech/ml/multi_horizon_feature_extraction.py:247-325`).
- هیچ نرمال‌سازی یا ذخیره‌ی Min/Max برای ویژگی‌ها انجام نمی‌شود و مدل‌ها باید خودشان با داده‌ی مقیاس‌نشده کنار بیایند، که باعث ناپایداری وزن‌های یادگرفته‌شده می‌شود.

---

## 3. آموزش وزن‌ها و مدل‌ها
**جریان:** `MLTrainingPipeline` چهار گام (دریافت داده، آموزش سطح اندیکاتور، آموزش سطح ابعاد، خلاصه) را روی داده‌ی تاریخی اجرا می‌کند؛ برای تحلیل چند‌افقی هم `MultiHorizonWeightLearner` به‌کار می‌رود.

- **ورودی:** `symbol`, `days`, lookback و forward horizon.
- **خروجی:** فایل‌های `.pkl` وزن اندیکاتور و ابعاد، گزارش JSON.
- **وابستگی‌ها:** LightGBM/XGBoost/Sklearn، `FeatureExtractor`, `DataConnector`.

**مشکلات و ریسک‌ها**
- `CompleteAnalysisPipeline` هیچ جایی weight learner آموزش‌دیده را بارگذاری نمی‌کند و حتی در سازنده‌ی `MultiHorizonAnalyzer` پارامتر الزامی `weight_learner` فراهم نشده است (نگاه کنید به `src/gravity_tech/ml/complete_analysis_pipeline.py:173` در برابر تعریف `MultiHorizonTrendAnalyzer.__init__` در `src/gravity_tech/ml/multi_horizon_analysis.py:127-151`).
- ذخیره‌سازی مدل‌ها/وزن‌ها تنها برای Indicator/Dimension learners تعبیه شده و `MultiHorizonWeightLearner.save_weights` هرگز در pipeline فراخوانی نمی‌شود؛ راهی برای نسخه‌بندی یا اشتراک وزن‌ها وجود ندارد (`src/gravity_tech/ml/multi_horizon_weights.py:214-274`).
- اسکریپت آموزش فقط خروجی CLI دارد؛ هیچ هوک CI/CD، هیچ گزارش متریک ساختارمند و هیچ اعتبارسنجی حداقلی برای توقف آموزش در صورت بدتر بودن نسبت baseline وجود ندارد (`src/gravity_tech/ml/train_pipeline.py:34-168`).

---

## 4. تحلیل چند‌افقی و ترکیبی
**جریان:** وزن‌های یادگرفته‌شده روی ویژگی‌های جدید اعمال می‌شوند، `MultiHorizonAnalyzer` و `MultiHorizonMomentumAnalyzer` سیگنال هر افق را می‌سازند و `CombinedTrendMomentumAnalyzer` آن‌ها را ادغام می‌کند. در نهایت `CompleteAnalysisPipeline` قرار است همین نتایج را به 5 بُعد و Volume Matrix پاس دهد.

- **ورودی:** دیکشنری ویژگی برای هر بعد.
- **خروجی:** `MultiHorizonAnalysis`, `MultiHorizonMomentumAnalysis`, `CombinedAnalysis`.
- **وابستگی‌ها:** وزن‌های آموزشی، `pandas`, `numpy`.

**مشکلات و ریسک‌ها**
- سازنده‌ی `CompleteAnalysisPipeline` با `MultiHorizonAnalyzer()` بدون پارامتر کار می‌کند و بلافاصله TypeError پرتاب می‌شود؛ عملاً pipeline غیرقابل اجراست (`src/gravity_tech/ml/complete_analysis_pipeline.py:173`).
- متد `to_dict` در `MultiHorizonMomentumAnalysis` به فیلدهای غیرموجود (`trend_signal`) ارجاع می‌دهد و serialization را هربار با `AttributeError` متوقف می‌کند (`src/gravity_tech/ml/multi_horizon_momentum_analysis.py:66-78`).
- هر فراخوانی `analyze` یک `DataFrame` جدید می‌سازد (`src/gravity_tech/ml/multi_horizon_analysis.py:188-206` و `multi_horizon_momentum_analysis.py:73-98`)؛ در درخواست‌های پرتکرار FastAPI، این تبدیل‌های پرتعداد به گلوگاه CPU تبدیل می‌شود.

---

## 5. ماتریس حجم-بعد و تصمیم 5بعدی
**جریان:** `VolumeDimensionMatrix` روی 20 کندل آخر، نسبت‌های حجمی و تعامل هر بعد با حجم را محاسبه می‌کند. سپس `FiveDimensionalDecisionMatrix` وزن هر بعد را (با اعمال Volume Matrix) تصحیح و سیگنال نهایی/ریسک را تولید می‌کند.

- **ورودی:** `TrendScore`, `MomentumScore`, `VolatilityScore`, `CycleScore`, `SupportResistanceScore`, لیست کندل.
- **خروجی:** `FiveDimensionalDecision` شامل سیگنال، ریسک و توصیه.
- **وابستگی‌ها:** خروجی لایه‌های قبلی، numpy.

**مشکلات و ریسک‌ها**
- تعاملات حجم دو بار در هر اجرا محاسبه می‌شوند: یک‌بار در `_calculate_volume_interactions` (`src/gravity_tech/ml/complete_analysis_pipeline.py:211-243`) و بار دیگر داخل `_apply_volume_adjustments` (`src/gravity_tech/ml/five_dimensional_decision_matrix.py:320-368`). این کار زمان اجرای تصمیم‌گیری را تقریباً دو برابر می‌کند و خروجی ذخیره‌شده‌ی pipeline بی‌استفاده می‌ماند.
- پنجره‌ی حجم به‌صورت ثابت 20 کندل است و هیچ بررسی‌ای برای کافی بودن داده یا تنظیم نسبت با تایم‌فریم انجام نمی‌شود (`src/gravity_tech/ml/volume_dimension_matrix.py:54-142`). در تایم‌فریم‌های پایین‌تر، نسبت‌ها نویزی می‌شوند.
- آستانه‌ها و ضرایب (مثلاً نرمال‌سازی OBV یا حدود InteractionType) تماما هاردکد هستند و هیچ مسیری برای یادگیری یا تِست حساسیت وجود ندارد؛ تنظیمات اشتباه می‌تواند خروجی 5D را ناپایدار کند.

---

## 6. تشخیص الگو و API ML
**جریان:** endpointهای FastAPI در `api/v1/ml.py` مدل پیکل‌شده را بارگذاری، ویژگی‌ها را به numpy تبدیل و پیش‌بینی را بازمی‌گردانند. امکاناتی مثل batch prediction، model info و backtest نیز از همین فایل ارائه می‌شود.

- **ورودی:** `PredictionRequest`, `BatchPredictionRequest`, داده‌های بک‌تست.
- **خروجی:** الگوی پیش‌بینی‌شده، اعتماد، متریک‌های backtest.
- **وابستگی‌ها:** FastAPI, structlog, pickle, numpy.

**مشکلات و ریسک‌ها**
- تابع `load_ml_model()` هر بار که endpoint صدا زده می‌شود فایل مدل را از دیسک می‌خواند (`src/gravity_tech/api/v1/ml.py:132-166`). در محیط تولید این کار I/O سنگینی ایجاد می‌کند و هیچ caching ساده‌ای (مثلاً متغیر ماژولی) فعال نیست.
- `predict_pattern` و `predict_batch` به‌صورت `async` تعریف شده‌اند اما تمام کارها (I/O فایل، numpy، LightGBM) را به شکل همگام روی event loop اجرا می‌کنند (`src/gravity_tech/api/v1/ml.py:185-370`). در نتیجه درخواست پرهزینه باعث Block شدن سایر endpointهای FastAPI می‌شود.
- Batch prediction تک‌تک نمونه‌ها را در یک حلقه پردازش می‌کند و هیچ بردارسازی یا parallelism ندارد؛ به‌خصوص وقتی در هر حلقه مدل از دیسک بارگذاری شود، تاخیر انفجاری خواهد شد.

---

## 7. سرویس TechnicalAnalysisService
**جریان:** `TechnicalAnalysisService.analyze` درخواست REST را گرفته، در thread pool تمام اندیکاتورها را محاسبه و سپس فاز مارکت و سیگنال کلی را تولید می‌کند.

- **ورودی:** `AnalysisRequest` با شمع‌های تنظیم‌شده.
- **خروجی:** `TechnicalAnalysisResult` شامل ده‌ها اندیکاتور، الگو، phase و سیگنال نهایی.
- **وابستگی‌ها:** Indicators مختلف، `FastBatchAnalyzer`, Dow Theory analyzer.

**مشکلات و ریسک‌ها**
- ماژول حمایت/مقاومت هنوز به pipeline متصل نشده و `TODO` فعال است؛ یعنی بعد پنجم تصمیم‌گیری همیشه غایب است (`src/gravity_tech/services/analysis_service.py:101-108`).
- تمام محاسبات حتی در حالت Fast Indicators فاقد کش اشتراکی هستند؛ برای هر درخواست تمام اندیکاتورها دوباره ساخته می‌شوند و هیچ throttling یا warm-cache وجود ندارد. در بار بالا thread pool سریعاً اشباع می‌شود.

---

## 8. پیشنهادگر ابزار و سناریو
**جریان:** `DynamicToolRecommender` با توجه به زمینه بازار وزن هر دسته ابزار را محاسبه کرده، API `/tools` در FastAPI باید نتایج را برگرداند. به موازات، `scenario_weight_optimizer` قرار است وزن‌های سناریو را تنظیم کند.

- **ورودی:** `MarketContext`, وزن‌های ML, candle data.
- **خروجی:** لیست ابزار اولویت‌بندی‌شده، استراتژی پیشنهادی، وزن سناریو.
- **وابستگی‌ها:** LightGBM/XGBoost (اختیاری), numpy, pandas.

**مشکلات و ریسک‌ها**
- مسیر آموزش و ذخیره‌سازی عملاً پیاده‌سازی نشده است؛ `train_recommender`, `save_model` و `load_model` فقط پیام هشدار چاپ می‌کنند (`src/gravity_tech/ml/ml_tool_recommender.py:610-649`).
- router مربوطه هنوز به سیستم واقعی وصل نشده و سرتاسر فایل `api/v1/tools.py` دارای TODO برای بارگذاری registry یا اجرای تحلیل است (خطوط 22، 260، 360، 475، 558 و 639). در نتیجه فرآیند پیشنهاد ابزار صرفاً Mock است.
- در `scenario_weight_optimizer` مقدار `volume_trend` هنوز مقدار ثابت 1.0 دارد و از Volume Matrix نمی‌آید (`src/gravity_tech/ml/scenario_weight_optimizer.py:170`)، بنابراین وزن‌های خروجی سناریو فاقد ارتباط با داده‌ی حجمی واقعی هستند.

---

## 9. بک‌تست و بهینه‌سازی سناریو
**جریان:** `PatternBacktester` الگوها را تشخیص داده، معاملات فرضی را شبیه‌سازی و متریک‌هایی مثل Win Rate/Sharpe را محاسبه می‌کند. سناریو اپتیمایزر نیز وزن‌های ریسک را برای شرایط مختلف تنظیم می‌کند.

- **ورودی:** سری قیمت/حجم، الگو یا مدل طبقه‌بندی.
- **خروجی:** لیست معاملات فرضی، متریک‌های عملکرد، وزن سناریو.

**مشکلات و ریسک‌ها**
- `ml/backtesting.py` برای دسترسی به ماژول‌ها از `sys.path.append` استفاده می‌کند (`src/gravity_tech/ml/backtesting.py:18-33`) که در بسته‌بندی ماژول یا اجرای تست‌ها رفتار غیرقابل‌پیش‌بینی ایجاد می‌کند.
- الگوریتم بک‌تست به‌شدت سریالی و مبتنی بر لیست است؛ هیچ استفاده‌ای از numpy/pandas برای بردارسازی نشده و برای دیتاست‌های بزرگ بسیار کند است.
- سناریو اپتیمایزر همان‌طور که در بخش 8 اشاره شد، هنوز ورودی‌های کلیدی (حجم/رژیم) را به‌طور واقعی مصرف نمی‌کند و بنابراین تصمیمات ریسک آن قابل اعتماد نیست.

---

## 10. استقرار و عملیات
**جریان:** راهنمای استقرار (Kubernetes + Helm) و Runbook عملیاتی در `docs/operations` نگهداری می‌شود؛ pipeline استقرار باید دستی اجرا شود.

- **ورودی:** دسترسی به کلاستر K8s، کانفیگ Helm، Secrets.
- **خروجی:** سرویس FastAPI و ML مستقر در چند ریجن.

**مشکلات و ریسک‌ها**
- راهنمای استقرار صرفاً لیست دستورهای دستی است (`docs/operations/DEPLOYMENT_GUIDE.md:20-120`)، هیچ IaC (Terraform/Helm values versioned) یا CI/CD اتوماتیکی تعریف نشده و احتمال خطای انسانی بالا است.
- مدیریت Secrets/ConfigMaps به کاربر واگذار شده و فرآیند استانداردی برای گردش کلیدها یا rollout امن توصیف نشده است.

---

## 11. تضمین کیفیت و پوشش تست
**جریان:** تست‌های موجود عمدتاً در `tests/integration/test_combined_system.py` و چند تست واحد محدود قرار دارند. گزارش‌های README وضعیت فعلی را نشان می‌دهند.

- **ورودی:** pytest, پوشه‌ی `tests`.
- **خروجی:** گزارش پاس/فیل و کاوریج.

**مشکلات و ریسک‌ها**
- پوشش تست طبق نشان README فقط 11.71٪ است و در badge خط 9 مستند شده است (`README.md:9`). این سطح پوشش با حجم منطق بحرانی ML/مالی کاملاً نامطمئن است.
- هیچ تستی برای `CompleteAnalysisPipeline`، `FiveDimensionalDecisionMatrix` یا Volume Matrix وجود ندارد (جستجوی `CompleteAnalysisPipeline` در مسیر `tests/` خروجی ندارد). بنابراین بحرانی‌ترین لایه تصمیم‌گیری بدون Regression Test باقی مانده است.
- تست‌های موجود فقط سناریوهای خوش‌بینانه (صعودی/نزولی) را پوشش می‌دهند و سناریوهای Edge مانند داده‌ی ناقص، وزن‌های خالی یا مدل بارگذاری‌نشده را بررسی نکرده‌اند.

---

## پیشنهاد بعدی
- اتصال خروجی بسته آموزشی به pipeline تحلیلی (تزریق Weight Learner و حذف دوباره‌کاری‌ها).
- جایگزینی `print` با logging ساختاری و افزودن Retry/Telemetry در ورودی داده.
- پیاده‌سازی caching سبک برای مدل‌های API و vectorization برای batchها.
- تکمیل TODOهای Service/Tools و پوشش تست ماژول‌های بدون تست (به‌خصوص Volume/5D و Analysis Service).
