# گزارش اعتبارسنجی بچ (50 نماد)
- زمان اجرا: 2025-12-19T04:07:49.651905+00:00
- DSN: postgresql://gravity:gravity_db_pass@127.0.0.1:5545/tech_analysis
- سورس: E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db
- limit کندل: 150

## شمارش و پوشش جداول
- analysis_results: rows=8190, symbols=50
- historical_scores: rows=33091, symbols=50
- historical_indicator_scores: rows=66199, symbols=50
- tool_performance_history: rows=33091, symbols=50
- backtest_runs: rows=38112, symbols=50
- pattern_detection_results: rows=33090, symbols=50
- ml_weights_history: rows=33090, symbols=50

- analysis_results range: (datetime.datetime(2025, 9, 19, 3, 24, 16, 448117), datetime.datetime(2025, 12, 19, 3, 51, 48, 559982))
- historical_scores range: (datetime.datetime(2011, 3, 26, 0, 0, tzinfo=datetime.timezone.utc), datetime.datetime(2025, 12, 13, 0, 0, tzinfo=datetime.timezone.utc))
- ml_weights_history range: (datetime.datetime(2011, 3, 26, 0, 0, tzinfo=datetime.timezone.utc), datetime.datetime(2025, 12, 13, 0, 0, tzinfo=datetime.timezone.utc))

## تکرار داده (duplicates)
- historical_scores duplicates (by symbol, ts, timeframe): 0
- historical_indicator_scores duplicates (by symbol, ts, timeframe, indicator_name, coalesce(indicator_params::text,'__NULL__')): 12
- ml_weights_history duplicates (by symbol, ts, model_name, timeframe): 0
- tool_performance_history duplicates (by symbol, timeframe, prediction_timestamp, tool_name): 0
- backtest_runs duplicates (by symbol, interval, period_start, period_end, model_version): 0
- pattern_detection_results duplicates (by symbol, timeframe, timestamp, pattern_type, pattern_name): 0

## منفی بودن نوسان (volatility_score < 0)
- historical_scores: 0 ردیف منفی

## تطابق جهت trend_score با بازده روز بعد
- total مقایسه: 135
- توافق جهت: 35 (25.93%)

## پوشش کندل سورس
- میانگین کندل (تا limit): 150.0
- حداقل کندل (تا limit): 150