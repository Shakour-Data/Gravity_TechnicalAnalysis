"""
Complete Project Database Schema - Schema کامل دیتابیس پروژه

این فایل تمام جداول مورد نیاز پروژه Gravity Tech را تعریف می‌کند

SQLite Schema with all 7 main tables:
1. historical_scores
2. tool_performance_history
3. tool_performance_stats
4. ml_weights_history
5. tool_recommendations_log
6. market_data_cache
7. pattern_detection_results

Author: Gravity Tech Team
Date: December 5, 2025
Version: 1.0.0
License: MIT
"""

# Table 1: Historical Scores
TABLE_HISTORICAL_SCORES = """
CREATE TABLE IF NOT EXISTS historical_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- شناسایی
    ticker TEXT NOT NULL,
    analysis_date DATETIME NOT NULL,
    timeframe TEXT NOT NULL,

    -- امتیازهای کلی
    trend_score REAL DEFAULT 0.0,
    trend_confidence REAL DEFAULT 0.0,
    momentum_score REAL DEFAULT 0.0,
    momentum_confidence REAL DEFAULT 0.0,
    combined_score REAL DEFAULT 0.0,
    combined_confidence REAL DEFAULT 0.0,

    -- وزن‌ها
    trend_weight REAL DEFAULT 0.5,
    momentum_weight REAL DEFAULT 0.5,

    -- سیگنال‌ها
    trend_signal TEXT DEFAULT 'NEUTRAL',
    momentum_signal TEXT DEFAULT 'NEUTRAL',
    combined_signal TEXT DEFAULT 'NEUTRAL',

    -- امتیازات ابعاد دیگر
    volume_score REAL DEFAULT 0.0,
    volatility_score REAL DEFAULT 0.0,
    cycle_score REAL DEFAULT 0.0,
    support_resistance_score REAL DEFAULT 0.0,

    -- اطلاعات تکمیلی
    recommendation TEXT,
    action TEXT,
    price_at_analysis REAL,

    -- داده‌های کامل (JSON)
    raw_data TEXT,

    -- متادیتا
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_historical_scores_ticker
    ON historical_scores(ticker);
CREATE INDEX IF NOT EXISTS idx_historical_scores_timeframe
    ON historical_scores(timeframe);
CREATE INDEX IF NOT EXISTS idx_historical_scores_date
    ON historical_scores(analysis_date);
CREATE INDEX IF NOT EXISTS idx_historical_scores_ticker_date
    ON historical_scores(ticker, analysis_date);
CREATE INDEX IF NOT EXISTS idx_historical_scores_combined_score
    ON historical_scores(combined_score);
CREATE INDEX IF NOT EXISTS idx_historical_scores_created_at
    ON historical_scores(created_at);
"""

# Table 2: Tool Performance History
TABLE_TOOL_PERFORMANCE_HISTORY = """
CREATE TABLE IF NOT EXISTS tool_performance_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    tool_name TEXT NOT NULL,
    tool_category TEXT NOT NULL,

    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    market_regime TEXT NOT NULL,
    volatility_level REAL,
    trend_strength REAL,
    volume_profile TEXT,

    prediction_type TEXT NOT NULL,
    prediction_value REAL,
    confidence_score REAL,

    actual_result TEXT,
    actual_price_change REAL,
    success INTEGER,
    accuracy REAL,

    prediction_timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    result_timestamp TEXT,
    evaluation_period_hours INTEGER,

    metadata TEXT,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tool_perf_tool_name
    ON tool_performance_history(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_perf_symbol
    ON tool_performance_history(symbol);
CREATE INDEX IF NOT EXISTS idx_tool_perf_regime
    ON tool_performance_history(market_regime);
CREATE INDEX IF NOT EXISTS idx_tool_perf_timestamp
    ON tool_performance_history(prediction_timestamp);
"""

# Table 3: Tool Performance Stats
TABLE_TOOL_PERFORMANCE_STATS = """
CREATE TABLE IF NOT EXISTS tool_performance_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT NOT NULL,
    tool_category TEXT NOT NULL,

    market_regime TEXT,
    timeframe TEXT,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,

    total_predictions INTEGER NOT NULL DEFAULT 0,
    correct_predictions INTEGER NOT NULL DEFAULT 0,
    accuracy REAL,

    avg_confidence REAL,
    avg_actual_change REAL,

    bullish_predictions INTEGER DEFAULT 0,
    bearish_predictions INTEGER DEFAULT 0,
    neutral_predictions INTEGER DEFAULT 0,

    bullish_success_rate REAL,
    bearish_success_rate REAL,
    neutral_success_rate REAL,

    best_accuracy REAL,
    worst_accuracy REAL,
    best_symbol TEXT,
    worst_symbol TEXT,

    last_updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(tool_name, market_regime, timeframe, period_start, period_end)
);

CREATE INDEX IF NOT EXISTS idx_tool_stats_tool_name
    ON tool_performance_stats(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_stats_regime
    ON tool_performance_stats(market_regime);
CREATE INDEX IF NOT EXISTS idx_tool_stats_accuracy
    ON tool_performance_stats(accuracy);
"""

# Table 4: ML Weights History
TABLE_ML_WEIGHTS_HISTORY = """
CREATE TABLE IF NOT EXISTS ml_weights_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,

    market_regime TEXT,
    timeframe TEXT,

    weights TEXT NOT NULL,

    training_accuracy REAL,
    validation_accuracy REAL,
    r2_score REAL,
    mae REAL,

    training_samples INTEGER,
    training_date TEXT NOT NULL,

    metadata TEXT,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ml_weights_model
    ON ml_weights_history(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_ml_weights_regime
    ON ml_weights_history(market_regime);
CREATE INDEX IF NOT EXISTS idx_ml_weights_date
    ON ml_weights_history(training_date);
"""

# Table 5: Tool Recommendations Log
TABLE_TOOL_RECOMMENDATIONS_LOG = """
CREATE TABLE IF NOT EXISTS tool_recommendations_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    request_id TEXT NOT NULL,
    user_id TEXT,

    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    analysis_goal TEXT,
    trading_style TEXT,

    market_regime TEXT NOT NULL,
    volatility_level REAL,
    trend_strength REAL,

    recommended_tools TEXT NOT NULL,
    ml_weights TEXT,

    user_feedback TEXT,
    tools_actually_used TEXT,
    trade_result TEXT,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    feedback_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_recommendations_request
    ON tool_recommendations_log(request_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_user
    ON tool_recommendations_log(user_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_symbol
    ON tool_recommendations_log(symbol);
CREATE INDEX IF NOT EXISTS idx_recommendations_created
    ON tool_recommendations_log(created_at);
"""

# Table 6: Market Data Cache
TABLE_MARKET_DATA_CACHE = """
CREATE TABLE IF NOT EXISTS market_data_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TEXT NOT NULL,

    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(symbol, timeframe, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_market_data_symbol
    ON market_data_cache(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_timeframe
    ON market_data_cache(timeframe);
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp
    ON market_data_cache(timestamp);
"""

# Table 7: Pattern Detection Results
TABLE_PATTERN_DETECTION_RESULTS = """
CREATE TABLE IF NOT EXISTS pattern_detection_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TEXT NOT NULL,

    pattern_type TEXT NOT NULL,
    pattern_name TEXT NOT NULL,

    confidence REAL,
    strength REAL,

    start_time TEXT,
    end_time TEXT,
    start_price REAL,
    end_price REAL,

    prediction TEXT,
    target_price REAL,
    stop_loss REAL,

    metadata TEXT,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pattern_symbol
    ON pattern_detection_results(symbol);
CREATE INDEX IF NOT EXISTS idx_pattern_type
    ON pattern_detection_results(pattern_type);
CREATE INDEX IF NOT EXISTS idx_pattern_timestamp
    ON pattern_detection_results(timestamp);
"""

# Table 8: Historical Indicator Scores
TABLE_HISTORICAL_INDICATOR_SCORES = """
CREATE TABLE IF NOT EXISTS historical_indicator_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    score_id INTEGER,
    ticker TEXT NOT NULL,
    analysis_date DATETIME NOT NULL,
    timeframe TEXT NOT NULL,

    indicator_name TEXT NOT NULL,
    indicator_category TEXT,
    indicator_params TEXT,

    value REAL NOT NULL,
    signal TEXT,
    confidence REAL,

    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (score_id) REFERENCES historical_scores(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_indicator_ticker_date
    ON historical_indicator_scores(ticker, analysis_date);
CREATE INDEX IF NOT EXISTS idx_indicator_name
    ON historical_indicator_scores(indicator_name);
CREATE INDEX IF NOT EXISTS idx_indicator_category
    ON historical_indicator_scores(indicator_category);
CREATE INDEX IF NOT EXISTS idx_indicator_score_id
    ON historical_indicator_scores(score_id);
"""

# All tables SQL combined
ALL_TABLES_SQL = f"""
{TABLE_HISTORICAL_SCORES}

{TABLE_TOOL_PERFORMANCE_HISTORY}

{TABLE_TOOL_PERFORMANCE_STATS}

{TABLE_ML_WEIGHTS_HISTORY}

{TABLE_TOOL_RECOMMENDATIONS_LOG}

{TABLE_MARKET_DATA_CACHE}

{TABLE_PATTERN_DETECTION_RESULTS}

{TABLE_HISTORICAL_INDICATOR_SCORES}
"""

if __name__ == "__main__":
    # Print all SQL for reference
    print(ALL_TABLES_SQL)
