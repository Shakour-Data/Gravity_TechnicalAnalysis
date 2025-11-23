-- =====================================================
-- Historical Analysis Results Database Schema
-- =====================================================
--
-- این schema برای ذخیره نتایج تحلیل‌های تکنیکال historical
-- در رویکرد هیبریدی طراحی شده است.
--
-- هدف: پشتیبانی از دسترسی historical به تحلیل‌ها
-- =====================================================

-- جدول اصلی نتایج تحلیل historical
CREATE TABLE IF NOT EXISTS historical_scores (
    id SERIAL PRIMARY KEY,

    -- شناسایی
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    timeframe VARCHAR(10) NOT NULL,

    -- امتیازهای کلی
    trend_score DECIMAL(5, 4) DEFAULT 0.0,
    trend_confidence DECIMAL(5, 4) DEFAULT 0.0,
    momentum_score DECIMAL(5, 4) DEFAULT 0.0,
    momentum_confidence DECIMAL(5, 4) DEFAULT 0.0,
    combined_score DECIMAL(5, 4) DEFAULT 0.0,
    combined_confidence DECIMAL(5, 4) DEFAULT 0.0,

    -- وزن‌ها
    trend_weight DECIMAL(5, 4) DEFAULT 0.5,
    momentum_weight DECIMAL(5, 4) DEFAULT 0.5,

    -- سیگنال‌ها
    trend_signal VARCHAR(20) DEFAULT 'NEUTRAL',
    momentum_signal VARCHAR(20) DEFAULT 'NEUTRAL',
    combined_signal VARCHAR(20) DEFAULT 'NEUTRAL',

    -- امتیازات اضافی (برای آینده)
    volume_score DECIMAL(5, 4) DEFAULT 0.0,
    volatility_score DECIMAL(5, 4) DEFAULT 0.0,
    cycle_score DECIMAL(5, 4) DEFAULT 0.0,
    support_resistance_score DECIMAL(5, 4) DEFAULT 0.0,

    -- اطلاعات تکمیلی
    recommendation VARCHAR(20), -- BUY, SELL, HOLD
    action VARCHAR(20), -- ENTER, EXIT, WAIT
    price_at_analysis DECIMAL(15, 8), -- قیمت در زمان تحلیل

    -- داده‌های کامل (JSON)
    raw_data JSONB,

    -- متادیتا
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_historical_scores_symbol ON historical_scores(symbol);
CREATE INDEX IF NOT EXISTS idx_historical_scores_timeframe ON historical_scores(timeframe);
CREATE INDEX IF NOT EXISTS idx_historical_scores_timestamp ON historical_scores(timestamp);
CREATE INDEX IF NOT EXISTS idx_historical_scores_symbol_timeframe ON historical_scores(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_historical_scores_combined_score ON historical_scores(combined_score);
CREATE INDEX IF NOT EXISTS idx_historical_scores_created_at ON historical_scores(created_at);

-- Partitioning by month (for better performance with large datasets)
-- CREATE TABLE historical_scores_y2025m11 PARTITION OF historical_scores
--     FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

-- Comments
COMMENT ON TABLE historical_scores IS 'Historical technical analysis results for hybrid architecture';
COMMENT ON COLUMN historical_scores.symbol IS 'Trading pair symbol (e.g., BTCUSDT)';
COMMENT ON COLUMN historical_scores.timestamp IS 'Analysis timestamp';
COMMENT ON COLUMN historical_scores.timeframe IS 'Candle timeframe (1m, 5m, 1h, 1d, etc.)';
COMMENT ON COLUMN historical_scores.combined_score IS 'Overall analysis score (-1 to 1)';
COMMENT ON COLUMN historical_scores.combined_confidence IS 'Confidence in the analysis (0-1)';
COMMENT ON COLUMN historical_scores.raw_data IS 'Complete analysis results as JSON';