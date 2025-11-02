-- ═══════════════════════════════════════════════════════════════════
-- Historical Scoring System Database Schema
-- ═══════════════════════════════════════════════════════════════════
-- هدف: ذخیره تمام امتیازها، اندیکاتورها، و ضرایب به صورت تاریخی
-- تا کاربر بتواند امتیاز هر تاریخی را بازیابی کند
-- ═══════════════════════════════════════════════════════════════════

-- 1️⃣ جدول اصلی امتیازهای تاریخی (Historical Scores)
CREATE TABLE IF NOT EXISTS historical_scores (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,                    -- مثل: BTCUSDT, ETHUSDT
    timestamp TIMESTAMPTZ NOT NULL,                 -- زمان دقیق محاسبه
    timeframe VARCHAR(10) NOT NULL,                 -- مثل: 1h, 4h, 1d
    
    -- امتیازهای کلی (Overall)
    trend_score DECIMAL(5,3) NOT NULL,              -- [-1, +1]
    trend_confidence DECIMAL(5,3) NOT NULL,         -- [0, 1]
    momentum_score DECIMAL(5,3) NOT NULL,           -- [-1, +1]
    momentum_confidence DECIMAL(5,3) NOT NULL,      -- [0, 1]
    combined_score DECIMAL(5,3) NOT NULL,           -- [-1, +1]
    combined_confidence DECIMAL(5,3) NOT NULL,      -- [0, 1]
    
    -- وزن‌ها (Weights)
    trend_weight DECIMAL(4,3) NOT NULL,             -- [0, 1]
    momentum_weight DECIMAL(4,3) NOT NULL,          -- [0, 1]
    
    -- سیگنال‌ها
    trend_signal VARCHAR(20) NOT NULL,              -- VERY_BULLISH, BULLISH, ...
    momentum_signal VARCHAR(20) NOT NULL,
    combined_signal VARCHAR(20) NOT NULL,
    
    -- توصیه نهایی
    recommendation VARCHAR(20) NOT NULL,            -- STRONG_BUY, BUY, HOLD, ...
    action VARCHAR(20) NOT NULL,                    -- STRONG_BUY, ACCUMULATE, ...
    
    -- قیمت در زمان تحلیل
    price_at_analysis DECIMAL(20,8) NOT NULL,
    
    -- برای جستجوی سریع
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Index برای جستجوی سریع
    CONSTRAINT unique_score_entry UNIQUE (symbol, timestamp, timeframe)
);

-- Index برای query های سریع
CREATE INDEX idx_historical_scores_symbol_time ON historical_scores(symbol, timestamp DESC);
CREATE INDEX idx_historical_scores_timeframe ON historical_scores(timeframe);
CREATE INDEX idx_historical_scores_date ON historical_scores(DATE(timestamp));


-- 2️⃣ جدول امتیازهای Multi-Horizon (برای 3d, 7d, 30d)
CREATE TABLE IF NOT EXISTS historical_horizon_scores (
    id BIGSERIAL PRIMARY KEY,
    score_id BIGINT NOT NULL REFERENCES historical_scores(id) ON DELETE CASCADE,
    
    -- مشخصات horizon
    horizon VARCHAR(5) NOT NULL,                    -- '3d', '7d', '30d'
    analysis_type VARCHAR(20) NOT NULL,             -- 'TREND', 'MOMENTUM'
    
    -- امتیازها
    score DECIMAL(5,3) NOT NULL,                    -- [-1, +1]
    confidence DECIMAL(5,3) NOT NULL,               -- [0, 1]
    signal VARCHAR(20) NOT NULL,                    -- VERY_BULLISH, ...
    
    CONSTRAINT unique_horizon_entry UNIQUE (score_id, horizon, analysis_type)
);

CREATE INDEX idx_horizon_scores_score_id ON historical_horizon_scores(score_id);


-- 3️⃣ جدول امتیازهای اندیکاتورها (تک تک)
CREATE TABLE IF NOT EXISTS historical_indicator_scores (
    id BIGSERIAL PRIMARY KEY,
    score_id BIGINT NOT NULL REFERENCES historical_scores(id) ON DELETE CASCADE,
    
    -- مشخصات اندیکاتور
    indicator_name VARCHAR(50) NOT NULL,            -- SMA, EMA, MACD, RSI, ...
    indicator_category VARCHAR(20) NOT NULL,        -- TREND, MOMENTUM
    indicator_params JSONB,                         -- پارامترها: {period: 20, ...}
    
    -- امتیازها
    score DECIMAL(5,3) NOT NULL,                    -- [-1, +1]
    confidence DECIMAL(5,3) NOT NULL,               -- [0, 1]
    signal VARCHAR(20) NOT NULL,
    
    -- مقادیر خام اندیکاتور (برای debugging)
    raw_value DECIMAL(20,8),                        -- مقدار واقعی اندیکاتور
    
    CONSTRAINT unique_indicator_entry UNIQUE (score_id, indicator_name, indicator_params)
);

CREATE INDEX idx_indicator_scores_score_id ON historical_indicator_scores(score_id);
CREATE INDEX idx_indicator_scores_name ON historical_indicator_scores(indicator_name);


-- 4️⃣ جدول الگوهای تشخیص داده شده (Patterns)
CREATE TABLE IF NOT EXISTS historical_patterns (
    id BIGSERIAL PRIMARY KEY,
    score_id BIGINT NOT NULL REFERENCES historical_scores(id) ON DELETE CASCADE,
    
    -- مشخصات الگو
    pattern_type VARCHAR(20) NOT NULL,              -- CANDLESTICK, CLASSICAL, ELLIOTT
    pattern_name VARCHAR(100) NOT NULL,             -- Bullish Engulfing, Head and Shoulders, ...
    
    -- امتیازها
    score DECIMAL(5,3) NOT NULL,
    confidence DECIMAL(5,3) NOT NULL,
    signal VARCHAR(20) NOT NULL,
    
    -- جزئیات الگو
    description TEXT,
    candle_indices JSONB,                           -- شماره کندل‌های مربوطه
    price_levels JSONB,                             -- سطوح قیمتی مهم: {support: 48000, resistance: 52000}
    projected_target DECIMAL(20,8),                 -- هدف قیمتی پیش‌بینی شده
    
    detected_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_patterns_score_id ON historical_patterns(score_id);
CREATE INDEX idx_patterns_type ON historical_patterns(pattern_type);


-- 5️⃣ جدول وزن‌های یادگیری شده (ML Weights)
CREATE TABLE IF NOT EXISTS historical_ml_weights (
    id BIGSERIAL PRIMARY KEY,
    
    -- مشخصات
    model_type VARCHAR(20) NOT NULL,                -- TREND, MOMENTUM
    horizon VARCHAR(5) NOT NULL,                    -- '3d', '7d', '30d'
    
    -- وزن‌های اندیکاتورها (JSON)
    indicator_weights JSONB NOT NULL,               -- {SMA: 0.15, EMA: 0.18, ...}
    
    -- متادیتا
    training_date TIMESTAMPTZ NOT NULL,
    accuracy DECIMAL(5,3),                          -- دقت مدل
    samples_count INTEGER,                          -- تعداد نمونه‌های آموزش
    
    -- برای version control
    version INTEGER NOT NULL DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT unique_weight_version UNIQUE (model_type, horizon, version)
);

CREATE INDEX idx_ml_weights_active ON historical_ml_weights(model_type, horizon, is_active);


-- 6️⃣ جدول اهداف قیمتی و نتایج (Price Targets & Results)
CREATE TABLE IF NOT EXISTS historical_price_targets (
    id BIGSERIAL PRIMARY KEY,
    score_id BIGINT NOT NULL REFERENCES historical_scores(id) ON DELETE CASCADE,
    
    -- اهداف
    target_type VARCHAR(20) NOT NULL,               -- SHORT_TERM, MID_TERM, LONG_TERM
    target_price DECIMAL(20,8) NOT NULL,
    stop_loss DECIMAL(20,8),
    
    -- پیش‌بینی
    expected_timeframe VARCHAR(20),                 -- 1d, 3d, 7d, 1mo
    confidence DECIMAL(5,3),
    
    -- نتیجه واقعی (برای backtesting)
    actual_reached BOOLEAN,
    reached_at TIMESTAMPTZ,
    actual_high DECIMAL(20,8),
    actual_low DECIMAL(20,8),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_price_targets_score_id ON historical_price_targets(score_id);


-- 7️⃣ جدول حجم و تایید (Volume Confirmation)
CREATE TABLE IF NOT EXISTS historical_volume_analysis (
    id BIGSERIAL PRIMARY KEY,
    score_id BIGINT NOT NULL REFERENCES historical_scores(id) ON DELETE CASCADE,
    
    -- تحلیل حجم
    volume_score DECIMAL(5,3) NOT NULL,             -- [-1, +1]
    volume_confidence DECIMAL(5,3) NOT NULL,        -- [0, 1]
    
    -- جزئیات
    avg_volume DECIMAL(20,2),                       -- میانگین حجم
    current_volume DECIMAL(20,2),                   -- حجم فعلی
    volume_ratio DECIMAL(6,3),                      -- نسبت به میانگین
    
    -- تایید روند
    confirms_trend BOOLEAN,
    
    CONSTRAINT unique_volume_entry UNIQUE (score_id)
);


-- 8️⃣ جدول Metadata برای کش و بهینه‌سازی
CREATE TABLE IF NOT EXISTS analysis_metadata (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    
    -- آخرین تحلیل
    last_analysis_at TIMESTAMPTZ NOT NULL,
    last_score_id BIGINT REFERENCES historical_scores(id),
    
    -- آمار
    total_analyses INTEGER DEFAULT 0,
    last_price DECIMAL(20,8),
    
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT unique_metadata UNIQUE (symbol, timeframe)
);


-- ═══════════════════════════════════════════════════════════════════
-- Views برای جستجوی آسان
-- ═══════════════════════════════════════════════════════════════════

-- نمای کامل امتیازها با horizons
CREATE VIEW v_complete_scores AS
SELECT 
    hs.*,
    json_agg(
        json_build_object(
            'horizon', hhs.horizon,
            'type', hhs.analysis_type,
            'score', hhs.score,
            'confidence', hhs.confidence,
            'signal', hhs.signal
        )
    ) as horizon_details
FROM historical_scores hs
LEFT JOIN historical_horizon_scores hhs ON hs.id = hhs.score_id
GROUP BY hs.id;


-- نمای آخرین تحلیل هر symbol
CREATE VIEW v_latest_scores AS
SELECT DISTINCT ON (symbol, timeframe)
    *
FROM historical_scores
ORDER BY symbol, timeframe, timestamp DESC;


-- ═══════════════════════════════════════════════════════════════════
-- Functions برای جستجوی تاریخی
-- ═══════════════════════════════════════════════════════════════════

-- تابع برای دریافت امتیاز در یک تاریخ خاص
CREATE OR REPLACE FUNCTION get_score_at_date(
    p_symbol VARCHAR,
    p_timeframe VARCHAR,
    p_date TIMESTAMPTZ
)
RETURNS TABLE (
    score_data JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        jsonb_build_object(
            'id', hs.id,
            'symbol', hs.symbol,
            'timestamp', hs.timestamp,
            'trend_score', hs.trend_score,
            'momentum_score', hs.momentum_score,
            'combined_score', hs.combined_score,
            'recommendation', hs.recommendation,
            'price', hs.price_at_analysis
        ) as score_data
    FROM historical_scores hs
    WHERE hs.symbol = p_symbol
      AND hs.timeframe = p_timeframe
      AND hs.timestamp <= p_date
    ORDER BY hs.timestamp DESC
    LIMIT 1;
END;
$$;


-- تابع برای دریافت سری زمانی امتیازها
CREATE OR REPLACE FUNCTION get_score_timeseries(
    p_symbol VARCHAR,
    p_timeframe VARCHAR,
    p_from TIMESTAMPTZ,
    p_to TIMESTAMPTZ
)
RETURNS TABLE (
    timestamp TIMESTAMPTZ,
    trend_score DECIMAL,
    momentum_score DECIMAL,
    combined_score DECIMAL,
    price DECIMAL
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        hs.timestamp,
        hs.trend_score,
        hs.momentum_score,
        hs.combined_score,
        hs.price_at_analysis
    FROM historical_scores hs
    WHERE hs.symbol = p_symbol
      AND hs.timeframe = p_timeframe
      AND hs.timestamp BETWEEN p_from AND p_to
    ORDER BY hs.timestamp ASC;
END;
$$;


-- ═══════════════════════════════════════════════════════════════════
-- Cleanup و Maintenance
-- ═══════════════════════════════════════════════════════════════════

-- تابع برای حذف داده‌های قدیمی (بیشتر از X روز)
CREATE OR REPLACE FUNCTION cleanup_old_scores(days_to_keep INTEGER DEFAULT 365)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM historical_scores
    WHERE timestamp < NOW() - (days_to_keep || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$;


-- ═══════════════════════════════════════════════════════════════════
-- Partitioning (اختیاری - برای دیتای خیلی زیاد)
-- ═══════════════════════════════════════════════════════════════════

-- اگر دیتا خیلی زیاد شد، می‌توان جداول را partition کرد:
-- CREATE TABLE historical_scores_2024_01 PARTITION OF historical_scores
--     FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
-- CREATE TABLE historical_scores_2024_02 PARTITION OF historical_scores
--     FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
-- ...


-- ═══════════════════════════════════════════════════════════════════
-- نمونه Queries مفید
-- ═══════════════════════════════════════════════════════════════════

-- 1. دریافت آخرین امتیاز یک symbol:
-- SELECT * FROM v_latest_scores WHERE symbol = 'BTCUSDT' AND timeframe = '1h';

-- 2. دریافت امتیاز در تاریخ خاص:
-- SELECT * FROM get_score_at_date('BTCUSDT', '1h', '2024-01-15 10:00:00+00');

-- 3. دریافت سری زمانی برای نمودار:
-- SELECT * FROM get_score_timeseries('BTCUSDT', '1h', '2024-01-01', '2024-01-31');

-- 4. مقایسه عملکرد اندیکاتورها:
-- SELECT 
--     indicator_name,
--     AVG(confidence) as avg_confidence,
--     COUNT(*) as usage_count
-- FROM historical_indicator_scores
-- WHERE score_id IN (
--     SELECT id FROM historical_scores 
--     WHERE symbol = 'BTCUSDT' 
--     AND timestamp > NOW() - INTERVAL '30 days'
-- )
-- GROUP BY indicator_name
-- ORDER BY avg_confidence DESC;

-- 5. دریافت الگوهای موفق:
-- SELECT 
--     hp.pattern_name,
--     COUNT(*) as detected_count,
--     AVG(hp.confidence) as avg_confidence,
--     AVG(hpt.actual_high - hs.price_at_analysis) / hs.price_at_analysis * 100 as avg_gain_pct
-- FROM historical_patterns hp
-- JOIN historical_scores hs ON hp.score_id = hs.id
-- LEFT JOIN historical_price_targets hpt ON hp.score_id = hpt.score_id
-- WHERE hpt.actual_reached = TRUE
-- GROUP BY hp.pattern_name
-- ORDER BY avg_gain_pct DESC;
