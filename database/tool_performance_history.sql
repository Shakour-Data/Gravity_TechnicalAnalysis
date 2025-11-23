-- =====================================================
-- Tool Performance History Database Schema
-- =====================================================
-- 
-- این schema برای ذخیره عملکرد تاریخی ابزارهای تحلیل تکنیکال
-- در شرایط مختلف بازار طراحی شده است.
-- 
-- هدف: یادگیری ML از عملکرد واقعی ابزارها
-- =====================================================

-- جدول اصلی عملکرد ابزارها
CREATE TABLE IF NOT EXISTS tool_performance_history (
    id SERIAL PRIMARY KEY,
    tool_name VARCHAR(100) NOT NULL,
    tool_category VARCHAR(50) NOT NULL,
    
    -- کانتکست بازار
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    market_regime VARCHAR(30) NOT NULL, -- trending_bullish, trending_bearish, ranging, volatile
    volatility_level DECIMAL(5, 2), -- 0-100
    trend_strength DECIMAL(5, 2), -- 0-100
    volume_profile VARCHAR(20), -- high, medium, low
    
    -- پیش‌بینی ابزار
    prediction_type VARCHAR(30) NOT NULL, -- buy, sell, hold, bullish, bearish, neutral
    prediction_value DECIMAL(10, 4), -- مقدار عددی پیش‌بینی (اگر وجود دارد)
    confidence_score DECIMAL(5, 4), -- 0-1
    
    -- نتیجه واقعی
    actual_result VARCHAR(30), -- buy, sell, hold, bullish, bearish, neutral
    actual_price_change DECIMAL(10, 4), -- تغییر قیمت واقعی (%)
    success BOOLEAN, -- آیا پیش‌بینی درست بود؟
    accuracy DECIMAL(5, 4), -- 0-1
    
    -- متادیتا
    prediction_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    result_timestamp TIMESTAMP, -- زمان مشخص شدن نتیجه
    evaluation_period_hours INTEGER, -- مدت زمان ارزیابی (ساعت)
    
    -- اطلاعات اضافی
    metadata JSONB, -- اطلاعات اضافی (parameters، شرایط خاص، etc.)
    
    -- Indexes
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_tool_performance_tool_name ON tool_performance_history(tool_name);
CREATE INDEX idx_tool_performance_symbol ON tool_performance_history(symbol);
CREATE INDEX idx_tool_performance_regime ON tool_performance_history(market_regime);
CREATE INDEX idx_tool_performance_timeframe ON tool_performance_history(timeframe);
CREATE INDEX idx_tool_performance_timestamp ON tool_performance_history(prediction_timestamp DESC);
CREATE INDEX idx_tool_performance_success ON tool_performance_history(success);

-- Composite index for common queries
CREATE INDEX idx_tool_performance_lookup ON tool_performance_history(
    tool_name, 
    market_regime, 
    timeframe, 
    prediction_timestamp DESC
);

-- =====================================================
-- جدول آمار تجمیعی ابزارها
-- =====================================================
-- این جدول برای سرعت بخشیدن به کوئری‌های آماری استفاده می‌شود
CREATE TABLE IF NOT EXISTS tool_performance_stats (
    id SERIAL PRIMARY KEY,
    tool_name VARCHAR(100) NOT NULL,
    tool_category VARCHAR(50) NOT NULL,
    
    -- فیلترهای آماری
    market_regime VARCHAR(30),
    timeframe VARCHAR(10),
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    
    -- آمار عملکرد
    total_predictions INTEGER NOT NULL DEFAULT 0,
    correct_predictions INTEGER NOT NULL DEFAULT 0,
    accuracy DECIMAL(5, 4), -- correct / total
    
    avg_confidence DECIMAL(5, 4),
    avg_actual_change DECIMAL(10, 4),
    
    bullish_predictions INTEGER DEFAULT 0,
    bearish_predictions INTEGER DEFAULT 0,
    neutral_predictions INTEGER DEFAULT 0,
    
    bullish_success_rate DECIMAL(5, 4),
    bearish_success_rate DECIMAL(5, 4),
    neutral_success_rate DECIMAL(5, 4),
    
    -- بهترین و بدترین
    best_accuracy DECIMAL(5, 4),
    worst_accuracy DECIMAL(5, 4),
    best_symbol VARCHAR(20),
    worst_symbol VARCHAR(20),
    
    -- متادیتا
    last_updated TIMESTAMP NOT NULL DEFAULT NOW(),
    
    UNIQUE(tool_name, market_regime, timeframe, period_start, period_end)
);

CREATE INDEX idx_tool_stats_tool_name ON tool_performance_stats(tool_name);
CREATE INDEX idx_tool_stats_regime ON tool_performance_stats(market_regime);
CREATE INDEX idx_tool_stats_period ON tool_performance_stats(period_start, period_end);

-- =====================================================
-- جدول وزن‌های ML تاریخی
-- =====================================================
-- ذخیره وزن‌های یادگرفته شده توسط ML در طول زمان
CREATE TABLE IF NOT EXISTS ml_weights_history (
    id SERIAL PRIMARY KEY,
    
    -- شناسایی مدل
    model_name VARCHAR(100) NOT NULL, -- ml_indicator_weights, ml_dimension_weights, etc.
    model_version VARCHAR(20) NOT NULL,
    
    -- کانتکست
    market_regime VARCHAR(30),
    timeframe VARCHAR(10),
    
    -- وزن‌ها (JSON)
    weights JSONB NOT NULL, -- {"MACD": 0.28, "RSI": 0.24, ...}
    
    -- متریک‌های مدل
    training_accuracy DECIMAL(5, 4),
    validation_accuracy DECIMAL(5, 4),
    r2_score DECIMAL(5, 4),
    mae DECIMAL(10, 4),
    
    -- اطلاعات آموزش
    training_samples INTEGER,
    training_date TIMESTAMP NOT NULL,
    
    -- متادیتا
    metadata JSONB, -- hyperparameters, feature importance, etc.
    
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ml_weights_model ON ml_weights_history(model_name, model_version);
CREATE INDEX idx_ml_weights_regime ON ml_weights_history(market_regime);
CREATE INDEX idx_ml_weights_date ON ml_weights_history(training_date DESC);

-- =====================================================
-- جدول پیشنهادات ابزارها
-- =====================================================
-- ذخیره پیشنهادات داده شده به کاربران
CREATE TABLE IF NOT EXISTS tool_recommendations_log (
    id SERIAL PRIMARY KEY,
    
    -- شناسایی درخواست
    request_id UUID NOT NULL,
    user_id VARCHAR(100), -- اگر احراز هویت وجود دارد
    
    -- کانتکست درخواست
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    analysis_goal VARCHAR(50), -- entry_signal, exit_signal, etc.
    trading_style VARCHAR(20), -- scalp, day, swing, position
    
    -- کانتکست بازار
    market_regime VARCHAR(30) NOT NULL,
    volatility_level DECIMAL(5, 2),
    trend_strength DECIMAL(5, 2),
    
    -- پیشنهادات
    recommended_tools JSONB NOT NULL, -- لیست ابزارهای پیشنهادی با جزئیات
    ml_weights JSONB, -- وزن‌های ML استفاده شده
    
    -- نتیجه (اگر feedback وجود دارد)
    user_feedback VARCHAR(20), -- helpful, not_helpful, neutral
    tools_actually_used TEXT[], -- ابزارهایی که کاربر استفاده کرد
    trade_result VARCHAR(20), -- profit, loss, no_trade
    
    -- زمان
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    feedback_at TIMESTAMP
);

CREATE INDEX idx_recommendations_request ON tool_recommendations_log(request_id);
CREATE INDEX idx_recommendations_symbol ON tool_recommendations_log(symbol);
CREATE INDEX idx_recommendations_regime ON tool_recommendations_log(market_regime);
CREATE INDEX idx_recommendations_created ON tool_recommendations_log(created_at DESC);

-- =====================================================
-- Views for Common Queries
-- =====================================================

-- View: عملکرد هر ابزار در 30 روز گذشته
CREATE OR REPLACE VIEW tool_performance_last_30_days AS
SELECT 
    tool_name,
    tool_category,
    market_regime,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN success THEN 1 ELSE 0 END) as correct_predictions,
    AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as accuracy,
    AVG(confidence_score) as avg_confidence,
    AVG(actual_price_change) as avg_price_change
FROM tool_performance_history
WHERE prediction_timestamp >= NOW() - INTERVAL '30 days'
GROUP BY tool_name, tool_category, market_regime
ORDER BY accuracy DESC;

-- View: بهترین ابزارها برای هر رژیم بازار
CREATE OR REPLACE VIEW best_tools_by_regime AS
SELECT 
    market_regime,
    tool_name,
    tool_category,
    COUNT(*) as total_predictions,
    AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as accuracy,
    AVG(confidence_score) as avg_confidence
FROM tool_performance_history
WHERE prediction_timestamp >= NOW() - INTERVAL '90 days'
  AND success IS NOT NULL
GROUP BY market_regime, tool_name, tool_category
HAVING COUNT(*) >= 10 -- حداقل 10 پیش‌بینی
ORDER BY market_regime, accuracy DESC;

-- View: وزن‌های ML فعلی
CREATE OR REPLACE VIEW current_ml_weights AS
SELECT DISTINCT ON (model_name, market_regime)
    model_name,
    model_version,
    market_regime,
    weights,
    training_accuracy,
    validation_accuracy,
    training_date
FROM ml_weights_history
ORDER BY model_name, market_regime, training_date DESC;

-- =====================================================
-- Functions
-- =====================================================

-- Function: محاسبه accuracy یک ابزار
CREATE OR REPLACE FUNCTION calculate_tool_accuracy(
    p_tool_name VARCHAR,
    p_market_regime VARCHAR DEFAULT NULL,
    p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    tool_name VARCHAR,
    market_regime VARCHAR,
    total_predictions BIGINT,
    correct_predictions BIGINT,
    accuracy NUMERIC,
    avg_confidence NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        tph.tool_name::VARCHAR,
        tph.market_regime::VARCHAR,
        COUNT(*) as total_predictions,
        SUM(CASE WHEN tph.success THEN 1 ELSE 0 END) as correct_predictions,
        AVG(CASE WHEN tph.success THEN 1.0 ELSE 0.0 END)::NUMERIC as accuracy,
        AVG(tph.confidence_score)::NUMERIC as avg_confidence
    FROM tool_performance_history tph
    WHERE tph.tool_name = p_tool_name
      AND (p_market_regime IS NULL OR tph.market_regime = p_market_regime)
      AND tph.prediction_timestamp >= NOW() - (p_days || ' days')::INTERVAL
      AND tph.success IS NOT NULL
    GROUP BY tph.tool_name, tph.market_regime;
END;
$$ LANGUAGE plpgsql;

-- Function: ثبت عملکرد ابزار
CREATE OR REPLACE FUNCTION record_tool_performance(
    p_tool_name VARCHAR,
    p_tool_category VARCHAR,
    p_symbol VARCHAR,
    p_timeframe VARCHAR,
    p_market_regime VARCHAR,
    p_prediction_type VARCHAR,
    p_confidence DECIMAL,
    p_metadata JSONB DEFAULT NULL
)
RETURNS INTEGER AS $$
DECLARE
    v_id INTEGER;
BEGIN
    INSERT INTO tool_performance_history (
        tool_name,
        tool_category,
        symbol,
        timeframe,
        market_regime,
        prediction_type,
        confidence_score,
        metadata
    ) VALUES (
        p_tool_name,
        p_tool_category,
        p_symbol,
        p_timeframe,
        p_market_regime,
        p_prediction_type,
        p_confidence,
        p_metadata
    )
    RETURNING id INTO v_id;
    
    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- Function: به‌روزرسانی نتیجه
CREATE OR REPLACE FUNCTION update_tool_result(
    p_id INTEGER,
    p_actual_result VARCHAR,
    p_actual_change DECIMAL,
    p_success BOOLEAN
)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE tool_performance_history
    SET 
        actual_result = p_actual_result,
        actual_price_change = p_actual_change,
        success = p_success,
        accuracy = CASE WHEN p_success THEN 1.0 ELSE 0.0 END,
        result_timestamp = NOW(),
        updated_at = NOW()
    WHERE id = p_id;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- Triggers
-- =====================================================

-- Trigger: به‌روزرسانی updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_tool_performance_updated_at
    BEFORE UPDATE ON tool_performance_history
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- Sample Data (for testing)
-- =====================================================

-- نمونه داده برای تست
INSERT INTO tool_performance_history (
    tool_name, tool_category, symbol, timeframe, market_regime,
    volatility_level, trend_strength, volume_profile,
    prediction_type, confidence_score,
    actual_result, actual_price_change, success, accuracy,
    result_timestamp, evaluation_period_hours
) VALUES 
    ('MACD', 'trend_indicators', 'BTCUSDT', '1d', 'trending_bullish', 45.5, 72.3, 'high',
     'bullish', 0.85, 'bullish', 3.2, true, 1.0, NOW() - INTERVAL '1 day', 24),
    
    ('RSI', 'momentum_indicators', 'BTCUSDT', '1h', 'ranging', 32.1, 25.5, 'medium',
     'neutral', 0.72, 'neutral', 0.5, true, 1.0, NOW() - INTERVAL '2 days', 4),
    
    ('ADX', 'trend_indicators', 'ETHUSDT', '4h', 'trending_bearish', 55.2, 68.9, 'high',
     'bearish', 0.89, 'bearish', -4.1, true, 1.0, NOW() - INTERVAL '3 days', 12),
    
    ('Bollinger_Bands', 'volatility_indicators', 'BTCUSDT', '1d', 'volatile', 78.3, 45.2, 'medium',
     'neutral', 0.65, 'bullish', 5.2, false, 0.0, NOW() - INTERVAL '4 days', 24);

-- Sample ML weights
INSERT INTO ml_weights_history (
    model_name, model_version, market_regime,
    weights, training_accuracy, validation_accuracy,
    r2_score, training_samples, training_date
) VALUES (
    'ml_indicator_weights', '1.0.0', 'trending_bullish',
    '{"MACD": 0.28, "ADX": 0.24, "RSI": 0.18, "EMA": 0.15, "Stochastic": 0.10, "CCI": 0.05}'::jsonb,
    0.84, 0.81, 0.78, 1500, NOW() - INTERVAL '7 days'
);

-- =====================================================
-- Comments
-- =====================================================

COMMENT ON TABLE tool_performance_history IS 'ذخیره عملکرد تاریخی ابزارهای تحلیل تکنیکال';
COMMENT ON TABLE tool_performance_stats IS 'آمار تجمیعی عملکرد ابزارها برای سرعت بیشتر';
COMMENT ON TABLE ml_weights_history IS 'وزن‌های یادگرفته شده ML در طول زمان';
COMMENT ON TABLE tool_recommendations_log IS 'لاگ پیشنهادات داده شده به کاربران';

COMMENT ON COLUMN tool_performance_history.market_regime IS 'رژیم بازار: trending_bullish, trending_bearish, ranging, volatile';
COMMENT ON COLUMN tool_performance_history.success IS 'آیا پیش‌بینی ابزار درست بود؟';
COMMENT ON COLUMN tool_performance_history.metadata IS 'اطلاعات اضافی به صورت JSON';

-- =====================================================
-- Grants (adjust based on your security needs)
-- =====================================================

-- GRANT SELECT, INSERT, UPDATE ON tool_performance_history TO gravity_app;
-- GRANT SELECT ON tool_performance_last_30_days TO gravity_app;
-- GRANT EXECUTE ON FUNCTION calculate_tool_accuracy TO gravity_app;

-- =====================================================
-- END OF SCHEMA
-- =====================================================
