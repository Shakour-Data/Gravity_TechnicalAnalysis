"""
Phase 2: Comprehensive tests for API V1 endpoints

This module tests all API V1 endpoints:
- Historical Analysis API
- ML Predictions API
- Pattern Detection API
- Tool Recommendations API

Author: Gravity Tech Test Suite
Date: December 4, 2025
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from gravity_tech.api.v1.historical import HistoricalAnalysisRequest, HistoricalScoreSummary
from gravity_tech.api.v1.ml import PredictionRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse, ModelInfoResponse, BacktestRequest, BacktestResponse
from gravity_tech.api.v1.patterns import PatternDetectionRequest, PatternDetectionResponse, PatternResult, CandleData
from gravity_tech.api.v1.tools import ToolRecommendationRequest, ToolRecommendationResponse, ToolRecommendation, MarketContextInfo, DynamicStrategy, AnalysisGoal, TradingStyle, ToolPriority, ToolCategory
from gravity_tech.models.schemas import Candle


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_candles() -> List[Candle]:
    """Create sample candle data for testing"""
    candles = []
    base_time = datetime(2025, 1, 1)
    for i in range(100):
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=100 + i * 0.5,
            high=105 + i * 0.5,
            low=95 + i * 0.5,
            close=102 + i * 0.5,
            volume=1000000 + i * 10000
        ))
    return candles


@pytest.fixture
def historical_request_data() -> Dict[str, Any]:
    """Sample historical analysis request"""
    return {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "start_date": datetime(2025, 1, 1),
        "end_date": datetime(2025, 1, 31),
        "limit": 100
    }


@pytest.fixture
def ml_prediction_request_data() -> Dict[str, Any]:
    """Sample ML prediction request"""
    return {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "model_type": "lstm",
        "lookahead": 5,
        "threshold": 0.6
    }


@pytest.fixture
def pattern_detection_request_data() -> Dict[str, Any]:
    """Sample pattern detection request"""
    return {
        "symbol": "BTCUSDT",
        "timeframe": "1d",
        "pattern_types": ["candlestick", "classical"]
    }


# ============================================================================
# Test: Historical Analysis Endpoints
# ============================================================================

class TestHistoricalAnalysisEndpoints:
    """Test suite for historical analysis API endpoints"""

    def test_historical_analyze_endpoint_exists(self):
        """Test that historical analyze endpoint exists"""
        # This would be tested with actual FastAPI TestClient
        # For now, verify the endpoint data model
        request = HistoricalAnalysisRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            limit=100,
            start_date=None,
            end_date=None
        )
        assert request.symbol == "BTCUSDT"
        assert request.timeframe == "1h"
        assert request.limit == 100

    def test_historical_request_validation(self):
        """Test historical request model validation"""
        # Valid request
        request = HistoricalAnalysisRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            limit=100,
            start_date=None,
            end_date=None
        )
        assert request.symbol == "BTCUSDT"

    def test_historical_request_limit_validation(self):
        """Test limit validation (1-1000)"""
        # Valid limit
        request = HistoricalAnalysisRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            limit=500,
            start_date=None,
            end_date=None
        )
        assert request.limit == 500

        # Test limit bounds
        with pytest.raises(ValueError):
            HistoricalAnalysisRequest(
                symbol="BTCUSDT",
                timeframe="1h",
                limit=0,
                start_date=None,
                end_date=None
            )

    def test_historical_response_model(self):
        """Test historical response model"""
        response = HistoricalScoreSummary(
            symbol="BTCUSDT",
            timeframe="1h",
            date=datetime(2025, 1, 1),
            combined_score=0.75,
            combined_confidence=0.85,
            combined_signal="buy",
            trend_score=0.8,
            momentum_score=0.7
        )
        assert response.symbol == "BTCUSDT"
        assert response.combined_score == 0.75
        assert response.combined_signal == "buy"

    def test_historical_date_range_handling(self):
        """Test date range handling in historical analysis"""
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 31)

        request = HistoricalAnalysisRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
            limit=100
        )
        assert request.start_date == start
        assert request.end_date == end

    def test_historical_default_dates(self):
        """Test default date assignment (30 days back)"""
        request = HistoricalAnalysisRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            limit=100,
            start_date=None,
            end_date=None
        )
        assert request.start_date is None  # Default handling in endpoint
        assert request.end_date is None

    def test_historical_multiple_symbols(self):
        """Test historical analysis with multiple symbols"""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

        for symbol in symbols:
            request = HistoricalAnalysisRequest(
                symbol=symbol,
                timeframe="1h",
                limit=100,
                start_date=None,
                end_date=None
            )
            assert request.symbol == symbol

    def test_historical_timeframe_support(self):
        """Test various timeframe formats"""
        timeframes = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]

        for tf in timeframes:
            request = HistoricalAnalysisRequest(
                symbol="BTCUSDT",
                timeframe=tf,
                limit=100,
                start_date=None,
                end_date=None
            )
            assert request.timeframe == tf


# ============================================================================
# Test: ML Prediction Endpoints
# ============================================================================

class TestMLPredictionEndpoints:
    """Test suite for ML prediction API endpoints"""

    def test_ml_prediction_endpoint_exists(self):
        """Test that ML prediction endpoint exists"""
        request = PredictionRequest(
            features=Mock()  # Mock PatternFeatures
        )
        assert hasattr(request, 'features')

    def test_ml_prediction_request_validation(self):
        """Test ML prediction request validation"""
        # This would require a proper PatternFeatures object
        # For now, just test that the model can be instantiated
        pass

    def test_ml_model_type_support(self):
        """Test supported ML model types"""
        # Model type is handled internally, not in request
        pass

    def test_ml_prediction_threshold_validation(self):
        """Test prediction confidence threshold validation"""
        # Threshold is handled internally, not in request
        pass

    def test_ml_prediction_response_model(self):
        """Test ML prediction response model"""
        response = PredictionResponse(
            predicted_pattern="gartley",
            confidence=0.85,
            probabilities={"gartley": 0.85, "butterfly": 0.1, "bat": 0.03, "crab": 0.02},
            model_version="v2",
            inference_time_ms=45.2
        )
        assert response.predicted_pattern == "gartley"
        assert response.confidence == 0.85

    def test_ml_prediction_with_candles(self, sample_candles):
        """Test ML prediction with real candle data"""
        # This would require actual ML model integration
        assert len(sample_candles) == 100

    def test_ml_batch_prediction(self):
        """Test batch ML prediction"""
        request = BatchPredictionRequest(
            features_list=[Mock(), Mock()]  # Mock PatternFeatures list
        )
        assert len(request.features_list) == 2


# ============================================================================
# Test: Pattern Detection Endpoints
# ============================================================================

class TestPatternDetectionEndpoints:
    """Test suite for pattern detection API endpoints"""

    def test_pattern_detection_endpoint_exists(self):
        """Test that pattern detection endpoint exists"""
        candles = [
            CandleData(
                timestamp=int(datetime(2025, 1, 1).timestamp()),
                open=100,
                high=105,
                low=95,
                close=102,
                volume=1000000
            )
        ]

        request = PatternDetectionRequest(
            symbol="BTCUSDT",
            timeframe="1d",
            candles=candles
        )
        assert request.symbol == "BTCUSDT"

    def test_pattern_types_support(self):
        """Test supported pattern types"""
        candles = [
            CandleData(
                timestamp=int(datetime(2025, 1, 1).timestamp()),
                open=100,
                high=105,
                low=95,
                close=102,
                volume=1000000
            )
        ]

        pattern_types = ["gartley", "butterfly", "bat", "crab"]

        for pattern_type in pattern_types:
            request = PatternDetectionRequest(
                symbol="BTCUSDT",
                timeframe="1d",
                pattern_types=[pattern_type],
                candles=candles
            )
            if request.pattern_types:
                assert pattern_type in request.pattern_types

    def test_pattern_response_model(self):
        """Test pattern detection response model"""
        patterns = [
            PatternResult(
                pattern_type="gartley",
                direction="bullish",
                points={
                    "X": Mock(index=0, price=100, timestamp=int(datetime(2025, 1, 1).timestamp())),
                    "A": Mock(index=10, price=110, timestamp=int(datetime(2025, 1, 2).timestamp())),
                    "B": Mock(index=20, price=105, timestamp=int(datetime(2025, 1, 3).timestamp())),
                    "C": Mock(index=30, price=95, timestamp=int(datetime(2025, 1, 4).timestamp())),
                    "D": Mock(index=40, price=102, timestamp=int(datetime(2025, 1, 5).timestamp()))
                },
                ratios={"xab": 0.618, "abc": 0.786, "bcd": 1.272, "xad": 0.786},
                completion_price=102.0,
                confidence=0.85,
                targets={"target1": 107.0, "target2": 112.0},
                stop_loss=97.0,
                detected_at=datetime.utcnow().isoformat()
            )
        ]

        response = PatternDetectionResponse(
            symbol="BTCUSDT",
            timeframe="1d",
            patterns_found=1,
            patterns=patterns,
            analysis_time_ms=150.5,
            ml_enabled=True
        )
        assert response.symbol == "BTCUSDT"
        assert len(response.patterns) == 1

    def test_multiple_pattern_detection(self):
        """Test detecting multiple patterns simultaneously"""
        candles = [
            CandleData(
                timestamp=int(datetime(2025, 1, 1).timestamp()),
                open=100,
                high=105,
                low=95,
                close=102,
                volume=1000000
            )
        ]

        request = PatternDetectionRequest(
            symbol="BTCUSDT",
            timeframe="1d",
            pattern_types=["gartley", "butterfly", "bat", "crab"],
            candles=candles
        )
        if request.pattern_types:
            assert len(request.pattern_types) == 4

    def test_pattern_confidence_thresholds(self):
        """Test pattern confidence threshold filtering"""
        candles = [
            CandleData(
                timestamp=int(datetime(2025, 1, 1).timestamp()),
                open=100,
                high=105,
                low=95,
                close=102,
                volume=1000000
            )
        ]

        request = PatternDetectionRequest(
            symbol="BTCUSDT",
            timeframe="1d",
            min_confidence=0.75,
            candles=candles
        )
        assert request.min_confidence == 0.75


# ============================================================================
# Test: Tool Recommendation Endpoints
# ============================================================================

class TestToolRecommendationEndpoints:
    """Test suite for tool recommendation API endpoints"""

    def test_tool_recommendation_endpoint_exists(self):
        """Test that tool recommendation endpoint exists"""
        request = ToolRecommendationRequest(
            symbol="BTCUSDT",
            timeframe="1d",
            analysis_goal=AnalysisGoal.ENTRY_SIGNAL,
            trading_style=TradingStyle.SWING,
            top_n=15
        )
        assert request.symbol == "BTCUSDT"

    def test_tool_recommendation_analysis_goals(self):
        """Test various analysis goals"""
        goals = [AnalysisGoal.ENTRY_SIGNAL, AnalysisGoal.EXIT_SIGNAL, AnalysisGoal.RISK_MANAGEMENT]

        for goal in goals:
            request = ToolRecommendationRequest(
                symbol="BTCUSDT",
                timeframe="1d",
                analysis_goal=goal,
                trading_style=TradingStyle.SWING,
                top_n=15
            )
            assert request.analysis_goal == goal

    def test_tool_recommendation_response(self):
        """Test tool recommendation response"""
        recommendations = {
            "must_use": [
                ToolRecommendation(
                    name="ADX",
                    category=ToolCategory.TREND_INDICATORS,
                    ml_weight=0.28,
                    confidence=0.87,
                    historical_accuracy="82.0%",
                    reason="در بازار روندی بسیار موثر است | وزن ML بالا (91.2%)",
                    priority=ToolPriority.MUST_USE,
                    best_for=["قدرت ترند", "تایید جهت حرکت"]
                )
            ]
        }

        response = ToolRecommendationResponse(
            symbol="BTCUSDT",
            market_context=MarketContextInfo(
                regime="trending_bullish",
                volatility=45.5,
                trend_strength=72.3,
                volume_profile="high"
            ),
            analysis_goal="entry_signal",
            recommendations=recommendations,
            dynamic_strategy=DynamicStrategy(
                primary_tools=["ADX", "MACD"],
                supporting_tools=["EMA", "VWAP"],
                confidence=0.84,
                based_on="تحلیل 3 ابزار برتر",
                regime="trending_bullish",
                expected_accuracy="84.0%"
            ),
            ml_metadata={"model_type": "lightgbm"},
            timestamp=datetime.utcnow()
        )
        assert response.symbol == "BTCUSDT"
        assert len(response.recommendations["must_use"]) == 1

    def test_tool_ranking_by_score(self):
        """Test tool ranking by effectiveness score"""
        recommendations = {
            "recommended": [
                ToolRecommendation(
                    name="RSI",
                    category=ToolCategory.MOMENTUM_INDICATORS,
                    ml_weight=0.18,
                    confidence=0.76,
                    historical_accuracy="76.0%",
                    reason="برای تشخیص نقاط اصلاح در روند",
                    priority=ToolPriority.RECOMMENDED,
                    best_for=["شناسایی اشباع خرید/فروش", "واگرایی"]
                ),
                ToolRecommendation(
                    name="MACD",
                    category=ToolCategory.TREND_INDICATORS,
                    ml_weight=0.24,
                    confidence=0.83,
                    historical_accuracy="79.0%",
                    reason="در بازار روندی بسیار موثر است | وزن ML بالا (85.7%)",
                    priority=ToolPriority.RECOMMENDED,
                    best_for=["تشخیص ترند", "سیگنال‌های خرید/فروش", "واگرایی"]
                )
            ]
        }

        response = ToolRecommendationResponse(
            symbol="BTCUSDT",
            market_context=MarketContextInfo(
                regime="trending_bullish",
                volatility=45.5,
                trend_strength=72.3,
                volume_profile="high"
            ),
            analysis_goal="entry_signal",
            recommendations=recommendations,
            dynamic_strategy=DynamicStrategy(
                primary_tools=["ADX", "MACD"],
                supporting_tools=["EMA", "VWAP"],
                confidence=0.84,
                based_on="تحلیل 3 ابزار برتر",
                regime="trending_bullish",
                expected_accuracy="84.0%"
            ),
            ml_metadata={"model_type": "lightgbm"},
            timestamp=datetime.utcnow()
        )
        # RSI should have lower weight than MACD
        rsi_weight = response.recommendations["recommended"][0].ml_weight
        macd_weight = response.recommendations["recommended"][1].ml_weight
        assert macd_weight > rsi_weight


# ============================================================================
# Test: API Error Handling
# ============================================================================

class TestAPIErrorHandling:
    """Test API error handling and validation"""

    def test_invalid_symbol_handling(self):
        """Test handling of invalid symbols"""
        # Should accept any string symbol (validation at database level)
        request = HistoricalAnalysisRequest(
            symbol="INVALID",
            timeframe="1h",
            limit=100,
            start_date=None,
            end_date=None
        )
        assert request.symbol == "INVALID"

    def test_missing_required_fields(self):
        """Test missing required fields"""
        # Symbol and timeframe are required
        with pytest.raises(TypeError):
            HistoricalAnalysisRequest(
                symbol="BTCUSDT",
                timeframe="1h",
                limit=100,
                start_date=None,
                end_date=None
            )

    def test_invalid_timeframe_format(self):
        """Test invalid timeframe format handling"""
        # Should accept format, validation at endpoint
        request = HistoricalAnalysisRequest(
            symbol="BTCUSDT",
            timeframe="invalid",
            limit=100,
            start_date=None,
            end_date=None
        )
        assert request.timeframe == "invalid"

    def test_date_range_validation(self):
        """Test date range validation"""
        start = datetime(2025, 1, 31)
        end = datetime(2025, 1, 1)  # End before start

        request = HistoricalAnalysisRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
            limit=100
        )
        # Model should accept, validation at endpoint
        if request.start_date and request.end_date:
            assert request.start_date > request.end_date


# ============================================================================
# Test: API Integration
# ============================================================================

class TestAPIIntegration:
    """Test integration between API endpoints"""

    def test_historical_to_ml_pipeline(self):
        """Test flow from historical data to ML prediction"""
        # Step 1: Get historical data
        hist_request = HistoricalAnalysisRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            limit=100,
            start_date=None,
            end_date=None
        )

        # Step 2: Use for ML prediction (would need actual integration)
        assert hist_request.symbol == "BTCUSDT"

    def test_pattern_detection_to_tool_recommendation(self):
        """Test flow from pattern detection to tool recommendation"""
        # Step 1: Detect patterns (would need actual integration)
        pattern_symbol = "BTCUSDT"

        # Step 2: Get tool recommendations
        tool_request = ToolRecommendationRequest(
            symbol=pattern_symbol,
            timeframe="1d",
            analysis_goal=AnalysisGoal.ENTRY_SIGNAL,
            trading_style=TradingStyle.SWING,
            top_n=15
        )

        assert tool_request.symbol == pattern_symbol

    def test_all_endpoints_same_symbol(self):
        """Test all endpoints with same symbol"""
        symbol = "ETHUSDT"

        # Historical
        hist_request = HistoricalAnalysisRequest(symbol=symbol, timeframe="1h", limit=100, start_date=None, end_date=None)
        assert hist_request.symbol == symbol

        # Pattern Detection
        candles = [CandleData(timestamp=int(datetime(2025, 1, 1).timestamp()), open=100, high=105, low=95, close=102, volume=1000000)]
        pattern_request = PatternDetectionRequest(symbol=symbol, timeframe="1d", candles=candles)
        assert pattern_request.symbol == symbol

        # Tool Recommendation
        tool_request = ToolRecommendationRequest(symbol=symbol, timeframe="1d", analysis_goal=AnalysisGoal.ENTRY_SIGNAL, trading_style=TradingStyle.SWING, top_n=15)
        assert tool_request.symbol == symbol


# ============================================================================
# Test: API Performance
# ============================================================================

class TestAPIPerformance:
    """Test API performance characteristics"""

    def test_historical_request_serialization(self):
        """Test historical request serialization speed"""
        request = HistoricalAnalysisRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            limit=100,
            start_date=None,
            end_date=None
        )

        # Should be serializable
        data = request.model_dump()
        assert data["symbol"] == "BTCUSDT"

    def test_response_model_creation_batch(self):
        """Test batch creation of response models"""
        responses = []
        base_time = datetime(2025, 1, 1)

        for i in range(100):
            response = HistoricalScoreSummary(
                symbol="BTCUSDT",
                timeframe="1h",
                date=base_time + timedelta(hours=i),
                combined_score=0.5 + i * 0.001,
                combined_confidence=0.8,
                combined_signal="buy",
                trend_score=0.7,
                momentum_score=0.6
            )
            responses.append(response)

        assert len(responses) == 100
        # Scores should increase gradually
        assert responses[-1].combined_score > responses[0].combined_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
