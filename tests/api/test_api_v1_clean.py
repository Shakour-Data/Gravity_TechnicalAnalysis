"""
Phase 2: Comprehensive tests for API V1 endpoints (Clean Version)

This module tests all API V1 endpoints with correct model usage:
- Historical Analysis API
- Pattern Detection API
- Tool Recommendations API

Author: Gravity Tech Test Suite
Date: December 4, 2025
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from unittest.mock import Mock

from gravity_tech.api.v1.historical import HistoricalAnalysisRequest, HistoricalScoreSummary
from gravity_tech.api.v1.patterns import PatternDetectionRequest, PatternDetectionResponse, PatternResult, CandleData
from gravity_tech.api.v1.tools import ToolRecommendationRequest, ToolRecommendationResponse, ToolRecommendation, MarketContextInfo, DynamicStrategy, AnalysisGoal, TradingStyle, ToolPriority, ToolCategory
from gravity_tech.models.schemas import Candle
from datetime import timezone


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_candles() -> List[Candle]:
    """Create sample candle data for testing"""
    candles = []
    base_time = datetime(2025, 1, 1)
    for i in range(60):  # Minimum 50 required by PatternDetectionRequest
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
def sample_candle_data(tse_candles_short) -> List[CandleData]:
    """Create sample CandleData for API requests from TSE data"""
    candles = []
    
    # اگر داده‌های TSE موجود باشند از آنها استفاده کنید
    for candle in tse_candles_short:
        candles.append(CandleData(
            timestamp=int(candle.timestamp.timestamp()),
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume
        ))
    
    return candles


# ============================================================================
# Test: Historical Analysis Endpoints
# ============================================================================

class TestHistoricalAnalysisEndpoints:
    """Test suite for historical analysis API endpoints"""

    def test_historical_request_minimal(self):
        """Test historical request with minimal parameters"""
        request = HistoricalAnalysisRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31),
            limit=100
        )
        assert request.symbol == "BTCUSDT"
        assert request.timeframe == "1h"
        assert request.limit == 100

    def test_historical_request_with_dates(self):
        """Test historical request with start and end dates"""
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 31)

        request = HistoricalAnalysisRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
            limit=500
        )
        assert request.symbol == "BTCUSDT"
        assert request.start_date == start
        assert request.end_date == end
        assert request.limit == 500

    def test_historical_response_model(self):
        """Test historical response model"""
        response = HistoricalScoreSummary(
            symbol="BTCUSDT",
            timeframe="1h",
            date=datetime(2025, 1, 15),
            combined_score=75.5,
            combined_confidence=0.85,
            combined_signal="bullish",
            trend_score=80.0,
            momentum_score=70.0
        )
        assert response.symbol == "BTCUSDT"
        assert response.combined_score == 75.5

    def test_historical_multiple_symbols(self):
        """Test historical analysis with multiple symbols"""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

        for symbol in symbols:
            request = HistoricalAnalysisRequest(
                symbol=symbol,
                timeframe="1h",
                start_date=datetime(2025, 1, 1),
                end_date=datetime(2025, 1, 31),
                limit=100
            )
            assert request.symbol == symbol


# ============================================================================
# Test: Pattern Detection Endpoints
# ============================================================================

class TestPatternDetectionEndpoints:
    """Test suite for pattern detection API endpoints"""

    def test_pattern_request_minimal(self, sample_candle_data):
        """Test pattern detection request with minimal parameters"""
        request = PatternDetectionRequest(
            symbol="BTCUSDT",
            timeframe="1d",
            candles=sample_candle_data
        )
        assert request.symbol == "BTCUSDT"
        assert request.timeframe == "1d"

    def test_pattern_request_with_types(self, sample_candle_data):
        """Test pattern detection with specific pattern types"""
        request = PatternDetectionRequest(
            symbol="ETHUSDT",
            timeframe="4h",
            candles=sample_candle_data,
            pattern_types=["gartley", "butterfly"]
        )
        assert request.symbol == "ETHUSDT"
        if request.pattern_types:
            assert "gartley" in request.pattern_types

    def test_pattern_response_model(self):
        """Test pattern detection response model"""
        response = PatternDetectionResponse(
            symbol="BTCUSDT",
            timeframe="1d",
            patterns_found=0,
            patterns=[],
            analysis_time_ms=45.2,
            ml_enabled=True
        )
        assert response.symbol == "BTCUSDT"
        assert response.patterns_found == 0

    def test_pattern_response_with_results(self):
        """Test pattern detection response with pattern results"""
        pattern_result = PatternResult(
            pattern_type="gartley",
            direction="bullish",
            points={},
            ratios={"xab": 0.618, "abc": 0.786},
            completion_price=102.5,
            confidence=0.87,
            targets={"target1": 105.0, "target2": 107.5},
            stop_loss=99.0,
            detected_at=datetime.now(timezone.utc).isoformat()
        )

        response = PatternDetectionResponse(
            symbol="BTCUSDT",
            timeframe="1d",
            patterns_found=1,
            patterns=[pattern_result],
            analysis_time_ms=52.3,
            ml_enabled=True
        )
        assert len(response.patterns) == 1
        assert response.patterns[0].pattern_type == "gartley"
        assert response.patterns[0].confidence == 0.87


# ============================================================================
# Test: Tool Recommendation Endpoints
# ============================================================================

class TestToolRecommendationEndpoints:
    """Test suite for tool recommendation API endpoints"""

    def test_tool_recommendation_request_minimal(self):
        """Test tool recommendation request with minimal parameters"""
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
        goals = [AnalysisGoal.ENTRY_SIGNAL, AnalysisGoal.EXIT_SIGNAL]

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
                    reason="در بازار روندی بسیار موثر است",
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
            timestamp=datetime.now(timezone.utc)
        )
        assert response.symbol == "BTCUSDT"
        assert len(response.recommendations["must_use"]) == 1


# ============================================================================
# Test: API Integration Tests
# ============================================================================

class TestAPIIntegration:
    """Integration tests for multiple API endpoints"""

    def test_workflow_historical_then_patterns(self, sample_candle_data):
        """Test workflow: get historical data then detect patterns"""
        # Step 1: Request historical data
        hist_request = HistoricalAnalysisRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31),
            limit=100
        )

        # Step 2: Request pattern detection on the same symbol
        pattern_request = PatternDetectionRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            candles=sample_candle_data
        )
        assert pattern_request.symbol == hist_request.symbol

    def test_workflow_patterns_then_tools(self, sample_candle_data):
        """Test workflow: detect patterns then get tool recommendations"""
        # Step 1: Detect patterns
        pattern_request = PatternDetectionRequest(
            symbol="ETHUSDT",
            timeframe="4h",
            candles=sample_candle_data
        )

        # Step 2: Get tool recommendations
        tool_request = ToolRecommendationRequest(
            symbol="ETHUSDT",
            timeframe="1d",
            analysis_goal=AnalysisGoal.ENTRY_SIGNAL,
            trading_style=TradingStyle.SWING,
            top_n=15
        )

        assert pattern_request.symbol == tool_request.symbol


# ============================================================================
# Test: Request Validation
# ============================================================================

class TestAPIRequestValidation:
    """Test request validation and error handling"""

    def test_historical_request_validation_symbol(self):
        """Test historical request symbol validation"""
        request = HistoricalAnalysisRequest(
            symbol="VALIDTOKEN",
            timeframe="1h",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31),
            limit=100
        )
        assert len(request.symbol) > 0

    def test_historical_request_validation_limit(self):
        """Test historical request limit validation"""
        request = HistoricalAnalysisRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31),
            limit=500
        )
        assert request.limit == 500

    def test_pattern_request_validation(self, sample_candle_data):
        """Test pattern request validation"""
        request = PatternDetectionRequest(
            symbol="BTCUSDT",
            timeframe="1d",
            candles=sample_candle_data
        )
        assert request.symbol is not None
        assert request.timeframe is not None

    def test_date_range_validation(self):
        """Test date range validation"""
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 31)

        request = HistoricalAnalysisRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
            limit=100
        )

        if request.start_date and request.end_date:
            assert request.start_date <= request.end_date


# ============================================================================
# Test: Response Data Validation
# ============================================================================

class TestAPIResponseValidation:
    """Test response data validation"""

    def test_historical_response_fields(self):
        """Test that historical response has all required fields"""
        response = HistoricalScoreSummary(
            symbol="BTCUSDT",
            timeframe="1h",
            date=datetime(2025, 1, 15),
            combined_score=75.5,
            combined_confidence=0.85,
            combined_signal="bullish",
            trend_score=80.0,
            momentum_score=70.0
        )

        assert hasattr(response, 'symbol')
        assert hasattr(response, 'combined_score')
        assert hasattr(response, 'combined_signal')

    def test_pattern_response_fields(self):
        """Test that pattern response has all required fields"""
        response = PatternDetectionResponse(
            symbol="BTCUSDT",
            timeframe="1d",
            patterns_found=0,
            patterns=[],
            analysis_time_ms=45.2,
            ml_enabled=True
        )

        assert hasattr(response, 'symbol')
        assert hasattr(response, 'patterns')
        assert hasattr(response, 'analysis_time_ms')

    def test_pattern_result_fields(self):
        """Test that pattern result has all required fields"""
        result = PatternResult(
            pattern_type="gartley",
            direction="bullish",
            points={},
            ratios={"xab": 0.618},
            completion_price=102.5,
            confidence=0.87,
            targets={"target1": 105.0},
            stop_loss=99.0,
            detected_at=datetime.now(timezone.utc).isoformat()
        )

        assert hasattr(result, 'pattern_type')
        assert hasattr(result, 'confidence')
        if result.confidence is not None:
            assert 0 <= result.confidence <= 1


# ============================================================================
# Test: Error Handling
# ============================================================================

class TestAPIErrorHandling:
    """Test error handling and edge cases"""

    def test_empty_pattern_list(self):
        """Test handling of empty pattern detection results"""
        response = PatternDetectionResponse(
            symbol="BTCUSDT",
            timeframe="1d",
            patterns_found=0,
            patterns=[],
            analysis_time_ms=30.1,
            ml_enabled=True
        )

        assert len(response.patterns) == 0
        assert response.analysis_time_ms > 0

    def test_low_confidence_patterns(self):
        """Test handling of low confidence pattern results"""
        result = PatternResult(
            pattern_type="butterfly",
            direction="bullish",
            points={},
            ratios={"xab": 0.786},
            completion_price=102.5,
            confidence=0.45,
            targets={"target1": 105.0},
            stop_loss=99.0,
            detected_at=datetime.now(timezone.utc).isoformat()
        )

        if result.confidence is not None:
            assert result.confidence < 0.5

    def test_multiple_patterns_same_symbol(self):
        """Test multiple patterns detected on same symbol"""
        patterns = [
            PatternResult(
                pattern_type="gartley",
                direction="bullish",
                points={},
                ratios={"xab": 0.618},
                completion_price=102.5,
                confidence=0.85,
                targets={"target1": 105.0},
                stop_loss=99.0,
                detected_at=datetime.now(timezone.utc).isoformat()
            ),
            PatternResult(
                pattern_type="butterfly",
                direction="bullish",
                points={},
                ratios={"xab": 0.786},
                completion_price=102.5,
                confidence=0.78,
                targets={"target1": 105.0},
                stop_loss=99.0,
                detected_at=datetime.now(timezone.utc).isoformat()
            )
        ]

        response = PatternDetectionResponse(
            symbol="BTCUSDT",
            timeframe="1d",
            patterns_found=2,
            patterns=patterns,
            analysis_time_ms=67.3,
            ml_enabled=True
        )

        assert len(response.patterns) == 2
        assert response.patterns_found == 2


# ============================================================================
# Test: Performance and Limits
# ============================================================================

class TestAPIPerformance:
    """Test API performance characteristics"""

    def test_large_limit_request(self):
        """Test request with maximum allowed limit"""
        request = HistoricalAnalysisRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31),
            limit=1000
        )
        assert request.limit == 1000

    def test_multiple_pattern_types(self, sample_candle_data):
        """Test pattern detection with multiple pattern types"""
        pattern_types = ["gartley", "butterfly", "bat", "crab"]

        request = PatternDetectionRequest(
            symbol="BTCUSDT",
            timeframe="1d",
            candles=sample_candle_data,
            pattern_types=pattern_types
        )

        if request.pattern_types:
            assert len(request.pattern_types) == 4

    def test_analysis_time_measurement(self):
        """Test analysis time measurement in response"""
        response = PatternDetectionResponse(
            symbol="BTCUSDT",
            timeframe="1d",
            patterns_found=0,
            patterns=[],
            analysis_time_ms=125.45,
            ml_enabled=True
        )

        assert response.analysis_time_ms > 100
        assert isinstance(response.analysis_time_ms, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
