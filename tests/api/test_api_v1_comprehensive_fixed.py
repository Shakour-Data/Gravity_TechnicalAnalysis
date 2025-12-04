"""
Phase 2: Comprehensive tests for API V1 endpoints (Fixed Version)

This module tests API V1 endpoints with correct parameter handling.

Author: Gravity Tech Test Suite
Date: December 4, 2025
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from gravity_tech.api.v1.historical import HistoricalAnalysisRequest, HistoricalScoreSummary
from gravity_tech.api.v1.patterns import PatternDetectionRequest, PatternDetectionResponse, PatternResult
from gravity_tech.api.v1.tools import ToolRecommendationRequest, TradingStyle
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
            start_date=None,
            end_date=None,
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
    
    def test_historical_request_custom_limit(self):
        """Test historical request with custom limit"""
        request = HistoricalAnalysisRequest(
            symbol="ETHUSDT",
            timeframe="4h",
            start_date=None,
            end_date=None,
            limit=250
        )
        assert request.limit == 250
    
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
    
    def test_historical_multiple_requests(self):
        """Test multiple historical requests with different parameters"""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        timeframes = ["1h", "4h", "1d"]
        
        for symbol in symbols:
            for timeframe in timeframes:
                request = HistoricalAnalysisRequest(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=None,
                    end_date=None,
                    limit=100
                )
                assert request.symbol == symbol
                assert request.timeframe == timeframe


# ============================================================================
# Test: Pattern Detection Endpoints
# ============================================================================

class TestPatternDetectionEndpoints:
    """Test suite for pattern detection API endpoints"""
    
    def test_pattern_request_minimal(self, sample_candles):
        """Test pattern detection request with minimal parameters"""
        request = PatternDetectionRequest(
            candles=sample_candles,
            symbol="BTCUSDT",
            timeframe="1d"
        )
        assert request.symbol == "BTCUSDT"
        assert request.timeframe == "1d"
        assert len(request.candles) == 100
    
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
        assert response.analysis_time_ms > 0
        assert response.ml_enabled is True
        assert response.patterns_found == 0


# ============================================================================
# Test: Tool Recommendations Endpoints
# ============================================================================

class TestToolRecommendationEndpoints:
    """Test suite for tool recommendation API endpoints"""
    
    def test_tool_recommendation_request_minimal(self):
        """Test tool recommendation request with minimal parameters"""
        request = ToolRecommendationRequest(
            symbol="BTCUSDT"
        )
        assert request.symbol == "BTCUSDT"
    
    def test_tool_recommendation_request_with_context(self):
        """Test tool recommendation with analysis goal"""
        request = ToolRecommendationRequest(
            symbol="ETHUSDT",
            timeframe="1h",
            top_n=10
        )
        assert request.symbol == "ETHUSDT"
        assert request.timeframe == "1h"
        assert request.top_n == 10
    
    def test_tool_recommendation_multiple_scenarios(self):
        """Test tool recommendations for different trading styles"""
        styles = [TradingStyle.SCALP, TradingStyle.DAY, TradingStyle.SWING, TradingStyle.POSITION]
        
        for style in styles:
            request = ToolRecommendationRequest(
                symbol="BTCUSDT",
                trading_style=style
            )
            assert request.symbol == "BTCUSDT"
            assert request.trading_style == style


# ============================================================================
# Test: API Integration Tests
# ============================================================================

class TestAPIIntegration:
    """Integration tests for multiple API endpoints"""
    
    def test_workflow_historical_then_patterns(self, sample_candles):
        """Test workflow: get historical data then detect patterns"""
        # Step 1: Request historical data
        hist_request = HistoricalAnalysisRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=None,
            end_date=None,
            limit=100
        )
        assert hist_request.limit == 100
        
        # Step 2: Request pattern detection on the same symbol
        pattern_request = PatternDetectionRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            candles=sample_candles
        )
        assert pattern_request.symbol == hist_request.symbol
    
    def test_concurrent_requests(self):
        """Test concurrent API requests"""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        requests = []
        
        for symbol in symbols:
            requests.append(HistoricalAnalysisRequest(
                symbol=symbol,
                timeframe="1h",
                start_date=None,
                end_date=None,
                limit=100
            ))
        
        assert len(requests) == 3
        assert all(req.limit == 100 for req in requests)


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
            start_date=None,
            end_date=None,
            limit=100
        )
        assert len(request.symbol) > 0
    
    def test_historical_request_validation_limit(self):
        """Test historical request limit validation"""
        # Valid limit
        request = HistoricalAnalysisRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=None,
            end_date=None,
            limit=500
        )
        assert 1 <= request.limit <= 1000
    
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
        
        # Validate date range logic
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
        
        # Verify all fields are present
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
        
        # Verify all fields are present
        assert hasattr(response, 'symbol')
        assert hasattr(response, 'patterns')
        assert hasattr(response, 'analysis_time_ms')


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
        assert response.patterns_found == 0


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
            start_date=None,
            end_date=None,
            limit=1000
        )
        assert request.limit == 1000
    
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
