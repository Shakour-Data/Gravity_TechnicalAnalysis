"""
Unit tests for response_formatters.py in API module.

Tests cover all functions to achieve >50% coverage.
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest
from gravity_tech.api.response_formatters import (
    format_analysis_summary,
    format_combined_response,
    format_error_response,
    format_horizon_score,
    format_trend_response,
)


class TestResponseFormatters:
    """Test suite for API response formatters."""

    def test_format_horizon_score_basic(self):
        """Test basic horizon score formatting."""
        # Mock horizon score object
        horizon_score = Mock()
        horizon_score.horizon = "3d"
        horizon_score.score = 0.75
        horizon_score.confidence = 0.85

        result = format_horizon_score(horizon_score)

        assert isinstance(result, dict)
        assert result["horizon"] == "3d"
        assert "score" in result
        assert "confidence" in result
        assert "signal" in result
        assert "confidence_quality" in result
        assert result["raw_score"] == 0.75
        assert result["raw_confidence"] == 0.85

    def test_format_horizon_score_persian(self):
        """Test horizon score formatting with Persian labels."""
        horizon_score = Mock()
        horizon_score.horizon = "7d"
        horizon_score.score = -0.3
        horizon_score.confidence = 0.6

        result = format_horizon_score(horizon_score, use_persian=True)

        assert isinstance(result, dict)
        assert result["horizon"] == "7d"
        assert "signal" in result
        assert "confidence_quality" in result

    def test_format_horizon_score_edge_cases(self):
        """Test horizon score formatting with edge case values."""
        test_cases = [
            (1.0, 1.0),  # Max positive
            (-1.0, 0.0),  # Max negative, min confidence
            (0.0, 0.5),  # Neutral
        ]

        for score, confidence in test_cases:
            horizon_score = Mock()
            horizon_score.horizon = "30d"
            horizon_score.score = score
            horizon_score.confidence = confidence

            result = format_horizon_score(horizon_score)
            assert isinstance(result, dict)
            assert result["raw_score"] == score
            assert result["raw_confidence"] == confidence

    def test_format_trend_response_basic(self):
        """Test basic trend response formatting."""
        # Mock analysis result
        analysis_result = Mock()
        analysis_result.score_3d = Mock()
        analysis_result.score_3d.horizon = "3d"
        analysis_result.score_3d.score = 0.6
        analysis_result.score_3d.confidence = 0.8

        analysis_result.score_7d = Mock()
        analysis_result.score_7d.horizon = "7d"
        analysis_result.score_7d.score = 0.4
        analysis_result.score_7d.confidence = 0.7

        analysis_result.score_30d = Mock()
        analysis_result.score_30d.horizon = "30d"
        analysis_result.score_30d.score = 0.2
        analysis_result.score_30d.confidence = 0.6

        result = format_trend_response(analysis_result)

        assert isinstance(result, dict)
        assert "type" in result
        assert result["type"] == "trend_analysis"
        assert "horizons" in result
        assert len(result["horizons"]) == 3
        assert "3d" in result["horizons"]
        assert "7d" in result["horizons"]
        assert "30d" in result["horizons"]

    def test_format_trend_response_persian(self):
        """Test trend response formatting with Persian."""
        analysis_result = Mock()
        analysis_result.score_3d = Mock()
        analysis_result.score_3d.horizon = "3d"
        analysis_result.score_3d.score = 0.5
        analysis_result.score_3d.confidence = 0.75

        analysis_result.score_7d = Mock()
        analysis_result.score_7d.horizon = "7d"
        analysis_result.score_7d.score = 0.3
        analysis_result.score_7d.confidence = 0.65

        analysis_result.score_30d = Mock()
        analysis_result.score_30d.horizon = "30d"
        analysis_result.score_30d.score = 0.1
        analysis_result.score_30d.confidence = 0.55

        result = format_trend_response(analysis_result, use_persian=True)

        assert isinstance(result, dict)
        assert result["type"] == "trend_analysis"

    def test_format_trend_response_with_raw(self):
        """Test trend response formatting including raw data."""
        analysis_result = Mock()
        analysis_result.score_3d = Mock()
        analysis_result.score_3d.horizon = "3d"
        analysis_result.score_3d.score = 0.4
        analysis_result.score_3d.confidence = 0.7

        analysis_result.score_7d = Mock()
        analysis_result.score_7d.horizon = "7d"
        analysis_result.score_7d.score = 0.2
        analysis_result.score_7d.confidence = 0.6

        analysis_result.score_30d = Mock()
        analysis_result.score_30d.horizon = "30d"
        analysis_result.score_30d.score = 0.0
        analysis_result.score_30d.confidence = 0.5

        result = format_trend_response(analysis_result, include_raw=True)

        assert isinstance(result, dict)
        assert "raw_data" in result

    def test_format_combined_response_basic(self):
        """Test basic combined response formatting."""
        # Mock combined analysis
        combined_analysis = Mock()
        combined_analysis.final_action = "BUY"
        combined_analysis.final_confidence = 0.8
        combined_analysis.combined_score_3d = 0.6
        combined_analysis.combined_score_7d = 0.4
        combined_analysis.combined_score_30d = 0.2

        # Mock trend and momentum analyses
        trend_analysis = Mock()
        trend_analysis.score_3d = Mock()
        trend_analysis.score_3d.score = 0.5
        trend_analysis.score_3d.confidence = 0.75

        momentum_analysis = Mock()
        momentum_analysis.momentum_3d = Mock()
        momentum_analysis.momentum_3d.score = 0.7
        momentum_analysis.momentum_3d.confidence = 0.8

        result = format_combined_response(combined_analysis, trend_analysis, momentum_analysis)

        assert isinstance(result, dict)
        assert "type" in result
        assert result["type"] == "combined_analysis"
        assert "recommendation" in result
        assert result["recommendation"]["action"] == "BUY"
        assert "trend_analysis" in result
        assert "momentum_analysis" in result

    def test_format_combined_response_persian(self):
        """Test combined response formatting with Persian."""
        combined_analysis = Mock()
        combined_analysis.final_action = "SELL"
        combined_analysis.final_confidence = 0.7
        combined_analysis.combined_score_3d = -0.3
        combined_analysis.combined_score_7d = -0.1
        combined_analysis.combined_score_30d = 0.0

        trend_analysis = Mock()
        trend_analysis.score_3d = Mock()
        trend_analysis.score_3d.score = -0.2
        trend_analysis.score_3d.confidence = 0.6

        momentum_analysis = Mock()
        momentum_analysis.momentum_3d = Mock()
        momentum_analysis.momentum_3d.score = -0.4
        momentum_analysis.momentum_3d.confidence = 0.7

        result = format_combined_response(combined_analysis, trend_analysis, momentum_analysis, use_persian=True)

        assert isinstance(result, dict)
        assert result["recommendation"]["action"] == "SELL"

    def test_format_analysis_summary_basic(self):
        """Test basic analysis summary formatting."""
        # Mock summary data
        summary = {
            "total_candles": 1000,
            "analysis_time": 2.5,
            "success_rate": 0.95,
            "error_count": 5
        }

        result = format_analysis_summary(summary)

        assert isinstance(result, dict)
        assert "type" in result
        assert result["type"] == "analysis_summary"
        assert "metrics" in result
        assert result["metrics"]["total_candles"] == 1000
        assert result["metrics"]["analysis_time"] == 2.5

    def test_format_analysis_summary_empty(self):
        """Test analysis summary formatting with empty data."""
        summary = {}

        result = format_analysis_summary(summary)

        assert isinstance(result, dict)
        assert result["type"] == "analysis_summary"
        assert "metrics" in result

    def test_format_error_response_basic(self):
        """Test basic error response formatting."""
        error_message = "Invalid input data"
        error_code = "VALIDATION_ERROR"

        result = format_error_response(error_message, error_code)

        assert isinstance(result, dict)
        assert "type" in result
        assert result["type"] == "error"
        assert result["error"]["message"] == error_message
        assert result["error"]["code"] == error_code
        assert "timestamp" in result

    def test_format_error_response_with_details(self):
        """Test error response formatting with additional details."""
        error_message = "Database connection failed"
        error_code = "DB_ERROR"
        details = {"host": "localhost", "port": 5432}

        result = format_error_response(error_message, error_code, details)

        assert isinstance(result, dict)
        assert result["error"]["details"] == details

    def test_format_error_response_default_code(self):
        """Test error response formatting with default error code."""
        error_message = "Unknown error occurred"

        result = format_error_response(error_message)

        assert isinstance(result, dict)
        assert result["error"]["code"] == "INTERNAL_ERROR"

    # Additional tests for edge cases
    def test_format_horizon_score_none_values(self):
        """Test horizon score formatting with None values."""
        horizon_score = Mock()
        horizon_score.horizon = "3d"
        horizon_score.score = None
        horizon_score.confidence = None

        result = format_horizon_score(horizon_score)

        assert isinstance(result, dict)
        assert result["horizon"] == "3d"

    def test_format_trend_response_missing_attributes(self):
        """Test trend response formatting with missing attributes."""
        analysis_result = Mock()
        # Missing some attributes
        analysis_result.score_3d = Mock()
        analysis_result.score_3d.horizon = "3d"
        analysis_result.score_3d.score = 0.5
        analysis_result.score_3d.confidence = 0.8

        # Missing score_7d and score_30d
        with pytest.raises(AttributeError):
            format_trend_response(analysis_result)

    def test_format_combined_response_none_analyses(self):
        """Test combined response formatting with None analyses."""
        combined_analysis = Mock()
        combined_analysis.final_action = "HOLD"
        combined_analysis.final_confidence = 0.5
        combined_analysis.combined_score_3d = 0.0
        combined_analysis.combined_score_7d = 0.0
        combined_analysis.combined_score_30d = 0.0

        result = format_combined_response(combined_analysis, None, None)

        assert isinstance(result, dict)
        assert result["recommendation"]["action"] == "HOLD"

    def test_format_analysis_summary_large_numbers(self):
        """Test analysis summary formatting with large numbers."""
        summary = {
            "total_candles": 1000000,
            "analysis_time": 999.99,
            "success_rate": 1.0,
            "error_count": 0
        }

        result = format_analysis_summary(summary)

        assert isinstance(result, dict)
        assert result["metrics"]["total_candles"] == 1000000

    def test_format_error_response_empty_message(self):
        """Test error response formatting with empty message."""
        result = format_error_response("")

        assert isinstance(result, dict)
        assert result["error"]["message"] == ""

    # Tests for different data types
    def test_format_horizon_score_decimal_values(self):
        """Test horizon score formatting with Decimal values."""
        horizon_score = Mock()
        horizon_score.horizon = "7d"
        horizon_score.score = Decimal('0.618')
        horizon_score.confidence = Decimal('0.85')

        result = format_horizon_score(horizon_score)

        assert isinstance(result, dict)
        assert result["raw_score"] == Decimal('0.618')
        assert result["raw_confidence"] == Decimal('0.85')

    def test_format_trend_response_list_horizons(self):
        """Test trend response formatting with list of horizons."""
        analysis_result = Mock()
        analysis_result.score_3d = Mock()
        analysis_result.score_3d.horizon = "3d"
        analysis_result.score_3d.score = 0.5
        analysis_result.score_3d.confidence = 0.8

        analysis_result.score_7d = Mock()
        analysis_result.score_7d.horizon = "7d"
        analysis_result.score_7d.score = 0.3
        analysis_result.score_7d.confidence = 0.7

        analysis_result.score_30d = Mock()
        analysis_result.score_30d.horizon = "30d"
        analysis_result.score_30d.score = 0.1
        analysis_result.score_30d.confidence = 0.6

        result = format_trend_response(analysis_result)

        horizons = result["horizons"]
        assert isinstance(horizons, dict)
        assert len(horizons) == 3

    def test_format_combined_response_different_actions(self):
        """Test combined response formatting with different actions."""
        actions = ["BUY", "SELL", "HOLD", "TAKE_PROFIT"]

        for action in actions:
            combined_analysis = Mock()
            combined_analysis.final_action = action
            combined_analysis.final_confidence = 0.8
            combined_analysis.combined_score_3d = 0.5
            combined_analysis.combined_score_7d = 0.3
            combined_analysis.combined_score_30d = 0.1

            trend_analysis = Mock()
            momentum_analysis = Mock()

            result = format_combined_response(combined_analysis, trend_analysis, momentum_analysis)

            assert result["recommendation"]["action"] == action

    def test_format_analysis_summary_zero_values(self):
        """Test analysis summary formatting with zero values."""
        summary = {
            "total_candles": 0,
            "analysis_time": 0.0,
            "success_rate": 0.0,
            "error_count": 0
        }

        result = format_analysis_summary(summary)

        assert result["metrics"]["total_candles"] == 0
        assert result["metrics"]["analysis_time"] == 0.0

    def test_format_error_response_special_characters(self):
        """Test error response formatting with special characters."""
        error_message = "Error with special chars: @#$%^&*()"

        result = format_error_response(error_message)

        assert result["error"]["message"] == error_message
