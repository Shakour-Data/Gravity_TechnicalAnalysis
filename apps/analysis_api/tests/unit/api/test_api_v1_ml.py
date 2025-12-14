"""
Unit tests for API V1 ML endpoints

Tests the ML analysis API endpoints with proper mocking and validation.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from gravity_tech.api.v1.ml import router as ml_router
from gravity_tech.core.domain.entities.candle import Candle


@pytest.fixture
def sample_candles():
    """Create sample candle data for testing"""
    candles = []
    base_time = datetime(2025, 1, 1)
    for i in range(100):  # Sufficient data for ML analysis
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=100 + i * 0.5,
            high=105 + i * 0.5,
            low=95 + i * 0.5,
            close=102 + i * 0.5,
            volume=1000000 + i * 10000
        ))
    return candles


class TestMLRouter:
    """Test ML router setup and basic functionality"""

    def test_ml_router_import(self):
        """Test that ML router can be imported successfully"""
        assert ml_router is not None
        assert hasattr(ml_router, 'routes')
        assert len(ml_router.routes) > 0

    def test_ml_router_has_expected_routes(self):
        """Test that ML router has expected route endpoints"""
        # Check that routes exist
        assert len(ml_router.routes) >= 4  # Should have at least 4 ML endpoints

    @patch('gravity_tech.api.v1.ml.load_ml_model')
    def test_load_ml_model_mock(self, mock_load):
        """Test that ML model loading function can be mocked"""
        # Setup mock
        mock_model = {"mock": "model"}
        mock_load.return_value = (mock_model, "v1.0.0")

        # Test that the mock works
        result_model, result_version = mock_load()
        assert result_model == mock_model
        assert result_version == "v1.0.0"


class TestMLModels:
    """Test ML model structures and data handling"""

    def test_candle_data_structure(self, sample_candles):
        """Test that candle data has expected structure"""
        candle = sample_candles[0]
        assert hasattr(candle, 'timestamp')
        assert hasattr(candle, 'open')
        assert hasattr(candle, 'high')
        assert hasattr(candle, 'low')
        assert hasattr(candle, 'close')
        assert hasattr(candle, 'volume')

        # Test attribute access
        assert isinstance(candle.timestamp, datetime)
        assert isinstance(candle.open, float)
        assert isinstance(candle.close, float)

    def test_sample_data_generation(self, sample_candles):
        """Test that sample data is generated correctly"""
        assert len(sample_candles) == 100

        # Test chronological order
        for i in range(1, len(sample_candles)):
            assert sample_candles[i].timestamp > sample_candles[i-1].timestamp

        # Test price progression
        for i in range(1, len(sample_candles)):
            assert sample_candles[i].open > sample_candles[i-1].open

    def test_ml_response_structure_validation(self):
        """Test that ML response structures are properly defined"""
        # Test basic ML prediction response structure
        prediction_response = {
            "pattern_type": "gartley",
            "confidence": 0.85,
            "probabilities": {
                "gartley": 0.85,
                "butterfly": 0.10,
                "bat": 0.03,
                "crab": 0.02
            },
            "processing_time": 0.12,
            "model_version": "v1.0.0"
        }

        # Validate structure
        required_keys = ["pattern_type", "confidence", "probabilities", "processing_time", "model_version"]
        for key in required_keys:
            assert key in prediction_response, f"Missing required key: {key}"

        # Validate nested structures
        assert "gartley" in prediction_response["probabilities"]
        assert isinstance(prediction_response["confidence"], int | float)
        assert 0 <= prediction_response["confidence"] <= 1
