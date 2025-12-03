"""
Unit tests for src/core/domain/entities/indicator_result.py

Tests IndicatorResult dataclass creation, validation, and properties.
"""

import pytest
from datetime import datetime
from unittest.mock import patch
from src.core.domain.entities.indicator_result import IndicatorResult
from src.core.domain.entities.indicator_category import IndicatorCategory
from src.core.domain.entities.signal_strength import SignalStrength


class TestIndicatorResultCreation:
    """Test suite for IndicatorResult creation and validation"""

    def test_indicator_result_creation_valid(self):
        """Test creating a valid IndicatorResult"""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        result = IndicatorResult(
            indicator_name="RSI",
            category=IndicatorCategory.MOMENTUM,
            signal=SignalStrength.BULLISH,
            value=65.5,
            additional_values={"signal": 60.0},
            confidence=0.85,
            description="RSI shows bullish momentum",
            timestamp=timestamp
        )
        assert result.indicator_name == "RSI"
        assert result.category == IndicatorCategory.MOMENTUM
        assert result.signal == SignalStrength.BULLISH
        assert result.value == 65.5
        assert result.additional_values == {"signal": 60.0}
        assert result.confidence == 0.85
        assert result.description == "RSI shows bullish momentum"
        assert result.timestamp == timestamp

    def test_indicator_result_creation_defaults(self):
        """Test IndicatorResult creation with default values"""
        result = IndicatorResult(
            indicator_name="MACD",
            category=IndicatorCategory.MOMENTUM,
            signal=SignalStrength.NEUTRAL,
            value=1.5
        )
        assert result.additional_values is None
        assert result.confidence == 0.75
        assert result.description is None
        assert isinstance(result.timestamp, datetime)

    def test_indicator_result_immutable(self):
        """Test that IndicatorResult is immutable (frozen dataclass)"""
        result = IndicatorResult(
            indicator_name="RSI",
            category=IndicatorCategory.MOMENTUM,
            signal=SignalStrength.BULLISH,
            value=65.5
        )
        with pytest.raises(AttributeError):
            result.value = 70.0

    def test_indicator_result_validation_confidence_too_low(self):
        """Test validation for confidence below 0.0"""
        with pytest.raises(ValueError, match="confidence must be 0.0-1.0"):
            IndicatorResult(
                indicator_name="RSI",
                category=IndicatorCategory.MOMENTUM,
                signal=SignalStrength.BULLISH,
                value=65.5,
                confidence=-0.1
            )

    def test_indicator_result_validation_confidence_too_high(self):
        """Test validation for confidence above 1.0"""
        with pytest.raises(ValueError, match="confidence must be 0.0-1.0"):
            IndicatorResult(
                indicator_name="RSI",
                category=IndicatorCategory.MOMENTUM,
                signal=SignalStrength.BULLISH,
                value=65.5,
                confidence=1.5
            )

    def test_indicator_result_validation_empty_name(self):
        """Test validation for empty indicator name"""
        with pytest.raises(ValueError, match="indicator_name cannot be empty"):
            IndicatorResult(
                indicator_name="",
                category=IndicatorCategory.MOMENTUM,
                signal=SignalStrength.BULLISH,
                value=65.5
            )

    def test_indicator_result_validation_whitespace_name(self):
        """Test validation for whitespace-only indicator name"""
        with pytest.raises(ValueError, match="indicator_name cannot be empty"):
            IndicatorResult(
                indicator_name="   ",
                category=IndicatorCategory.MOMENTUM,
                signal=SignalStrength.BULLISH,
                value=65.5
            )

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_indicator_result_valid_confidence_values(self, confidence):
        """Test valid confidence values"""
        result = IndicatorResult(
            indicator_name="RSI",
            category=IndicatorCategory.MOMENTUM,
            signal=SignalStrength.BULLISH,
            value=65.5,
            confidence=confidence
        )
        assert result.confidence == confidence

    @pytest.mark.parametrize("invalid_confidence", [-0.1, 1.1, 2.0, -1.0])
    def test_indicator_result_invalid_confidence_values(self, invalid_confidence):
        """Test invalid confidence values"""
        with pytest.raises(ValueError, match="confidence must be 0.0-1.0"):
            IndicatorResult(
                indicator_name="RSI",
                category=IndicatorCategory.MOMENTUM,
                signal=SignalStrength.BULLISH,
                value=65.5,
                confidence=invalid_confidence
            )


class TestIndicatorResultProperties:
    """Test suite for IndicatorResult properties and methods"""

    @pytest.fixture
    def sample_result(self):
        """Fixture for a sample IndicatorResult"""
        return IndicatorResult(
            indicator_name="RSI(14)",
            category=IndicatorCategory.MOMENTUM,
            signal=SignalStrength.BULLISH,
            value=65.5,
            additional_values={"signal": 60.0, "histogram": 5.5},
            confidence=0.85,
            description="RSI indicates bullish momentum",
            timestamp=datetime(2023, 1, 1, 12, 0, 0)
        )

    def test_indicator_result_attributes_access(self, sample_result):
        """Test accessing all attributes"""
        assert sample_result.indicator_name == "RSI(14)"
        assert sample_result.category == IndicatorCategory.MOMENTUM
        assert sample_result.signal == SignalStrength.BULLISH
        assert sample_result.value == 65.5
        assert sample_result.additional_values == {"signal": 60.0, "histogram": 5.5}
        assert sample_result.confidence == 0.85
        assert sample_result.description == "RSI indicates bullish momentum"
        assert sample_result.timestamp == datetime(2023, 1, 1, 12, 0, 0)

    def test_indicator_result_equality(self):
        """Test IndicatorResult equality"""
        fixed_timestamp = datetime(2023, 1, 1, 12, 0, 0)
        result1 = IndicatorResult(
            indicator_name="RSI",
            category=IndicatorCategory.MOMENTUM,
            signal=SignalStrength.BULLISH,
            value=65.5,
            timestamp=fixed_timestamp
        )
        result2 = IndicatorResult(
            indicator_name="RSI",
            category=IndicatorCategory.MOMENTUM,
            signal=SignalStrength.BULLISH,
            value=65.5,
            timestamp=fixed_timestamp
        )
        result3 = IndicatorResult(
            indicator_name="MACD",
            category=IndicatorCategory.MOMENTUM,
            signal=SignalStrength.BULLISH,
            value=65.5,
            timestamp=fixed_timestamp
        )
        assert result1 == result2
        assert result1 != result3

    def test_indicator_result_hashable(self):
        """Test that IndicatorResult is hashable"""
        result = IndicatorResult(
            indicator_name="RSI",
            category=IndicatorCategory.MOMENTUM,
            signal=SignalStrength.BULLISH,
            value=65.5
        )
        result_set = {result}
        assert len(result_set) == 1

    def test_indicator_result_repr(self, sample_result):
        """Test string representation of IndicatorResult"""
        repr_str = repr(sample_result)
        assert "IndicatorResult" in repr_str
        assert "RSI(14)" in repr_str
        assert "MOMENTUM" in repr_str

    def test_indicator_result_timestamp_auto_generation(self):
        """Test automatic timestamp generation"""
        before = datetime.now()
        result = IndicatorResult(
            indicator_name="RSI",
            category=IndicatorCategory.MOMENTUM,
            signal=SignalStrength.BULLISH,
            value=65.5
        )
        after = datetime.now()
        assert before <= result.timestamp <= after


class TestIndicatorResultEdgeCases:
    """Test suite for IndicatorResult edge cases"""

    def test_indicator_result_with_none_additional_values(self):
        """Test IndicatorResult with None additional_values"""
        result = IndicatorResult(
            indicator_name="RSI",
            category=IndicatorCategory.MOMENTUM,
            signal=SignalStrength.BULLISH,
            value=65.5,
            additional_values=None
        )
        assert result.additional_values is None

    def test_indicator_result_with_empty_additional_values(self):
        """Test IndicatorResult with empty additional_values dict"""
        result = IndicatorResult(
            indicator_name="RSI",
            category=IndicatorCategory.MOMENTUM,
            signal=SignalStrength.BULLISH,
            value=65.5,
            additional_values={}
        )
        assert result.additional_values == {}

    def test_indicator_result_extreme_values(self):
        """Test IndicatorResult with extreme numeric values"""
        result = IndicatorResult(
            indicator_name="Test",
            category=IndicatorCategory.MOMENTUM,
            signal=SignalStrength.NEUTRAL,
            value=float('inf'),
            confidence=1.0
        )
        assert result.value == float('inf')

        result_neg = IndicatorResult(
            indicator_name="Test",
            category=IndicatorCategory.MOMENTUM,
            signal=SignalStrength.NEUTRAL,
            value=float('-inf'),
            confidence=0.0
        )
        assert result_neg.value == float('-inf')

    def test_indicator_result_special_float_values(self):
        """Test IndicatorResult with NaN values"""
        result = IndicatorResult(
            indicator_name="Test",
            category=IndicatorCategory.MOMENTUM,
            signal=SignalStrength.NEUTRAL,
            value=float('nan'),
            confidence=0.5
        )
        assert str(result.value) == 'nan'  # NaN != NaN, so check string representation

    @pytest.mark.parametrize("category", [
        IndicatorCategory.TREND,
        IndicatorCategory.MOMENTUM,
        IndicatorCategory.CYCLE,
        IndicatorCategory.VOLUME,
        IndicatorCategory.VOLATILITY,
        IndicatorCategory.SUPPORT_RESISTANCE,
    ])
    def test_indicator_result_all_categories(self, category):
        """Test IndicatorResult with all indicator categories"""
        result = IndicatorResult(
            indicator_name="Test",
            category=category,
            signal=SignalStrength.NEUTRAL,
            value=50.0
        )
        assert result.category == category

    @pytest.mark.parametrize("signal", [
        SignalStrength.VERY_BULLISH,
        SignalStrength.BULLISH,
        SignalStrength.BULLISH_BROKEN,
        SignalStrength.NEUTRAL,
        SignalStrength.BEARISH_BROKEN,
        SignalStrength.BEARISH,
        SignalStrength.VERY_BEARISH,
    ])
    def test_indicator_result_all_signals(self, signal):
        """Test IndicatorResult with all signal strengths"""
        result = IndicatorResult(
            indicator_name="Test",
            category=IndicatorCategory.MOMENTUM,
            signal=signal,
            value=50.0
        )
        assert result.signal == signal
