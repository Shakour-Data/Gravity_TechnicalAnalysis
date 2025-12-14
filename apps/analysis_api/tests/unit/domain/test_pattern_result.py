"""
Unit tests for src/core/domain/entities/pattern_result.py

Tests PatternResult dataclass creation, validation, and properties.
"""

import pytest
from datetime import datetime, timedelta
from gravity_tech.core.domain.entities.pattern_result import PatternResult
from gravity_tech.core.domain.entities.pattern_type import PatternType
from gravity_tech.core.domain.entities.signal_strength import SignalStrength


class TestPatternResultCreation:
    """Test suite for PatternResult creation and validation"""

    def test_pattern_result_creation_valid(self):
        """Test creating a valid PatternResult"""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 13, 0, 0)
        result = PatternResult(
            pattern_name="Head and Shoulders",
            pattern_type=PatternType.CLASSICAL,
            signal=SignalStrength.BEARISH,
            confidence=0.85,
            start_time=start_time,
            end_time=end_time,
            description="Bearish reversal pattern detected",
            price_target=95.0,
            stop_loss=110.0
        )
        assert result.pattern_name == "Head and Shoulders"
        assert result.pattern_type == PatternType.CLASSICAL
        assert result.signal == SignalStrength.BEARISH
        assert result.confidence == 0.85
        assert result.start_time == start_time
        assert result.end_time == end_time
        assert result.description == "Bearish reversal pattern detected"
        assert result.price_target == 95.0
        assert result.stop_loss == 110.0

    def test_pattern_result_creation_minimal(self):
        """Test PatternResult creation with minimal required fields"""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 13, 0, 0)
        result = PatternResult(
            pattern_name="Doji",
            pattern_type=PatternType.CANDLESTICK,
            signal=SignalStrength.NEUTRAL,
            confidence=0.6,
            start_time=start_time,
            end_time=end_time,
            description="Indecision pattern"
        )
        assert result.price_target is None
        assert result.stop_loss is None

    def test_pattern_result_immutable(self):
        """Test that PatternResult is immutable (frozen dataclass)"""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 13, 0, 0)
        result = PatternResult(
            pattern_name="Bullish Engulfing",
            pattern_type=PatternType.CANDLESTICK,
            signal=SignalStrength.BULLISH,
            confidence=0.75,
            start_time=start_time,
            end_time=end_time,
            description="Bullish reversal pattern"
        )
        with pytest.raises(AttributeError):
            result.confidence = 0.8

    def test_pattern_result_validation_confidence_too_low(self):
        """Test validation for confidence below 0.0"""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 13, 0, 0)
        with pytest.raises(ValueError, match="confidence must be 0.0-1.0"):
            PatternResult(
                pattern_name="Test",
                pattern_type=PatternType.CLASSICAL,
                signal=SignalStrength.NEUTRAL,
                confidence=-0.1,
                start_time=start_time,
                end_time=end_time,
                description="Test pattern"
            )

    def test_pattern_result_validation_confidence_too_high(self):
        """Test validation for confidence above 1.0"""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 13, 0, 0)
        with pytest.raises(ValueError, match="confidence must be 0.0-1.0"):
            PatternResult(
                pattern_name="Test",
                pattern_type=PatternType.CLASSICAL,
                signal=SignalStrength.NEUTRAL,
                confidence=1.5,
                start_time=start_time,
                end_time=end_time,
                description="Test pattern"
            )

    def test_pattern_result_validation_end_time_before_start(self):
        """Test validation when end_time is before start_time"""
        start_time = datetime(2023, 1, 1, 13, 0, 0)
        end_time = datetime(2023, 1, 1, 12, 0, 0)  # Before start_time
        with pytest.raises(ValueError, match="end_time .* must be >= start_time"):
            PatternResult(
                pattern_name="Test",
                pattern_type=PatternType.CLASSICAL,
                signal=SignalStrength.NEUTRAL,
                confidence=0.5,
                start_time=start_time,
                end_time=end_time,
                description="Test pattern"
            )

    def test_pattern_result_validation_empty_name(self):
        """Test validation for empty pattern name"""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 13, 0, 0)
        with pytest.raises(ValueError, match="pattern_name cannot be empty"):
            PatternResult(
                pattern_name="",
                pattern_type=PatternType.CLASSICAL,
                signal=SignalStrength.NEUTRAL,
                confidence=0.5,
                start_time=start_time,
                end_time=end_time,
                description="Test pattern"
            )

    def test_pattern_result_validation_whitespace_name(self):
        """Test validation for whitespace-only pattern name"""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 13, 0, 0)
        with pytest.raises(ValueError, match="pattern_name cannot be empty"):
            PatternResult(
                pattern_name="   ",
                pattern_type=PatternType.CLASSICAL,
                signal=SignalStrength.NEUTRAL,
                confidence=0.5,
                start_time=start_time,
                end_time=end_time,
                description="Test pattern"
            )

    def test_pattern_result_validation_price_target_negative(self):
        """Test validation for negative price_target"""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 13, 0, 0)
        with pytest.raises(ValueError, match="price_target must be positive"):
            PatternResult(
                pattern_name="Test",
                pattern_type=PatternType.CLASSICAL,
                signal=SignalStrength.NEUTRAL,
                confidence=0.5,
                start_time=start_time,
                end_time=end_time,
                description="Test pattern",
                price_target=-10.0
            )

    def test_pattern_result_validation_price_target_zero(self):
        """Test validation for zero price_target"""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 13, 0, 0)
        with pytest.raises(ValueError, match="price_target must be positive"):
            PatternResult(
                pattern_name="Test",
                pattern_type=PatternType.CLASSICAL,
                signal=SignalStrength.NEUTRAL,
                confidence=0.5,
                start_time=start_time,
                end_time=end_time,
                description="Test pattern",
                price_target=0.0
            )

    def test_pattern_result_validation_stop_loss_negative(self):
        """Test validation for negative stop_loss"""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 13, 0, 0)
        with pytest.raises(ValueError, match="stop_loss must be positive"):
            PatternResult(
                pattern_name="Test",
                pattern_type=PatternType.CLASSICAL,
                signal=SignalStrength.NEUTRAL,
                confidence=0.5,
                start_time=start_time,
                end_time=end_time,
                description="Test pattern",
                stop_loss=-5.0
            )

    def test_pattern_result_validation_stop_loss_zero(self):
        """Test validation for zero stop_loss"""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 13, 0, 0)
        with pytest.raises(ValueError, match="stop_loss must be positive"):
            PatternResult(
                pattern_name="Test",
                pattern_type=PatternType.CLASSICAL,
                signal=SignalStrength.NEUTRAL,
                confidence=0.5,
                start_time=start_time,
                end_time=end_time,
                description="Test pattern",
                stop_loss=0.0
            )

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_pattern_result_valid_confidence_values(self, confidence):
        """Test valid confidence values"""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 13, 0, 0)
        result = PatternResult(
            pattern_name="Test",
            pattern_type=PatternType.CLASSICAL,
            signal=SignalStrength.NEUTRAL,
            confidence=confidence,
            start_time=start_time,
            end_time=end_time,
            description="Test pattern"
        )
        assert result.confidence == confidence

    @pytest.mark.parametrize("invalid_confidence", [-0.1, 1.1, 2.0, -1.0])
    def test_pattern_result_invalid_confidence_values(self, invalid_confidence):
        """Test invalid confidence values"""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 13, 0, 0)
        with pytest.raises(ValueError, match="confidence must be 0.0-1.0"):
            PatternResult(
                pattern_name="Test",
                pattern_type=PatternType.CLASSICAL,
                signal=SignalStrength.NEUTRAL,
                confidence=invalid_confidence,
                start_time=start_time,
                end_time=end_time,
                description="Test pattern"
            )


class TestPatternResultProperties:
    """Test suite for PatternResult properties and methods"""

    @pytest.fixture
    def sample_result(self):
        """Fixture for a sample PatternResult"""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 15, 30, 0)
        return PatternResult(
            pattern_name="Double Bottom",
            pattern_type=PatternType.CLASSICAL,
            signal=SignalStrength.BULLISH,
            confidence=0.8,
            start_time=start_time,
            end_time=end_time,
            description="Bullish reversal pattern with high confidence",
            price_target=120.0,
            stop_loss=95.0
        )

    def test_pattern_result_attributes_access(self, sample_result):
        """Test accessing all attributes"""
        assert sample_result.pattern_name == "Double Bottom"
        assert sample_result.pattern_type == PatternType.CLASSICAL
        assert sample_result.signal == SignalStrength.BULLISH
        assert sample_result.confidence == 0.8
        assert sample_result.start_time == datetime(2023, 1, 1, 12, 0, 0)
        assert sample_result.end_time == datetime(2023, 1, 1, 15, 30, 0)
        assert sample_result.description == "Bullish reversal pattern with high confidence"
        assert sample_result.price_target == 120.0
        assert sample_result.stop_loss == 95.0

    def test_pattern_result_equality(self):
        """Test PatternResult equality"""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 13, 0, 0)
        result1 = PatternResult(
            pattern_name="Test",
            pattern_type=PatternType.CLASSICAL,
            signal=SignalStrength.NEUTRAL,
            confidence=0.5,
            start_time=start_time,
            end_time=end_time,
            description="Test pattern"
        )
        result2 = PatternResult(
            pattern_name="Test",
            pattern_type=PatternType.CLASSICAL,
            signal=SignalStrength.NEUTRAL,
            confidence=0.5,
            start_time=start_time,
            end_time=end_time,
            description="Test pattern"
        )
        result3 = PatternResult(
            pattern_name="Different",
            pattern_type=PatternType.CLASSICAL,
            signal=SignalStrength.NEUTRAL,
            confidence=0.5,
            start_time=start_time,
            end_time=end_time,
            description="Test pattern"
        )
        assert result1 == result2
        assert result1 != result3

    def test_pattern_result_hashable(self):
        """Test that PatternResult is hashable"""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 13, 0, 0)
        result = PatternResult(
            pattern_name="Test",
            pattern_type=PatternType.CLASSICAL,
            signal=SignalStrength.NEUTRAL,
            confidence=0.5,
            start_time=start_time,
            end_time=end_time,
            description="Test pattern"
        )
        result_set = {result}
        assert len(result_set) == 1

    def test_pattern_result_repr(self, sample_result):
        """Test string representation of PatternResult"""
        repr_str = repr(sample_result)
        assert "PatternResult" in repr_str
        assert "Double Bottom" in repr_str
        assert "CLASSICAL" in repr_str


class TestPatternResultEdgeCases:
    """Test suite for PatternResult edge cases"""

    def test_pattern_result_same_start_end_time(self):
        """Test PatternResult with same start and end time"""
        time = datetime(2023, 1, 1, 12, 0, 0)
        result = PatternResult(
            pattern_name="Single Candle Pattern",
            pattern_type=PatternType.CANDLESTICK,
            signal=SignalStrength.NEUTRAL,
            confidence=0.5,
            start_time=time,
            end_time=time,
            description="Single candle pattern"
        )
        assert result.start_time == result.end_time

    def test_pattern_result_long_duration(self):
        """Test PatternResult with long time duration"""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 10, 12, 0, 0)  # 9 days later
        result = PatternResult(
            pattern_name="Long-term Pattern",
            pattern_type=PatternType.CLASSICAL,
            signal=SignalStrength.BULLISH,
            confidence=0.9,
            start_time=start_time,
            end_time=end_time,
            description="Long-term bullish pattern"
        )
        assert (result.end_time - result.start_time).days == 9

    def test_pattern_result_extreme_values(self):
        """Test PatternResult with extreme numeric values"""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 13, 0, 0)
        result = PatternResult(
            pattern_name="Extreme Values",
            pattern_type=PatternType.CLASSICAL,
            signal=SignalStrength.NEUTRAL,
            confidence=1.0,
            start_time=start_time,
            end_time=end_time,
            description="Test extreme values",
            price_target=float('inf'),
            stop_loss=float('inf')
        )
        assert result.price_target == float('inf')
        assert result.stop_loss == float('inf')

    @pytest.mark.parametrize("pattern_type", [PatternType.CLASSICAL, PatternType.CANDLESTICK])
    def test_pattern_result_all_pattern_types(self, pattern_type):
        """Test PatternResult with all pattern types"""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 13, 0, 0)
        result = PatternResult(
            pattern_name="Test",
            pattern_type=pattern_type,
            signal=SignalStrength.NEUTRAL,
            confidence=0.5,
            start_time=start_time,
            end_time=end_time,
            description="Test pattern"
        )
        assert result.pattern_type == pattern_type

    @pytest.mark.parametrize("signal", [
        SignalStrength.VERY_BULLISH,
        SignalStrength.BULLISH,
        SignalStrength.BULLISH_BROKEN,
        SignalStrength.NEUTRAL,
        SignalStrength.BEARISH_BROKEN,
        SignalStrength.BEARISH,
        SignalStrength.VERY_BEARISH,
    ])
    def test_pattern_result_all_signals(self, signal):
        """Test PatternResult with all signal strengths"""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 13, 0, 0)
        result = PatternResult(
            pattern_name="Test",
            pattern_type=PatternType.CLASSICAL,
            signal=signal,
            confidence=0.5,
            start_time=start_time,
            end_time=end_time,
            description="Test pattern"
        )
        assert result.signal == signal

    def test_pattern_result_very_long_name(self):
        """Test PatternResult with very long pattern name"""
        long_name = "A" * 1000  # Very long name
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 13, 0, 0)
        result = PatternResult(
            pattern_name=long_name,
            pattern_type=PatternType.CLASSICAL,
            signal=SignalStrength.NEUTRAL,
            confidence=0.5,
            start_time=start_time,
            end_time=end_time,
            description="Test long name"
        )
        assert result.pattern_name == long_name

    def test_pattern_result_very_long_description(self):
        """Test PatternResult with very long description"""
        long_desc = "A" * 10000  # Very long description
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 13, 0, 0)
        result = PatternResult(
            pattern_name="Test",
            pattern_type=PatternType.CLASSICAL,
            signal=SignalStrength.NEUTRAL,
            confidence=0.5,
            start_time=start_time,
            end_time=end_time,
            description=long_desc
        )
        assert result.description == long_desc

