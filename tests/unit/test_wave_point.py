"""
Unit tests for WavePoint entity (src/core/domain/entities/wave_point.py)

Tests cover:
- WavePoint dataclass creation and validation
- Immutable properties
- Validation of wave_type ("PEAK" or "TROUGH")
- Validation of positive price values
- Edge cases and error conditions
"""

import pytest
from datetime import datetime
from gravity_tech.core.domain.entities.wave_point import WavePoint


class TestWavePoint:
    """Test suite for WavePoint entity"""

    @pytest.fixture
    def valid_timestamp(self):
        """Fixture for a valid timestamp"""
        return datetime(2024, 1, 1, 12, 0, 0)

    @pytest.fixture
    def valid_wave_point_peak(self, valid_timestamp):
        """Fixture for a valid peak wave point"""
        return WavePoint(
            wave_number=1,
            price=100.0,
            timestamp=valid_timestamp,
            wave_type="PEAK"
        )

    @pytest.fixture
    def valid_wave_point_trough(self, valid_timestamp):
        """Fixture for a valid trough wave point"""
        return WavePoint(
            wave_number=2,
            price=95.0,
            timestamp=valid_timestamp,
            wave_type="TROUGH"
        )

    def test_wave_point_creation_peak(self, valid_wave_point_peak):
        """Test creating a valid peak wave point"""
        assert valid_wave_point_peak.wave_number == 1
        assert valid_wave_point_peak.price == 100.0
        assert valid_wave_point_peak.wave_type == "PEAK"
        assert isinstance(valid_wave_point_peak.timestamp, datetime)

    def test_wave_point_creation_trough(self, valid_wave_point_trough):
        """Test creating a valid trough wave point"""
        assert valid_wave_point_trough.wave_number == 2
        assert valid_wave_point_trough.price == 95.0
        assert valid_wave_point_trough.wave_type == "TROUGH"
        assert isinstance(valid_wave_point_trough.timestamp, datetime)

    def test_wave_point_immutability(self, valid_wave_point_peak):
        """Test that WavePoint is immutable (frozen dataclass)"""
        with pytest.raises(AttributeError):
            valid_wave_point_peak.wave_number = 2

        with pytest.raises(AttributeError):
            valid_wave_point_peak.price = 101.0

        with pytest.raises(AttributeError):
            valid_wave_point_peak.wave_type = "TROUGH"

    def test_wave_point_equality(self, valid_timestamp):
        """Test WavePoint equality"""
        point1 = WavePoint(1, 100.0, valid_timestamp, "PEAK")
        point2 = WavePoint(1, 100.0, valid_timestamp, "PEAK")
        point3 = WavePoint(2, 100.0, valid_timestamp, "PEAK")

        assert point1 == point2
        assert point1 != point3

    def test_wave_point_hashable(self, valid_wave_point_peak):
        """Test that WavePoint is hashable (for use in sets, dict keys)"""
        # Should not raise an exception
        hash(valid_wave_point_peak)

        # Can be used as dict key
        wave_dict = {valid_wave_point_peak: "test_value"}
        assert wave_dict[valid_wave_point_peak] == "test_value"

    @pytest.mark.parametrize("invalid_wave_type", [
        "peak",      # lowercase
        "PEAKS",     # plural
        "TROUGHES",  # misspelled
        "INVALID",   # completely wrong
        "",          # empty string
        None,        # None value
        123,         # integer
    ])
    def test_invalid_wave_type_validation(self, valid_timestamp, invalid_wave_type):
        """Test validation of wave_type field"""
        with pytest.raises(ValueError, match="wave_type must be 'PEAK' or 'TROUGH'"):
            WavePoint(
                wave_number=1,
                price=100.0,
                timestamp=valid_timestamp,
                wave_type=invalid_wave_type
            )

    @pytest.mark.parametrize("invalid_price", [
        0,           # zero
        -1.0,        # negative
        -100.0,      # negative large
        -0.01,       # negative small
    ])
    def test_invalid_price_validation(self, valid_timestamp, invalid_price):
        """Test validation of price field (must be positive)"""
        with pytest.raises(ValueError, match="price must be positive"):
            WavePoint(
                wave_number=1,
                price=invalid_price,
                timestamp=valid_timestamp,
                wave_type="PEAK"
            )

    @pytest.mark.parametrize("valid_price", [
        0.01,        # very small positive
        1.0,         # small positive
        100.0,       # normal price
        1000000.0,   # large price
        0.000001,    # very small decimal
    ])
    def test_valid_price_values(self, valid_timestamp, valid_price):
        """Test that valid positive prices are accepted"""
        point = WavePoint(
            wave_number=1,
            price=valid_price,
            timestamp=valid_timestamp,
            wave_type="PEAK"
        )
        assert point.price == valid_price

    @pytest.mark.parametrize("wave_number", [
        1, 2, 3, 4, 5,  # impulse waves
        0,              # wave 0 (sometimes used)
        -1,             # negative (though unusual)
        100,            # large number
    ])
    def test_wave_number_values(self, valid_timestamp, wave_number):
        """Test various wave number values"""
        point = WavePoint(
            wave_number=wave_number,
            price=100.0,
            timestamp=valid_timestamp,
            wave_type="PEAK"
        )
        assert point.wave_number == wave_number

    def test_wave_point_string_representation(self, valid_wave_point_peak):
        """Test string representation of WavePoint"""
        str_repr = str(valid_wave_point_peak)
        assert "WavePoint" in str_repr
        assert "wave_number=1" in str_repr
        assert "price=100.0" in str_repr
        assert "wave_type='PEAK'" in str_repr

    def test_wave_point_repr(self, valid_wave_point_peak):
        """Test repr representation of WavePoint"""
        repr_str = repr(valid_wave_point_peak)
        assert "WavePoint(" in repr_str
        assert "wave_number=1" in repr_str
        assert "price=100.0" in repr_str
        assert "wave_type='PEAK'" in repr_str

    def test_wave_point_with_different_timestamps(self, valid_timestamp):
        """Test WavePoint with different timestamps are different objects"""
        timestamp1 = valid_timestamp
        timestamp2 = datetime(2024, 1, 2, 12, 0, 0)

        point1 = WavePoint(1, 100.0, timestamp1, "PEAK")
        point2 = WavePoint(1, 100.0, timestamp2, "PEAK")

        assert point1 != point2  # Different timestamps make them different

    def test_wave_point_copy_behavior(self, valid_wave_point_peak):
        """Test that WavePoint behaves correctly with copying"""
        import copy

        # Test shallow copy
        copied = copy.copy(valid_wave_point_peak)
        assert copied == valid_wave_point_peak
        assert copied is not valid_wave_point_peak  # Different objects

        # Test deep copy
        deep_copied = copy.deepcopy(valid_wave_point_peak)
        assert deep_copied == valid_wave_point_peak
        assert deep_copied is not valid_wave_point_peak

