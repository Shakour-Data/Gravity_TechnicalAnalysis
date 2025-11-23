"""
Unit tests for src/core/domain/entities/pattern_type.py

Tests PatternType enum values and behavior.
"""

import pytest
from src.core.domain.entities.pattern_type import PatternType


class TestPatternType:
    """Test suite for PatternType enum"""

    def test_pattern_type_values(self):
        """Test all PatternType enum values"""
        assert PatternType.CLASSICAL == "CLASSICAL"
        assert PatternType.CANDLESTICK == "CANDLESTICK"

    def test_pattern_type_str_values(self):
        """Test string representation of PatternType values"""
        assert str(PatternType.CLASSICAL) == "CLASSICAL"
        assert str(PatternType.CANDLESTICK) == "CANDLESTICK"

    def test_pattern_type_enum_members(self):
        """Test that all expected enum members exist"""
        expected_members = {'CLASSICAL', 'CANDLESTICK'}
        actual_members = set(PatternType.__members__.keys())
        assert actual_members == expected_members

    def test_pattern_type_inheritance(self):
        """Test that PatternType inherits from str"""
        assert isinstance(PatternType.CLASSICAL, str)
        assert isinstance(PatternType.CLASSICAL.value, str)

    @pytest.mark.parametrize("pattern_type,expected_value", [
        (PatternType.CLASSICAL, "CLASSICAL"),
        (PatternType.CANDLESTICK, "CANDLESTICK"),
    ])
    def test_pattern_type_parametrized_values(self, pattern_type, expected_value):
        """Parametrized test for all PatternType values"""
        assert pattern_type == expected_value
        assert pattern_type.value == expected_value

    def test_pattern_type_uniqueness(self):
        """Test that all PatternType values are unique"""
        values = [pt.value for pt in PatternType]
        assert len(values) == len(set(values))

    def test_pattern_type_iteration(self):
        """Test that PatternType can be iterated over"""
        types = list(PatternType)
        assert len(types) == 2
        assert PatternType.CLASSICAL in types
        assert PatternType.CANDLESTICK in types

    def test_pattern_type_comparison(self):
        """Test PatternType comparison"""
        assert PatternType.CLASSICAL != PatternType.CANDLESTICK
        assert PatternType.CLASSICAL == PatternType.CLASSICAL

    def test_pattern_type_hashable(self):
        """Test that PatternType is hashable (can be used in sets/dicts)"""
        type_set = {PatternType.CLASSICAL, PatternType.CANDLESTICK}
        assert len(type_set) == 2

        type_dict = {PatternType.CLASSICAL: "classical_value"}
        assert type_dict[PatternType.CLASSICAL] == "classical_value"
