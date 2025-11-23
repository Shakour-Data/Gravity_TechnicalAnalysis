"""
Unit tests for src/core/domain/entities/indicator_category.py

Tests IndicatorCategory enum values and behavior.
"""

import pytest
from src.core.domain.entities.indicator_category import IndicatorCategory


class TestIndicatorCategory:
    """Test suite for IndicatorCategory enum"""

    def test_indicator_category_values(self):
        """Test all IndicatorCategory enum values"""
        assert IndicatorCategory.TREND == "TREND"
        assert IndicatorCategory.MOMENTUM == "MOMENTUM"
        assert IndicatorCategory.CYCLE == "CYCLE"
        assert IndicatorCategory.VOLUME == "VOLUME"
        assert IndicatorCategory.VOLATILITY == "VOLATILITY"
        assert IndicatorCategory.SUPPORT_RESISTANCE == "SUPPORT_RESISTANCE"

    def test_indicator_category_str_values(self):
        """Test string representation of IndicatorCategory values"""
        assert str(IndicatorCategory.TREND) == "TREND"
        assert str(IndicatorCategory.MOMENTUM) == "MOMENTUM"
        assert str(IndicatorCategory.CYCLE) == "CYCLE"
        assert str(IndicatorCategory.VOLUME) == "VOLUME"
        assert str(IndicatorCategory.VOLATILITY) == "VOLATILITY"
        assert str(IndicatorCategory.SUPPORT_RESISTANCE) == "SUPPORT_RESISTANCE"

    def test_indicator_category_enum_members(self):
        """Test that all expected enum members exist"""
        expected_members = {
            'TREND', 'MOMENTUM', 'CYCLE', 'VOLUME', 'VOLATILITY', 'SUPPORT_RESISTANCE'
        }
        actual_members = set(IndicatorCategory.__members__.keys())
        assert actual_members == expected_members

    def test_indicator_category_inheritance(self):
        """Test that IndicatorCategory inherits from str"""
        assert isinstance(IndicatorCategory.TREND, str)
        assert isinstance(IndicatorCategory.TREND.value, str)

    @pytest.mark.parametrize("category,expected_value", [
        (IndicatorCategory.TREND, "TREND"),
        (IndicatorCategory.MOMENTUM, "MOMENTUM"),
        (IndicatorCategory.CYCLE, "CYCLE"),
        (IndicatorCategory.VOLUME, "VOLUME"),
        (IndicatorCategory.VOLATILITY, "VOLATILITY"),
        (IndicatorCategory.SUPPORT_RESISTANCE, "SUPPORT_RESISTANCE"),
    ])
    def test_indicator_category_parametrized_values(self, category, expected_value):
        """Parametrized test for all IndicatorCategory values"""
        assert category == expected_value
        assert category.value == expected_value

    def test_indicator_category_uniqueness(self):
        """Test that all IndicatorCategory values are unique"""
        values = [cat.value for cat in IndicatorCategory]
        assert len(values) == len(set(values))

    def test_indicator_category_iteration(self):
        """Test that IndicatorCategory can be iterated over"""
        categories = list(IndicatorCategory)
        assert len(categories) == 6
        assert IndicatorCategory.TREND in categories
        assert IndicatorCategory.MOMENTUM in categories

    def test_indicator_category_comparison(self):
        """Test IndicatorCategory comparison"""
        assert IndicatorCategory.TREND != IndicatorCategory.MOMENTUM
        assert IndicatorCategory.TREND == IndicatorCategory.TREND

    def test_indicator_category_hashable(self):
        """Test that IndicatorCategory is hashable (can be used in sets/dicts)"""
        category_set = {IndicatorCategory.TREND, IndicatorCategory.MOMENTUM}
        assert len(category_set) == 2

        category_dict = {IndicatorCategory.TREND: "trend_value"}
        assert category_dict[IndicatorCategory.TREND] == "trend_value"
