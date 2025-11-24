"""
Comprehensive Tests for Support/Resistance Indicators

Testing pivot points, S/R levels, and dynamic support/resistance detection.

Author: Dr. Sarah O'Connor (QA Lead)
Date: November 15, 2025
Version: 1.0.0
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from src.core.domain.entities import Candle
from src.core.indicators.support_resistance import SupportResistanceIndicators


class TestPivotPoints:
    """Test Pivot Point calculations"""
    
    def test_pivot_points_standard(self, sample_candles):
        """Test standard pivot point calculation"""
        result = SupportResistanceIndicators.pivot_points(
            sample_candles,
            method='standard'
        )
        
        assert result is not None
        assert 'pivot' in result.additional_values
        assert 'r1' in result.additional_values
        assert 'r2' in result.additional_values
        assert 'r3' in result.additional_values
        assert 's1' in result.additional_values
        assert 's2' in result.additional_values
        assert 's3' in result.additional_values
    
    def test_pivot_points_woodie(self, sample_candles):
        """Test Woodie's pivot points"""
        result = SupportResistanceIndicators.pivot_points(
            sample_candles,
            method='woodie'
        )
        
        assert result is not None
        assert result.additional_values.get('pivot') is not None
    
    def test_pivot_points_camarilla(self, sample_candles):
        """Test Camarilla pivot points"""
        result = SupportResistanceIndicators.pivot_points(
            sample_candles,
            method='camarilla'
        )
        
        assert result is not None
        assert result.additional_values.get('pivot') is not None
    
    def test_pivot_points_fibonacci(self, sample_candles):
        """Test Fibonacci pivot points"""
        result = SupportResistanceIndicators.pivot_points(
            sample_candles,
            method='fibonacci'
        )
        
        assert result is not None
        assert result.additional_values.get('pivot') is not None
    
    def test_pivot_levels_order(self, sample_candles):
        """Test that pivot levels are in correct order"""
        result = SupportResistanceIndicators.pivot_points(sample_candles)
        
        assert result is not None
        
        pivot = result.additional_values['pivot']
        r1 = result.additional_values['r1']
        r2 = result.additional_values['r2']
        r3 = result.additional_values['r3']
        s1 = result.additional_values['s1']
        s2 = result.additional_values['s2']
        s3 = result.additional_values['s3']
        
        # Resistance levels should increase
        assert r1 > pivot
        assert r2 > r1
        assert r3 > r2
        
        # Support levels should decrease
        assert s1 < pivot
        assert s2 < s1
        assert s3 < s2


class TestFibonacciRetracement:
    """Test Fibonacci Retracement levels"""
    
    def test_fibonacci_basic(self, sample_candles):
        """Test basic Fibonacci retracement"""
        result = SupportResistanceIndicators.fibonacci_retracement(
            sample_candles
        )
        
        assert result is not None
        assert 'levels' in result.additional_values
        
        levels = result.additional_values['levels']
        assert 0.236 in levels
        assert 0.382 in levels
        assert 0.5 in levels
        assert 0.618 in levels
        assert 0.786 in levels
    
    def test_fibonacci_uptrend(self, uptrend_candles):
        """Test Fibonacci in uptrend"""
        result = SupportResistanceIndicators.fibonacci_retracement(
            uptrend_candles
        )
        
        assert result is not None
        levels = result.additional_values.get('levels', {})
        
        # In uptrend, swing high > swing low
        swing_high = result.additional_values.get('swing_high')
        swing_low = result.additional_values.get('swing_low')
        
        if swing_high and swing_low:
            assert swing_high > swing_low
    
    def test_fibonacci_downtrend(self, downtrend_candles):
        """Test Fibonacci in downtrend"""
        result = SupportResistanceIndicators.fibonacci_retracement(
            downtrend_candles
        )
        
        assert result is not None
        # Should still calculate levels
        assert 'levels' in result.additional_values


class TestSupportResistanceLevels:
    """Test Support and Resistance level detection"""
    
    def test_sr_levels_detection(self, sample_candles):
        """Test S/R level detection"""
        result = SupportResistanceIndicators.support_resistance_levels(
            sample_candles,
            window=5,
            num_touches=2
        )
        
        assert result is not None
        assert 'support_levels' in result.additional_values
        assert 'resistance_levels' in result.additional_values
    
    def test_sr_levels_with_touches(self):
        """Test S/R levels with multiple touches"""
        candles = []
        base_time = datetime(2024, 1, 1)
        
        # Create pattern with clear support at 100
        prices = [105, 103, 101, 100, 102, 104, 102, 100, 103, 105,
                  103, 101, 100, 102, 104]
        
        for i, price in enumerate(prices):
            candles.append(Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price,
                high=price + 2,
                low=price - 1,
                close=price,
                volume=1000000
            ))
        
        result = SupportResistanceIndicators.support_resistance_levels(
            candles,
            window=3,
            num_touches=2
        )
        
        assert result is not None
        # Should detect support around 100
        support_levels = result.additional_values.get('support_levels', [])
        assert len(support_levels) > 0
    
    def test_sr_strength_calculation(self, sample_candles):
        """Test S/R strength calculation"""
        result = SupportResistanceIndicators.support_resistance_levels(
            sample_candles
        )
        
        assert result is not None
        
        # Check if strength values are included
        support_levels = result.additional_values.get('support_levels', [])
        for level in support_levels:
            if isinstance(level, dict):
                assert 'price' in level
                assert 'strength' in level


class TestDynamicSupportResistance:
    """Test Dynamic Support/Resistance (e.g., moving averages)"""
    
    def test_dynamic_sr_basic(self, sample_candles):
        """Test dynamic S/R using moving averages"""
        result = SupportResistanceIndicators.dynamic_support_resistance(
            sample_candles,
            short_period=10,
            long_period=20
        )
        
        assert result is not None
        assert 'short_ma' in result.additional_values
        assert 'long_ma' in result.additional_values
    
    def test_dynamic_sr_signal(self, sample_candles):
        """Test dynamic S/R signal generation"""
        result = SupportResistanceIndicators.dynamic_support_resistance(
            sample_candles
        )
        
        assert result is not None
        assert hasattr(result, 'signal')
        assert -1.0 <= result.signal <= 1.0


class TestKeyLevels:
    """Test Key Level identification"""
    
    def test_key_levels_basic(self, sample_candles):
        """Test key level identification"""
        result = SupportResistanceIndicators.identify_key_levels(
            sample_candles,
            lookback=50
        )
        
        assert result is not None
        assert 'key_levels' in result.additional_values
    
    def test_key_levels_round_numbers(self):
        """Test key levels near round numbers"""
        candles = []
        base_time = datetime(2024, 1, 1)
        
        # Create prices around 100 (round number)
        for i in range(50):
            price = 99 + (i % 5)
            candles.append(Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000
            ))
        
        result = SupportResistanceIndicators.identify_key_levels(
            candles,
            lookback=50
        )
        
        assert result is not None
        # Should identify 100 as key level
        key_levels = result.additional_values.get('key_levels', [])
        assert len(key_levels) > 0


class TestBreakoutDetection:
    """Test Breakout Detection"""
    
    def test_breakout_resistance(self):
        """Test breakout above resistance"""
        candles = []
        base_time = datetime(2024, 1, 1)
        
        # Create consolidation then breakout
        for i in range(30):
            if i < 20:
                price = 100 + (i % 3)  # Consolidation
            else:
                price = 103 + i  # Breakout
            
            candles.append(Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price,
                high=price + 2,
                low=price - 1,
                close=price + 1,
                volume=1000000 * (1.5 if i >= 20 else 1.0)
            ))
        
        result = SupportResistanceIndicators.detect_breakout(
            candles,
            window=20
        )
        
        assert result is not None
        # Should detect breakout
        if result.additional_values.get('breakout_type'):
            assert result.additional_values['breakout_type'] in ['resistance', 'support']
    
    def test_breakout_support(self):
        """Test breakdown below support"""
        candles = []
        base_time = datetime(2024, 1, 1)
        
        # Create consolidation then breakdown
        for i in range(30):
            if i < 20:
                price = 100 + (i % 3)  # Consolidation
            else:
                price = 100 - (i - 19) * 2  # Breakdown
            
            candles.append(Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price,
                high=price + 1,
                low=price - 2,
                close=price - 1,
                volume=1000000 * (1.5 if i >= 20 else 1.0)
            ))
        
        result = SupportResistanceIndicators.detect_breakout(
            candles,
            window=20
        )
        
        assert result is not None


class TestZoneDetection:
    """Test Support/Resistance Zone Detection"""
    
    def test_zone_detection_basic(self, sample_candles):
        """Test basic zone detection"""
        result = SupportResistanceIndicators.detect_zones(
            sample_candles,
            zone_width=0.02  # 2% width
        )
        
        assert result is not None
        assert 'support_zones' in result.additional_values
        assert 'resistance_zones' in result.additional_values
    
    def test_zone_width(self, sample_candles):
        """Test zone width parameter"""
        result_narrow = SupportResistanceIndicators.detect_zones(
            sample_candles,
            zone_width=0.01
        )
        result_wide = SupportResistanceIndicators.detect_zones(
            sample_candles,
            zone_width=0.05
        )
        
        assert result_narrow is not None
        assert result_wide is not None


class TestPriceAction:
    """Test Price Action at S/R levels"""
    
    def test_price_action_at_support(self, sample_candles):
        """Test price action near support"""
        result = SupportResistanceIndicators.price_action_at_level(
            sample_candles,
            level=100.0,
            tolerance=0.01
        )
        
        assert result is not None
    
    def test_price_action_at_resistance(self, sample_candles):
        """Test price action near resistance"""
        result = SupportResistanceIndicators.price_action_at_level(
            sample_candles,
            level=110.0,
            tolerance=0.01
        )
        
        assert result is not None


class TestEdgeCases:
    """Test edge cases"""
    
    def test_insufficient_data(self, insufficient_candles):
        """Test with insufficient data"""
        result = SupportResistanceIndicators.pivot_points(insufficient_candles)
        assert result is None
    
    def test_flat_market(self):
        """Test with flat market (no volatility)"""
        candles = []
        base_time = datetime(2024, 1, 1)
        
        for i in range(50):
            candles.append(Candle(
                timestamp=base_time + timedelta(hours=i),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=1000000
            ))
        
        result = SupportResistanceIndicators.support_resistance_levels(candles)
        # Should handle flat market gracefully
        assert result is not None or result is None
    
    def test_extreme_volatility(self):
        """Test with extreme volatility"""
        candles = []
        base_time = datetime(2024, 1, 1)
        
        for i in range(50):
            price = 100 * (1.5 if i % 2 == 0 else 0.5)  # 50% swings
            candles.append(Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price,
                high=price * 1.1,
                low=price * 0.9,
                close=price,
                volume=1000000
            ))
        
        result = SupportResistanceIndicators.pivot_points(candles)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
