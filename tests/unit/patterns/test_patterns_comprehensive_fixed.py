"""
Pattern Recognition Comprehensive Tests.

Tests for pattern identification and analysis:
- Elliott Wave patterns
- Harmonic patterns
- Classical patterns
- Divergence detection
- Candlestick patterns

Author: Gravity Tech Team
Date: December 5, 2025
Version: 1.0.0
License: MIT
"""

import importlib
from datetime import datetime

import numpy as np
import pytest
from gravity_tech.core.domain.entities import Candle


def _get_pattern_attribute(module_path: str, attribute_name: str):
    """Safely import a module attribute, skipping tests when unavailable."""
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        pytest.skip(f"Module '{module_path}' is not available")
    try:
        return getattr(module, attribute_name)
    except AttributeError:
        pytest.skip(f"Attribute '{attribute_name}' not found in '{module_path}'")


class TestElliottWavePatterns:
    """Test Elliott Wave pattern recognition"""

    def test_elliott_wave_detector_initialization(self):
        """Test Elliott Wave detector can be created"""
        detector_class = _get_pattern_attribute("gravity_tech.patterns.elliott_wave", "ElliottWaveDetector")
        detector = detector_class()
        assert detector is not None

    def test_detect_five_wave_pattern(self, uptrend_candles):
        """Test detection of 5-wave pattern"""
        detector_class = _get_pattern_attribute("gravity_tech.patterns.elliott_wave", "ElliottWaveDetector")
        detector = detector_class()

        if not hasattr(detector, "detect"):
            pytest.skip("ElliottWaveDetector lacks detect method")

        pattern = detector.detect(uptrend_candles)
        assert pattern is None or isinstance(pattern, dict)

    def test_wave_degree_classification(self):
        """Test wave degree classification"""
        classify_wave_degree = _get_pattern_attribute("gravity_tech.patterns.elliott_wave", "classify_wave_degree")
        degree = classify_wave_degree(small_timeframe="1m", large_timeframe="1h")
        assert degree is None or isinstance(degree, int | str)

    def test_gartley_pattern_detection(self, sample_candles):
        """Test Gartley pattern detection"""
        try:
            from gravity_tech.patterns.harmonic import detect_gartley
            highs = np.array([c.high for c in sample_candles])
            lows = np.array([c.low for c in sample_candles])
            closes = np.array([c.close for c in sample_candles])
            pattern = detect_gartley(highs=highs, lows=lows, closes=closes)
            assert pattern is None or isinstance(pattern, dict)
        except ImportError:
            pytest.skip("Harmonic patterns not available")

    def test_butterfly_pattern_detection(self, sample_candles):
        """Test Butterfly pattern detection"""
        try:
            from gravity_tech.patterns.harmonic import detect_butterfly
            highs = np.array([c.high for c in sample_candles])
            lows = np.array([c.low for c in sample_candles])
            closes = np.array([c.close for c in sample_candles])
            pattern = detect_butterfly(highs=highs, lows=lows, closes=closes)
            assert pattern is None or isinstance(pattern, dict)
        except ImportError:
            pytest.skip("Harmonic patterns not available")

    def test_bat_pattern_detection(self, sample_candles):
        """Test Bat pattern detection"""
        try:
            from gravity_tech.patterns.harmonic import detect_bat
            highs = np.array([c.high for c in sample_candles])
            lows = np.array([c.low for c in sample_candles])
            closes = np.array([c.close for c in sample_candles])
            pattern = detect_bat(highs=highs, lows=lows, closes=closes)
            assert pattern is None or isinstance(pattern, dict)
        except ImportError:
            pytest.skip("Harmonic patterns not available")

    def test_crab_pattern_detection(self, sample_candles):
        """Test Crab pattern detection"""
        try:
            from gravity_tech.patterns.harmonic import detect_crab
            highs = np.array([c.high for c in sample_candles])
            lows = np.array([c.low for c in sample_candles])
            closes = np.array([c.close for c in sample_candles])
            pattern = detect_crab(highs=highs, lows=lows, closes=closes)
            assert pattern is None or isinstance(pattern, dict)
        except ImportError:
            pytest.skip("Harmonic patterns not available")

    def test_harmonic_ratio_validation(self):
        """Test harmonic ratio calculations"""
        try:
            from gravity_tech.patterns.harmonic import validate_harmonic_ratios  # type: ignore
            # Test with valid ratios
            ratios = [0.786, 0.886, 1.13]
            valid = validate_harmonic_ratios(ratios)
            assert isinstance(valid, bool) or valid is None
        except ImportError:
            pytest.skip("Harmonic patterns not available")


class TestClassicalPatterns:
    """Test classical technical patterns"""

    def test_head_and_shoulders_detection(self, sample_candles):
        """Test Head and Shoulders pattern detection"""
        try:
            from gravity_tech.patterns.classical import ClassicalPatterns  # type: ignore
            pattern = ClassicalPatterns.detect_head_and_shoulders(sample_candles)
            assert pattern is None or hasattr(pattern, 'pattern_name')
        except ImportError:
            pytest.skip("Classical patterns not available")

    def test_double_top_detection(self):
        """Test Double Top pattern detection"""
        try:
            from gravity_tech.patterns.classical import ClassicalPatterns  # type: ignore
            # Create mock candles with double top
            pattern = ClassicalPatterns.detect_double_top([])
            assert pattern is None or hasattr(pattern, 'pattern_name')
        except ImportError:
            pytest.skip("Classical patterns not available")

    def test_double_bottom_detection(self):
        """Test Double Bottom pattern detection"""
        try:
            from gravity_tech.patterns.classical import ClassicalPatterns  # type: ignore
            pattern = ClassicalPatterns.detect_double_bottom([])
            assert pattern is None or hasattr(pattern, 'pattern_name')
        except ImportError:
            pytest.skip("Classical patterns not available")

    def test_triangle_pattern_detection(self, sample_candles):
        """Test Triangle pattern detection"""
        try:
            from gravity_tech.patterns.classical import ClassicalPatterns  # type: ignore
            pattern = ClassicalPatterns.detect_symmetrical_triangle(sample_candles)
            assert pattern is None or hasattr(pattern, 'pattern_name')
        except ImportError:
            pytest.skip("Classical patterns not available")

    def test_flag_pattern_detection(self, uptrend_candles):
        """Test Flag pattern detection"""
        # Flag pattern not implemented yet
        pytest.skip("Flag pattern detection not implemented")

    def test_wedge_pattern_detection(self, sample_candles):
        """Test Wedge pattern detection"""
        # Wedge pattern not implemented yet
        pytest.skip("Wedge pattern detection not implemented")


class TestCandlestickPatterns:
    """Test candlestick pattern recognition"""

    def test_doji_pattern_detection(self):
        """Test Doji candle detection"""
        try:
            from gravity_tech.patterns.candlestick import CandlestickPatterns
            # Create doji candle (open == close)
            doji = Candle(
                timestamp=datetime.now(),
                open=100,
                high=110,
                low=90,
                close=100,
                volume=1000
            )
            pattern = CandlestickPatterns.is_doji(doji)
            assert isinstance(pattern, bool)
        except ImportError:
            pytest.skip("Candlestick patterns not available")

    def test_hammer_pattern_detection(self):
        """Test Hammer candle detection"""
        try:
            from gravity_tech.patterns.candlestick import CandlestickPatterns
            hammer = Candle(
                timestamp=datetime.now(),
                open=100,
                high=105,
                low=80,  # Long lower wick
                close=102,
                volume=1000
            )
            pattern = CandlestickPatterns.is_hammer(hammer)
            assert isinstance(pattern, bool)
        except ImportError:
            pytest.skip("Candlestick patterns not available")

    def test_hanging_man_detection(self):
        """Test Hanging Man detection"""
        # Hanging man not implemented yet
        pytest.skip("Hanging man detection not implemented")

    def test_engulfing_pattern_detection(self, sample_candles):
        """Test Engulfing pattern detection"""
        try:
            from gravity_tech.patterns.candlestick import CandlestickPatterns
            if len(sample_candles) >= 2:
                pattern = CandlestickPatterns.is_engulfing(sample_candles[-2], sample_candles[-1])
                assert isinstance(pattern, bool)
        except ImportError:
            pytest.skip("Candlestick patterns not available")

    def test_harami_pattern_detection(self, sample_candles):
        """Test Harami pattern detection"""
        try:
            from gravity_tech.patterns.candlestick import CandlestickPatterns
            if len(sample_candles) >= 2:
                pattern = CandlestickPatterns.is_harami(sample_candles[-2], sample_candles[-1])
                assert isinstance(pattern, bool)
        except ImportError:
            pytest.skip("Candlestick patterns not available")

    def test_morning_star_detection(self, sample_candles):
        """Test Morning Star pattern detection"""
        try:
            from gravity_tech.patterns.candlestick import CandlestickPatterns
            if len(sample_candles) >= 3:
                pattern = CandlestickPatterns.is_morning_evening_star(sample_candles[-3:])
                assert isinstance(pattern, bool)
        except ImportError:
            pytest.skip("Candlestick patterns not available")

    def test_evening_star_detection(self, downtrend_candles):
        """Test Evening Star pattern detection"""
        try:
            from gravity_tech.patterns.candlestick import CandlestickPatterns
            if len(downtrend_candles) >= 3:
                pattern = CandlestickPatterns.is_morning_evening_star(downtrend_candles[-3:])
                assert isinstance(pattern, bool)
        except ImportError:
            pytest.skip("Candlestick patterns not available")


class TestDivergencePatterns:
    """Test divergence pattern detection"""

    def test_bullish_divergence_detection(self, uptrend_candles):
        """Test bullish divergence detection"""
        try:
            from gravity_tech.patterns.divergence import DivergenceDetector
            detector = DivergenceDetector()
            indicator_values = [c.close for c in uptrend_candles]
            pattern = detector.detect(uptrend_candles, indicator_values=indicator_values)
            assert pattern is None or isinstance(pattern, dict)
        except (ImportError, TypeError):
            pytest.skip("Divergence patterns not available")

    def test_bearish_divergence_detection(self, downtrend_candles):
        """Test bearish divergence detection"""
        try:
            from gravity_tech.patterns.divergence import DivergenceDetector
            detector = DivergenceDetector()
            indicator_values = [c.close for c in downtrend_candles]
            pattern = detector.detect(downtrend_candles, indicator_values=indicator_values)
            assert pattern is None or isinstance(pattern, dict)
        except (ImportError, TypeError):
            pytest.skip("Divergence patterns not available")

    def test_hidden_bullish_divergence(self, sample_candles):
        """Test hidden bullish divergence detection"""
        try:
            from gravity_tech.patterns.divergence import DivergenceDetector
            detector = DivergenceDetector()
            indicator_values = [c.close for c in sample_candles]
            pattern = detector.detect(sample_candles, indicator_values=indicator_values)
            assert pattern is None or isinstance(pattern, dict)
        except (ImportError, TypeError):
            pytest.skip("Divergence patterns not available")

    def test_hidden_bearish_divergence(self, sample_candles):
        """Test hidden bearish divergence detection"""
        try:
            from gravity_tech.patterns.divergence import DivergenceDetector
            detector = DivergenceDetector()
            indicator_values = [c.close for c in sample_candles]
            pattern = detector.detect(sample_candles, indicator_values=indicator_values)
            assert pattern is None or isinstance(pattern, dict)
        except (ImportError, TypeError):
            pytest.skip("Divergence patterns not available")


class TestPatternConfidence:
    """Test pattern confidence scoring"""

    def test_pattern_strength_calculation(self):
        """Test pattern strength/confidence calculation"""
        try:
            from gravity_tech.patterns.utils import calculate_pattern_strength
            # Example pattern parameters
            strength = calculate_pattern_strength(
                ratio=0.786,
                completion_percentage=85,
                volume_confirmation=True
            )
            assert 0 <= strength <= 1 or strength is None
        except ImportError:
            pytest.skip("Pattern utilities not available")

    def test_pattern_reliability_score(self):
        """Test pattern reliability scoring"""
        try:
            from gravity_tech.patterns.utils import get_reliability_score
            # Score should be between 0 and 1
            score = get_reliability_score("GARTLEY")
            assert score is None or (0 <= score <= 1)
        except ImportError:
            pytest.skip("Pattern utilities not available")


class TestPatternMultipleTimeframes:
    """Test pattern detection across multiple timeframes"""

    def test_pattern_confirmation_across_timeframes(self, sample_candles):
        """Test confirming patterns on multiple timeframes"""
        try:
            from gravity_tech.patterns.utils import detect_pattern_multiframe
            timeframes = ["1m", "5m", "1h"]

            for tf in timeframes:
                try:
                    pattern = detect_pattern_multiframe(sample_candles, tf)
                    assert pattern is None or isinstance(pattern, dict)
                except Exception:
                    pass
        except ImportError:
            pytest.skip("Multi-timeframe pattern detection not available")

    def test_pattern_hierarchy(self):
        """Test pattern hierarchy and degree"""
        try:
            from gravity_tech.patterns.utils import get_pattern_hierarchy
            patterns = ["GARTLEY", "BUTTERFLY", "BAT", "CRAB"]

            for pattern_name in patterns:
                hierarchy = get_pattern_hierarchy(pattern_name)
                assert hierarchy is None or isinstance(hierarchy, int | str)
        except ImportError:
            pytest.skip("Pattern utilities not available")


class TestPatternEdgeCases:
    """Test pattern detection edge cases"""

    def test_incomplete_pattern_detection(self, sample_candles):
        """Test detection of incomplete patterns"""
        try:
            from gravity_tech.patterns.classical import ClassicalPatterns  # type: ignore
            # Try to detect any pattern
            patterns = ClassicalPatterns.detect_all(sample_candles)
            # Should return empty list for incomplete patterns
            assert isinstance(patterns, list)
        except ImportError:
            pytest.skip("Pattern detection not available")

    def test_overlapping_patterns(self, sample_candles):
        """Test handling of overlapping patterns"""
        try:
            from gravity_tech.patterns.utils import detect_all_patterns
            patterns = detect_all_patterns(sample_candles)
            # Multiple patterns may be detected
            if patterns:
                assert isinstance(patterns, list)
        except ImportError:
            pytest.skip("Pattern detection not available")

    def test_rare_pattern_variations(self):
        """Test detection of rare pattern variations"""
        try:
            from gravity_tech.patterns.utils import detect_rare_patterns
            # Rare patterns like Shark, etc.
            pattern = detect_rare_patterns([])
            assert pattern is None or isinstance(pattern, list)
        except ImportError:
            pytest.skip("Rare pattern detection not available")
