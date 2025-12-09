"""
Unit tests for src/gravity_tech/ml/pattern_features.py

Tests pattern feature extraction functionality.
"""

from unittest.mock import MagicMock

import numpy as np

from src.gravity_tech.ml.pattern_features import PatternFeatureExtractor, PatternFeatures


class TestPatternFeatureExtractor:
    """Test PatternFeatureExtractor functionality."""

    def test_init(self):
        """Test PatternFeatureExtractor initialization."""
        extractor = PatternFeatureExtractor()

        # Should initialize without errors
        assert extractor is not None
        assert hasattr(extractor, 'extract_features')

    def test_extract_features_basic(self):
        """Test basic feature extraction."""
        extractor = PatternFeatureExtractor()

        # Mock a harmonic pattern
        mock_pattern = MagicMock()

        # Create mock points with index and price attributes
        class MockPoint:
            def __init__(self, index, price):
                self.index = index
                self.price = price

        mock_pattern.points = {
            'X': MockPoint(0, 100.0),
            'A': MockPoint(10, 90.0),
            'B': MockPoint(20, 110.0),
            'C': MockPoint(30, 95.0),
            'D': MockPoint(40, 105.0)
        }
        mock_pattern.direction = MagicMock()
        mock_pattern.direction.value = 'bullish'
        mock_pattern.confidence = 85.0
        # Mock pattern_type as an enum-like object
        mock_pattern.pattern_type = MagicMock()
        mock_pattern.pattern_type.value = 'gartley'
        # Mock ratios
        mock_pattern.ratios = {
            'XA_BC': 0.62,
            'AB_CD': 0.78,
            'XA_AD': 0.79
        }

        # Mock price data
        highs = np.array([105.0, 103.0, 108.0, 100.0, 110.0, 108.0, 113.0, 102.0, 111.0, 109.0] * 5)
        lows = np.array([95.0, 93.0, 97.0, 90.0, 100.0, 98.0, 103.0, 92.0, 101.0, 99.0] * 5)
        closes = np.array([100.0, 98.0, 102.0, 95.0, 105.0, 103.0, 108.0, 97.0, 106.0, 104.0] * 5)
        volume = np.array([1000, 1200, 800, 1500, 1100, 1300, 900, 1600, 1000, 1400] * 5)

        features = extractor.extract_features(mock_pattern, highs, lows, closes, volume)

        assert isinstance(features, PatternFeatures)
        assert hasattr(features, 'xab_ratio_accuracy')
        assert hasattr(features, 'pattern_symmetry')
        assert hasattr(features, 'pattern_duration')

    def test_calculate_angle(self):
        """Test angle calculation."""
        extractor = PatternFeatureExtractor()

        # Horizontal line (0 degrees)
        angle = extractor._calculate_angle(0, 100, 10, 100)
        assert abs(angle) < 0.1  # Nearly 0

        # Vertical line (90 degrees)
        angle = extractor._calculate_angle(0, 100, 0, 110)
        assert abs(angle - 90) < 1

    def test_normalize_angle(self):
        """Test angle normalization."""
        extractor = PatternFeatureExtractor()

        # Test various angles
        assert extractor._normalize_angle(0) == 0.5  # Center (horizontal)
        assert extractor._normalize_angle(90) == 1.0  # Vertical up
        assert extractor._normalize_angle(-90) == 0.0  # Vertical down

    def test_normalize_slope(self):
        """Test slope normalization."""
        extractor = PatternFeatureExtractor()

        # Test various slopes
        assert extractor._normalize_slope(0) == 0.5  # Flat
        assert extractor._normalize_slope(1) > 0.5   # Positive
        assert extractor._normalize_slope(-1) < 0.5  # Negative

    def test_calculate_rsi(self):
        """Test RSI calculation."""
        extractor = PatternFeatureExtractor()

        # Create test price data with upward trend
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                          110, 111, 112, 113, 114, 115, 116, 117, 118, 119])

        rsi = extractor._calculate_rsi(prices, period=14)

        # RSI calculation starts after the period
        assert len(rsi) == len(prices) - 14
        assert rsi[-1] > 50  # Should be high RSI for upward trend

    def test_calculate_ema(self):
        """Test EMA calculation."""
        extractor = PatternFeatureExtractor()

        prices = np.array([100, 101, 102, 103, 104, 105])
        ema = extractor._calculate_ema(prices, period=3)

        assert len(ema) == len(prices)
        # EMA should be close to the recent prices
        assert abs(ema[-1] - 104.5) < 1

    def test_calculate_momentum(self):
        """Test momentum calculation."""
        extractor = PatternFeatureExtractor()

        # Upward momentum
        prices = np.array([100, 102, 104, 106, 108, 110, 112, 114, 116, 118])
        momentum = extractor._calculate_momentum(prices, period=5)

        assert momentum > 0  # Positive momentum

        # Downward momentum
        prices = np.array([120, 118, 116, 114, 112, 110, 108, 106, 104, 102])
        momentum = extractor._calculate_momentum(prices, period=5)

        assert momentum < 0  # Negative momentum

    def test_features_to_array(self):
        """Test converting features to array."""
        extractor = PatternFeatureExtractor()

        # Create mock features
        features = PatternFeatures(
            xab_ratio_accuracy=0.95,
            abc_ratio_accuracy=0.88,
            bcd_ratio_accuracy=0.92,
            xad_ratio_accuracy=0.85,
            pattern_symmetry=0.75,
            pattern_slope=0.3,
            xa_angle=0.2,
            ab_angle=0.4,
            bc_angle=0.6,
            cd_angle=0.8,
            pattern_duration=50,
            xa_magnitude=5.0,
            ab_magnitude=3.0,
            bc_magnitude=2.5,
            cd_magnitude=4.0,
            volume_at_d=1200.0,
            volume_trend=0.8,
            volume_confirmation=0.9,
            rsi_at_d=65.0,
            macd_at_d=0.5,
            momentum_divergence=0.7,
            pattern_type='gartley',
            direction='bullish',
            confidence=0.85
        )

        feature_array = extractor.features_to_array(features)

        assert isinstance(feature_array, np.ndarray)
        assert len(feature_array) > 20  # Should have many features

    def test_get_feature_names(self):
        """Test getting feature names."""
        extractor = PatternFeatureExtractor()

        names = extractor.get_feature_names()

        assert isinstance(names, list)
        assert len(names) > 20  # Should have many feature names
        assert 'xab_ratio_accuracy' in names
        assert 'pattern_symmetry' in names
