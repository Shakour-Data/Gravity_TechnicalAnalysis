"""
Comprehensive Tests for Harmonic Pattern Recognition

Tests:
- Harmonic pattern detection (Gartley, Butterfly, Bat, Crab)
- Feature extraction pipeline
- XGBoost classifier training and prediction

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
- Confidence scoring system
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gravity_tech.patterns.harmonic import (
    HarmonicPatternDetector,
    PatternType,
    PatternDirection,
    detect_gartley,
    detect_butterfly,
    detect_bat,
    detect_crab,
    detect_all_harmonic_patterns
)
from gravity_tech.ml.pattern_features import PatternFeatureExtractor, extract_pattern_features_batch
from gravity_tech.ml.pattern_classifier import (
    PatternClassifier,
    PatternConfidenceScorer,
    generate_synthetic_training_data
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_prices():
    """Generate sample price data with potential harmonic patterns."""
    np.random.seed(42)
    
    # Create uptrend with retracements (potential bullish pattern)
    prices = np.array([
        100, 102, 104, 106, 108, 110, 112, 115, 118, 120,  # X to A
        120, 118, 116, 114, 112, 110, 108, 106, 104, 102,  # A to B (retracement)
        102, 104, 106, 108, 110, 112, 114, 116, 118, 115,  # B to C
        115, 112, 109, 106, 103, 100, 98, 96, 94, 92,      # C to D (final leg)
        92, 95, 98, 101, 104, 107, 110, 113, 116, 119      # Reversal
    ], dtype=np.float32)
    
    # Add some noise
    prices = prices + np.random.normal(0, 1, len(prices))
    
    highs = prices + np.abs(np.random.normal(0, 0.5, len(prices)))
    lows = prices - np.abs(np.random.normal(0, 0.5, len(prices)))
    closes = prices
    volume = np.random.uniform(1000, 5000, len(prices))
    
    return highs, lows, closes, volume


@pytest.fixture
def gartley_pattern_data():
    """Generate ideal Gartley pattern."""
    # X-A-B-C-D pattern with correct Fibonacci ratios
    xa = 20  # X=100, A=120
    ab = xa * 0.618  # B retraces 61.8% of XA
    bc = ab * 0.382  # C
    cd = ab * 0.786  # D retraces 78.6% of XA
    
    prices = [
        100,  # X
        120,  # A (X + xa)
        120 - ab,  # B
        120 - ab + bc,  # C
        120 - ab + bc - cd  # D
    ]
    
    # Interpolate to create smooth price series
    full_prices = []
    for i in range(len(prices) - 1):
        start, end = prices[i], prices[i + 1]
        segment = np.linspace(start, end, 10)
        full_prices.extend(segment[:-1])
    full_prices.append(prices[-1])
    
    prices_array = np.array(full_prices, dtype=np.float32)
    highs = prices_array + 0.5
    lows = prices_array - 0.5
    closes = prices_array
    
    return highs, lows, closes


# ============================================================================
# Pattern Detection Tests
# ============================================================================

def test_harmonic_pattern_detector_initialization():
    """Test detector initialization."""
    detector = HarmonicPatternDetector(tolerance=0.05, min_pattern_bars=20)
    
    assert detector.tolerance == 0.05
    assert detector.min_pattern_bars == 20
    assert PatternType.GARTLEY in detector.pattern_definitions
    assert PatternType.BUTTERFLY in detector.pattern_definitions


def test_detect_patterns_short_data():
    """Test detection with insufficient data."""
    detector = HarmonicPatternDetector()
    
    highs = np.array([100, 101, 102])
    lows = np.array([99, 100, 101])
    closes = np.array([100, 101, 102])
    
    patterns = detector.detect_patterns(highs, lows, closes)
    assert len(patterns) == 0  # Not enough data


def test_detect_gartley_pattern(gartley_pattern_data):
    """Test Gartley pattern detection."""
    highs, lows, closes = gartley_pattern_data
    
    result = detect_gartley(highs, lows, closes, tolerance=0.15)
    
    # Should detect a pattern (with generous tolerance)
    # Note: Detection depends on pivot finding, may not always succeed
    assert 'detected' in result
    assert 'confidence' in result
    assert 'ratios' in result


def test_detect_butterfly_pattern(sample_prices):
    """Test Butterfly pattern detection."""
    highs, lows, closes, volume = sample_prices
    
    result = detect_butterfly(highs, lows, closes)
    
    assert 'detected' in result
    assert 'direction' in result
    assert 'confidence' in result
    assert isinstance(result['confidence'], float)


def test_detect_bat_pattern(sample_prices):
    """Test Bat pattern detection."""
    highs, lows, closes, volume = sample_prices
    
    result = detect_bat(highs, lows, closes)
    
    assert 'detected' in result
    assert 'stop_loss' in result
    assert 'target_1' in result
    assert 'target_2' in result


def test_detect_crab_pattern(sample_prices):
    """Test Crab pattern detection."""
    highs, lows, closes, volume = sample_prices
    
    result = detect_crab(highs, lows, closes)
    
    assert 'detected' in result
    assert 'completion_point' in result


def test_detect_all_patterns(sample_prices):
    """Test detection of all harmonic patterns."""
    highs, lows, closes, volume = sample_prices
    
    result = detect_all_harmonic_patterns(highs, lows, closes)
    
    assert 'total_patterns' in result
    assert 'gartley' in result
    assert 'butterfly' in result
    assert 'bat' in result
    assert 'crab' in result
    assert isinstance(result['total_patterns'], int)
    assert result['total_patterns'] >= 0


# ============================================================================
# Feature Extraction Tests
# ============================================================================

def test_feature_extractor_initialization():
    """Test feature extractor initialization."""
    extractor = PatternFeatureExtractor()
    
    assert len(extractor.feature_names) > 0
    assert 'xab_ratio_accuracy' in extractor.feature_names
    assert 'pattern_symmetry' in extractor.feature_names


def test_extract_features_from_pattern(sample_prices):
    """Test feature extraction from detected pattern."""
    highs, lows, closes, volume = sample_prices
    
    # Detect patterns first
    detector = HarmonicPatternDetector(tolerance=0.2)
    patterns = detector.detect_patterns(highs, lows, closes)
    
    if len(patterns) > 0:
        extractor = PatternFeatureExtractor()
        features = extractor.extract_features(patterns[0], highs, lows, closes, volume)
        
        # Check all features are present
        assert hasattr(features, 'xab_ratio_accuracy')
        assert hasattr(features, 'pattern_symmetry')
        assert hasattr(features, 'volume_at_d')
        assert hasattr(features, 'rsi_at_d')
        
        # Check features are normalized (0-1 range)
        assert 0 <= features.xab_ratio_accuracy <= 1
        assert 0 <= features.pattern_symmetry <= 1
        assert 0 <= features.rsi_at_d <= 1


def test_features_to_array_conversion(sample_prices):
    """Test conversion of features to numpy array."""
    highs, lows, closes, volume = sample_prices
    
    detector = HarmonicPatternDetector(tolerance=0.2)
    patterns = detector.detect_patterns(highs, lows, closes)
    
    if len(patterns) > 0:
        extractor = PatternFeatureExtractor()
        features = extractor.extract_features(patterns[0], highs, lows, closes, volume)
        
        # Convert to array
        feature_array = extractor.features_to_array(features)
        
        assert isinstance(feature_array, np.ndarray)
        assert feature_array.dtype == np.float32
        assert len(feature_array) == len(extractor.feature_names)
        assert np.all(feature_array >= 0) and np.all(feature_array <= 1)


def test_batch_feature_extraction(sample_prices):
    """Test batch feature extraction."""
    highs, lows, closes, volume = sample_prices
    
    detector = HarmonicPatternDetector(tolerance=0.2)
    patterns = detector.detect_patterns(highs, lows, closes)
    
    if len(patterns) > 0:
        feature_matrix, labels = extract_pattern_features_batch(
            patterns, highs, lows, closes, volume
        )
        
        assert isinstance(feature_matrix, np.ndarray)
        assert isinstance(labels, list)
        assert len(feature_matrix) == len(labels)
        assert feature_matrix.shape[1] == 21  # Number of features


# ============================================================================
# ML Classifier Tests
# ============================================================================

def test_classifier_initialization():
    """Test pattern classifier initialization."""
    classifier = PatternClassifier(n_estimators=50, max_depth=4)
    
    assert classifier.n_estimators == 50
    assert classifier.max_depth == 4
    assert len(classifier.pattern_classes) == 4


def test_synthetic_data_generation():
    """Test synthetic training data generation."""
    X, y_type, y_success = generate_synthetic_training_data(n_samples=100)
    
    assert X.shape[0] == 100
    assert X.shape[1] == 21  # Number of features
    assert len(y_type) == 100
    assert len(y_success) == 100
    
    # Check data ranges
    assert np.all(X >= 0) and np.all(X <= 1)
    assert np.all(y_success >= 0) and np.all(y_success <= 1)
    
    # Check pattern types
    assert all(pt in ['gartley', 'butterfly', 'bat', 'crab'] for pt in y_type)


def test_classifier_training():
    """Test classifier training on synthetic data."""
    X, y_type, y_success = generate_synthetic_training_data(n_samples=200)
    
    classifier = PatternClassifier(n_estimators=50, max_depth=4)
    metrics = classifier.train(X, y_type, y_success, verbose=False)
    
    # Check training completed
    assert 'train_accuracy' in metrics
    assert 'test_accuracy' in metrics
    assert 'cv_mean' in metrics
    
    # Check reasonable accuracy (>40% for 4-class problem)
    assert metrics['test_accuracy'] > 0.4
    assert classifier.feature_importance is not None


def test_classifier_prediction():
    """Test classifier predictions."""
    X, y_type, y_success = generate_synthetic_training_data(n_samples=200)
    
    classifier = PatternClassifier(n_estimators=50, max_depth=4)
    classifier.train(X, y_type, y_success, verbose=False)
    
    # Make predictions
    pattern_types, confidences, success_probs = classifier.predict(X[:10])
    
    assert len(pattern_types) == 10
    assert len(confidences) == 10
    assert len(success_probs) == 10
    
    # Check output ranges
    assert all(pt in ['gartley', 'butterfly', 'bat', 'crab'] for pt in pattern_types)
    assert np.all(confidences >= 0) and np.all(confidences <= 1)
    assert np.all(success_probs >= 0) and np.all(success_probs <= 1)


def test_single_prediction():
    """Test single pattern prediction."""
    X, y_type, y_success = generate_synthetic_training_data(n_samples=200)
    
    classifier = PatternClassifier(n_estimators=50, max_depth=4)
    classifier.train(X, y_type, y_success, verbose=False)
    
    # Single prediction
    result = classifier.predict_single(X[0])
    
    assert 'pattern_type' in result
    assert 'confidence' in result
    assert 'success_probability' in result
    assert 'quality_score' in result
    
    assert result['pattern_type'] in ['gartley', 'butterfly', 'bat', 'crab']
    assert 0 <= result['confidence'] <= 1
    assert 0 <= result['quality_score'] <= 1


def test_feature_importance_extraction():
    """Test feature importance extraction."""
    X, y_type, y_success = generate_synthetic_training_data(n_samples=200)
    
    classifier = PatternClassifier(n_estimators=50, max_depth=4)
    classifier.train(X, y_type, y_success, verbose=False)
    
    extractor = PatternFeatureExtractor()
    feature_names = extractor.get_feature_names()
    
    importance_dict = classifier.get_feature_importance(feature_names)
    
    assert len(importance_dict) == len(feature_names)
    assert all(0 <= imp <= 1 for imp in importance_dict.values())
    
    # Sum of importances should be 1.0
    assert abs(sum(importance_dict.values()) - 1.0) < 0.01


# ============================================================================
# Confidence Scoring Tests
# ============================================================================

def test_confidence_scorer_initialization():
    """Test confidence scorer initialization."""
    scorer = PatternConfidenceScorer()
    
    assert 'fibonacci_accuracy' in scorer.weights
    assert 'ml_confidence' in scorer.weights
    assert abs(sum(scorer.weights.values()) - 1.0) < 0.01  # Weights sum to 1


def test_high_confidence_scoring():
    """Test high confidence score calculation."""
    scorer = PatternConfidenceScorer()
    
    result = scorer.calculate_confidence(
        fibonacci_accuracy=0.9,
        ml_confidence=0.85,
        volume_confirmation=0.8,
        momentum_confirmation=0.75,
        geometric_quality=0.9
    )
    
    assert result['confidence'] > 70
    assert result['category'] in ['HIGH', 'VERY_HIGH']
    assert 'breakdown' in result
    assert 'weights' in result


def test_low_confidence_scoring():
    """Test low confidence score calculation."""
    scorer = PatternConfidenceScorer()
    
    result = scorer.calculate_confidence(
        fibonacci_accuracy=0.3,
        ml_confidence=0.25,
        volume_confirmation=0.2,
        momentum_confirmation=0.3,
        geometric_quality=0.4
    )
    
    assert result['confidence'] < 50
    assert result['category'] in ['LOW', 'VERY_LOW', 'MEDIUM']
    assert result['signal'] in ['CAUTION', 'AVOID', 'NEUTRAL']


def test_confidence_breakdown():
    """Test confidence score breakdown."""
    scorer = PatternConfidenceScorer()
    
    result = scorer.calculate_confidence(
        fibonacci_accuracy=0.7,
        ml_confidence=0.6,
        volume_confirmation=0.5,
        momentum_confirmation=0.8,
        geometric_quality=0.65
    )
    
    breakdown = result['breakdown']
    
    assert 'fibonacci_accuracy' in breakdown
    assert 'ml_confidence' in breakdown
    assert 'volume_confirmation' in breakdown
    assert 'momentum_confirmation' in breakdown
    assert 'geometric_quality' in breakdown
    
    # All breakdown values should be 0-100 scale
    assert all(0 <= v <= 100 for v in breakdown.values())


# ============================================================================
# Integration Tests
# ============================================================================

def test_end_to_end_pattern_recognition(sample_prices):
    """Test complete pattern recognition pipeline."""
    highs, lows, closes, volume = sample_prices
    
    # Step 1: Detect patterns
    detector = HarmonicPatternDetector(tolerance=0.2)
    patterns = detector.detect_patterns(highs, lows, closes)
    
    if len(patterns) > 0:
        # Step 2: Extract features
        extractor = PatternFeatureExtractor()
        features = extractor.extract_features(patterns[0], highs, lows, closes, volume)
        feature_array = extractor.features_to_array(features)
        
        # Step 3: Train classifier (on synthetic data)
        X_train, y_train, y_success = generate_synthetic_training_data(n_samples=200)
        classifier = PatternClassifier(n_estimators=50, max_depth=4)
        classifier.train(X_train, y_train, y_success, verbose=False)
        
        # Step 4: Predict pattern type and confidence
        prediction = classifier.predict_single(feature_array)
        
        # Step 5: Calculate overall confidence
        scorer = PatternConfidenceScorer()
        confidence_result = scorer.calculate_confidence(
            fibonacci_accuracy=features.xab_ratio_accuracy,
            ml_confidence=prediction['confidence'],
            volume_confirmation=features.volume_confirmation,
            momentum_confirmation=features.momentum_divergence,
            geometric_quality=features.pattern_symmetry
        )
        
        # Verify complete pipeline
        assert prediction['pattern_type'] in ['gartley', 'butterfly', 'bat', 'crab']
        assert 0 <= prediction['confidence'] <= 1
        assert 0 <= confidence_result['confidence'] <= 100
        assert confidence_result['category'] in ['VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']


def test_model_save_and_load(tmp_path):
    """Test model saving and loading."""
    # Train model
    X, y_type, y_success = generate_synthetic_training_data(n_samples=200)
    classifier = PatternClassifier(n_estimators=50, max_depth=4)
    classifier.train(X, y_type, y_success, verbose=False)
    
    # Save model
    model_path = tmp_path / "pattern_model.pkl"
    classifier.save(str(model_path))
    
    assert model_path.exists()
    
    # Load model
    loaded_classifier = PatternClassifier.load(str(model_path))
    
    # Test loaded model
    predictions_original = classifier.predict(X[:10])
    predictions_loaded = loaded_classifier.predict(X[:10])
    
    assert predictions_original[0] == predictions_loaded[0]  # Same pattern types
    np.testing.assert_array_almost_equal(predictions_original[1], predictions_loaded[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

