"""
Unit tests for src/gravity_tech/ml/pattern_classifier.py

Tests pattern classification functionality.
"""

import tempfile
from unittest.mock import MagicMock, patch

import numpy as np

from src.gravity_tech.ml.pattern_classifier import PatternClassifier


class TestPatternClassifier:
    """Test PatternClassifier functionality."""

    def test_init(self):
        """Test PatternClassifier initialization."""
        classifier = PatternClassifier(n_estimators=50, max_depth=4, learning_rate=0.05)

        assert classifier.n_estimators == 50
        assert classifier.max_depth == 4
        assert classifier.learning_rate == 0.05
        assert classifier.pattern_classes == ['gartley', 'butterfly', 'bat', 'crab']
        assert classifier.class_to_idx == {'gartley': 0, 'butterfly': 1, 'bat': 2, 'crab': 3}
        assert classifier.idx_to_class == {0: 'gartley', 1: 'butterfly', 2: 'bat', 3: 'crab'}

    def test_init_default_params(self):
        """Test PatternClassifier initialization with default parameters."""
        classifier = PatternClassifier()

        assert classifier.n_estimators == 100
        assert classifier.max_depth == 6
        assert classifier.learning_rate == 0.1

    @patch('src.gravity_tech.ml.pattern_classifier.cross_val_score')
    @patch('xgboost.XGBClassifier')
    @patch('xgboost.XGBRegressor')
    def test_train_basic(self, mock_regressor, mock_classifier, mock_cross_val):
        """Test basic training functionality."""
        # Mock the classifiers
        mock_type_classifier = MagicMock()
        mock_success_regressor = MagicMock()
        mock_classifier.return_value = mock_type_classifier
        mock_regressor.return_value = mock_success_regressor

        # Set up mock predictions and attributes
        def mock_predict(X):
            # Return predictions based on input size
            n_samples = X.shape[0]
            return np.random.randint(0, 4, n_samples)  # Random predictions for 4 classes

        mock_type_classifier.predict.side_effect = mock_predict
        mock_type_classifier.feature_importances_ = np.array([0.1, 0.2, 0.3])
        mock_success_regressor.predict.side_effect = lambda X: np.random.rand(X.shape[0])
        mock_cross_val.return_value = np.array([0.8, 0.85, 0.82, 0.87, 0.83])  # Mock CV scores

        # Mock training data
        X = np.random.rand(100, 20)
        y_type = ['gartley'] * 25 + ['butterfly'] * 25 + ['bat'] * 25 + ['crab'] * 25
        y_success = np.random.rand(100)

        classifier = PatternClassifier()
        result = classifier.train(X, y_type, y_success, test_size=0.3, verbose=False)

        # Verify the classifiers were trained
        mock_type_classifier.fit.assert_called_once()
        mock_success_regressor.fit.assert_called_once()

        # Verify result structure
        assert isinstance(result, dict)
        assert 'test_accuracy' in result
        assert 'r2_score' in result

    def test_train_without_success_labels(self):
        """Test training without success labels."""
        classifier = PatternClassifier()

        # Mock training data
        X = np.random.rand(50, 15)
        y_type = ['gartley'] * 25 + ['butterfly'] * 25

        result = classifier.train(X, y_type, verbose=False)

        # Should still return results
        assert isinstance(result, dict)
        assert 'test_accuracy' in result

    def test_predict(self):
        """Test prediction functionality."""
        classifier = PatternClassifier()

        # Mock trained classifiers
        classifier.type_classifier = MagicMock()
        classifier.success_regressor = MagicMock()

        # Mock predictions
        classifier.type_classifier.predict.return_value = np.array([0, 1, 2])
        classifier.type_classifier.predict_proba.return_value = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.7, 0.1, 0.1],
            [0.05, 0.1, 0.8, 0.05]
        ])
        classifier.success_regressor.predict.return_value = np.array([0.75, 0.82, 0.65])

        X = np.random.rand(3, 10)
        pattern_types, probabilities, success_probs = classifier.predict(X)

        assert pattern_types == ['gartley', 'butterfly', 'bat']
        assert len(probabilities) == 3
        assert len(success_probs) == 3
        assert success_probs[0] == 0.75

    def test_predict_single(self):
        """Test single prediction functionality."""
        classifier = PatternClassifier()

        # Mock trained classifiers
        classifier.type_classifier = MagicMock()
        classifier.success_regressor = MagicMock()

        # Mock predictions
        classifier.type_classifier.predict.return_value = np.array([1])
        classifier.type_classifier.predict_proba.return_value = np.array([[0.1, 0.8, 0.05, 0.05]])
        classifier.success_regressor.predict.return_value = np.array([0.85])

        features = np.random.rand(10)
        result = classifier.predict_single(features)

        assert isinstance(result, dict)
        assert result['pattern_type'] == 'butterfly'
        assert result['confidence'] == 0.8
        assert result['success_probability'] == 0.85

    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        classifier = PatternClassifier()

        # Mock trained classifier with feature importance
        classifier.feature_importance = np.array([0.1, 0.2, 0.15, 0.05])

        feature_names = ['feature1', 'feature2', 'feature3', 'feature4']
        importance = classifier.get_feature_importance(feature_names)

        assert isinstance(importance, dict)
        assert len(importance) == 4
        assert importance['feature2'] == 0.2

    def test_save_and_load(self):
        """Test saving and loading classifier."""
        classifier = PatternClassifier()

        # Set some attributes
        classifier.train_accuracy = 0.85
        classifier.feature_importance = np.array([0.3, 0.2, 0.1])

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            # Save
            classifier.save(tmp.name)

            # Load
            loaded_classifier = PatternClassifier.load(tmp.name)

            assert loaded_classifier.train_accuracy == 0.85
            np.testing.assert_array_equal(loaded_classifier.feature_importance, np.array([0.3, 0.2, 0.1]))
