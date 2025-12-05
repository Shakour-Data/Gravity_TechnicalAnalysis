"""
XGBoost Pattern Recognition Classifier

Trains XGBoost classifier to:
- Identify harmonic pattern types
- Predict pattern success probability
- Calculate confidence scores
- Validate pattern quality

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import os
import pickle
from typing import Optional

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split


class PatternClassifier:
    """
    XGBoost-based classifier for harmonic pattern recognition.

    Uses gradient boosting to learn pattern characteristics and predict:
    - Pattern type (Gartley, Butterfly, Bat, Crab)
    - Success probability (will pattern complete successfully?)
    - Confidence score (how reliable is this pattern?)
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
        """
        Initialize pattern classifier.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate for gradient boosting
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

        # Pattern type classifier
        self.type_classifier = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='multi:softmax',
            num_class=4,  # 4 pattern types
            random_state=42
        )

        # Success probability regressor
        self.success_regressor = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='reg:squarederror',
            random_state=42
        )

        # Pattern classes
        self.pattern_classes = ['gartley', 'butterfly', 'bat', 'crab']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.pattern_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Feature importance
        self.feature_importance = None

        # Training metrics
        self.train_accuracy = None
        self.test_accuracy = None
        self.cv_scores = None

    def train(
        self,
        X: np.ndarray,
        y_type: list[str],
        y_success: Optional[np.ndarray] = None,
        test_size: float = 0.2,
        verbose: bool = True
    ) -> dict:
        """
        Train pattern classifier on labeled data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y_type: Pattern type labels
            y_success: Optional success labels (0-1) for success prediction
            test_size: Fraction of data for testing
            verbose: Print training progress

        Returns:
            Dictionary with training metrics
        """
        if len(X) == 0:
            raise ValueError("No training data provided")

        # Convert pattern types to indices
        y_type_idx = np.array([self.class_to_idx[t] for t in y_type])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_type_idx, test_size=test_size, random_state=42, stratify=y_type_idx
        )

        if verbose:
            print("Training pattern classifier...")
            print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Train type classifier
        self.type_classifier.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = self.type_classifier.predict(X_test)
        self.test_accuracy = accuracy_score(y_test, y_pred)
        self.train_accuracy = accuracy_score(y_train, self.type_classifier.predict(X_train))

        # Cross-validation
        self.cv_scores = cross_val_score(self.type_classifier, X, y_type_idx, cv=5)

        # Feature importance
        self.feature_importance = self.type_classifier.feature_importances_

        if verbose:
            print("\n✅ Training Complete!")
            print(f"Train Accuracy: {self.train_accuracy:.4f}")
            print(f"Test Accuracy: {self.test_accuracy:.4f}")
            print(f"CV Accuracy: {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std():.4f})")

            # Classification report
            print("\nClassification Report:")
            y_test_labels = [self.idx_to_class[idx] for idx in y_test]
            y_pred_labels = [self.idx_to_class[idx] for idx in y_pred]
            print(classification_report(y_test_labels, y_pred_labels))

        # Train success regressor if labels provided
        success_metrics = {}
        if y_success is not None and len(y_success) == len(X):
            X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
                X, y_success, test_size=test_size, random_state=42
            )

            self.success_regressor.fit(X_train_s, y_train_s)

            y_pred_s = self.success_regressor.predict(X_test_s)
            success_r2 = self.success_regressor.score(X_test_s, y_test_s)
            success_metrics['r2_score'] = success_r2

            if verbose:
                print(f"\n✅ Success Regressor R² Score: {success_r2:.4f}")

        return {
            'train_accuracy': self.train_accuracy,
            'test_accuracy': self.test_accuracy,
            'cv_mean': self.cv_scores.mean(),
            'cv_std': self.cv_scores.std(),
            'feature_importance': self.feature_importance,
            **success_metrics
        }

    def predict(self, X: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
        """
        Predict pattern type and confidence.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Tuple of (pattern_types, confidences, success_probabilities)
        """
        if not hasattr(self.type_classifier, 'n_features_in_'):
            raise ValueError("Model not trained. Call train() first.")

        # Predict pattern types
        y_pred_idx = self.type_classifier.predict(X)
        pattern_types = [self.idx_to_class[idx] for idx in y_pred_idx]

        # Get prediction probabilities (confidence)
        y_pred_proba = self.type_classifier.predict_proba(X)
        confidences = np.max(y_pred_proba, axis=1)  # Max probability

        # Predict success probability
        success_probs = self.success_regressor.predict(X)
        success_probs = np.clip(success_probs, 0, 1)

        return pattern_types, confidences, success_probs

    def predict_single(self, features: np.ndarray) -> dict:
        """
        Predict for a single pattern.

        Args:
            features: 1D feature array

        Returns:
            Dictionary with prediction results
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        pattern_types, confidences, success_probs = self.predict(features)

        return {
            'pattern_type': pattern_types[0],
            'confidence': float(confidences[0]),
            'success_probability': float(success_probs[0]),
            'quality_score': float(confidences[0] * success_probs[0])  # Combined score
        }

    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """
        Get feature importance scores.

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.feature_importance is None:
            return {}

        return {
            name: float(importance)
            for name, importance in zip(feature_names, self.feature_importance)
        }

    def save(self, filepath: str):
        """Save trained model to file."""
        model_data = {
            'type_classifier': self.type_classifier,
            'success_regressor': self.success_regressor,
            'pattern_classes': self.pattern_classes,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'feature_importance': self.feature_importance,
            'train_accuracy': self.train_accuracy,
            'test_accuracy': self.test_accuracy,
            'cv_scores': self.cv_scores
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, filepath: str) -> 'PatternClassifier':
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        classifier = cls()
        classifier.type_classifier = model_data['type_classifier']
        classifier.success_regressor = model_data['success_regressor']
        classifier.pattern_classes = model_data['pattern_classes']
        classifier.class_to_idx = model_data['class_to_idx']
        classifier.idx_to_class = model_data['idx_to_class']
        classifier.feature_importance = model_data['feature_importance']
        classifier.train_accuracy = model_data.get('train_accuracy')
        classifier.test_accuracy = model_data.get('test_accuracy')
        classifier.cv_scores = model_data.get('cv_scores')

        return classifier


class PatternConfidenceScorer:
    """
    Advanced confidence scoring system for harmonic patterns.

    Combines multiple signals:
    - Fibonacci ratio accuracy
    - ML model confidence
    - Volume confirmation
    - Momentum indicators
    - Historical success rate
    """

    def __init__(self):
        """Initialize confidence scorer."""
        self.weights = {
            'fibonacci_accuracy': 0.30,
            'ml_confidence': 0.25,
            'volume_confirmation': 0.15,
            'momentum_confirmation': 0.15,
            'geometric_quality': 0.15
        }

    def calculate_confidence(
        self,
        fibonacci_accuracy: float,
        ml_confidence: float,
        volume_confirmation: float,
        momentum_confirmation: float,
        geometric_quality: float
    ) -> dict:
        """
        Calculate overall pattern confidence score.

        All inputs should be in 0-1 range.

        Returns:
            Dictionary with confidence breakdown
        """
        # Weighted average
        total_score = (
            fibonacci_accuracy * self.weights['fibonacci_accuracy'] +
            ml_confidence * self.weights['ml_confidence'] +
            volume_confirmation * self.weights['volume_confirmation'] +
            momentum_confirmation * self.weights['momentum_confirmation'] +
            geometric_quality * self.weights['geometric_quality']
        )

        # Convert to 0-100 scale
        confidence = total_score * 100

        # Categorize confidence
        if confidence >= 80:
            category = "VERY_HIGH"
            signal = "STRONG_BUY" if momentum_confirmation > 0.6 else "STRONG_SELL"
        elif confidence >= 65:
            category = "HIGH"
            signal = "BUY" if momentum_confirmation > 0.5 else "SELL"
        elif confidence >= 50:
            category = "MEDIUM"
            signal = "NEUTRAL"
        elif confidence >= 35:
            category = "LOW"
            signal = "CAUTION"
        else:
            category = "VERY_LOW"
            signal = "AVOID"

        return {
            'confidence': float(confidence),
            'category': category,
            'signal': signal,
            'breakdown': {
                'fibonacci_accuracy': float(fibonacci_accuracy * 100),
                'ml_confidence': float(ml_confidence * 100),
                'volume_confirmation': float(volume_confirmation * 100),
                'momentum_confirmation': float(momentum_confirmation * 100),
                'geometric_quality': float(geometric_quality * 100)
            },
            'weights': self.weights
        }


def train_pattern_classifier(
    X_train: np.ndarray,
    y_train: list[str],
    y_success: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
) -> PatternClassifier:
    """
    Convenience function to train and optionally save pattern classifier.

    Args:
        X_train: Training feature matrix
        y_train: Training labels (pattern types)
        y_success: Optional success labels
        save_path: Optional path to save trained model

    Returns:
        Trained PatternClassifier
    """
    classifier = PatternClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )

    metrics = classifier.train(X_train, y_train, y_success, verbose=True)

    if save_path:
        classifier.save(save_path)
        print(f"\n✅ Model saved to: {save_path}")

    return classifier


def generate_synthetic_training_data(n_samples: int = 1000) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Generate synthetic training data for initial model training.

    This creates realistic-looking pattern features based on known
    Fibonacci ratios and pattern characteristics.

    Args:
        n_samples: Number of samples to generate

    Returns:
        Tuple of (X, y_type, y_success)
    """
    np.random.seed(42)

    X = []
    y_type = []
    y_success = []

    patterns = ['gartley', 'butterfly', 'bat', 'crab']

    for _ in range(n_samples):
        # Randomly select pattern type
        pattern = np.random.choice(patterns)

        # Generate features based on pattern type
        if pattern == 'gartley':
            fib_acc = np.random.beta(5, 2)  # High accuracy
            xab = np.random.normal(0.85, 0.1)
            abc = np.random.normal(0.85, 0.1)
            bcd = np.random.normal(0.85, 0.1)
            xad = np.random.normal(0.85, 0.1)
        elif pattern == 'butterfly':
            fib_acc = np.random.beta(4, 3)
            xab = np.random.normal(0.80, 0.12)
            abc = np.random.normal(0.75, 0.15)
            bcd = np.random.normal(0.75, 0.15)
            xad = np.random.normal(0.80, 0.12)
        elif pattern == 'bat':
            fib_acc = np.random.beta(5, 2)
            xab = np.random.normal(0.88, 0.08)
            abc = np.random.normal(0.88, 0.08)
            bcd = np.random.normal(0.88, 0.08)
            xad = np.random.normal(0.88, 0.08)
        else:  # crab
            fib_acc = np.random.beta(3, 3)
            xab = np.random.normal(0.75, 0.15)
            abc = np.random.normal(0.70, 0.15)
            bcd = np.random.normal(0.70, 0.15)
            xad = np.random.normal(0.75, 0.15)

        # Generate other features
        features = [
            np.clip(xab, 0, 1),  # xab_ratio_accuracy
            np.clip(abc, 0, 1),  # abc_ratio_accuracy
            np.clip(bcd, 0, 1),  # bcd_ratio_accuracy
            np.clip(xad, 0, 1),  # xad_ratio_accuracy
            np.random.beta(3, 2),  # pattern_symmetry
            np.random.beta(3, 3),  # pattern_slope
            np.random.beta(3, 3),  # xa_angle
            np.random.beta(3, 3),  # ab_angle
            np.random.beta(3, 3),  # bc_angle
            np.random.beta(3, 3),  # cd_angle
            np.random.beta(2, 3),  # pattern_duration
            np.random.beta(3, 3),  # xa_magnitude
            np.random.beta(3, 3),  # ab_magnitude
            np.random.beta(3, 3),  # bc_magnitude
            np.random.beta(3, 3),  # cd_magnitude
            np.random.beta(3, 2),  # volume_at_d
            np.random.beta(3, 3),  # volume_trend
            np.random.beta(3, 2),  # volume_confirmation
            np.random.beta(3, 3),  # rsi_at_d
            np.random.beta(3, 3),  # macd_at_d
            np.random.beta(3, 2),  # momentum_divergence
        ]

        X.append(features)
        y_type.append(pattern)

        # Success probability based on feature quality
        success_prob = np.mean([fib_acc, features[4], features[17]]) + np.random.normal(0, 0.1)
        y_success.append(np.clip(success_prob, 0, 1))

    return np.array(X, dtype=np.float32), y_type, np.array(y_success, dtype=np.float32)
