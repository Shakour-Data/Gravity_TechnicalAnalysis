"""
ML-Based Weight Optimization for Signal Calculation

This module uses machine learning to dynamically calculate optimal weights
for combining different indicator categories (Trend, Momentum, Cycle, Volume).

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from gravity_tech.models.schemas import Candle, IndicatorResult
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


class MLWeightOptimizer:
    """
    Machine Learning based weight optimizer for indicator signals.

    Uses historical data and future returns to learn optimal weights
    for combining Trend, Momentum, Cycle, and Volume indicators.
    """

    def __init__(self, model_type: str = "gradient_boosting"):
        """
        Initialize the ML weight optimizer

        Args:
            model_type: Type of ML model ('random_forest', 'gradient_boosting',
                       'ridge', 'lasso', 'adaptive')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.optimal_weights = None
        self.model_path = Path("models/ml_weights")
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the ML model based on type"""
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == "ridge":
            self.model = Ridge(alpha=1.0)
        elif self.model_type == "lasso":
            self.model = Lasso(alpha=0.1)
        elif self.model_type == "adaptive":
            # Ensemble of models
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                random_state=42
            )

    def prepare_features(self,
                        trend_indicators: list[IndicatorResult],
                        momentum_indicators: list[IndicatorResult],
                        cycle_indicators: list[IndicatorResult],
                        volume_indicators: list[IndicatorResult],
                        market_phase: Optional[str] = None) -> np.ndarray:
        """
        Prepare feature vector from indicators

        Args:
            trend_indicators: List of trend indicator results
            momentum_indicators: List of momentum indicator results
            cycle_indicators: List of cycle indicator results
            volume_indicators: List of volume indicator results
            market_phase: Optional market phase (accumulation, markup, etc.)

        Returns:
            Feature vector for ML model
        """
        features = []

        # Aggregate signals for each category
        def aggregate_signals(indicators: list[IndicatorResult]) -> dict:
            if not indicators:
                return {
                    'mean_score': 0.0,
                    'weighted_score': 0.0,
                    'confidence': 0.0,
                    'std_dev': 0.0,
                    'agreement': 0.0
                }

            scores = [ind.signal.get_score() for ind in indicators]
            confidences = [ind.confidence for ind in indicators]

            mean_score = np.mean(scores)
            weighted_score = np.average(scores, weights=confidences)
            mean_confidence = np.mean(confidences)
            std_dev = np.std(scores)

            # Agreement: how many indicators point in same direction
            positive = sum(1 for s in scores if s > 0)
            negative = sum(1 for s in scores if s < 0)
            agreement = max(positive, negative) / len(scores)

            return {
                'mean_score': mean_score,
                'weighted_score': weighted_score,
                'confidence': mean_confidence,
                'std_dev': std_dev,
                'agreement': agreement
            }

        # Get aggregates for each category
        trend_agg = aggregate_signals(trend_indicators)
        momentum_agg = aggregate_signals(momentum_indicators)
        cycle_agg = aggregate_signals(cycle_indicators)
        volume_agg = aggregate_signals(volume_indicators)

        # Build feature vector
        features.extend([
            # Trend features
            trend_agg['mean_score'],
            trend_agg['weighted_score'],
            trend_agg['confidence'],
            trend_agg['std_dev'],
            trend_agg['agreement'],

            # Momentum features
            momentum_agg['mean_score'],
            momentum_agg['weighted_score'],
            momentum_agg['confidence'],
            momentum_agg['std_dev'],
            momentum_agg['agreement'],

            # Cycle features
            cycle_agg['mean_score'],
            cycle_agg['weighted_score'],
            cycle_agg['confidence'],
            cycle_agg['std_dev'],
            cycle_agg['agreement'],

            # Volume features
            volume_agg['mean_score'],
            volume_agg['weighted_score'],
            volume_agg['confidence'],
            volume_agg['std_dev'],
            volume_agg['agreement'],

            # Cross-category features
            abs(trend_agg['weighted_score'] - momentum_agg['weighted_score']),
            abs(trend_agg['weighted_score'] - cycle_agg['weighted_score']),
            abs(momentum_agg['weighted_score'] - cycle_agg['weighted_score']),

            # Volume confirmation
            1.0 if trend_agg['weighted_score'] * volume_agg['weighted_score'] > 0 else 0.0,
        ])

        # Market phase encoding (if provided)
        if market_phase:
            phase_encoding = {
                'انباشت': [1, 0, 0, 0],  # Accumulation
                'صعود': [0, 1, 0, 0],     # Markup
                'توزیع': [0, 0, 1, 0],     # Distribution
                'نزول': [0, 0, 0, 1],     # Markdown
                'انتقال': [0.5, 0.5, 0.5, 0.5]  # Transition
            }
            features.extend(phase_encoding.get(market_phase, [0, 0, 0, 0]))

        return np.array(features).reshape(1, -1)

    def calculate_future_return(self, candles: list[Candle],
                               current_idx: int,
                               horizon: int = 10) -> float:
        """
        Calculate future return for training

        Args:
            candles: List of candles
            current_idx: Current candle index
            horizon: Look-ahead period

        Returns:
            Future return (percentage)
        """
        if current_idx + horizon >= len(candles):
            return 0.0

        current_price = candles[current_idx].close
        future_price = candles[current_idx + horizon].close

        return ((future_price - current_price) / current_price) * 100

    def train(self,
              training_data: list[dict],
              validation_split: float = 0.2) -> dict:
        """
        Train the ML model on historical data

        Args:
            training_data: List of dicts with 'features' and 'target' (future return)
            validation_split: Fraction of data for validation

        Returns:
            Training metrics
        """
        if not training_data:
            raise ValueError("Training data is empty")

        # Prepare data
        X = np.array([d['features'].flatten() for d in training_data])
        y = np.array([d['target'] for d in training_data])

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        val_score = self.model.score(X_val_scaled, y_val)

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=5, scoring='r2'
        )

        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_

            # Calculate optimal weights from feature importance
            # Features 0-4: Trend, 5-9: Momentum, 10-14: Cycle, 15-19: Volume
            trend_importance = np.sum(self.feature_importance[0:5])
            momentum_importance = np.sum(self.feature_importance[5:10])
            cycle_importance = np.sum(self.feature_importance[10:15])
            volume_importance = np.sum(self.feature_importance[15:20])

            total = trend_importance + momentum_importance + cycle_importance + volume_importance

            if total > 0:
                self.optimal_weights = {
                    'trend': float(trend_importance / total),
                    'momentum': float(momentum_importance / total),
                    'cycle': float(cycle_importance / total),
                    'volume': float(volume_importance / total)
                }

        metrics = {
            'train_r2': float(train_score),
            'val_r2': float(val_score),
            'cv_mean_r2': float(cv_scores.mean()),
            'cv_std_r2': float(cv_scores.std()),
            'n_samples': len(X),
            'n_features': X.shape[1],
            'model_type': self.model_type,
            'optimal_weights': self.optimal_weights,
            'timestamp': datetime.utcnow().isoformat()
        }

        return metrics

    def predict_weights(self, features: np.ndarray) -> dict[str, float]:
        """
        Predict optimal weights for current market conditions

        Args:
            features: Feature vector

        Returns:
            Dictionary with weights for each category
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict expected return
        predicted_return = self.model.predict(features_scaled)[0]

        # Use feature importance or learned weights
        if self.optimal_weights:
            weights = self.optimal_weights.copy()
        else:
            # Default weights
            weights = {
                'trend': 0.30,
                'momentum': 0.25,
                'cycle': 0.25,
                'volume': 0.20
            }

        # Adjust weights based on predicted return confidence
        # If model is confident (high absolute return), use ML weights
        # Otherwise, blend with default weights
        confidence = min(1.0, abs(predicted_return) / 5.0)  # 5% return = full confidence

        default_weights = {
            'trend': 0.30,
            'momentum': 0.25,
            'cycle': 0.25,
            'volume': 0.20
        }

        final_weights = {}
        for key in weights:
            final_weights[key] = (
                weights[key] * confidence +
                default_weights[key] * (1 - confidence)
            )

        # Normalize to sum to 1.0
        total = sum(final_weights.values())
        final_weights = {k: v/total for k, v in final_weights.items()}

        return final_weights

    def save_model(self, name: str = "ml_weights_model"):
        """Save trained model and scaler"""
        model_file = self.model_path / f"{name}.pkl"
        scaler_file = self.model_path / f"{name}_scaler.pkl"
        weights_file = self.model_path / f"{name}_weights.json"

        joblib.dump(self.model, model_file)
        joblib.dump(self.scaler, scaler_file)

        if self.optimal_weights:
            with open(weights_file, 'w') as f:
                json.dump(self.optimal_weights, f, indent=2)

        print(f"✅ Model saved to {model_file}")

    def load_model(self, name: str = "ml_weights_model"):
        """Load trained model and scaler"""
        model_file = self.model_path / f"{name}.pkl"
        scaler_file = self.model_path / f"{name}_scaler.pkl"
        weights_file = self.model_path / f"{name}_weights.json"

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)

        if weights_file.exists():
            with open(weights_file) as f:
                self.optimal_weights = json.load(f)

        print(f"✅ Model loaded from {model_file}")


class AdaptiveWeightCalculator:
    """
    Adaptive weight calculator that adjusts weights based on:
    - Market phase (Dow Theory)
    - Volatility regime
    - ML predictions
    """

    def __init__(self, use_ml: bool = True):
        self.use_ml = use_ml
        self.ml_optimizer = MLWeightOptimizer() if use_ml else None

    def calculate_adaptive_weights(self,
                                   trend_indicators: list[IndicatorResult],
                                   momentum_indicators: list[IndicatorResult],
                                   cycle_indicators: list[IndicatorResult],
                                   volume_indicators: list[IndicatorResult],
                                   market_phase: Optional[str] = None,
                                   volatility: Optional[float] = None) -> dict[str, float]:
        """
        Calculate adaptive weights based on market conditions

        Args:
            trend_indicators: Trend indicator results
            momentum_indicators: Momentum indicator results
            cycle_indicators: Cycle indicator results
            volume_indicators: Volume indicator results
            market_phase: Current market phase
            volatility: Current volatility level

        Returns:
            Optimal weights for each category
        """
        # Base weights (from Dow Theory and technical analysis principles)
        base_weights = {
            'trend': 0.30,
            'momentum': 0.25,
            'cycle': 0.25,
            'volume': 0.20
        }

        # Adjust based on market phase
        if market_phase:
            phase_adjustments = {
                'انباشت': {'volume': 0.15, 'momentum': 0.05},  # Volume more important
                'صعود': {'trend': 0.10, 'momentum': -0.05},    # Trend more important
                'توزیع': {'volume': 0.15, 'trend': -0.05},     # Volume more important
                'نزول': {'trend': 0.10, 'cycle': -0.05},       # Trend more important
                'انتقال': {}  # Keep base weights
            }

            adjustments = phase_adjustments.get(market_phase, {})
            for key, adj in adjustments.items():
                base_weights[key] += adj

        # Adjust based on volatility
        if volatility:
            if volatility > 0.03:  # High volatility (>3%)
                base_weights['volatility_adjusted'] = True
                base_weights['momentum'] += 0.05
                base_weights['trend'] -= 0.05
            elif volatility < 0.01:  # Low volatility (<1%)
                base_weights['volatility_adjusted'] = True
                base_weights['cycle'] += 0.05
                base_weights['momentum'] -= 0.05

        # Use ML prediction if available
        if self.use_ml and self.ml_optimizer and self.ml_optimizer.model:
            try:
                features = self.ml_optimizer.prepare_features(
                    trend_indicators,
                    momentum_indicators,
                    cycle_indicators,
                    volume_indicators,
                    market_phase
                )
                ml_weights = self.ml_optimizer.predict_weights(features)

                # Blend ML weights with base weights (70% ML, 30% base)
                final_weights = {}
                for key in ['trend', 'momentum', 'cycle', 'volume']:
                    final_weights[key] = (
                        ml_weights[key] * 0.7 +
                        base_weights[key] * 0.3
                    )

                return final_weights

            except Exception as e:
                print(f"⚠️ ML prediction failed: {e}. Using base weights.")

        # Normalize weights to sum to 1.0
        total = sum(base_weights[k] for k in ['trend', 'momentum', 'cycle', 'volume'])
        normalized_weights = {
            k: base_weights[k] / total
            for k in ['trend', 'momentum', 'cycle', 'volume']
        }

        return normalized_weights
