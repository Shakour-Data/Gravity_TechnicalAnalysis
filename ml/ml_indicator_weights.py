"""
ML-Based Weight Learning for 10 Trend Indicators

This module uses Gradient Boosting (LightGBM/XGBoost) to learn optimal weights
for combining 10 trend indicators based on their ability to predict future returns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path
import json

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt


class IndicatorWeightLearner:
    """
    Learn optimal weights for 7 available trend indicators using ML
    """
    
    # 7 Available Trend Indicators
    INDICATORS = ['sma', 'ema', 'wma', 'dema', 'tema', 'macd', 'adx']
    
    def __init__(self, model_type: str = "lightgbm"):
        """
        Initialize learner
        
        Args:
            model_type: "lightgbm", "xgboost", or "sklearn"
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = {}
        self.learned_weights = {}
        self.model_path = Path("models/ml_weights")
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Validate model availability
        if model_type == "lightgbm" and not LIGHTGBM_AVAILABLE:
            print("⚠️ LightGBM not available, falling back to sklearn")
            self.model_type = "sklearn"
        elif model_type == "xgboost" and not XGBOOST_AVAILABLE:
            print("⚠️ XGBoost not available, falling back to sklearn")
            self.model_type = "sklearn"
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """
        Train model to predict future returns from indicator signals
        
        Args:
            X: Features (indicator signals, confidences, weighted signals)
            y: Labels (future returns)
            test_size: Test set ratio
            random_state: Random seed
            
        Returns:
            Training metrics
        """
        print(f"\n🎯 Training {self.model_type.upper()} model...")
        print(f"   Features: {X.shape[1]}")
        print(f"   Samples: {X.shape[0]}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        print(f"   Train: {X_train.shape[0]} samples")
        print(f"   Test:  {X_test.shape[0]} samples")
        
        # Train model
        if self.model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                random_state=random_state,
                verbose=-1
            )
        elif self.model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=random_state,
                verbosity=0
            )
        else:  # sklearn
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=random_state
            )
        
        # Fit model
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred)
        }
        
        print(f"\n📊 Training Results:")
        print(f"   Train R²: {metrics['train_r2']:.4f}")
        print(f"   Test R²:  {metrics['test_r2']:.4f}")
        print(f"   Train MAE: {metrics['train_mae']:.6f}")
        print(f"   Test MAE:  {metrics['test_mae']:.6f}")
        
        # Extract feature importance
        self._extract_feature_importance(X.columns)
        
        # Calculate learned weights
        self._calculate_weights_from_importance()
        
        return metrics
    
    def _extract_feature_importance(self, feature_names: List[str]):
        """
        Extract feature importance from trained model
        """
        if self.model_type in ["lightgbm", "xgboost"]:
            importance = self.model.feature_importances_
        else:
            importance = self.model.feature_importances_
        
        # Create importance dict
        self.feature_importance = {
            name: float(imp) 
            for name, imp in zip(feature_names, importance)
        }
        
        # Sort by importance
        sorted_importance = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        print(f"\n🔝 Top 10 Most Important Features:")
        for i, (feature, imp) in enumerate(sorted_importance[:10], 1):
            print(f"   {i:2d}. {feature:25s}: {imp:.4f}")
    
    def _calculate_weights_from_importance(self):
        """
        Calculate indicator weights from feature importance
        
        For each indicator, we have 3 features:
        - {indicator}_signal
        - {indicator}_confidence  
        - {indicator}_weighted
        
        We sum the importance of all 3 features for each indicator
        """
        indicator_importance = {}
        
        for indicator in self.INDICATORS:
            # Sum importance of all features for this indicator
            total_imp = 0.0
            
            for suffix in ['_signal', '_confidence', '_weighted']:
                feature_name = f'{indicator}{suffix}'
                if feature_name in self.feature_importance:
                    total_imp += self.feature_importance[feature_name]
            
            indicator_importance[indicator] = total_imp
        
        # Normalize to weights (sum to 1.0)
        total = sum(indicator_importance.values())
        
        if total > 0:
            self.learned_weights = {
                ind: imp / total 
                for ind, imp in indicator_importance.items()
            }
        else:
            # Fallback to equal weights
            self.learned_weights = {
                ind: 1.0 / len(self.INDICATORS) 
                for ind in self.INDICATORS
            }
        
        print(f"\n⚖️  Learned Indicator Weights:")
        sorted_weights = sorted(
            self.learned_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for indicator, weight in sorted_weights:
            bar = '█' * int(weight * 100)
            print(f"   {indicator:15s}: {weight:.4f} {bar}")
        
        # Verify sum
        weight_sum = sum(self.learned_weights.values())
        print(f"\n   Total weight: {weight_sum:.6f} (should be 1.0)")
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get learned weights for 7 indicators
        
        Returns:
            Dictionary mapping indicator name to weight
        """
        return self.learned_weights.copy()
    
    def get_weights_with_confidence(self, X_test, y_test) -> Dict[str, Dict[str, float]]:
        """
        Get learned weights with confidence metrics
        
        Returns:
            Dictionary with weights, R², MAE, and confidence intervals
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import r2_score, mean_absolute_error
        import numpy as np
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Calculate residuals for confidence intervals
        residuals = y_test - y_pred
        std_residual = np.std(residuals)
        
        # 95% confidence interval
        confidence_95 = 1.96 * std_residual
        
        # Weight reliability based on R²
        # R² > 0.7 → high reliability
        # R² 0.4-0.7 → medium reliability
        # R² < 0.4 → low reliability
        if r2 > 0.7:
            weight_reliability = "high"
            reliability_factor = 1.0
        elif r2 > 0.4:
            weight_reliability = "medium"
            reliability_factor = 0.7
        else:
            weight_reliability = "low"
            reliability_factor = 0.4
        
        result = {
            'weights': self.learned_weights.copy(),
            'metrics': {
                'r2_score': r2,
                'mae': mae,
                'confidence_interval_95': confidence_95,
                'reliability': weight_reliability,
                'reliability_factor': reliability_factor
            },
            'adjusted_weights': {
                ind: weight * reliability_factor
                for ind, weight in self.learned_weights.items()
            }
        }
        
        return result
    
    def save_model(self, filename: str = "indicator_weights_model.pkl"):
        """
        Save trained model and weights with confidence metrics
        """
        save_path = self.model_path / filename
        
        # Save model
        joblib.dump(self.model, save_path)
        
        # Save weights with metrics
        weights_data = {
            'weights': self.learned_weights,
            'feature_importance': self.feature_importance,
            'timestamp': datetime.utcnow().isoformat(),
            'model_type': self.model_type
        }
        
        weights_path = self.model_path / "indicator_weights.json"
        with open(weights_path, 'w') as f:
            json.dump(weights_data, f, indent=2)
        
        # Save feature importance
        importance_path = self.model_path / "indicator_feature_importance.json"
        with open(importance_path, 'w') as f:
            json.dump(self.feature_importance, f, indent=2)
        
        print(f"\n💾 Saved:")
        print(f"   Model:      {save_path}")
        print(f"   Weights:    {weights_path}")
        print(f"   Importance: {importance_path}")
    
    def load_model(self, filename: str = "indicator_weights_model.pkl"):
        """
        Load trained model and weights
        """
        load_path = self.model_path / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")
        
        self.model = joblib.load(load_path)
        
        # Load weights
        weights_path = self.model_path / "indicator_weights.json"
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                self.learned_weights = json.load(f)
        
        # Load feature importance
        importance_path = self.model_path / "indicator_feature_importance.json"
        if importance_path.exists():
            with open(importance_path, 'r') as f:
                self.feature_importance = json.load(f)
        
        print(f"✅ Loaded model from {load_path}")
    
    def compare_with_equal_weights(self, X: pd.DataFrame, y: pd.Series):
        """
        Compare ML weights vs equal weights (baseline)
        """
        print(f"\n📊 Comparison: ML Weights vs Equal Weights")
        print("=" * 60)
        
        # ML predictions
        y_pred_ml = self.model.predict(X)
        ml_r2 = r2_score(y, y_pred_ml)
        ml_mae = mean_absolute_error(y, y_pred_ml)
        
        # Equal weights baseline: simple average of all weighted signals
        weighted_features = [f'{ind}_weighted' for ind in self.INDICATORS]
        available_features = [f for f in weighted_features if f in X.columns]
        
        if available_features:
            equal_weight_pred = X[available_features].mean(axis=1)
            equal_r2 = r2_score(y, equal_weight_pred)
            equal_mae = mean_absolute_error(y, equal_weight_pred)
        else:
            equal_r2 = 0.0
            equal_mae = float('inf')
        
        print(f"\n   ML Weights:")
        print(f"      R²:  {ml_r2:.4f}")
        print(f"      MAE: {ml_mae:.6f}")
        
        print(f"\n   Equal Weights (Baseline):")
        print(f"      R²:  {equal_r2:.4f}")
        print(f"      MAE: {equal_mae:.6f}")
        
        print(f"\n   Improvement:")
        print(f"      R²:  {(ml_r2 - equal_r2):.4f} ({((ml_r2/equal_r2 - 1)*100 if equal_r2 != 0 else 0):.1f}%)")
        print(f"      MAE: {(equal_mae - ml_mae):.6f} ({((1 - ml_mae/equal_mae)*100 if equal_mae != 0 else 0):.1f}%)")
    
    def plot_feature_importance(self, top_n: int = 15):
        """
        Plot feature importance
        """
        if not self.feature_importance:
            print("⚠️ No feature importance available. Train model first.")
            return
        
        # Get top N features
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        features, importances = zip(*sorted_features)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), importances, color='steelblue')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - Indicator Weights')
        plt.tight_layout()
        
        plot_path = self.model_path / "indicator_feature_importance.png"
        plt.savefig(plot_path, dpi=150)
        print(f"📈 Saved plot: {plot_path}")
        plt.close()
    
    def plot_weights_comparison(self):
        """
        Plot learned weights vs equal weights
        """
        if not self.learned_weights:
            print("⚠️ No learned weights available. Train model first.")
            return
        
        indicators = list(self.learned_weights.keys())
        ml_weights = [self.learned_weights[ind] for ind in indicators]
        equal_weights = [1.0 / len(indicators)] * len(indicators)
        
        x = np.arange(len(indicators))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, ml_weights, width, label='ML Learned Weights', color='steelblue')
        ax.bar(x + width/2, equal_weights, width, label='Equal Weights', color='coral')
        
        ax.set_xlabel('Indicator')
        ax.set_ylabel('Weight')
        ax.set_title('Learned Weights vs Equal Weights (10 Indicators)')
        ax.set_xticks(x)
        ax.set_xticklabels(indicators, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.model_path / "indicator_weights_comparison.png"
        plt.savefig(plot_path, dpi=150)
        print(f"📈 Saved plot: {plot_path}")
        plt.close()


# Example usage and testing
if __name__ == "__main__":
    from ml.data_connector import DataConnector
    from ml.feature_extraction import FeatureExtractor
    from datetime import datetime, timedelta
    
    print("=" * 70)
    print("🤖 ML-Based Indicator Weight Learning")
    print("=" * 70)
    
    # Step 1: Fetch data
    print("\n📥 Step 1: Fetching Bitcoin historical data...")
    connector = DataConnector()
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    candles = connector.fetch_daily_candles("BTCUSDT", start_date, end_date)
    print(f"✅ Loaded {len(candles)} daily candles")
    
    # Step 2: Extract features
    print("\n🔧 Step 2: Extracting features from indicators...")
    extractor = FeatureExtractor(lookback_period=100, forward_days=5)
    X, y = extractor.extract_training_dataset(candles, level="indicators")
    
    print(f"✅ Extracted {X.shape[0]} training samples with {X.shape[1]} features")
    
    # Step 3: Train model
    print("\n🎓 Step 3: Training ML model...")
    learner = IndicatorWeightLearner(model_type="lightgbm")
    metrics = learner.train(X, y, test_size=0.2)
    
    # Step 4: Compare with baseline
    print("\n🔍 Step 4: Comparing with baseline...")
    X_test = X.iloc[int(len(X) * 0.8):]
    y_test = y.iloc[int(len(y) * 0.8):]
    learner.compare_with_equal_weights(X_test, y_test)
    
    # Step 5: Visualize
    print("\n📊 Step 5: Creating visualizations...")
    learner.plot_feature_importance(top_n=15)
    learner.plot_weights_comparison()
    
    # Step 6: Save model
    print("\n💾 Step 6: Saving model...")
    learner.save_model()
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    
    # Display final weights
    print("\n🎯 Final Learned Weights:")
    for ind, weight in sorted(learner.get_weights().items(), key=lambda x: x[1], reverse=True):
        print(f"   {ind:15s}: {weight:.4f} ({weight*100:.1f}%)")
