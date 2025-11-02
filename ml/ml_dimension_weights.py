"""
ML-Based Weight Learning for 4 Trend Analysis Dimensions

This module uses Gradient Boosting to learn optimal weights for combining
4 dimensions of trend analysis:
1. Technical Indicators (10 indicators combined)
2. Candlestick Patterns
3. Elliott Wave Theory
4. Classical Chart Patterns
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

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt


class DimensionWeightLearner:
    """
    Learn optimal weights for 4 trend analysis dimensions using ML
    """
    
    # 4 Dimensions of Trend Analysis
    DIMENSIONS = ['indicators', 'candlestick', 'elliott', 'classical']
    
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
        Train model to predict future returns from 4 dimension signals
        
        Args:
            X: Features (4 dimension scores, confidences, weighted signals)
            y: Labels (future returns)
            test_size: Test set ratio
            random_state: Random seed
            
        Returns:
            Training metrics
        """
        print(f"\n🎯 Training {self.model_type.upper()} model for 4 dimensions...")
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
                max_depth=4,  # Shallower for fewer features
                num_leaves=15,
                random_state=random_state,
                verbose=-1
            )
        elif self.model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                random_state=random_state,
                verbosity=0
            )
        else:  # sklearn
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
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
        
        print(f"\n🔝 Feature Importance (All {len(sorted_importance)} features):")
        for i, (feature, imp) in enumerate(sorted_importance, 1):
            print(f"   {i:2d}. {feature:30s}: {imp:.4f}")
    
    def _calculate_weights_from_importance(self):
        """
        Calculate dimension weights from feature importance
        
        For each dimension, we have 3 features:
        - dim{N}_{dimension}_score
        - dim{N}_{dimension}_confidence
        - dim{N}_{dimension}_weighted
        
        We sum the importance of all 3 features for each dimension
        """
        dimension_importance = {}
        
        for i, dimension in enumerate(self.DIMENSIONS, 1):
            # Sum importance of all features for this dimension
            total_imp = 0.0
            
            for suffix in ['_score', '_confidence', '_weighted']:
                feature_name = f'dim{i}_{dimension}{suffix}'
                if feature_name in self.feature_importance:
                    total_imp += self.feature_importance[feature_name]
            
            dimension_importance[dimension] = total_imp
        
        # Normalize to weights (sum to 1.0)
        total = sum(dimension_importance.values())
        
        if total > 0:
            self.learned_weights = {
                dim: imp / total 
                for dim, imp in dimension_importance.items()
            }
        else:
            # Fallback to proposed weights: 40%, 30%, 20%, 10%
            self.learned_weights = {
                'indicators': 0.40,
                'elliott': 0.30,
                'classical': 0.20,
                'candlestick': 0.10
            }
        
        print(f"\n⚖️  Learned Dimension Weights:")
        sorted_weights = sorted(
            self.learned_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for dimension, weight in sorted_weights:
            bar = '█' * int(weight * 100)
            print(f"   {dimension:15s}: {weight:.4f} ({weight*100:.1f}%) {bar}")
        
        # Verify sum
        weight_sum = sum(self.learned_weights.values())
        print(f"\n   Total weight: {weight_sum:.6f} (should be 1.0)")
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get learned weights for 4 dimensions
        
        Returns:
            Dictionary mapping dimension name to weight
        """
        return self.learned_weights.copy()
    
    def get_weights_with_confidence(self, X_test, y_test) -> Dict[str, Dict[str, float]]:
        """
        Get learned weights with confidence metrics and reliability factor
        
        Returns:
            Dictionary with weights, R², MAE, confidence intervals, and adjusted weights
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
        # R² > 0.7 → high reliability (use ML weights fully)
        # R² 0.4-0.7 → medium reliability (blend with proposed weights)
        # R² < 0.4 → low reliability (use proposed weights more)
        if r2 > 0.7:
            weight_reliability = "high"
            reliability_factor = 1.0
            blend_ml = 1.0
        elif r2 > 0.4:
            weight_reliability = "medium"
            reliability_factor = 0.7
            blend_ml = 0.6  # 60% ML, 40% proposed
        else:
            weight_reliability = "low"
            reliability_factor = 0.4
            blend_ml = 0.3  # 30% ML, 70% proposed
        
        # Proposed weights as fallback
        proposed_weights = {
            'indicators': 0.40,
            'elliott': 0.30,
            'classical': 0.20,
            'candlestick': 0.10
        }
        
        # Blend ML weights with proposed weights based on reliability
        blended_weights = {}
        for dim in self.DIMENSIONS:
            ml_weight = self.learned_weights.get(dim, proposed_weights.get(dim, 0.25))
            prop_weight = proposed_weights.get(dim, 0.25)
            blended_weights[dim] = (ml_weight * blend_ml) + (prop_weight * (1 - blend_ml))
        
        # Normalize blended weights
        total_blended = sum(blended_weights.values())
        if total_blended > 0:
            blended_weights = {
                dim: weight / total_blended
                for dim, weight in blended_weights.items()
            }
        
        result = {
            'weights': self.learned_weights.copy(),
            'metrics': {
                'r2_score': r2,
                'mae': mae,
                'confidence_interval_95': confidence_95,
                'reliability': weight_reliability,
                'reliability_factor': reliability_factor,
                'blend_ratio': f"{int(blend_ml*100)}% ML / {int((1-blend_ml)*100)}% Proposed"
            },
            'adjusted_weights': blended_weights,
            'proposed_weights': proposed_weights
        }
        
        return result
    
    def save_model(self, filename: str = "dimension_weights_model.pkl"):
        """
        Save trained model and weights with metadata
        """
        from datetime import datetime
        
        save_path = self.model_path / filename
        
        # Save model
        joblib.dump(self.model, save_path)
        
        # Save weights with metadata
        weights_path = self.model_path / "dimension_weights.json"
        weights_data = {
            'weights': self.learned_weights,
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'feature_importance': self.feature_importance
        }
        
        with open(weights_path, 'w') as f:
            json.dump(weights_data, f, indent=2)
        
        # Save feature importance
        importance_path = self.model_path / "dimension_feature_importance.json"
        with open(importance_path, 'w') as f:
            json.dump(self.feature_importance, f, indent=2)
        
        print(f"\n💾 Saved:")
        print(f"   Model:      {save_path}")
        print(f"   Weights:    {weights_path} (with timestamp and metadata)")
        print(f"   Importance: {importance_path}")
    
    def load_model(self, filename: str = "dimension_weights_model.pkl"):
        """
        Load trained model and weights
        """
        load_path = self.model_path / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")
        
        self.model = joblib.load(load_path)
        
        # Load weights
        weights_path = self.model_path / "dimension_weights.json"
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                self.learned_weights = json.load(f)
        
        # Load feature importance
        importance_path = self.model_path / "dimension_feature_importance.json"
        if importance_path.exists():
            with open(importance_path, 'r') as f:
                self.feature_importance = json.load(f)
        
        print(f"✅ Loaded model from {load_path}")
    
    def compare_with_proposed_weights(self, X: pd.DataFrame, y: pd.Series):
        """
        Compare ML weights vs proposed weights (40%, 30%, 20%, 10%)
        """
        print(f"\n📊 Comparison: ML Weights vs Proposed Weights")
        print("=" * 60)
        
        # ML predictions
        y_pred_ml = self.model.predict(X)
        ml_r2 = r2_score(y, y_pred_ml)
        ml_mae = mean_absolute_error(y, y_pred_ml)
        
        # Proposed weights: 40%, 30%, 20%, 10%
        proposed_weights = {
            'indicators': 0.40,
            'elliott': 0.30,
            'classical': 0.20,
            'candlestick': 0.10
        }
        
        # Calculate weighted average with proposed weights
        weighted_features = []
        for i, dim in enumerate(self.DIMENSIONS, 1):
            feature_name = f'dim{i}_{dim}_weighted'
            if feature_name in X.columns:
                weighted_features.append(X[feature_name] * proposed_weights[dim])
        
        if weighted_features:
            proposed_pred = sum(weighted_features)
            proposed_r2 = r2_score(y, proposed_pred)
            proposed_mae = mean_absolute_error(y, proposed_pred)
        else:
            proposed_r2 = 0.0
            proposed_mae = float('inf')
        
        print(f"\n   ML Learned Weights:")
        print(f"      R²:  {ml_r2:.4f}")
        print(f"      MAE: {ml_mae:.6f}")
        
        print(f"\n   Proposed Weights (40-30-20-10):")
        print(f"      R²:  {proposed_r2:.4f}")
        print(f"      MAE: {proposed_mae:.6f}")
        
        print(f"\n   Improvement:")
        print(f"      R²:  {(ml_r2 - proposed_r2):.4f} ({((ml_r2/proposed_r2 - 1)*100 if proposed_r2 != 0 else 0):.1f}%)")
        print(f"      MAE: {(proposed_mae - ml_mae):.6f} ({((1 - ml_mae/proposed_mae)*100 if proposed_mae != 0 else 0):.1f}%)")
    
    def plot_feature_importance(self):
        """
        Plot feature importance for all dimension features
        """
        if not self.feature_importance:
            print("⚠️ No feature importance available. Train model first.")
            return
        
        # Sort features
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        features, importances = zip(*sorted_features)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importances, color='steelblue')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance - 4 Dimension Weights')
        plt.tight_layout()
        
        plot_path = self.model_path / "dimension_feature_importance.png"
        plt.savefig(plot_path, dpi=150)
        print(f"📈 Saved plot: {plot_path}")
        plt.close()
    
    def plot_weights_comparison(self):
        """
        Plot learned weights vs proposed weights (40-30-20-10)
        """
        if not self.learned_weights:
            print("⚠️ No learned weights available. Train model first.")
            return
        
        dimensions = ['indicators', 'elliott', 'classical', 'candlestick']
        ml_weights = [self.learned_weights.get(dim, 0.0) for dim in dimensions]
        proposed_weights = [0.40, 0.30, 0.20, 0.10]
        
        x = np.arange(len(dimensions))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, ml_weights, width, label='ML Learned Weights', color='steelblue')
        ax.bar(x + width/2, proposed_weights, width, label='Proposed Weights (40-30-20-10)', color='coral')
        
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Weight')
        ax.set_title('Learned Weights vs Proposed Weights (4 Dimensions)')
        ax.set_xticks(x)
        ax.set_xticklabels(dimensions, rotation=0)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for i, (ml, prop) in enumerate(zip(ml_weights, proposed_weights)):
            ax.text(i - width/2, ml + 0.01, f'{ml*100:.1f}%', ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, prop + 0.01, f'{prop*100:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plot_path = self.model_path / "dimension_weights_comparison.png"
        plt.savefig(plot_path, dpi=150)
        print(f"📈 Saved plot: {plot_path}")
        plt.close()


# Example usage and testing
if __name__ == "__main__":
    from ml.data_connector import DataConnector
    from ml.feature_extraction import FeatureExtractor
    from datetime import datetime, timedelta
    
    print("=" * 70)
    print("🤖 ML-Based Dimension Weight Learning (4 Dimensions)")
    print("=" * 70)
    
    # Step 1: Fetch data
    print("\n📥 Step 1: Fetching Bitcoin historical data...")
    connector = DataConnector()
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    candles = connector.fetch_daily_candles("BTCUSDT", start_date, end_date)
    print(f"✅ Loaded {len(candles)} daily candles")
    
    # Step 2: Extract features
    print("\n🔧 Step 2: Extracting features from 4 dimensions...")
    extractor = FeatureExtractor(lookback_period=100, forward_days=5)
    X, y = extractor.extract_training_dataset(candles, level="dimensions")
    
    print(f"✅ Extracted {X.shape[0]} training samples with {X.shape[1]} features")
    
    # Step 3: Train model
    print("\n🎓 Step 3: Training ML model...")
    learner = DimensionWeightLearner(model_type="lightgbm")
    metrics = learner.train(X, y, test_size=0.2)
    
    # Step 4: Compare with proposed weights
    print("\n🔍 Step 4: Comparing with proposed weights...")
    X_test = X.iloc[int(len(X) * 0.8):]
    y_test = y.iloc[int(len(y) * 0.8):]
    learner.compare_with_proposed_weights(X_test, y_test)
    
    # Step 5: Visualize
    print("\n📊 Step 5: Creating visualizations...")
    learner.plot_feature_importance()
    learner.plot_weights_comparison()
    
    # Step 6: Save model
    print("\n💾 Step 6: Saving model...")
    learner.save_model()
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    
    # Display final weights
    print("\n🎯 Final Learned Weights:")
    for dim, weight in sorted(learner.get_weights().items(), key=lambda x: x[1], reverse=True):
        print(f"   {dim:15s}: {weight:.4f} ({weight*100:.1f}%)")
