"""
Model Interpretability with SHAP

SHAP (SHapley Additive exPlanations) for:
- Feature importance visualization
- Individual prediction explanation
- Pattern decision analysis
- Model transparency and trust

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import sys
import os

try:
    import matplotlib.pyplot as plt
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available. Install with: pip install shap matplotlib")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PatternModelInterpreter:
    """
    SHAP-based interpreter for pattern recognition models.
    
    Provides explanations for:
    - Why a specific pattern was predicted
    - Which features contributed most
    - How features interact
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize interpreter.
        
        Args:
            model: Trained classification model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self, X_background: np.ndarray, model_type: str = 'tree'):
        """
        Create SHAP explainer for the model.
        
        Args:
            X_background: Background dataset for SHAP (subset of training data)
            model_type: Type of explainer ('tree', 'kernel', 'linear')
        """
        if not SHAP_AVAILABLE:
            print("⚠️ SHAP not available. Skipping explainer creation.")
            return
        
        print(f"\n🔍 Creating SHAP {model_type.upper()} Explainer...")
        
        if model_type == 'tree':
            # For tree-based models (XGBoost, RandomForest, etc.)
            self.explainer = shap.TreeExplainer(self.model)
        elif model_type == 'kernel':
            # For any model (slower but more general)
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                X_background[:100]  # Use subset for speed
            )
        else:
            raise ValueError(f"Unknown explainer type: {model_type}")
        
        print("✅ Explainer created")
    
    def explain_predictions(
        self,
        X: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Generate SHAP values for predictions.
        
        Args:
            X: Features to explain
            class_names: Optional class names
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        print(f"\n📊 Computing SHAP values for {len(X)} samples...")
        
        self.shap_values = self.explainer.shap_values(X)
        
        print("✅ SHAP values computed")
        return self.shap_values
    
    def plot_summary(
        self,
        X: np.ndarray,
        max_display: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Create summary plot showing feature importance.
        
        Args:
            X: Feature data
            max_display: Maximum features to display
            save_path: Optional path to save plot
        """
        if self.shap_values is None:
            self.explain_predictions(X)
        
        print("\n📈 Creating SHAP Summary Plot...")
        
        plt.figure(figsize=(12, 8))
        
        # For multi-class, use class 0 (or combine)
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        shap.summary_plot(
            shap_vals,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Summary plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance(
        self,
        X: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Create bar plot of mean absolute SHAP values (feature importance).
        
        Args:
            X: Feature data
            save_path: Optional path to save plot
        """
        if self.shap_values is None:
            self.explain_predictions(X)
        
        print("\n📊 Creating Feature Importance Plot...")
        
        plt.figure(figsize=(10, 8))
        
        # For multi-class, average across classes
        if isinstance(self.shap_values, list):
            shap_vals = np.abs(np.array(self.shap_values)).mean(axis=0)
        else:
            shap_vals = np.abs(self.shap_values)
        
        # Calculate mean importance
        importance = np.mean(shap_vals, axis=0)
        
        # Sort by importance
        indices = np.argsort(importance)[::-1]
        
        # Plot top 15
        top_n = min(15, len(indices))
        top_indices = indices[:top_n]
        
        plt.barh(range(top_n), importance[top_indices])
        plt.yticks(range(top_n), [self.feature_names[i] for i in top_indices])
        plt.xlabel('Mean |SHAP Value| (Feature Importance)')
        plt.title('Top Feature Importance (SHAP)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Feature importance plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def explain_single_prediction(
        self,
        X_single: np.ndarray,
        class_idx: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Explain a single prediction with waterfall plot.
        
        Args:
            X_single: Single sample to explain (1D array)
            class_idx: Class index to explain
            save_path: Optional path to save plot
        """
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)
        
        if self.explainer is None:
            raise ValueError("Explainer not created")
        
        print(f"\n🔍 Explaining Single Prediction (Class {class_idx})...")
        
        shap_values_single = self.explainer.shap_values(X_single)
        
        # For multi-class, select specific class
        if isinstance(shap_values_single, list):
            shap_vals = shap_values_single[class_idx][0]
        else:
            shap_vals = shap_values_single[0]
        
        # Create waterfall plot
        plt.figure(figsize=(10, 8))
        
        # Sort by absolute value
        indices = np.argsort(np.abs(shap_vals))[::-1][:15]
        
        feature_values = X_single[0][indices]
        shap_contributions = shap_vals[indices]
        feature_labels = [self.feature_names[i] for i in indices]
        
        # Create waterfall
        cumsum = np.cumsum(shap_contributions)
        base_value = self.explainer.expected_value
        
        if isinstance(base_value, np.ndarray):
            base_value = base_value[class_idx]
        
        plt.barh(range(len(indices)), shap_contributions, 
                color=['red' if x < 0 else 'green' for x in shap_contributions])
        plt.yticks(range(len(indices)), 
                  [f"{label}={val:.3f}" for label, val in zip(feature_labels, feature_values)])
        plt.xlabel('SHAP Value (Impact on Prediction)')
        plt.title(f'Single Prediction Explanation (Class {class_idx})')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Single prediction plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Print explanation
        print("\n📋 Prediction Explanation:")
        print(f"   Base Value: {base_value:.4f}")
        print(f"   Final Prediction: {base_value + np.sum(shap_vals):.4f}")
        print(f"\n   Top 5 Contributing Features:")
        for i, (idx, contrib) in enumerate(zip(indices[:5], shap_contributions[:5]), 1):
            direction = "↑" if contrib > 0 else "↓"
            print(f"   {i}. {self.feature_names[idx]}: {contrib:+.4f} {direction}")
    
    def get_feature_importance_dict(self, X: np.ndarray) -> Dict[str, float]:
        """
        Get feature importance as dictionary.
        
        Args:
            X: Feature data
            
        Returns:
            Dictionary of feature_name -> importance
        """
        if self.shap_values is None:
            self.explain_predictions(X)
        
        # For multi-class, average across classes
        if isinstance(self.shap_values, list):
            shap_vals = np.abs(np.array(self.shap_values)).mean(axis=0)
        else:
            shap_vals = np.abs(self.shap_values)
        
        # Calculate mean importance
        importance = np.mean(shap_vals, axis=0)
        
        # Normalize to sum to 1
        importance = importance / importance.sum()
        
        return {
            name: float(imp)
            for name, imp in zip(self.feature_names, importance)
        }
    
    def analyze_feature_interactions(
        self,
        X: np.ndarray,
        feature_idx1: int,
        feature_idx2: int,
        save_path: Optional[str] = None
    ):
        """
        Analyze interaction between two features.
        
        Args:
            X: Feature data
            feature_idx1: First feature index
            feature_idx2: Second feature index
            save_path: Optional path to save plot
        """
        if self.shap_values is None:
            self.explain_predictions(X)
        
        print(f"\n🔗 Analyzing Feature Interaction...")
        print(f"   {self.feature_names[feature_idx1]} × {self.feature_names[feature_idx2]}")
        
        plt.figure(figsize=(10, 6))
        
        # For multi-class, use first class
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        shap.dependence_plot(
            feature_idx1,
            shap_vals,
            X,
            feature_names=self.feature_names,
            interaction_index=feature_idx2,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Interaction plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()


def demo_shap_analysis():
    """Demonstration of SHAP analysis."""
    print("=" * 80)
    print("🔍 SHAP Model Interpretability Demo")
    print("=" * 80)
    
    # Load model and generate test data
    from ml.advanced_pattern_training import AdvancedPatternTrainer
    
    trainer = AdvancedPatternTrainer()
    X, y_type, y_success = trainer.generate_enhanced_training_data(n_samples=500)
    
    # Train simple model for demo
    from sklearn.ensemble import RandomForestClassifier
    
    class_to_idx = {'gartley': 0, 'butterfly': 1, 'bat': 2, 'crab': 3}
    y_idx = np.array([class_to_idx[t] for t in y_type])
    
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X, y_idx)
    
    print("✅ Model trained for demo")
    
    # Feature names
    feature_names = [
        'xab_ratio_accuracy', 'abc_ratio_accuracy', 'bcd_ratio_accuracy', 'xad_ratio_accuracy',
        'pattern_symmetry', 'pattern_slope', 'xa_angle', 'ab_angle', 'bc_angle', 'cd_angle',
        'pattern_duration', 'xa_magnitude', 'ab_magnitude', 'bc_magnitude', 'cd_magnitude',
        'volume_at_d', 'volume_trend', 'volume_confirmation',
        'rsi_at_d', 'macd_at_d', 'momentum_divergence'
    ]
    
    # Create interpreter
    interpreter = PatternModelInterpreter(model, feature_names)
    interpreter.create_explainer(X[:100], model_type='tree')
    
    # Explain predictions
    interpreter.explain_predictions(X[:100])
    
    # Get feature importance
    importance_dict = interpreter.get_feature_importance_dict(X[:100])
    
    print("\n📊 Feature Importance (Top 10):")
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(sorted_features[:10], 1):
        print(f"   {i:2d}. {feature:30s} {importance:.4f}")
    
    # Explain single prediction
    print("\n" + "=" * 80)
    interpreter.explain_single_prediction(X[0], class_idx=0)
    
    print("\n✅ SHAP Analysis Complete!")


if __name__ == "__main__":
    demo_shap_analysis()
