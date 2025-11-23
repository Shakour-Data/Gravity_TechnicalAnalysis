"""
Complete Training Pipeline for ML-Based Weight Learning

This module provides end-to-end training pipeline for both levels:
- Level 1: Learning weights for 10 trend indicators
- Level 2: Learning weights for 4 trend analysis dimensions

Usage:
    python ml/train_pipeline.py --symbol BTCUSDT --days 730
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import json

from gravity_tech.ml.data_connector import DataConnector
from gravity_tech.ml.feature_extraction import FeatureExtractor
from gravity_tech.ml.ml_indicator_weights import IndicatorWeightLearner
from gravity_tech.ml.ml_dimension_weights import DimensionWeightLearner


class MLTrainingPipeline:
    """
    Complete ML training pipeline for weight learning
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        days: int = 730,
        model_type: str = "lightgbm",
        lookback_period: int = 100,
        forward_days: int = 5
    ):
        """
        Initialize training pipeline
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            days: Number of historical days to fetch
            model_type: ML model type (lightgbm, xgboost, sklearn)
            lookback_period: Candles for indicator calculation
            forward_days: Days ahead for return prediction
        """
        self.symbol = symbol
        self.days = days
        self.model_type = model_type
        self.lookback_period = lookback_period
        self.forward_days = forward_days
        
        self.connector = DataConnector()
        self.extractor = FeatureExtractor(lookback_period, forward_days)
        
        self.candles = None
        self.indicator_learner = None
        self.dimension_learner = None
    
    def step1_fetch_data(self):
        """
        Step 1: Fetch historical candle data
        """
        print("\n" + "=" * 70)
        print("📥 STEP 1: Fetching Historical Data")
        print("=" * 70)
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=self.days)
        
        print(f"\nSymbol: {self.symbol}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Days:   {self.days}")
        
        self.candles = self.connector.fetch_daily_candles(
            symbol=self.symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"\n✅ Fetched {len(self.candles)} daily candles")
        print(f"   First: {self.candles[0].timestamp.date()} @ ${self.candles[0].close:,.2f}")
        print(f"   Last:  {self.candles[-1].timestamp.date()} @ ${self.candles[-1].close:,.2f}")
        
        # Calculate price change
        price_change = ((self.candles[-1].close - self.candles[0].close) 
                       / self.candles[0].close * 100)
        print(f"   Change: {price_change:+.2f}%")
    
    def step2_train_indicator_weights(self):
        """
        Step 2: Train Level 1 - 10 Indicator Weights
        """
        print("\n" + "=" * 70)
        print("🎓 STEP 2: Training Level 1 - 10 Indicator Weights")
        print("=" * 70)
        
        # Extract features
        print("\n📊 Extracting indicator-level features...")
        X, y = self.extractor.extract_training_dataset(
            self.candles,
            level="indicators"
        )
        
        print(f"✅ Features: {X.shape}")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Future returns: mean={y.mean():.4f}, std={y.std():.4f}")
        
        # Train model
        print(f"\n🤖 Training {self.model_type.upper()} model...")
        self.indicator_learner = IndicatorWeightLearner(model_type=self.model_type)
        metrics = self.indicator_learner.train(X, y, test_size=0.2)
        
        # Compare with baseline
        print("\n🔍 Comparing with equal weights baseline...")
        X_test = X.iloc[int(len(X) * 0.8):]
        y_test = y.iloc[int(len(y) * 0.8):]
        self.indicator_learner.compare_with_equal_weights(X_test, y_test)
        
        # Visualize
        print("\n📊 Creating visualizations...")
        self.indicator_learner.plot_feature_importance(top_n=15)
        self.indicator_learner.plot_weights_comparison()
        
        # Save
        print("\n💾 Saving model...")
        self.indicator_learner.save_model("indicator_weights_model.pkl")
        
        print("\n✅ Level 1 training complete!")
        
        return metrics
    
    def step3_train_dimension_weights(self):
        """
        Step 3: Train Level 2 - 4 Dimension Weights
        """
        print("\n" + "=" * 70)
        print("🎓 STEP 3: Training Level 2 - 4 Dimension Weights")
        print("=" * 70)
        
        # Extract features
        print("\n📊 Extracting dimension-level features...")
        X, y = self.extractor.extract_training_dataset(
            self.candles,
            level="dimensions"
        )
        
        print(f"✅ Features: {X.shape}")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Future returns: mean={y.mean():.4f}, std={y.std():.4f}")
        
        # Train model
        print(f"\n🤖 Training {self.model_type.upper()} model...")
        self.dimension_learner = DimensionWeightLearner(model_type=self.model_type)
        metrics = self.dimension_learner.train(X, y, test_size=0.2)
        
        # Compare with proposed weights
        print("\n🔍 Comparing with proposed weights (40-30-20-10)...")
        X_test = X.iloc[int(len(X) * 0.8):]
        y_test = y.iloc[int(len(y) * 0.8):]
        self.dimension_learner.compare_with_proposed_weights(X_test, y_test)
        
        # Visualize
        print("\n📊 Creating visualizations...")
        self.dimension_learner.plot_feature_importance()
        self.dimension_learner.plot_weights_comparison()
        
        # Save
        print("\n💾 Saving model...")
        self.dimension_learner.save_model("dimension_weights_model.pkl")
        
        print("\n✅ Level 2 training complete!")
        
        return metrics
    
    def step4_generate_summary(self, metrics_level1, metrics_level2):
        """
        Step 4: Generate training summary
        """
        print("\n" + "=" * 70)
        print("📋 STEP 4: Training Summary")
        print("=" * 70)
        
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': self.symbol,
            'days': self.days,
            'model_type': self.model_type,
            'lookback_period': self.lookback_period,
            'forward_days': self.forward_days,
            'total_candles': len(self.candles),
            'level1_metrics': metrics_level1,
            'level2_metrics': metrics_level2,
            'level1_weights': self.indicator_learner.get_weights() if self.indicator_learner else {},
            'level2_weights': self.dimension_learner.get_weights() if self.dimension_learner else {}
        }
        
        # Save summary
        summary_path = Path("models/ml_weights/training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Saved training summary: {summary_path}")
        
        # Display summary
        print("\n📊 Training Results:")
        print("\n   Level 1: 10 Indicator Weights")
        print(f"      Test R²:  {metrics_level1['test_r2']:.4f}")
        print(f"      Test MAE: {metrics_level1['test_mae']:.6f}")
        print("\n      Learned Weights:")
        for ind, weight in sorted(
            summary['level1_weights'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]:
            print(f"         {ind:15s}: {weight:.4f} ({weight*100:.1f}%)")
        
        print("\n   Level 2: 4 Dimension Weights")
        print(f"      Test R²:  {metrics_level2['test_r2']:.4f}")
        print(f"      Test MAE: {metrics_level2['test_mae']:.6f}")
        print("\n      Learned Weights:")
        for dim, weight in sorted(
            summary['level2_weights'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"         {dim:15s}: {weight:.4f} ({weight*100:.1f}%)")
        
        return summary
    
    def run_complete_pipeline(self):
        """
        Run complete training pipeline
        """
        print("\n" + "=" * 70)
        print("🚀 MACHINE LEARNING WEIGHT TRAINING PIPELINE")
        print("=" * 70)
        print(f"\nSymbol:          {self.symbol}")
        print(f"Model Type:      {self.model_type}")
        print(f"Lookback Period: {self.lookback_period} candles")
        print(f"Forward Days:    {self.forward_days} days")
        
        try:
            # Step 1: Fetch data
            self.step1_fetch_data()
            
            # Step 2: Train indicator weights
            metrics_level1 = self.step2_train_indicator_weights()
            
            # Step 3: Train dimension weights
            metrics_level2 = self.step3_train_dimension_weights()
            
            # Step 4: Summary
            summary = self.step4_generate_summary(metrics_level1, metrics_level2)
            
            print("\n" + "=" * 70)
            print("✅ TRAINING PIPELINE COMPLETE!")
            print("=" * 70)
            
            print("\n📂 Generated Files:")
            print("   models/ml_weights/")
            print("   ├── indicator_weights_model.pkl")
            print("   ├── indicator_weights.json")
            print("   ├── indicator_feature_importance.json")
            print("   ├── indicator_feature_importance.png")
            print("   ├── indicator_weights_comparison.png")
            print("   ├── dimension_weights_model.pkl")
            print("   ├── dimension_weights.json")
            print("   ├── dimension_feature_importance.json")
            print("   ├── dimension_feature_importance.png")
            print("   ├── dimension_weights_comparison.png")
            print("   └── training_summary.json")
            
            return summary
            
        except Exception as e:
            print(f"\n❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """
    Command-line interface for training pipeline
    """
    parser = argparse.ArgumentParser(
        description='ML-Based Weight Training Pipeline'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='Trading symbol (default: BTCUSDT)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=730,
        help='Number of historical days (default: 730 = 2 years)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='lightgbm',
        choices=['lightgbm', 'xgboost', 'sklearn'],
        help='ML model type (default: lightgbm)'
    )
    
    parser.add_argument(
        '--lookback',
        type=int,
        default=100,
        help='Lookback period for indicators (default: 100)'
    )
    
    parser.add_argument(
        '--forward',
        type=int,
        default=5,
        help='Forward days for return prediction (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = MLTrainingPipeline(
        symbol=args.symbol,
        days=args.days,
        model_type=args.model,
        lookback_period=args.lookback,
        forward_days=args.forward
    )
    
    summary = pipeline.run_complete_pipeline()
    
    if summary:
        print("\n🎯 Use the learned weights in your analysis!")
        print("   Load with:")
        print("   >>> from ml.ml_indicator_weights import IndicatorWeightLearner")
        print("   >>> learner = IndicatorWeightLearner()")
        print("   >>> learner.load_model()")
        print("   >>> weights = learner.get_weights()")


if __name__ == "__main__":
    main()
