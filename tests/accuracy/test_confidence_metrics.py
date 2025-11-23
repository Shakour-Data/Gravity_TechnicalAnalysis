"""
Test confidence metrics for both indicator and dimension weights
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gravity_tech.ml.data_connector import DataConnector
from gravity_tech.ml.feature_extraction import FeatureExtractor
from gravity_tech.ml.ml_indicator_weights import IndicatorWeightLearner
from gravity_tech.ml.ml_dimension_weights import DimensionWeightLearner

def main():
    print("="*60)
    print("Testing Confidence Metrics for ML Weight Learning")
    print("="*60)
    
    # 1. Get data
    print("\n📊 Step 1: Fetching data...")
    connector = DataConnector()
    candles = connector.fetch_daily_candles(symbol="BTCUSDT", limit=200)
    print(f"   Retrieved {len(candles)} candles")
    
    # 2. Extract features
    print("\n🔧 Step 2: Extracting features...")
    extractor = FeatureExtractor(lookback_period=50, forward_days=3)
    
    # Level 1 features (indicators)
    X_train_l1, y_train_l1 = extractor.extract_training_dataset(
        candles, 
        level='indicators'
    )
    
    # Split for testing
    split_idx = int(len(X_train_l1) * 0.8)
    X_train_ind = X_train_l1[:split_idx]
    y_train_ind = y_train_l1[:split_idx]
    X_test_ind = X_train_l1[split_idx:]
    y_test_ind = y_train_l1[split_idx:]
    
    print(f"   Level 1 - Training: {len(X_train_ind)}, Test: {len(X_test_ind)}")
    
    # 3. Train Level 1 (Indicator Weights)
    print("\n🎯 Step 3: Training Level 1 - Indicator Weights...")
    learner_l1 = IndicatorWeightLearner(model_type='lightgbm')
    learner_l1.train(X_train_ind, y_train_ind)
    
    # Get weights with confidence
    print("\n📈 Level 1 Results:")
    result_l1 = learner_l1.get_weights_with_confidence(X_test_ind, y_test_ind)
    
    print("\n   Raw ML Weights:")
    for ind, weight in sorted(result_l1['weights'].items(), key=lambda x: x[1], reverse=True):
        print(f"      {ind:10s}: {weight*100:5.1f}%")
    
    print("\n   Quality Metrics:")
    metrics_l1 = result_l1['metrics']
    print(f"      R² Score:          {metrics_l1['r2_score']:.4f}")
    print(f"      MAE:               {metrics_l1['mae']:.6f}")
    print(f"      95% Confidence:    ±{metrics_l1['confidence_interval_95']:.6f}")
    print(f"      Reliability:       {metrics_l1['reliability']} (factor={metrics_l1['reliability_factor']})")
    
    print("\n   Confidence-Adjusted Weights:")
    for ind, weight in sorted(result_l1['adjusted_weights'].items(), key=lambda x: x[1], reverse=True):
        print(f"      {ind:10s}: {weight*100:5.1f}%")
    
    # 4. Extract Level 2 features
    print("\n🔧 Step 4: Extracting Level 2 features...")
    X_train_l2, y_train_l2 = extractor.extract_training_dataset(
        candles,
        level='dimensions'
    )
    
    # Split for testing
    split_idx = int(len(X_train_l2) * 0.8)
    X_train_dim = X_train_l2[:split_idx]
    y_train_dim = y_train_l2[:split_idx]
    X_test_dim = X_train_l2[split_idx:]
    y_test_dim = y_train_l2[split_idx:]
    
    print(f"   Level 2 - Training: {len(X_train_dim)}, Test: {len(X_test_dim)}")
    
    # 5. Train Level 2 (Dimension Weights)
    print("\n🎯 Step 5: Training Level 2 - Dimension Weights...")
    learner_l2 = DimensionWeightLearner(model_type='lightgbm')
    learner_l2.train(X_train_dim, y_train_dim)
    
    # Get weights with confidence
    print("\n📈 Level 2 Results:")
    result_l2 = learner_l2.get_weights_with_confidence(X_test_dim, y_test_dim)
    
    print("\n   Raw ML Weights:")
    for dim, weight in sorted(result_l2['weights'].items(), key=lambda x: x[1], reverse=True):
        print(f"      {dim:12s}: {weight*100:5.1f}%")
    
    print("\n   Quality Metrics:")
    metrics_l2 = result_l2['metrics']
    print(f"      R² Score:          {metrics_l2['r2_score']:.4f}")
    print(f"      MAE:               {metrics_l2['mae']:.6f}")
    print(f"      95% Confidence:    ±{metrics_l2['confidence_interval_95']:.6f}")
    print(f"      Reliability:       {metrics_l2['reliability']} (factor={metrics_l2['reliability_factor']})")
    print(f"      Blend Ratio:       {metrics_l2['blend_ratio']}")
    
    print("\n   Proposed Weights (Fallback):")
    for dim, weight in sorted(result_l2['proposed_weights'].items(), key=lambda x: x[1], reverse=True):
        print(f"      {dim:12s}: {weight*100:5.1f}%")
    
    print("\n   Blended Weights (ML + Proposed):")
    for dim, weight in sorted(result_l2['adjusted_weights'].items(), key=lambda x: x[1], reverse=True):
        print(f"      {dim:12s}: {weight*100:5.1f}%")
    
    # 6. Interpretation
    print("\n" + "="*60)
    print("📊 Interpretation:")
    print("="*60)
    
    print("\n🔍 Level 1 (Indicators):")
    if metrics_l1['r2_score'] > 0.7:
        print("   ✅ HIGH reliability - Using ML weights at full strength")
    elif metrics_l1['r2_score'] > 0.4:
        print("   ⚠️  MEDIUM reliability - Reducing ML weight influence by 30%")
    else:
        print("   ❌ LOW reliability - Reducing ML weight influence by 60%")
    
    print("\n🔍 Level 2 (Dimensions):")
    if metrics_l2['r2_score'] > 0.7:
        print("   ✅ HIGH reliability - Using ML weights at full strength")
        print("   Strategy: 100% ML weights")
    elif metrics_l2['r2_score'] > 0.4:
        print("   ⚠️  MEDIUM reliability - Blending ML with proposed weights")
        print("   Strategy: 60% ML + 40% Proposed")
    else:
        print("   ❌ LOW reliability - Favoring proposed weights")
        print("   Strategy: 30% ML + 70% Proposed")
    
    print("\n💡 Why Confidence Metrics Matter:")
    print("   • R² < 0: Model performs worse than baseline (use fallback)")
    print("   • Small datasets → overfitting → low reliability")
    print("   • Confidence intervals show prediction uncertainty")
    print("   • Reliability factor prevents poor models from misleading analysis")
    
    print("\n✅ Test Complete!")

if __name__ == "__main__":
    main()
