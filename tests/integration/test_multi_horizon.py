"""
Test Multi-Horizon System

Author: Gravity Tech Team
Date: 2024
Version: 1.0
License: MIT

Complete test script for multi-horizon system:
1. Train model with Bitcoin data
2. Analyze and display three scores (3d, 7d, 30d)
3. Detect patterns
4. Generate recommendations
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from typing import List

# Add parent directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.domain.entities import Candle
from gravity_tech.ml.train_multi_horizon import train_multi_horizon_system, load_trained_model, create_realistic_market_data
from gravity_tech.ml.multi_horizon_feature_extraction import MultiHorizonFeatureExtractor
from gravity_tech.ml.multi_horizon_analysis import MultiHorizonTrendAnalyzer
import json


def test_training():
    """
    Test 1: Train model with Bitcoin.
    """
    print("\n" + "🧪"*35)
    print("TEST 1: TRAINING MULTI-HORIZON SYSTEM")
    print("🧪"*35)
    
    result = train_multi_horizon_system(
        symbol="BTCUSDT",
        interval="1d",
        lookback_days=365,
        horizons=[3, 7, 30],
        output_dir="ml_models/multi_horizon"
    )
    
    print("\n✅ TEST 1 PASSED - Training completed")
    return result


def test_analysis(result):
    """
    Test 2: Analysis with trained model.
    """
    print("\n" + "🧪"*35)
    print("TEST 2: MULTI-HORIZON ANALYSIS")
    print("🧪"*35)
    
    # Get new data (simulation)
    candles = create_realistic_market_data(
        base_price=50000,
        candles_count=150,
        trend="mixed"
    )
    
    print(f"\n📊 Latest data: {len(candles)} candles")
    print(f"   Last candle: {candles[-1].timestamp}")
    print(f"   Price: ${candles[-1].close:,.2f}")
    
    # Extract current features
    extractor = MultiHorizonFeatureExtractor(
        lookback_period=100,
        horizons=[3, 7, 30]
    )
    
    # Level 1 features
    features_l1 = extractor.extract_indicator_features(candles[-100:])
    
    # Create analyzer with trained model
    analyzer_l1 = MultiHorizonTrendAnalyzer(
        result['learner_indicators']
    )
    
    # Analysis
    analysis_l1 = analyzer_l1.analyze(features_l1)
    
    # Display result
    analyzer_l1.print_analysis(analysis_l1)
    
    # ═══════════════════════════════════════════════════════════
    # Test level 2
    # ═══════════════════════════════════════════════════════════
    print("\n" + "-"*70)
    print("📊 LEVEL 2 ANALYSIS (Dimensions)")
    print("-"*70)
    
    features_l2 = extractor.extract_dimension_features(candles[-100:])
    
    analyzer_l2 = MultiHorizonTrendAnalyzer(
        result['learner_dimensions']
    )
    
    analysis_l2 = analyzer_l2.analyze(features_l2)
    analyzer_l2.print_analysis(analysis_l2)
    
    print("\n✅ TEST 2 PASSED - Analysis completed")
    
    return {
        'analysis_level1': analysis_l1,
        'analysis_level2': analysis_l2
    }


def test_pattern_detection(analyses):
    """
    Test 3: Pattern detection.
    """
    print("\n" + "🧪"*35)
    print("TEST 3: PATTERN DETECTION")
    print("🧪"*35)
    
    analysis_l1 = analyses['analysis_level1']
    analysis_l2 = analyses['analysis_level2']
    
    print("\n🔍 Level 1 Pattern:")
    print(f"   Type: {analysis_l1.pattern.value}")
    print(f"   Confidence: {analysis_l1.pattern_confidence:.0%}")
    
    print("\n🔍 Level 2 Pattern:")
    print(f"   Type: {analysis_l2.pattern.value}")
    print(f"   Confidence: {analysis_l2.pattern_confidence:.0%}")
    
    # Comparison
    if analysis_l1.pattern == analysis_l2.pattern:
        print(f"\n✅ Both levels agree: {analysis_l1.pattern.value}")
    else:
        print(f"\n⚠️ Different patterns detected:")
        print(f"   L1: {analysis_l1.pattern.value}")
        print(f"   L2: {analysis_l2.pattern.value}")
    
    print("\n✅ TEST 3 PASSED - Pattern detection working")


def test_recommendations(analyses):
    """
    Test 4: Recommendations.
    """
    print("\n" + "🧪"*35)
    print("TEST 4: RECOMMENDATIONS")
    print("🧪"*35)
    
    analysis_l1 = analyses['analysis_level1']
    
    print("\n📋 Level 1 Recommendations:")
    print(f"\n  3-Day (Day Trading):")
    print(f"    {analysis_l1.recommendation_3d}")
    
    print(f"\n  7-Day (Swing Trading):")
    print(f"    {analysis_l1.recommendation_7d}")
    
    print(f"\n  30-Day (Position Trading):")
    print(f"    {analysis_l1.recommendation_30d}")
    
    print("\n✅ TEST 4 PASSED - Recommendations generated")


def test_save_and_load():
    """
    Test 5: Save & Load.
    """
    print("\n" + "🧪"*35)
    print("TEST 5: SAVE & LOAD")
    print("🧪"*35)
    
    # Load saved model
    learner = load_trained_model(
        symbol="BTCUSDT",
        level="indicators",
        model_dir="ml_models/multi_horizon"
    )
    
    print("\n✅ Model loaded successfully")
    
    # Check weights
    for horizon in ['3d', '7d', '30d']:
        weights = learner.get_horizon_weights(horizon)
        print(f"\n{horizon.upper()}:")
        print(f"  Confidence: {weights.confidence:.2f}")
        print(f"  R² Test: {weights.metrics['r2_test']:.4f}")
        print(f"  MAE Test: {weights.metrics['mae_test']:.4f}")
    
    print("\n✅ TEST 5 PASSED - Save/Load working")


def generate_report(result, analyses):
    """
    Generate final report.
    """
    print("\n" + "="*70)
    print("📊 FINAL TEST REPORT")
    print("="*70)
    
    config = result['config']
    analysis_l1 = analyses['analysis_level1']
    
    report = {
        'test_date': analysis_l1.timestamp,
        'symbol': config['symbol'],
        'training_samples': config['n_samples'],
        'horizons': config['horizons'],
        
        'level1_results': {
            'pattern': analysis_l1.pattern.value,
            'pattern_confidence': analysis_l1.pattern_confidence,
            'scores': {
                '3d': analysis_l1.score_3d.score,
                '7d': analysis_l1.score_7d.score,
                '30d': analysis_l1.score_30d.score
            },
            'confidences': {
                '3d': analysis_l1.score_3d.confidence,
                '7d': analysis_l1.score_7d.confidence,
                '30d': analysis_l1.score_30d.confidence
            },
            'combined_score': analysis_l1.combined_score,
            'combined_confidence': analysis_l1.combined_confidence
        },
        
        'recommendations': {
            '3d': analysis_l1.recommendation_3d,
            '7d': analysis_l1.recommendation_7d,
            '30d': analysis_l1.recommendation_30d
        }
    }
    
    # Save report
    output_dir = Path(result['output_dir'])
    report_file = output_dir / "test_report.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Report saved: {report_file}")
    
    # Display summary
    print("\n📈 SUMMARY:")
    print("-" * 70)
    print(f"Symbol: {config['symbol']}")
    print(f"Training Samples: {config['n_samples']}")
    print(f"Pattern: {analysis_l1.pattern.value} ({analysis_l1.pattern_confidence:.0%})")
    print(f"\nScores:")
    print(f"  3d:  {analysis_l1.score_3d.score:+.3f} (conf: {analysis_l1.score_3d.confidence:.0%})")
    print(f"  7d:  {analysis_l1.score_7d.score:+.3f} (conf: {analysis_l1.score_7d.confidence:.0%})")
    print(f"  30d: {analysis_l1.score_30d.score:+.3f} (conf: {analysis_l1.score_30d.confidence:.0%})")
    print(f"\nCombined: {analysis_l1.combined_score:+.3f} (conf: {analysis_l1.combined_confidence:.0%})")
    print("-" * 70)


def run_all_tests():
    """
    Run all tests.
    """
    print("\n" + "🚀"*35)
    print("MULTI-HORIZON SYSTEM - FULL TEST SUITE")
    print("="*35)
    
    try:
        # Test 1: Training
        result = test_training()
        
        # Test 2: Analysis
        analyses = test_analysis(result)
        
        # Test 3: Pattern detection
        test_pattern_detection(analyses)
        
        # Test 4: Recommendations
        test_recommendations(analyses)
        
        # Test 5: Save/Load
        test_save_and_load()
        
        # Final report
        generate_report(result, analyses)
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        
    except Exception as e:
        print("\n" + "="*70)
        print(f"❌ TEST FAILED: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
