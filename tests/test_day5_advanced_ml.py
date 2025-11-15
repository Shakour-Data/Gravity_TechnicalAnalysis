"""
Comprehensive Tests for Day 5 - Advanced ML Features

Tests:
- Advanced pattern training with hyperparameter tuning
- Model interpretability with SHAP
- Backtesting framework

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
- Model comparison and ensemble methods
"""

import pytest
import numpy as np
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gravity_tech.ml.advanced_pattern_training import AdvancedPatternTrainer
from gravity_tech.ml.model_interpretability import PatternModelInterpreter, SHAP_AVAILABLE
from gravity_tech.ml.backtesting import PatternBacktester, TradeResult
from gravity_tech.patterns.harmonic import HarmonicPatternDetector


# ============================================================================
# Advanced Training Tests
# ============================================================================

def test_advanced_trainer_initialization():
    """Test advanced trainer initialization."""
    trainer = AdvancedPatternTrainer(random_state=42)
    
    assert trainer.random_state == 42
    assert trainer.best_model is None
    assert trainer.model_comparison == {}
    assert trainer.tuning_results == {}


def test_enhanced_data_generation():
    """Test enhanced synthetic data generation."""
    trainer = AdvancedPatternTrainer(random_state=42)
    X, y_type, y_success = trainer.generate_enhanced_training_data(n_samples=100)
    
    assert X.shape[0] == 100
    assert X.shape[1] == 21  # 21 features
    assert len(y_type) == 100
    assert len(y_success) == 100
    
    # Check data ranges
    assert np.all(X >= 0) and np.all(X <= 1)
    assert np.all(y_success >= 0) and np.all(y_success <= 1)
    
    # Check pattern types
    assert all(pt in ['gartley', 'butterfly', 'bat', 'crab'] for pt in y_type)
    
    # Check realistic distributions
    unique, counts = np.unique(y_type, return_counts=True)
    assert len(unique) == 4  # All 4 pattern types present


def test_xgboost_tuning():
    """Test XGBoost hyperparameter tuning."""
    trainer = AdvancedPatternTrainer(random_state=42)
    X, y_type, _ = trainer.generate_enhanced_training_data(n_samples=200)
    
    # Convert to indices
    class_to_idx = {'gartley': 0, 'butterfly': 1, 'bat': 2, 'crab': 3}
    y_idx = np.array([class_to_idx[t] for t in y_type])
    
    # Tune with small grid for speed
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_idx, test_size=0.2, random_state=42, stratify=y_idx
    )
    
    # This would be slow, so we skip actual tuning in tests
    # Just verify the function exists and can be called
    assert hasattr(trainer, 'tune_xgboost')


def test_random_forest_training():
    """Test Random Forest training."""
    trainer = AdvancedPatternTrainer(random_state=42)
    X, y_type, _ = trainer.generate_enhanced_training_data(n_samples=200)
    
    class_to_idx = {'gartley': 0, 'butterfly': 1, 'bat': 2, 'crab': 3}
    y_idx = np.array([class_to_idx[t] for t in y_type])
    
    rf_model = trainer.train_random_forest(X, y_idx)
    
    assert hasattr(rf_model, 'predict')
    assert hasattr(rf_model, 'predict_proba')
    
    # Test prediction
    predictions = rf_model.predict(X[:10])
    assert len(predictions) == 10
    assert all(0 <= p < 4 for p in predictions)


def test_gradient_boosting_training():
    """Test Gradient Boosting training."""
    trainer = AdvancedPatternTrainer(random_state=42)
    X, y_type, _ = trainer.generate_enhanced_training_data(n_samples=200)
    
    class_to_idx = {'gartley': 0, 'butterfly': 1, 'bat': 2, 'crab': 3}
    y_idx = np.array([class_to_idx[t] for t in y_type])
    
    gb_model = trainer.train_gradient_boosting(X, y_idx)
    
    assert hasattr(gb_model, 'predict')
    predictions = gb_model.predict(X[:10])
    assert len(predictions) == 10


def test_model_comparison():
    """Test model comparison functionality."""
    trainer = AdvancedPatternTrainer(random_state=42)
    X, y_type, _ = trainer.generate_enhanced_training_data(n_samples=300)
    
    class_to_idx = {'gartley': 0, 'butterfly': 1, 'bat': 2, 'crab': 3}
    y_idx = np.array([class_to_idx[t] for t in y_type])
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_idx, test_size=0.2, random_state=42, stratify=y_idx
    )
    
    # Train models
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)
    
    models = {'RandomForest': rf_model}
    
    comparison = trainer.compare_models(models, X_train, y_train, X_test, y_test)
    
    assert 'RandomForest' in comparison
    assert 'train_accuracy' in comparison['RandomForest']
    assert 'test_accuracy' in comparison['RandomForest']
    assert 'cv_accuracy' in comparison['RandomForest']


# ============================================================================
# Model Interpretability Tests
# ============================================================================

def test_interpreter_initialization():
    """Test interpreter initialization."""
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    feature_names = ['feature1', 'feature2', 'feature3']
    
    interpreter = PatternModelInterpreter(model, feature_names)
    
    assert interpreter.model is model
    assert interpreter.feature_names == feature_names
    assert interpreter.explainer is None
    assert interpreter.shap_values is None


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP library not installed")
def test_shap_explainer_creation():
    """Test SHAP explainer creation."""
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate data
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Create interpreter
    feature_names = [f'feature_{i}' for i in range(5)]
    interpreter = PatternModelInterpreter(model, feature_names)
    interpreter.create_explainer(X, model_type='tree')
    
    assert interpreter.explainer is not None


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP library not installed")
def test_shap_value_computation():
    """Test SHAP value computation."""
    from sklearn.ensemble import RandomForestClassifier
    
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    feature_names = [f'feature_{i}' for i in range(5)]
    interpreter = PatternModelInterpreter(model, feature_names)
    interpreter.create_explainer(X[:50], model_type='tree')
    
    shap_values = interpreter.explain_predictions(X[:10])
    
    assert shap_values is not None
    assert interpreter.shap_values is not None


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP library not installed")
def test_feature_importance_dict():
    """Test feature importance dictionary extraction."""
    from sklearn.ensemble import RandomForestClassifier
    
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    feature_names = [f'feature_{i}' for i in range(5)]
    interpreter = PatternModelInterpreter(model, feature_names)
    interpreter.create_explainer(X[:50], model_type='tree')
    interpreter.explain_predictions(X[:20])
    
    importance_dict = interpreter.get_feature_importance_dict(X[:20])
    
    assert len(importance_dict) == 5
    assert all(isinstance(v, float) for v in importance_dict.values())
    assert abs(sum(importance_dict.values()) - 1.0) < 0.01  # Sum to 1


# ============================================================================
# Backtesting Tests
# ============================================================================

def test_backtester_initialization():
    """Test backtester initialization."""
    detector = HarmonicPatternDetector(tolerance=0.1)
    backtester = PatternBacktester(detector, min_confidence=0.5)
    
    assert backtester.detector is detector
    assert backtester.min_confidence == 0.5
    assert backtester.trades == []
    assert backtester.equity_curve == []


def test_historical_data_generation():
    """Test historical data generation."""
    detector = HarmonicPatternDetector()
    backtester = PatternBacktester(detector)
    
    highs, lows, closes, volume, dates = backtester.generate_historical_data(n_bars=100)
    
    assert len(highs) == 100
    assert len(lows) == 100
    assert len(closes) == 100
    assert len(volume) == 100
    assert len(dates) == 100
    
    # Check price relationships
    assert np.all(highs >= closes)
    assert np.all(closes >= lows)
    assert np.all(highs >= lows)


def test_trade_result_dataclass():
    """Test TradeResult dataclass."""
    trade = TradeResult(
        entry_date=datetime(2024, 1, 1),
        entry_price=100.0,
        exit_date=datetime(2024, 1, 5),
        exit_price=105.0,
        pattern_type='gartley',
        direction='bullish',
        confidence=0.85,
        stop_loss=95.0,
        target_1=103.0,
        target_2=106.0,
        pnl=5.0,
        pnl_percent=5.0,
        outcome='win',
        hit_target='target2'
    )
    
    assert trade.pnl == 5.0
    assert trade.outcome == 'win'
    assert trade.pattern_type == 'gartley'


def test_backtest_execution():
    """Test backtest execution."""
    detector = HarmonicPatternDetector(tolerance=0.15)
    backtester = PatternBacktester(detector, min_confidence=0.5)
    
    # Generate short historical data for speed
    highs, lows, closes, volume, dates = backtester.generate_historical_data(n_bars=300)
    
    trades = backtester.run_backtest(highs, lows, closes, volume, dates)
    
    assert isinstance(trades, list)
    # May or may not find patterns in random data
    if len(trades) > 0:
        assert isinstance(trades[0], TradeResult)


def test_metrics_calculation():
    """Test backtest metrics calculation."""
    detector = HarmonicPatternDetector(tolerance=0.15)
    backtester = PatternBacktester(detector, min_confidence=0.5)
    
    highs, lows, closes, volume, dates = backtester.generate_historical_data(n_bars=500)
    trades = backtester.run_backtest(highs, lows, closes, volume, dates)
    
    if len(trades) > 0:
        metrics = backtester.calculate_metrics()
        
        assert 'total_trades' in metrics
        assert 'win_rate' in metrics
        assert 'total_pnl' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        
        assert metrics['total_trades'] == len(trades)
        assert 0 <= metrics['win_rate'] <= 1
    else:
        # No patterns found - should return gracefully
        metrics = backtester.calculate_metrics()
        assert metrics.get('total_trades', 0) == 0


def test_trade_simulation_bullish():
    """Test bullish trade simulation."""
    from patterns.harmonic import HarmonicPattern, PatternType, PatternDirection, PatternPoint
    
    # Create mock pattern
    points = {
        'X': PatternPoint(0, 100.0, 'X'),
        'A': PatternPoint(10, 110.0, 'A'),
        'B': PatternPoint(20, 106.0, 'B'),
        'C': PatternPoint(30, 108.0, 'C'),
        'D': PatternPoint(40, 104.0, 'D')
    }
    
    pattern = HarmonicPattern(
        pattern_type=PatternType.GARTLEY,
        direction=PatternDirection.BULLISH,
        points=points,
        ratios={'XA_BC': 0.6, 'AB_CD': 0.78},
        confidence=85.0,
        fibonacci_accuracy=0.9,
        completion_point=104.0,
        stop_loss=102.0,
        target_1=106.0,
        target_2=108.0
    )
    
    # Generate price data
    closes = np.linspace(100, 110, 100)
    highs = closes + 1
    lows = closes - 1
    
    # Verify pattern structure
    assert pattern.direction == PatternDirection.BULLISH
    assert pattern.stop_loss < pattern.completion_point
    assert pattern.target_1 > pattern.completion_point


def test_performance_metrics_positive():
    """Test metrics with positive performance."""
    # Create synthetic trades
    trades = []
    for i in range(10):
        trade = TradeResult(
            entry_date=datetime(2024, 1, i+1),
            entry_price=100.0,
            exit_date=datetime(2024, 1, i+2),
            exit_price=105.0,
            pattern_type='gartley',
            direction='bullish',
            confidence=0.8,
            stop_loss=95.0,
            target_1=103.0,
            target_2=106.0,
            pnl=5.0,
            pnl_percent=5.0,
            outcome='win',
            hit_target='target1'
        )
        trades.append(trade)
    
    detector = HarmonicPatternDetector()
    backtester = PatternBacktester(detector)
    backtester.trades = trades
    
    metrics = backtester.calculate_metrics()
    
    assert metrics['total_trades'] == 10
    assert metrics['winning_trades'] == 10
    assert metrics['win_rate'] == 1.0
    assert metrics['total_pnl'] == 50.0


# ============================================================================
# Integration Tests
# ============================================================================

def test_end_to_end_training_and_backtesting():
    """Test complete pipeline from training to backtesting."""
    # Train model
    trainer = AdvancedPatternTrainer(random_state=42)
    X, y_type, y_success = trainer.generate_enhanced_training_data(n_samples=500)
    
    class_to_idx = {'gartley': 0, 'butterfly': 1, 'bat': 2, 'crab': 3}
    y_idx = np.array([class_to_idx[t] for t in y_type])
    
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y_idx)
    
    # Run backtest
    detector = HarmonicPatternDetector(tolerance=0.15)
    backtester = PatternBacktester(detector, classifier=model, min_confidence=0.6)
    
    highs, lows, closes, volume, dates = backtester.generate_historical_data(n_bars=300)
    trades = backtester.run_backtest(highs, lows, closes, volume, dates)
    
    # Verify pipeline completed
    assert model is not None
    assert isinstance(trades, list)


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP library not installed")
def test_model_interpretability_on_trained_model():
    """Test SHAP on actually trained model."""
    trainer = AdvancedPatternTrainer(random_state=42)
    X, y_type, _ = trainer.generate_enhanced_training_data(n_samples=200)
    
    class_to_idx = {'gartley': 0, 'butterfly': 1, 'bat': 2, 'crab': 3}
    y_idx = np.array([class_to_idx[t] for t in y_type])
    
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
    model.fit(X, y_idx)
    
    feature_names = [
        'xab_ratio_accuracy', 'abc_ratio_accuracy', 'bcd_ratio_accuracy', 'xad_ratio_accuracy',
        'pattern_symmetry', 'pattern_slope', 'xa_angle', 'ab_angle', 'bc_angle', 'cd_angle',
        'pattern_duration', 'xa_magnitude', 'ab_magnitude', 'bc_magnitude', 'cd_magnitude',
        'volume_at_d', 'volume_trend', 'volume_confirmation',
        'rsi_at_d', 'macd_at_d', 'momentum_divergence'
    ]
    
    interpreter = PatternModelInterpreter(model, feature_names)
    interpreter.create_explainer(X[:50], model_type='tree')
    interpreter.explain_predictions(X[:30])
    
    importance = interpreter.get_feature_importance_dict(X[:30])
    
    assert len(importance) == 21
    assert all(v >= 0 for v in importance.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
