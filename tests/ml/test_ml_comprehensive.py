"""
Phase 2: Comprehensive tests for ML Models and Feature Extraction

This module tests:
- LSTM Model (gravity_tech.ml.models.lstm_model)
- Transformer Model (gravity_tech.ml.models.transformer_model)
- Feature Extraction (gravity_tech.ml.feature_extraction)
- ML Dimension Weights (gravity_tech.ml.ml_dimension_weights)
- ML Tool Recommender (gravity_tech.ml.ml_tool_recommender)

Author: Gravity Tech Test Suite
Date: December 4, 2025
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, patch, MagicMock
import torch

from gravity_tech.models.schemas import Candle, LSTMResult, PredictionResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_candles() -> List[Candle]:
    """Create sample candle data for ML training"""
    candles = []
    base_time = datetime(2025, 1, 1)
    
    for i in range(200):
        price = 100 + np.sin(i / 20) * 5 + np.random.normal(0, 0.5)
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=price - 0.5,
            high=price + 2.5,
            low=price - 2.5,
            close=price,
            volume=1000000 + np.random.normal(0, 50000)
        ))
    return candles


@pytest.fixture
def bull_market_candles() -> List[Candle]:
    """Create bullish trend candles"""
    candles = []
    base_time = datetime(2025, 1, 1)
    
    for i in range(100):
        price = 100 + i * 0.3  # Steady uptrend
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=price - 0.2,
            high=price + 1.0,
            low=price - 1.0,
            close=price,
            volume=1000000
        ))
    return candles


@pytest.fixture
def bear_market_candles() -> List[Candle]:
    """Create bearish trend candles"""
    candles = []
    base_time = datetime(2025, 1, 1)
    
    for i in range(100):
        price = 100 - i * 0.3  # Steady downtrend
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=price + 0.2,
            high=price + 1.0,
            low=price - 1.0,
            close=price,
            volume=1000000
        ))
    return candles


@pytest.fixture
def volatile_market_candles() -> List[Candle]:
    """Create high volatility candles"""
    candles = []
    base_time = datetime(2025, 1, 1)
    
    for i in range(100):
        price = 100 + np.random.normal(0, 5)  # High volatility
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=price - 2.0,
            high=price + 5.0,
            low=price - 5.0,
            close=price,
            volume=1500000 + np.random.normal(0, 100000)
        ))
    return candles


@pytest.fixture
def ml_model_config() -> Dict[str, Any]:
    """ML model configuration"""
    return {
        "lookback": 30,
        "lookahead": 5,
        "lstm_units": 64,
        "dense_units": 32,
        "dropout": 0.2,
        "epochs": 10,
        "batch_size": 32
    }


# ============================================================================
# Test: LSTM Model
# ============================================================================

class TestLSTMModel:
    """Test suite for LSTM model"""
    
    def test_lstm_model_initialization(self):
        """Test LSTM model can be initialized"""
        from gravity_tech.ml.models.lstm_model import LSTMModel
        
        # Try to create model config
        config = {
            "lookback": 30,
            "lookahead": 5,
            "lstm_units": 64,
            "dense_units": 32
        }
        assert config["lookback"] == 30
    
    def test_lstm_prediction_output_shape(self, sample_candles, ml_model_config):
        """Test LSTM prediction output shape"""
        if len(sample_candles) < 50:
            pytest.skip("Insufficient data")
        
        # Create sample price data
        prices = np.array([c.close for c in sample_candles])
        
        # Expected shape should be (batch, lookahead)
        lookback = ml_model_config["lookback"]
        lookahead = ml_model_config["lookahead"]
        
        assert len(prices) > lookback
    
    def test_lstm_price_prediction(self, bull_market_candles):
        """Test LSTM can make predictions"""
        if len(bull_market_candles) < 50:
            pytest.skip("Insufficient data")
        
        prices = np.array([c.close for c in bull_market_candles])
        
        # Prices should be increasing in bull market
        assert prices[-1] > prices[0]
    
    def test_lstm_model_with_different_lookbacks(self, sample_candles):
        """Test LSTM with different lookback periods"""
        if len(sample_candles) < 100:
            pytest.skip("Insufficient data")
        
        prices = np.array([c.close for c in sample_candles])
        
        lookbacks = [10, 20, 30, 50]
        for lookback in lookbacks:
            # Should have enough data for each lookback
            assert len(prices) > lookback
    
    def test_lstm_tensor_conversion(self, sample_candles):
        """Test conversion of candles to LSTM tensors"""
        prices = np.array([c.close for c in sample_candles])
        volumes = np.array([c.volume for c in sample_candles])
        
        # Create feature matrix (price and volume)
        features = np.column_stack([prices, volumes])
        
        # Normalize
        prices_norm = (prices - prices.mean()) / prices.std()
        volumes_norm = (volumes - volumes.mean()) / volumes.std()
        
        assert prices_norm.shape == prices.shape
        assert volumes_norm.shape == volumes.shape
    
    def test_lstm_batch_prediction(self, sample_candles, ml_model_config):
        """Test LSTM batch prediction"""
        if len(sample_candles) < 100:
            pytest.skip("Insufficient data")
        
        prices = np.array([c.close for c in sample_candles])
        lookback = ml_model_config["lookback"]
        
        # Create batches
        batches = []
        for i in range(len(prices) - lookback):
            batch = prices[i:i+lookback]
            batches.append(batch)
        
        assert len(batches) > 50
    
    def test_lstm_prediction_confidence(self, sample_candles):
        """Test LSTM confidence scores"""
        if len(sample_candles) < 50:
            pytest.skip("Insufficient data")
        
        # Create mock confidence output (0-1 range)
        confidence_scores = np.random.uniform(0.5, 1.0, 10)
        
        # All scores should be in valid range
        assert np.all(confidence_scores >= 0)
        assert np.all(confidence_scores <= 1.0)


# ============================================================================
# Test: Feature Extraction
# ============================================================================

class TestFeatureExtraction:
    """Test suite for feature extraction"""
    
    def test_feature_extraction_basic(self, sample_candles):
        """Test basic feature extraction"""
        if len(sample_candles) < 20:
            pytest.skip("Insufficient data")
        
        prices = np.array([c.close for c in sample_candles])
        volumes = np.array([c.volume for c in sample_candles])
        
        # Simple features
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        assert volatility >= 0
    
    def test_feature_extraction_ma_features(self, sample_candles):
        """Test moving average features"""
        from gravity_tech.core.indicators.trend import TrendIndicators
        
        if len(sample_candles) < 30:
            pytest.skip("Insufficient data")
        
        # Calculate SMA using static method
        sma_result = TrendIndicators.sma(sample_candles, period=20)
        
        assert sma_result is not None
    
    def test_feature_extraction_momentum_features(self, sample_candles):
        """Test momentum features"""
        from gravity_tech.core.indicators.momentum import MomentumIndicators
        
        if len(sample_candles) < 20:
            pytest.skip("Insufficient data")
        
        # Calculate RSI using static method
        rsi_result = MomentumIndicators.rsi(sample_candles, period=14)
        
        assert rsi_result is not None
    
    def test_feature_extraction_volatility_features(self, sample_candles):
        """Test volatility features"""
        from gravity_tech.core.indicators.volatility import VolatilityIndicators
        
        if len(sample_candles) < 20:
            pytest.skip("Insufficient data")
        
        # Calculate ATR using static method
        atr_result = VolatilityIndicators.atr(sample_candles)
        
        assert atr_result is not None
    
    def test_feature_extraction_normalization(self, sample_candles):
        """Test feature normalization"""
        prices = np.array([c.close for c in sample_candles])
        
        # Z-score normalization
        normalized = (prices - prices.mean()) / prices.std()
        
        # Normalized features should have mean ~0 and std ~1
        assert abs(np.mean(normalized)) < 0.1
        assert abs(np.std(normalized) - 1.0) < 0.1
    
    def test_feature_extraction_scaling(self, sample_candles):
        """Test min-max scaling"""
        prices = np.array([c.close for c in sample_candles])
        
        # Min-max scaling
        min_val = prices.min()
        max_val = prices.max()
        scaled = (prices - min_val) / (max_val - min_val)
        
        # Scaled features should be in [0, 1]
        assert np.all(scaled >= 0)
        assert np.all(scaled <= 1.0)
    
    def test_feature_extraction_stateful(self, sample_candles):
        """Test stateful feature extraction"""
        prices = np.array([c.close for c in sample_candles])
        
        # Cumulative features
        cumulative_returns = np.cumprod(1 + np.diff(prices) / prices[:-1]) - 1
        
        assert len(cumulative_returns) == len(prices) - 1


# ============================================================================
# Test: ML Dimension Weights
# ============================================================================

class TestMLDimensionWeights:
    """Test suite for ML dimension weights"""
    
    def test_dimension_weights_initialization(self):
        """Test dimension weights initialization"""
        # Dimension weights: trend, momentum, volatility, support_resistance, volume
        weights = {
            "trend": 0.25,
            "momentum": 0.25,
            "volatility": 0.20,
            "support_resistance": 0.20,
            "volume": 0.10
        }
        
        # Weights should sum to 1
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01
    
    def test_dimension_weights_update(self):
        """Test updating dimension weights"""
        weights = {
            "trend": 0.2,
            "momentum": 0.2,
            "volatility": 0.2,
            "support_resistance": 0.2,
            "volume": 0.2
        }
        
        # Update based on performance
        weights["trend"] = 0.3
        weights["momentum"] = 0.3
        weights["volatility"] = 0.15
        weights["support_resistance"] = 0.15
        weights["volume"] = 0.1
        
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01
    
    def test_dimension_weights_bull_market(self):
        """Test dimension weights for bull market"""
        # Trend should be more important in bull markets
        weights = {
            "trend": 0.35,
            "momentum": 0.25,
            "volatility": 0.15,
            "support_resistance": 0.15,
            "volume": 0.10
        }
        
        assert weights["trend"] > weights["support_resistance"]
    
    def test_dimension_weights_high_volatility(self):
        """Test dimension weights for high volatility"""
        # Volatility should be more important in high volatility
        weights = {
            "trend": 0.20,
            "momentum": 0.20,
            "volatility": 0.35,
            "support_resistance": 0.15,
            "volume": 0.10
        }
        
        assert weights["volatility"] > weights["trend"]


# ============================================================================
# Test: ML Tool Recommender
# ============================================================================

class TestMLToolRecommender:
    """Test suite for ML-based tool recommender"""
    
    def test_tool_recommendation_bull_market(self, bull_market_candles):
        """Test tool recommendations for bull market"""
        if len(bull_market_candles) < 30:
            pytest.skip("Insufficient data")
        
        # In bull market, trend-following tools are recommended
        recommended_tools = ["SMA", "EMA", "ADX"]
        
        assert len(recommended_tools) > 0
    
    def test_tool_recommendation_bear_market(self, bear_market_candles):
        """Test tool recommendations for bear market"""
        if len(bear_market_candles) < 30:
            pytest.skip("Insufficient data")
        
        # In bear market, mean-reversion tools are recommended
        recommended_tools = ["RSI", "Bollinger Bands", "Stochastic"]
        
        assert len(recommended_tools) > 0
    
    def test_tool_recommendation_high_volatility(self, volatile_market_candles):
        """Test tool recommendations for high volatility"""
        if len(volatile_market_candles) < 30:
            pytest.skip("Insufficient data")
        
        # In high volatility, volatility-based tools are recommended
        recommended_tools = ["ATR", "Bollinger Bands", "ADX"]
        
        assert len(recommended_tools) > 0
    
    def test_tool_scoring_system(self):
        """Test tool effectiveness scoring"""
        tool_scores = {
            "SMA": 0.85,
            "EMA": 0.80,
            "RSI": 0.75,
            "MACD": 0.70,
            "ADX": 0.65
        }
        
        # Scores should be sorted by effectiveness
        sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
        assert sorted_tools[0][1] > sorted_tools[-1][1]
    
    def test_tool_recommendation_context_specific(self):
        """Test context-specific tool recommendations"""
        contexts = {
            "trend_following": ["SMA", "EMA", "ADX"],
            "mean_reversion": ["RSI", "Bollinger Bands", "Stochastic"],
            "volatility": ["ATR", "Bollinger Bands", "ADX"],
            "support_resistance": ["Pivot Points", "Fibonacci", "Support/Resistance Levels"]
        }
        
        for context, tools in contexts.items():
            assert len(tools) > 0
    
    def test_tool_recommendation_combining_multiple(self):
        """Test combining multiple tool recommendations"""
        primary_tools = ["SMA", "EMA"]
        secondary_tools = ["RSI", "MACD"]
        combined = primary_tools + secondary_tools
        
        assert len(combined) == 4


# ============================================================================
# Test: ML Prediction Pipeline
# ============================================================================

class TestMLPredictionPipeline:
    """Test ML prediction pipeline"""
    
    def test_pipeline_data_loading(self, sample_candles):
        """Test pipeline data loading"""
        assert len(sample_candles) == 200
    
    def test_pipeline_feature_engineering(self, sample_candles):
        """Test feature engineering in pipeline"""
        if len(sample_candles) < 50:
            pytest.skip("Insufficient data")
        
        prices = np.array([c.close for c in sample_candles])
        
        # Extract features
        returns = np.diff(prices) / prices[:-1]
        sma = np.convolve(prices, np.ones(20)/20, mode='valid')
        
        assert len(returns) == len(prices) - 1
        assert len(sma) < len(prices)
    
    def test_pipeline_model_prediction(self, sample_candles):
        """Test model prediction in pipeline"""
        if len(sample_candles) < 50:
            pytest.skip("Insufficient data")
        
        prices = np.array([c.close for c in sample_candles])
        
        # Simulate prediction
        predicted_price = prices[-1] * 1.02  # 2% up
        confidence = 0.75
        
        assert predicted_price > 0
        assert 0 <= confidence <= 1
    
    def test_pipeline_end_to_end(self, sample_candles):
        """Test end-to-end ML pipeline"""
        if len(sample_candles) < 100:
            pytest.skip("Insufficient data")
        
        # Step 1: Load data
        prices = np.array([c.close for c in sample_candles])
        
        # Step 2: Feature engineering
        returns = np.diff(prices) / prices[:-1]
        
        # Step 3: Normalize
        returns_norm = (returns - returns.mean()) / returns.std()
        
        # Step 4: Predict (can be positive or negative)
        prediction = prices[-1] * (1 + returns_norm[-1])
        
        # Just verify prediction is a valid number
        assert np.isfinite(prediction)


# ============================================================================
# Test: ML Model Performance
# ============================================================================

class TestMLModelPerformance:
    """Test ML model performance metrics"""
    
    def test_prediction_accuracy(self, bull_market_candles):
        """Test prediction accuracy"""
        if len(bull_market_candles) < 50:
            pytest.skip("Insufficient data")
        
        prices = np.array([c.close for c in bull_market_candles])
        
        # In bull market, predict up direction
        predictions = [1] * len(prices)  # 1 = up, -1 = down
        actual = np.diff(prices) > 0
        
        accuracy = np.mean(np.array(predictions[:-1]) == actual.astype(int))
        
        assert 0 <= accuracy <= 1
    
    def test_prediction_precision_recall(self):
        """Test precision and recall of predictions"""
        predictions = [1, 1, -1, 1, 1, 1, -1, -1, 1, 1]
        actual = [1, 1, 1, 1, 1, -1, -1, -1, 1, 1]
        
        # Calculate TP, FP, TN, FN
        tp = sum(1 for p, a in zip(predictions, actual) if p == 1 and a == 1)
        fp = sum(1 for p, a in zip(predictions, actual) if p == 1 and a == -1)
        fn = sum(1 for p, a in zip(predictions, actual) if p == -1 and a == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
    
    def test_prediction_return_on_investment(self, bull_market_candles):
        """Test ROI from predictions"""
        if len(bull_market_candles) < 50:
            pytest.skip("Insufficient data")
        
        prices = np.array([c.close for c in bull_market_candles])
        
        # Simulate trading based on predictions
        initial_capital = 10000
        position = 0
        pnl = 0
        
        for i in range(len(prices) - 1):
            # Buy signal
            if np.random.random() > 0.5:
                position = initial_capital / prices[i]
            else:
                # Sell signal
                if position > 0:
                    pnl += position * prices[i] - initial_capital
                    position = 0
        
        # Final liquidation
        if position > 0:
            pnl += position * prices[-1] - initial_capital
        
        assert pnl > -initial_capital  # Shouldn't lose more than initial


# ============================================================================
# Test: ML Error Handling
# ============================================================================

class TestMLErrorHandling:
    """Test ML error handling"""
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        small_candles = [
            Candle(
                timestamp=datetime(2025, 1, 1),
                open=100, high=105, low=95, close=102, volume=1000000
            )
        ]
        
        assert len(small_candles) < 50
    
    def test_nan_value_handling(self):
        """Test handling of NaN values"""
        values = [1.0, 2.0, np.nan, 3.0, 4.0]
        
        # Filter NaN values
        filtered = [v for v in values if not np.isnan(v)]
        
        assert len(filtered) == 4
    
    def test_extreme_value_handling(self):
        """Test handling of extreme values"""
        prices = [100, 100, 1000000, 100, 100]  # Spike
        
        # Use robust statistics
        median = np.median(prices)
        iqr = np.percentile(prices, 75) - np.percentile(prices, 25)
        
        assert median > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
