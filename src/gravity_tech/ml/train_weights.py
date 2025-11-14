"""
Training Script for ML-Based Weight Optimization

This script trains a machine learning model to learn optimal weights
for combining indicator signals based on historical data.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import List
import matplotlib.pyplot as plt

from gravity_tech.models.schemas import Candle, AnalysisRequest
from gravity_tech.services.analysis_service import TechnicalAnalysisService
from gravity_tech.analysis.market_phase import analyze_market_phase
from gravity_tech.ml.weight_optimizer import MLWeightOptimizer


def generate_historical_data(days: int = 365, trend_type: str = "mixed") -> List[Candle]:
    """
    Generate synthetic historical data for training
    
    Args:
        days: Number of days of data
        trend_type: Type of trend ('bullish', 'bearish', 'mixed', 'ranging')
    """
    candles = []
    base_time = datetime.now() - timedelta(days=days)
    base_price = 100.0
    
    for i in range(days * 24):  # Hourly candles
        if trend_type == "bullish":
            drift = 0.01
        elif trend_type == "bearish":
            drift = -0.01
        elif trend_type == "ranging":
            drift = 0.0
        else:  # mixed
            # Change trend every 100 candles
            if (i // 100) % 2 == 0:
                drift = 0.01
            else:
                drift = -0.005
        
        # Random walk with drift
        price = base_price + drift * i + np.random.normal(0, 2)
        
        # Add volatility clusters
        if i % 200 < 50:  # High volatility period
            volatility = 3.0
        else:
            volatility = 1.0
        
        open_price = price + np.random.normal(0, volatility)
        close_price = price + np.random.normal(0, volatility)
        high_price = max(open_price, close_price) + abs(np.random.normal(0, volatility/2))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, volatility/2))
        
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=float(open_price),
            high=float(high_price),
            low=float(low_price),
            close=float(close_price),
            volume=float(1000 + np.random.normal(0, 200))
        ))
    
    return candles


async def prepare_training_dataset(num_samples: int = 1000,
                                   window_size: int = 100,
                                   prediction_horizon: int = 10) -> List[dict]:
    """
    Prepare training dataset from historical data
    
    Args:
        num_samples: Number of training samples to generate
        window_size: Size of candlestick window for analysis
        prediction_horizon: How many candles ahead to predict
        
    Returns:
        List of training samples with features and targets
    """
    print(f"Preparing {num_samples} training samples...")
    
    training_data = []
    
    # Generate different market conditions
    market_types = ["bullish", "bearish", "ranging", "mixed"]
    
    for sample_idx in range(num_samples):
        if sample_idx % 100 == 0:
            print(f"  Generated {sample_idx}/{num_samples} samples...")
        
        # Random market type
        market_type = np.random.choice(market_types)
        
        # Generate data with extra candles for prediction
        candles = generate_historical_data(
            days=int((window_size + prediction_horizon + 50) / 24),
            trend_type=market_type
        )
        
        # Random starting point
        start_idx = np.random.randint(50, len(candles) - window_size - prediction_horizon)
        window_candles = candles[start_idx:start_idx + window_size]
        
        # Perform analysis
        request = AnalysisRequest(
            symbol="TRAINING",
            timeframe="1h",
            candles=window_candles
        )
        
        try:
            result = await TechnicalAnalysisService.analyze(request)
            
            # Get market phase
            phase_result = analyze_market_phase(window_candles)
            market_phase = phase_result.get('market_phase', 'انتقال')
            
            # Prepare features
            optimizer = MLWeightOptimizer()
            features = optimizer.prepare_features(
                result.trend_indicators,
                result.momentum_indicators,
                result.cycle_indicators,
                result.volume_indicators,
                market_phase
            )
            
            # Calculate target (future return)
            current_price = window_candles[-1].close
            future_price = candles[start_idx + window_size + prediction_horizon].close
            target = ((future_price - current_price) / current_price) * 100
            
            training_data.append({
                'features': features,
                'target': target,
                'market_phase': market_phase,
                'sample_idx': sample_idx
            })
            
        except Exception as e:
            print(f"  ⚠️ Error in sample {sample_idx}: {e}")
            continue
    
    print(f"✅ Prepared {len(training_data)} valid training samples")
    return training_data


async def train_ml_model(num_samples: int = 1000,
                        model_type: str = "gradient_boosting",
                        save_model: bool = True):
    """
    Train ML model for weight optimization
    
    Args:
        num_samples: Number of training samples
        model_type: Type of ML model
        save_model: Whether to save trained model
    """
    print("="*70)
    print("ML Weight Optimization Training")
    print("="*70)
    print(f"\nModel Type: {model_type}")
    print(f"Training Samples: {num_samples}")
    print(f"Prediction Horizon: 10 candles ahead\n")
    
    # Prepare training data
    training_data = await prepare_training_dataset(num_samples=num_samples)
    
    if not training_data:
        print("❌ No training data generated!")
        return
    
    # Initialize optimizer
    print("\n📊 Training ML model...")
    optimizer = MLWeightOptimizer(model_type=model_type)
    
    # Train
    metrics = optimizer.train(training_data, validation_split=0.2)
    
    # Display results
    print("\n" + "="*70)
    print("Training Results")
    print("="*70)
    print(f"\nModel Performance:")
    print(f"  • Training R²: {metrics['train_r2']:.4f}")
    print(f"  • Validation R²: {metrics['val_r2']:.4f}")
    print(f"  • Cross-validation R² (mean): {metrics['cv_mean_r2']:.4f}")
    print(f"  • Cross-validation R² (std): {metrics['cv_std_r2']:.4f}")
    
    print(f"\nDataset Info:")
    print(f"  • Total samples: {metrics['n_samples']}")
    print(f"  • Number of features: {metrics['n_features']}")
    
    if metrics.get('optimal_weights'):
        print(f"\n🎯 Learned Optimal Weights:")
        weights = metrics['optimal_weights']
        print(f"  • Trend: {weights['trend']:.1%}")
        print(f"  • Momentum: {weights['momentum']:.1%}")
        print(f"  • Cycle: {weights['cycle']:.1%}")
        print(f"  • Volume: {weights['volume']:.1%}")
        
        # Compare with default weights
        print(f"\n📊 Comparison with Default Weights:")
        default = {'trend': 0.30, 'momentum': 0.25, 'cycle': 0.25, 'volume': 0.20}
        for key in weights:
            diff = weights[key] - default[key]
            sign = "+" if diff > 0 else ""
            print(f"  • {key.capitalize()}: {sign}{diff:.1%} ({weights[key]:.1%} vs {default[key]:.1%})")
    
    # Save model
    if save_model:
        print("\n💾 Saving model...")
        optimizer.save_model(name=f"ml_weights_{model_type}")
        print("✅ Model saved successfully!")
    
    print("\n" + "="*70)
    
    return optimizer, metrics


async def test_ml_model():
    """Test the trained ML model"""
    print("\n" + "="*70)
    print("Testing Trained ML Model")
    print("="*70)
    
    # Generate test data
    print("\nGenerating test market data...")
    candles = generate_historical_data(days=30, trend_type="mixed")
    
    # Perform analysis
    request = AnalysisRequest(
        symbol="TEST",
        timeframe="1h",
        candles=candles[-100:]  # Last 100 candles
    )
    
    result = await TechnicalAnalysisService.analyze(request)
    phase_result = analyze_market_phase(candles[-100:])
    
    # Load trained model
    print("Loading trained model...")
    optimizer = MLWeightOptimizer()
    try:
        optimizer.load_model(name="ml_weights_gradient_boosting")
    except FileNotFoundError:
        print("❌ No trained model found. Please train first.")
        return
    
    # Prepare features
    features = optimizer.prepare_features(
        result.trend_indicators,
        result.momentum_indicators,
        result.cycle_indicators,
        result.volume_indicators,
        phase_result.get('market_phase')
    )
    
    # Predict optimal weights
    predicted_weights = optimizer.predict_weights(features)
    
    print(f"\n🎯 ML-Predicted Optimal Weights:")
    print(f"  • Trend: {predicted_weights['trend']:.1%}")
    print(f"  • Momentum: {predicted_weights['momentum']:.1%}")
    print(f"  • Cycle: {predicted_weights['cycle']:.1%}")
    print(f"  • Volume: {predicted_weights['volume']:.1%}")
    
    print(f"\n📊 Market Context:")
    print(f"  • Phase: {phase_result.get('market_phase')}")
    print(f"  • Phase Strength: {phase_result.get('phase_strength')}")
    print(f"  • Overall Score: {phase_result.get('detailed_analysis', {}).get('overall_score', 0):.1f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode
        asyncio.run(test_ml_model())
    else:
        # Training mode
        num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 500
        asyncio.run(train_ml_model(num_samples=num_samples))
