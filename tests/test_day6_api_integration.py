"""
Day 6 API Integration Tests

Test Pattern Recognition and ML endpoints.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import requests
import numpy as np
from datetime import datetime, timedelta

BASE_URL = "http://127.0.0.1:8000"

def test_health_checks():
    """Test all health endpoints"""
    print("=" * 80)
    print("🏥 Testing Health Endpoints")
    print("=" * 80)
    
    # Main health
    response = requests.get(f"{BASE_URL}/health")
    print(f"\n✅ Main Health: {response.status_code}")
    print(response.json())
    
    # Pattern service health
    response = requests.get(f"{BASE_URL}/api/v1/patterns/health")
    print(f"\n✅ Pattern Service Health: {response.status_code}")
    print(response.json())
    
    # ML service health
    response = requests.get(f"{BASE_URL}/api/v1/ml/health")
    print(f"\n✅ ML Service Health: {response.status_code}")
    print(response.json())


def test_pattern_types():
    """Test pattern types listing"""
    print("\n" + "=" * 80)
    print("📊 Testing Pattern Types Endpoint")
    print("=" * 80)
    
    response = requests.get(f"{BASE_URL}/api/v1/patterns/types")
    data = response.json()
    
    print(f"\n✅ Status Code: {response.status_code}")
    print(f"Total Patterns: {data['total']}")
    for pattern in data['patterns']:
        print(f"\n{pattern['name']}:")
        print(f"  Type: {pattern['type']}")
        print(f"  Reliability: {pattern['reliability']}")


def generate_sample_candles(n_candles=100):
    """Generate sample candle data"""
    base_price = 50000
    candles = []
    timestamp = int(datetime.now().timestamp())
    
    for i in range(n_candles):
        # Generate realistic OHLCV
        open_price = base_price + np.random.randn() * 100
        close_price = open_price + np.random.randn() * 150
        high_price = max(open_price, close_price) + abs(np.random.randn()) * 50
        low_price = min(open_price, close_price) - abs(np.random.randn()) * 50
        volume = abs(np.random.randn()) * 1000 + 500
        
        candles.append({
            "timestamp": timestamp - (n_candles - i) * 3600,
            "open": float(open_price),
            "high": float(high_price),
            "low": float(low_price),
            "close": float(close_price),
            "volume": float(volume)
        })
    
    return candles


def test_pattern_detection():
    """Test pattern detection endpoint"""
    print("\n" + "=" * 80)
    print("🔍 Testing Pattern Detection")
    print("=" * 80)
    
    candles = generate_sample_candles(100)
    
    payload = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "candles": candles,
        "use_ml": True,
        "min_confidence": 0.5,
        "tolerance": 0.05
    }
    
    response = requests.post(f"{BASE_URL}/api/v1/patterns/detect", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Status Code: {response.status_code}")
        print(f"Symbol: {data['symbol']}")
        print(f"Timeframe: {data['timeframe']}")
        print(f"Patterns Found: {data['patterns_found']}")
        print(f"Analysis Time: {data['analysis_time_ms']:.2f}ms")
        print(f"ML Enabled: {data['ml_enabled']}")
        
        for i, pattern in enumerate(data['patterns'][:3]):  # Show first 3
            print(f"\nPattern {i+1}:")
            print(f"  Type: {pattern['pattern_type']}")
            print(f"  Direction: {pattern['direction']}")
            print(f"  Confidence: {pattern.get('confidence', 'N/A')}")
            print(f"  Completion Price: ${pattern['completion_price']:.2f}")
            if pattern.get('targets'):
                print(f"  Target 1: ${pattern['targets']['target1']:.2f}")
                print(f"  Target 2: ${pattern['targets']['target2']:.2f}")
            print(f"  Stop Loss: ${pattern.get('stop_loss', 'N/A'):.2f}")
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)


def test_ml_model_info():
    """Test ML model info endpoint"""
    print("\n" + "=" * 80)
    print("🤖 Testing ML Model Info")
    print("=" * 80)
    
    response = requests.get(f"{BASE_URL}/api/v1/ml/model/info")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Status Code: {response.status_code}")
        print(f"Model Name: {data['model_name']}")
        print(f"Model Version: {data['model_version']}")
        print(f"Model Type: {data['model_type']}")
        print(f"Training Date: {data.get('training_date', 'N/A')}")
        print(f"Accuracy: {data.get('accuracy', 'N/A')}")
        print(f"Features Count: {data['features_count']}")
        print(f"Supported Patterns: {', '.join(data['supported_patterns'])}")
    else:
        print(f"\n⚠️ Status Code: {response.status_code}")
        print(response.text)


def test_ml_prediction():
    """Test ML prediction endpoint"""
    print("\n" + "=" * 80)
    print("🔮 Testing ML Prediction")
    print("=" * 80)
    
    # Sample features
    features = {
        "xab_ratio_accuracy": 0.95,
        "abc_ratio_accuracy": 0.87,
        "bcd_ratio_accuracy": 0.92,
        "xad_ratio_accuracy": 0.88,
        "pattern_symmetry": 0.85,
        "pattern_slope": 0.3,
        "xa_angle": 45.0,
        "ab_angle": 30.0,
        "bc_angle": 40.0,
        "cd_angle": 35.0,
        "pattern_duration": 0.6,
        "xa_magnitude": 0.05,
        "ab_magnitude": 0.03,
        "bc_magnitude": 0.04,
        "cd_magnitude": 0.03,
        "volume_at_d": 0.7,
        "volume_trend": 0.6,
        "volume_confirmation": 0.8,
        "rsi_at_d": 55.0,
        "macd_at_d": 0.15,
        "momentum_divergence": 0.65
    }
    
    payload = {"features": features}
    
    response = requests.post(f"{BASE_URL}/api/v1/ml/predict", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Status Code: {response.status_code}")
        print(f"Predicted Pattern: {data['predicted_pattern']}")
        print(f"Confidence: {data['confidence']:.4f}")
        print(f"Model Version: {data['model_version']}")
        print(f"Inference Time: {data['inference_time_ms']:.2f}ms")
        print("\nProbabilities:")
        for pattern, prob in data['probabilities'].items():
            print(f"  {pattern}: {prob:.4f}")
    else:
        print(f"\n⚠️ Status Code: {response.status_code}")
        print(response.text)


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("🚀 Day 6 API Integration Tests")
    print("=" * 80)
    
    try:
        test_health_checks()
        test_pattern_types()
        test_pattern_detection()
        test_ml_model_info()
        test_ml_prediction()
        
        print("\n" + "=" * 80)
        print("✅ All API Tests Completed Successfully!")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server")
        print("Please make sure the server is running: python -m uvicorn main:app --port 8000")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
