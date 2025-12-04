"""
Feature Extraction for Machine Learning Weight Optimization

This module extracts features from historical candle data:
- 10 trend indicator signals and confidences
- 4 dimension scores (Indicators, Candlestick, Elliott, Classical)
- Future returns as labels for supervised learning

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

from gravity_tech.models.schemas import Candle, IndicatorResult, SignalStrength
from gravity_tech.indicators.trend import TrendIndicators
from gravity_tech.patterns.candlestick import CandlestickPatterns
from gravity_tech.patterns.elliott_wave import ElliottWaveAnalyzer
from gravity_tech.patterns.classical import ClassicalPatterns


class FeatureExtractor:
    """
    Extract features from candles for ML training
    """
    
    def __init__(self, lookback_period: int = 100, forward_days: int = 5):
        """
        Initialize feature extractor
        
        Args:
            lookback_period: Number of candles to use for indicator calculation
            forward_days: Number of days ahead to calculate returns (label)
        """
        self.lookback_period = lookback_period
        self.forward_days = forward_days
    
    def extract_10_indicator_features(
        self,
        candles: List[Candle]
    ) -> Dict[str, float]:
        """
        Extract features from 10 trend indicators
        
        Returns:
            Dictionary with indicator signals and confidences
        """
        if len(candles) < self.lookback_period:
            raise ValueError(f"Need at least {self.lookback_period} candles")
        
        # Use last lookback_period candles for indicators
        recent_candles = candles[-self.lookback_period:]
        
        # Calculate all 7 available trend indicators
        indicators = {
            'sma': TrendIndicators.sma(recent_candles, period=20),
            'ema': TrendIndicators.ema(recent_candles, period=20),
            'wma': TrendIndicators.wma(recent_candles, period=20),
            'dema': TrendIndicators.dema(recent_candles, period=20),
            'tema': TrendIndicators.tema(recent_candles, period=20),
            'macd': TrendIndicators.macd(recent_candles),
            'adx': TrendIndicators.adx(recent_candles)
        }
        
        # Extract features: signal score + confidence for each
        features = {}
        
        for name, result in indicators.items():
            # Signal score normalized to [-1, 1]
            signal_score = result.signal.get_score() / 2.0
            features[f'{name}_signal'] = signal_score
            features[f'{name}_confidence'] = result.confidence
            
            # Weighted signal (signal × confidence)
            features[f'{name}_weighted'] = signal_score * result.confidence
        
        return features
    
    def extract_4_dimension_features(
        self,
        candles: List[Candle]
    ) -> Dict[str, float]:
        """
        Extract features from 4 dimensions of trend analysis
        
        Returns:
            Dictionary with dimension scores and confidences
        """
        if len(candles) < self.lookback_period:
            raise ValueError(f"Need at least {self.lookback_period} candles")
        
        recent_candles = candles[-self.lookback_period:]
        features = {}
        
        # ═══════════════════════════════════════════════════════
        # Dimension 1: Technical Indicators (10 indicators)
        # ═══════════════════════════════════════════════════════
        indicator_results = []
        
        try:
            indicator_results = [
                TrendIndicators.sma(recent_candles, 20),
                TrendIndicators.ema(recent_candles, 20),
                TrendIndicators.wma(recent_candles, 20),
                TrendIndicators.dema(recent_candles, 20),
                TrendIndicators.tema(recent_candles, 20),
                TrendIndicators.macd(recent_candles),
                TrendIndicators.adx(recent_candles)
            ]
            
            # Calculate dimension score (weighted by confidence)
            weighted_sum = sum(ind.signal.get_score() * ind.confidence for ind in indicator_results)
            total_weight = sum(ind.confidence for ind in indicator_results)
            
            dim1_score = weighted_sum / total_weight if total_weight > 0 else 0.0
            dim1_confidence = total_weight / len(indicator_results)
            
        except Exception as e:
            print(f"Warning: Error in indicator calculation: {e}")
            dim1_score = 0.0
            dim1_confidence = 0.5
        
        features['dim1_indicators_score'] = dim1_score / 2.0  # Normalize to [-1, 1]
        features['dim1_indicators_confidence'] = dim1_confidence
        features['dim1_indicators_weighted'] = (dim1_score / 2.0) * dim1_confidence
        
        # ═══════════════════════════════════════════════════════
        # Dimension 2: Candlestick Patterns
        # ═══════════════════════════════════════════════════════
        try:
            patterns = CandlestickPatterns.detect_patterns(recent_candles)
            
            if patterns:
                # Most recent pattern
                latest_pattern = patterns[0]
                dim2_score = latest_pattern.signal.get_score()
                dim2_confidence = latest_pattern.confidence
            else:
                # No patterns detected
                dim2_score = 0.0
                dim2_confidence = 0.5
                
        except Exception as e:
            print(f"Warning: Error in candlestick pattern detection: {e}")
            dim2_score = 0.0
            dim2_confidence = 0.5
        
        features['dim2_candlestick_score'] = dim2_score / 2.0
        features['dim2_candlestick_confidence'] = dim2_confidence
        features['dim2_candlestick_weighted'] = (dim2_score / 2.0) * dim2_confidence
        
        # ═══════════════════════════════════════════════════════
        # Dimension 3: Elliott Wave Theory
        # ═══════════════════════════════════════════════════════
        try:
            elliott = ElliottWaveAnalyzer.analyze(recent_candles)
            
            dim3_score = elliott.signal.get_score()
            dim3_confidence = elliott.confidence
            
        except Exception as e:
            print(f"Warning: Error in Elliott Wave analysis: {e}")
            dim3_score = 0.0
            dim3_confidence = 0.5
        
        features['dim3_elliott_score'] = dim3_score / 2.0
        features['dim3_elliott_confidence'] = dim3_confidence
        features['dim3_elliott_weighted'] = (dim3_score / 2.0) * dim3_confidence
        
        # ═══════════════════════════════════════════════════════
        # Dimension 4: Classical Patterns
        # ═══════════════════════════════════════════════════════
        try:
            classical = ClassicalPatterns.detect_all(recent_candles)
            
            if classical:
                # Average signal from all detected patterns
                avg_score = sum(p.signal.get_score() for p in classical) / len(classical)
                avg_confidence = sum(p.confidence for p in classical) / len(classical)
                
                dim4_score = avg_score
                dim4_confidence = avg_confidence
            else:
                # No patterns detected
                dim4_score = 0.0
                dim4_confidence = 0.5
                
        except Exception as e:
            print(f"Warning: Error in classical pattern detection: {e}")
            dim4_score = 0.0
            dim4_confidence = 0.5
        
        features['dim4_classical_score'] = dim4_score / 2.0
        features['dim4_classical_confidence'] = dim4_confidence
        features['dim4_classical_weighted'] = (dim4_score / 2.0) * dim4_confidence
        
        return features
    
    def calculate_future_return(
        self,
        candles: List[Candle],
        forward_days: int = None
    ) -> float:
        """
        Calculate future return as label for supervised learning
        
        Args:
            candles: Historical candles
            forward_days: Days ahead to calculate return
            
        Returns:
            Future return percentage (e.g., 0.05 = 5% gain)
        """
        if forward_days is None:
            forward_days = self.forward_days
        
        if len(candles) < forward_days + 1:
            return 0.0
        
        current_price = candles[0].close
        future_price = candles[forward_days].close
        
        return (future_price - current_price) / current_price
    
    def extract_training_dataset(
        self,
        candles: List[Candle],
        level: str = "indicators"  # "indicators" or "dimensions"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract complete training dataset
        
        Args:
            candles: Historical candles (many days)
            level: "indicators" for 10-indicator level, "dimensions" for 4-dimension level
            
        Returns:
            (features_df, labels_series)
        """
        print(f"📊 Extracting features at '{level}' level...")
        
        all_features = []
        all_labels = []
        
        # Sliding window approach
        # Need lookback_period for features + forward_days for labels
        required_length = self.lookback_period + self.forward_days
        
        for i in range(len(candles) - required_length):
            # Extract features from [i : i + lookback_period]
            window = candles[i : i + self.lookback_period]
            
            try:
                if level == "indicators":
                    features = self.extract_10_indicator_features(window)
                elif level == "dimensions":
                    features = self.extract_4_dimension_features(window)
                else:
                    raise ValueError(f"Unknown level: {level}")
                
                # Calculate future return as label
                # Look forward from the end of the window
                future_window = candles[i + self.lookback_period : i + required_length + 1]
                future_return = self.calculate_future_return(
                    [window[-1]] + future_window,
                    self.forward_days
                )
                
                all_features.append(features)
                all_labels.append(future_return)
                
            except Exception as e:
                print(f"  ⚠️ Skipping window {i}: {e}")
                continue
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        labels_series = pd.Series(all_labels, name='future_return')
        
        print(f"✅ Extracted {len(features_df)} training samples")
        print(f"   Features: {list(features_df.columns)[:5]}... ({len(features_df.columns)} total)")
        print(f"   Returns: mean={labels_series.mean():.4f}, std={labels_series.std():.4f}")
        
        return features_df, labels_series
    
    def create_binary_labels(
        self,
        returns: pd.Series,
        threshold: float = 0.01  # 1%
    ) -> pd.Series:
        """
        Convert continuous returns to binary labels (bullish/bearish)
        
        Args:
            returns: Future return percentages
            threshold: Minimum return to classify as bullish (or bearish if negative)
            
        Returns:
            Binary labels: 1 = bullish, 0 = bearish
        """
        # 1 if return > threshold, 0 if return < -threshold, neutral otherwise
        labels = pd.Series(index=returns.index, dtype=int)
        labels[returns > threshold] = 1  # Bullish
        labels[returns < -threshold] = 0  # Bearish
        labels[(returns >= -threshold) & (returns <= threshold)] = -1  # Neutral (can be filtered)
        
        return labels
    
    def extract_market_context_features(
        self,
        candles: List[Candle]
    ) -> Dict[str, float]:
        """
        Extract additional market context features
        
        Returns:
            Dictionary with context features (volatility, trend strength, etc.)
        """
        closes = np.array([c.close for c in candles[-self.lookback_period:]])
        highs = np.array([c.high for c in candles[-self.lookback_period:]])
        lows = np.array([c.low for c in candles[-self.lookback_period:]])
        volumes = np.array([c.volume for c in candles[-self.lookback_period:]])
        
        features = {}
        
        # Volatility (ATR-based)
        true_ranges = []
        for i in range(1, len(candles[-self.lookback_period:])):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i-1]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        features['volatility_atr'] = np.mean(true_ranges) / closes[-1] if true_ranges else 0.0
        features['volatility_std'] = np.std(closes) / np.mean(closes) if len(closes) > 0 else 0.0
        
        # Trend strength (linear regression slope)
        x = np.arange(len(closes))
        if len(closes) > 1:
            slope, _ = np.polyfit(x, closes, 1)
            features['trend_strength'] = slope / closes[-1]
        else:
            features['trend_strength'] = 0.0
        
        # Volume trend
        if len(volumes) > 1:
            volume_slope, _ = np.polyfit(x, volumes, 1)
            features['volume_trend'] = volume_slope / np.mean(volumes) if np.mean(volumes) > 0 else 0.0
        else:
            features['volume_trend'] = 0.0
        
        # Price position (current vs. recent high/low)
        recent_high = np.max(highs[-20:]) if len(highs) >= 20 else np.max(highs)
        recent_low = np.min(lows[-20:]) if len(lows) >= 20 else np.min(lows)
        
        if recent_high > recent_low:
            features['price_position'] = (closes[-1] - recent_low) / (recent_high - recent_low)
        else:
            features['price_position'] = 0.5
        
        return features


# Example usage
if __name__ == "__main__":
    from gravity_tech.ml.data_connector import DataConnector
    from datetime import datetime, timedelta
    
    print("🔧 Testing Feature Extraction...")
    
    # Fetch data
    connector = DataConnector()
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365 * 2)  # 2 years
    
    candles = connector.fetch_daily_candles("BTCUSDT", start_date, end_date)
    print(f"✅ Loaded {len(candles)} daily candles\n")
    
    # Initialize extractor
    extractor = FeatureExtractor(lookback_period=100, forward_days=5)
    
    # Test indicator-level features
    print("=" * 60)
    print("LEVEL 1: 10 Indicator Features")
    print("=" * 60)
    
    X_indicators, y_indicators = extractor.extract_training_dataset(
        candles,
        level="indicators"
    )
    
    print(f"\nShape: {X_indicators.shape}")
    print(f"Features:\n{X_indicators.head()}\n")
    print(f"Labels (future returns):\n{y_indicators.head()}\n")
    
    # Test dimension-level features
    print("=" * 60)
    print("LEVEL 2: 4 Dimension Features")
    print("=" * 60)
    
    X_dimensions, y_dimensions = extractor.extract_training_dataset(
        candles,
        level="dimensions"
    )
    
    print(f"\nShape: {X_dimensions.shape}")
    print(f"Features:\n{X_dimensions.head()}\n")
    
    print("✅ Feature extraction complete!")
