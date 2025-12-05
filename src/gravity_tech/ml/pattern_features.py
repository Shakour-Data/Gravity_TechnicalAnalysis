"""
Pattern Feature Extraction for Machine Learning

Extracts features from harmonic patterns for ML model training:
- Fibonacci ratio accuracy
- Pattern geometry (angles, symmetry)
- Price action characteristics
- Volume confirmation
- Momentum indicators at pattern points

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gravity_tech.patterns.harmonic import HarmonicPattern, PatternDirection


@dataclass
class PatternFeatures:
    """Extracted features from a harmonic pattern."""
    # Fibonacci ratio features
    xab_ratio_accuracy: float
    abc_ratio_accuracy: float
    bcd_ratio_accuracy: float
    xad_ratio_accuracy: float

    # Geometric features
    pattern_symmetry: float
    pattern_slope: float
    xa_angle: float
    ab_angle: float
    bc_angle: float
    cd_angle: float

    # Price action features
    pattern_duration: int  # bars
    xa_magnitude: float  # price move
    ab_magnitude: float
    bc_magnitude: float
    cd_magnitude: float

    # Volume features
    volume_at_d: float
    volume_trend: float
    volume_confirmation: float

    # Momentum features
    rsi_at_d: float
    macd_at_d: float
    momentum_divergence: float

    # Pattern metadata
    pattern_type: str
    direction: str
    confidence: float


class PatternFeatureExtractor:
    """
    Extracts comprehensive features from harmonic patterns for ML training.

    Features are normalized to 0-1 range for better ML model performance.
    """

    def __init__(self):
        """Initialize feature extractor."""
        self.feature_names = [
            'xab_ratio_accuracy',
            'abc_ratio_accuracy',
            'bcd_ratio_accuracy',
            'xad_ratio_accuracy',
            'pattern_symmetry',
            'pattern_slope',
            'xa_angle',
            'ab_angle',
            'bc_angle',
            'cd_angle',
            'pattern_duration',
            'xa_magnitude',
            'ab_magnitude',
            'bc_magnitude',
            'cd_magnitude',
            'volume_at_d',
            'volume_trend',
            'volume_confirmation',
            'rsi_at_d',
            'macd_at_d',
            'momentum_divergence'
        ]

    def extract_features(
        self,
        pattern: HarmonicPattern,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volume: Optional[np.ndarray] = None
    ) -> PatternFeatures:
        """
        Extract all features from a harmonic pattern.

        Args:
            pattern: Detected harmonic pattern
            highs: High prices
            lows: Low prices
            closes: Close prices
            volume: Optional volume data

        Returns:
            PatternFeatures object with all extracted features
        """
        # Extract Fibonacci ratio accuracy
        fib_features = self._extract_fibonacci_features(pattern)

        # Extract geometric features
        geo_features = self._extract_geometric_features(pattern)

        # Extract price action features
        price_features = self._extract_price_features(pattern, closes)

        # Extract volume features
        if volume is not None:
            vol_features = self._extract_volume_features(pattern, volume)
        else:
            vol_features = {
                'volume_at_d': 0.5,
                'volume_trend': 0.5,
                'volume_confirmation': 0.5
            }

        # Extract momentum features
        momentum_features = self._extract_momentum_features(pattern, closes)

        # Combine all features
        return PatternFeatures(
            **fib_features,
            **geo_features,
            **price_features,
            **vol_features,
            **momentum_features,
            pattern_type=pattern.pattern_type.value,
            direction=pattern.direction.value,
            confidence=pattern.confidence / 100.0
        )

    def _extract_fibonacci_features(self, pattern: HarmonicPattern) -> dict:
        """
        Extract Fibonacci ratio accuracy features.

        Measures how closely actual ratios match ideal Fibonacci levels.
        """
        ratios = pattern.ratios

        # Target Fibonacci ratios based on pattern type
        targets = {
            'gartley': {'XA_BC': 0.618, 'AB_CD': 0.786, 'XA_AD': 0.786},
            'butterfly': {'XA_BC': 0.786, 'AB_CD': 1.618, 'XA_AD': 1.272},
            'bat': {'XA_BC': 0.500, 'AB_CD': 0.886, 'XA_AD': 0.886},
            'crab': {'XA_BC': 0.618, 'AB_CD': 2.618, 'XA_AD': 1.618}
        }

        pattern_targets = targets.get(pattern.pattern_type.value, targets['gartley'])

        # Calculate accuracy for each ratio (1.0 = perfect match)
        accuracies = {}
        for ratio_name, target in pattern_targets.items():
            if ratio_name in ratios:
                actual = ratios[ratio_name]
                deviation = abs(actual - target) / target if target != 0 else 0
                accuracy = max(0, 1 - deviation)
                accuracies[ratio_name] = accuracy
            else:
                accuracies[ratio_name] = 0.0

        return {
            'xab_ratio_accuracy': accuracies.get('XA_BC', 0.5),
            'abc_ratio_accuracy': accuracies.get('AB_CD', 0.5),
            'bcd_ratio_accuracy': accuracies.get('AB_CD', 0.5),  # Same as AB_CD
            'xad_ratio_accuracy': accuracies.get('XA_AD', 0.5)
        }

    def _extract_geometric_features(self, pattern: HarmonicPattern) -> dict:
        """
        Extract geometric features from pattern shape.

        Includes angles, symmetry, and slope.
        """
        points = pattern.points

        # Calculate angles between legs
        xa_angle = self._calculate_angle(
            points['X'].index, points['X'].price,
            points['A'].index, points['A'].price
        )
        ab_angle = self._calculate_angle(
            points['A'].index, points['A'].price,
            points['B'].index, points['B'].price
        )
        bc_angle = self._calculate_angle(
            points['B'].index, points['B'].price,
            points['C'].index, points['C'].price
        )
        cd_angle = self._calculate_angle(
            points['C'].index, points['C'].price,
            points['D'].index, points['D'].price
        )

        # Calculate symmetry (how balanced is the pattern)
        ab_duration = points['B'].index - points['A'].index
        cd_duration = points['D'].index - points['C'].index
        symmetry = 1 - abs(ab_duration - cd_duration) / max(ab_duration, cd_duration, 1)

        # Calculate overall pattern slope
        x_to_d_bars = points['D'].index - points['X'].index
        x_to_d_price = points['D'].price - points['X'].price
        slope = x_to_d_price / x_to_d_bars if x_to_d_bars > 0 else 0

        return {
            'pattern_symmetry': np.clip(symmetry, 0, 1),
            'pattern_slope': self._normalize_slope(slope),
            'xa_angle': self._normalize_angle(xa_angle),
            'ab_angle': self._normalize_angle(ab_angle),
            'bc_angle': self._normalize_angle(bc_angle),
            'cd_angle': self._normalize_angle(cd_angle)
        }

    def _extract_price_features(self, pattern: HarmonicPattern, closes: np.ndarray) -> dict:
        """
        Extract price action features.

        Includes pattern duration and magnitude of moves.
        """
        points = pattern.points

        # Pattern duration
        duration = points['D'].index - points['X'].index

        # Magnitudes of each leg (normalized by average price)
        avg_price = np.mean(closes[points['X'].index:points['D'].index + 1])

        xa_magnitude = abs(points['A'].price - points['X'].price) / avg_price
        ab_magnitude = abs(points['B'].price - points['A'].price) / avg_price
        bc_magnitude = abs(points['C'].price - points['B'].price) / avg_price
        cd_magnitude = abs(points['D'].price - points['C'].price) / avg_price

        return {
            'pattern_duration': min(duration / 100.0, 1.0),  # Normalize to 0-1
            'xa_magnitude': np.clip(xa_magnitude * 10, 0, 1),
            'ab_magnitude': np.clip(ab_magnitude * 10, 0, 1),
            'bc_magnitude': np.clip(bc_magnitude * 10, 0, 1),
            'cd_magnitude': np.clip(cd_magnitude * 10, 0, 1)
        }

    def _extract_volume_features(self, pattern: HarmonicPattern, volume: np.ndarray) -> dict:
        """
        Extract volume features.

        Includes volume at D point and volume trend.
        """
        points = pattern.points

        # Volume at D point (normalized by average)
        d_idx = points['D'].index
        avg_volume = np.mean(volume[max(0, d_idx - 20):d_idx + 1])
        volume_at_d = volume[d_idx] / avg_volume if avg_volume > 0 else 1.0

        # Volume trend (increasing or decreasing)
        volume_slice = volume[points['X'].index:points['D'].index + 1]
        if len(volume_slice) > 1:
            volume_trend = np.corrcoef(range(len(volume_slice)), volume_slice)[0, 1]
            volume_trend = (volume_trend + 1) / 2  # Normalize -1,1 to 0,1
        else:
            volume_trend = 0.5

        # Volume confirmation (higher volume at reversal point)
        recent_volume = np.mean(volume[max(0, d_idx - 5):d_idx + 1])
        previous_volume = np.mean(volume[max(0, d_idx - 25):d_idx - 5])
        volume_confirmation = recent_volume / previous_volume if previous_volume > 0 else 1.0

        return {
            'volume_at_d': np.clip(volume_at_d, 0, 2) / 2,
            'volume_trend': np.clip(volume_trend, 0, 1),
            'volume_confirmation': np.clip(volume_confirmation, 0, 2) / 2
        }

    def _extract_momentum_features(self, pattern: HarmonicPattern, closes: np.ndarray) -> dict:
        """
        Extract momentum indicator features.

        Includes RSI, MACD, and momentum divergence.
        """
        points = pattern.points
        d_idx = points['D'].index

        # Calculate RSI at D point
        rsi = self._calculate_rsi(closes[:d_idx + 1], period=14)
        rsi_at_d = rsi[-1] / 100.0 if len(rsi) > 0 else 0.5

        # Calculate MACD at D point
        macd_line = self._calculate_macd(closes[:d_idx + 1])
        macd_at_d = (macd_line[-1] + 1) / 2 if len(macd_line) > 0 else 0.5  # Normalize

        # Momentum divergence (price vs momentum)
        momentum_div = self._calculate_divergence(pattern, closes)

        return {
            'rsi_at_d': np.clip(rsi_at_d, 0, 1),
            'macd_at_d': np.clip(macd_at_d, 0, 1),
            'momentum_divergence': np.clip(momentum_div, 0, 1)
        }

    def _calculate_angle(self, x1: int, y1: float, x2: int, y2: float) -> float:
        """Calculate angle in degrees between two points."""
        if x2 - x1 == 0:
            return 90.0
        slope = (y2 - y1) / (x2 - x1)
        angle = np.degrees(np.arctan(slope))
        return angle

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to 0-1 range."""
        # Convert -90 to 90 degrees to 0-1
        return (angle + 90) / 180

    def _normalize_slope(self, slope: float) -> float:
        """Normalize slope to 0-1 range."""
        # Clip and normalize slope
        return np.clip((slope + 0.1) / 0.2, 0, 1)

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return np.array([50.0])

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        rsi = np.zeros(len(prices) - period)

        for i in range(len(rsi)):
            if i == 0:
                rs = avg_gain / avg_loss if avg_loss != 0 else 100
            else:
                avg_gain = (avg_gain * (period - 1) + gains[period + i - 1]) / period
                avg_loss = (avg_loss * (period - 1) + losses[period + i - 1]) / period
                rs = avg_gain / avg_loss if avg_loss != 0 else 100

            rsi[i] = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
        """Calculate MACD line."""
        if len(prices) < slow:
            return np.array([0.0])

        # Calculate EMAs
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)

        # MACD line
        macd = ema_fast - ema_slow
        return macd

    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return prices

        alpha = 2 / (period + 1)
        ema = np.zeros(len(prices))
        ema[0] = prices[0]

        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

        return ema

    def _calculate_divergence(self, pattern: HarmonicPattern, closes: np.ndarray) -> float:
        """
        Calculate momentum divergence.

        Positive divergence: price makes lower low, momentum makes higher low
        Negative divergence: price makes higher high, momentum makes lower high
        """
        points = pattern.points

        # Get price at A and D
        a_idx = points['A'].index
        d_idx = points['D'].index

        price_change = closes[d_idx] - closes[a_idx]

        # Calculate momentum at both points
        momentum_a = self._calculate_momentum(closes[:a_idx + 1])
        momentum_d = self._calculate_momentum(closes[:d_idx + 1])

        momentum_change = momentum_d - momentum_a

        # Check for divergence
        if pattern.direction == PatternDirection.BULLISH:
            # Expect price lower, momentum higher (positive divergence)
            if price_change < 0 and momentum_change > 0:
                return 1.0  # Strong bullish divergence
            elif price_change < 0 and momentum_change < 0:
                return 0.0  # No divergence
            else:
                return 0.5
        else:
            # Expect price higher, momentum lower (negative divergence)
            if price_change > 0 and momentum_change < 0:
                return 1.0  # Strong bearish divergence
            elif price_change > 0 and momentum_change > 0:
                return 0.0  # No divergence
            else:
                return 0.5

    def _calculate_momentum(self, prices: np.ndarray, period: int = 10) -> float:
        """Calculate price momentum."""
        if len(prices) < period:
            return 0.0
        return prices[-1] - prices[-period]

    def features_to_array(self, features: PatternFeatures) -> np.ndarray:
        """
        Convert PatternFeatures to numpy array for ML models.

        Returns:
            1D numpy array with all numerical features
        """
        return np.array([
            features.xab_ratio_accuracy,
            features.abc_ratio_accuracy,
            features.bcd_ratio_accuracy,
            features.xad_ratio_accuracy,
            features.pattern_symmetry,
            features.pattern_slope,
            features.xa_angle,
            features.ab_angle,
            features.bc_angle,
            features.cd_angle,
            features.pattern_duration,
            features.xa_magnitude,
            features.ab_magnitude,
            features.bc_magnitude,
            features.cd_magnitude,
            features.volume_at_d,
            features.volume_trend,
            features.volume_confirmation,
            features.rsi_at_d,
            features.macd_at_d,
            features.momentum_divergence
        ], dtype=np.float32)

    def get_feature_names(self) -> list[str]:
        """Get list of feature names for ML model."""
        return self.feature_names.copy()


def extract_pattern_features_batch(
    patterns: list[HarmonicPattern],
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volume: Optional[np.ndarray] = None
) -> tuple[np.ndarray, list[str]]:
    """
    Extract features from multiple patterns for batch ML processing.

    Args:
        patterns: List of harmonic patterns
        highs: High prices
        lows: Low prices
        closes: Close prices
        volume: Optional volume data

    Returns:
        Tuple of (feature_matrix, labels) where:
        - feature_matrix: 2D array (n_patterns, n_features)
        - labels: List of pattern types
    """
    extractor = PatternFeatureExtractor()

    feature_matrix = []
    labels = []

    for pattern in patterns:
        features = extractor.extract_features(pattern, highs, lows, closes, volume)
        feature_array = extractor.features_to_array(features)

        feature_matrix.append(feature_array)
        labels.append(pattern.pattern_type.value)

    if feature_matrix:
        return np.vstack(feature_matrix), labels
    else:
        return np.array([]), []
