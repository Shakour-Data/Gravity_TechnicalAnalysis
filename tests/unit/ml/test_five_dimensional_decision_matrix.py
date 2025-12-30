"""
Tests for Five Dimensional Decision Matrix

This module contains comprehensive tests for the FiveDimensionalDecisionMatrix class,
which combines trend, momentum, volatility, cycle, and support/resistance analysis
into a unified trading decision.
"""

from datetime import datetime

import pytest
from gravity_tech.core.domain.entities import Candle
from gravity_tech.core.domain.entities.signal_strength import SignalStrength
from gravity_tech.ml.five_dimensional_decision_matrix import (
    FiveDimensionalDecisionMatrix,
)
from gravity_tech.ml.multi_horizon_analysis import MarketPattern, TrendScore
from gravity_tech.ml.multi_horizon_cycle_analysis import CycleScore
from gravity_tech.ml.multi_horizon_momentum_analysis import MomentumScore
from gravity_tech.ml.multi_horizon_support_resistance_analysis import SupportResistanceScore
from gravity_tech.ml.multi_horizon_volatility_analysis import VolatilityScore


@pytest.fixture
def sample_candles():
    """Create sample candles for testing (150 candles to meet minimum requirement)."""
    candles = []
    base_price = 100.0
    for i in range(150):
        # Create a slight upward trend with some volatility
        price_change = (i - 75) * 0.01  # Slight trend
        noise = (i % 10 - 5) * 0.05  # Some noise
        close = base_price + price_change + noise

        candles.append(
            Candle(
                timestamp=datetime(2024, 1, 1, 0, 0, 0),
                open=close - 0.5,
                high=close + 1.0,
                low=close - 1.0,
                close=close,
                volume=1000 + i * 10,
            )
        )
    return candles


@pytest.fixture
def mock_trend_score():
    """Create a mock TrendScore for testing."""
    return TrendScore(
        score=0.6,
        confidence=0.8,
        signal=SignalStrength.BULLISH,
        pattern=MarketPattern.STRONG_UPTREND,
        recommendation="Strong upward trend detected",
    )


@pytest.fixture
def mock_momentum_score():
    """Create a mock MomentumScore for testing."""
    return MomentumScore(horizon="7d", score=0.4, confidence=0.7, signal=SignalStrength.BULLISH)


@pytest.fixture
def mock_volatility_score():
    """Create a mock VolatilityScore for testing."""
    return VolatilityScore(horizon="7d", score=-0.2, confidence=0.6, signal=SignalStrength.NEUTRAL)


@pytest.fixture
def mock_cycle_score():
    """Create a mock CycleScore for testing."""
    return CycleScore(
        horizon="7d",
        score=0.3,
        confidence=0.75,
        phase=45.0,  # Accumulation phase
        cycle_period=20.0,
        signal=SignalStrength.BULLISH,
    )


@pytest.fixture
def mock_sr_score():
    """Create a mock SupportResistanceScore for testing."""
    return SupportResistanceScore(
        horizon="7d",
        score=0.5,
        confidence=0.8,
        bounce_probability=0.7,
        breakout_probability=0.3,
        nearest_support=95.0,
        nearest_resistance=105.0,
        support_strength=0.8,
        resistance_strength=0.6,
        sr_position=0.3,
        distance_to_key_level=2.5,
        signal="NEAR_SUPPORT",
        recommendation="Consider Buy",
    )


class TestFiveDimensionalDecisionMatrix:
    """Test suite for FiveDimensionalDecisionMatrix."""

    def test_initialization_valid_candles(self, sample_candles):
        """Test matrix initialization with valid candles."""
        matrix = FiveDimensionalDecisionMatrix(sample_candles)
        assert len(matrix.candles) == 150
        # Compare weights with approximate equality due to floating point precision
        for key in FiveDimensionalDecisionMatrix.DEFAULT_WEIGHTS:
            assert (
                abs(matrix.weights[key] - FiveDimensionalDecisionMatrix.DEFAULT_WEIGHTS[key]) < 1e-9
            )
        assert matrix.use_volume_matrix is True

    def test_initialization_insufficient_candles(self):
        """Test matrix initialization with insufficient candles."""
        short_candles = [
            Candle(
                timestamp=datetime(2024, 1, 1), open=100, high=101, low=99, close=100.5, volume=1000
            )
            for _ in range(50)
        ]

        with pytest.raises(
            ValueError, match="FiveDimensionalDecisionMatrix requires at least 120 candles"
        ):
            FiveDimensionalDecisionMatrix(short_candles)

    def test_initialization_invalid_candle_data(self):
        """Test matrix initialization with invalid candle data."""
        invalid_candles = [
            Candle(
                timestamp=datetime(2024, 1, 1),
                open=float("nan"),
                high=101,
                low=99,
                close=100.5,
                volume=1000,
            )
            for _ in range(120)
        ]

        with pytest.raises(ValueError):
            FiveDimensionalDecisionMatrix(invalid_candles)

    def test_initialization_high_less_than_low(self):
        """Test matrix initialization with high < low."""
        with pytest.raises(ValueError, match="Low.*must be <= high"):
            invalid_candles = [
                Candle(
                    timestamp=datetime(2024, 1, 1),
                    open=100,
                    high=99,
                    low=100,
                    close=99.5,
                    volume=1000,
                )
                for _ in range(120)
            ]
            FiveDimensionalDecisionMatrix(invalid_candles)

    def test_initialization_negative_volume(self):
        """Test matrix initialization with negative volume."""
        with pytest.raises(ValueError, match="Volume.*cannot be negative"):
            invalid_candles = [
                Candle(
                    timestamp=datetime(2024, 1, 1),
                    open=100,
                    high=101,
                    low=99,
                    close=100.5,
                    volume=-100,
                )
                for _ in range(120)
            ]
            FiveDimensionalDecisionMatrix(invalid_candles)

    def test_custom_weights(self, sample_candles):
        """Test matrix with custom dimension weights."""
        custom_weights = {
            "trend": 0.4,
            "momentum": 0.3,
            "volatility": 0.1,
            "cycle": 0.1,
            "support_resistance": 0.1,
        }

        matrix = FiveDimensionalDecisionMatrix(sample_candles, dimension_weights=custom_weights)
        # Compare weights with approximate equality due to floating point precision
        for key in custom_weights:
            assert abs(matrix.weights[key] - custom_weights[key]) < 1e-9

    def test_analyze_strong_bullish_signal(
        self,
        sample_candles,
        mock_trend_score,
        mock_momentum_score,
        mock_volatility_score,
        mock_cycle_score,
        mock_sr_score,
    ):
        """Test analysis with strong bullish signals across dimensions."""
        matrix = FiveDimensionalDecisionMatrix(sample_candles)
        result = matrix.analyze(
            trend_score=mock_trend_score,
            momentum_score=mock_momentum_score,
            volatility_score=mock_volatility_score,
            cycle_score=mock_cycle_score,
            sr_score=mock_sr_score,
        )

        assert hasattr(result, "final_signal")
        assert hasattr(result, "final_score")
        assert result.final_confidence >= 0.5
        assert hasattr(result, "agreement")

    def test_analyze_strong_bearish_signal(self, sample_candles):
        """Test analysis with strong bearish signals across dimensions."""
        # Create strongly bearish scores
        strong_bearish_trend = TrendScore(
            score=-0.9,
            confidence=0.95,
            signal=SignalStrength.VERY_BEARISH,
            pattern=MarketPattern.STRONG_DOWNTREND,
            recommendation="Very strong downtrend",
        )
        strong_bearish_momentum = MomentumScore(
            horizon="7d", score=-0.8, confidence=0.9, signal=SignalStrength.VERY_BEARISH
        )
        neutral_volatility = VolatilityScore(
            horizon="7d", score=0.0, confidence=0.8, signal=SignalStrength.NEUTRAL
        )
        bearish_cycle = CycleScore(
            horizon="7d",
            score=-0.7,
            confidence=0.85,
            phase=270.0,
            cycle_period=25.0,
            signal=SignalStrength.BEARISH,
        )
        bearish_sr = SupportResistanceScore(
            horizon="7d",
            score=-0.6,
            confidence=0.9,
            bounce_probability=0.1,
            breakout_probability=0.9,
            nearest_support=90.0,
            nearest_resistance=100.0,
            support_strength=0.4,
            resistance_strength=0.8,
            sr_position=0.8,
            distance_to_key_level=2.5,
            signal="NEAR_RESISTANCE",
            recommendation="Strong Sell Signal",
        )

        matrix = FiveDimensionalDecisionMatrix(sample_candles)
        result = matrix.analyze(
            trend_score=strong_bearish_trend,
            momentum_score=strong_bearish_momentum,
            volatility_score=neutral_volatility,
            cycle_score=bearish_cycle,
            sr_score=bearish_sr,
        )

        assert hasattr(result, "final_signal")
        assert result.final_confidence >= 0.5

    def test_analyze_mixed_signals(self, sample_candles):
        """Test analysis with mixed/conflicting signals."""
        bullish_trend = TrendScore(
            score=0.6,
            confidence=0.8,
            signal=SignalStrength.BULLISH,
            pattern=MarketPattern.BUY_THE_DIP,
            recommendation="Buy the dip",
        )
        bearish_momentum = MomentumScore(
            horizon="7d", score=-0.5, confidence=0.7, signal=SignalStrength.BEARISH
        )
        high_volatility = VolatilityScore(
            horizon="7d", score=0.8, confidence=0.9, signal=SignalStrength.VERY_BULLISH
        )
        neutral_cycle = CycleScore(
            horizon="7d",
            score=0.0,
            confidence=0.6,
            phase=270.0,
            cycle_period=30.0,
            signal=SignalStrength.NEUTRAL,
        )
        neutral_sr = SupportResistanceScore(
            horizon="7d",
            score=0.1,
            confidence=0.5,
            bounce_probability=0.5,
            breakout_probability=0.5,
            nearest_support=98.0,
            nearest_resistance=102.0,
            support_strength=0.5,
            resistance_strength=0.5,
            sr_position=0.5,
            distance_to_key_level=0.0,
            signal="BALANCED",
            recommendation="Wait",
        )

        matrix = FiveDimensionalDecisionMatrix(sample_candles)
        result = matrix.analyze(
            trend_score=bullish_trend,
            momentum_score=bearish_momentum,
            volatility_score=high_volatility,
            cycle_score=neutral_cycle,
            sr_score=neutral_sr,
        )

        assert hasattr(result, "final_signal")
        assert hasattr(result, "agreement")

    def test_analyze_without_volume_matrix(
        self,
        sample_candles,
        mock_trend_score,
        mock_momentum_score,
        mock_volatility_score,
        mock_cycle_score,
        mock_sr_score,
    ):
        """Test analysis without volume-dimension matrix."""
        matrix = FiveDimensionalDecisionMatrix(sample_candles, use_volume_matrix=False)
        result = matrix.analyze(
            trend_score=mock_trend_score,
            momentum_score=mock_momentum_score,
            volatility_score=mock_volatility_score,
            cycle_score=mock_cycle_score,
            sr_score=mock_sr_score,
        )

        assert hasattr(result, "final_signal")
        assert hasattr(result, "dimensions")

    def test_result_structure(
        self,
        sample_candles,
        mock_trend_score,
        mock_momentum_score,
        mock_volatility_score,
        mock_cycle_score,
        mock_sr_score,
    ):
        """Test that result has all required attributes."""
        matrix = FiveDimensionalDecisionMatrix(sample_candles)
        result = matrix.analyze(
            trend_score=mock_trend_score,
            momentum_score=mock_momentum_score,
            volatility_score=mock_volatility_score,
            cycle_score=mock_cycle_score,
            sr_score=mock_sr_score,
        )

        # Check main attributes
        assert hasattr(result, "timestamp")
        assert hasattr(result, "dimensions")
        assert hasattr(result, "final_score")
        assert hasattr(result, "final_confidence")
        assert hasattr(result, "final_signal")
        assert hasattr(result, "signal_strength")
        assert hasattr(result, "agreement")
        assert hasattr(result, "risk_level")
        assert hasattr(result, "recommendation")

        # Check dimensions
        assert len(result.dimensions) == 5
        assert "trend" in result.dimensions
        assert "momentum" in result.dimensions
        assert "volatility" in result.dimensions
        assert "cycle" in result.dimensions
        assert "support_resistance" in result.dimensions

        # Check agreement structure
        assert hasattr(result.agreement, "overall_agreement")
        assert hasattr(result.agreement, "bullish_dimensions")
        assert hasattr(result.agreement, "bearish_dimensions")

    def test_signal_strength_calculation(self, sample_candles):
        """Test signal strength calculation based on score and confidence."""
        # High confidence, strong signal
        strong_trend = TrendScore(
            score=0.9,
            confidence=0.95,
            signal=SignalStrength.VERY_BULLISH,
            pattern=MarketPattern.STRONG_UPTREND,
            recommendation="Strong",
        )
        weak_momentum = MomentumScore(
            horizon="7d", score=0.1, confidence=0.4, signal=SignalStrength.NEUTRAL
        )
        neutral_volatility = VolatilityScore(
            horizon="7d", score=0.0, confidence=0.5, signal=SignalStrength.NEUTRAL
        )
        neutral_cycle = CycleScore(
            horizon="7d",
            score=0.0,
            confidence=0.5,
            phase=180.0,
            cycle_period=20.0,
            signal=SignalStrength.NEUTRAL,
        )
        neutral_sr = SupportResistanceScore(
            horizon="7d",
            score=0.0,
            confidence=0.5,
            bounce_probability=0.5,
            breakout_probability=0.5,
            nearest_support=98.0,
            nearest_resistance=102.0,
            support_strength=0.5,
            resistance_strength=0.5,
            sr_position=0.5,
            distance_to_key_level=2.0,
            signal="BALANCED",
            recommendation="Neutral Position",
        )

        matrix = FiveDimensionalDecisionMatrix(sample_candles)
        result = matrix.analyze(
            trend_score=strong_trend,
            momentum_score=weak_momentum,
            volatility_score=neutral_volatility,
            cycle_score=neutral_cycle,
            sr_score=neutral_sr,
        )

        # Signal strength is based on overall dimension agreement
        # With only trend bullish and rest neutral, signal strength is moderate
        assert result.signal_strength >= 0.0
        assert result.signal_strength <= 1.0
        assert hasattr(result, "agreement")
        # Check that trend is the strongest dimension
        assert result.agreement.strongest_dimension == "trend"
