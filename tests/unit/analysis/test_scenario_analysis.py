"""
Unit tests for scenario_analysis.py in analysis module.

Tests cover all methods in ThreeScenarioAnalysis class to achieve >50% coverage.
Uses real TSE market data and realistic scenarios.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from gravity_tech.analysis.scenario_analysis import (
    ScenarioAnalyzer,
    ScenarioResult,
    ThreeScenarioAnalysis,
)
from gravity_tech.core.domain.entities import Candle


@pytest.fixture
def real_tse_candles():
    """Realistic TSE market data for testing."""
    df = create_realistic_tse_data(num_samples=200, trend='uptrend', seed=42)
    return dataframe_to_candles(df)


@pytest.fixture
def downtrend_candles():
    """Downtrend TSE data."""
    df = create_realistic_tse_data(num_samples=200, trend='downtrend', seed=123)
    return dataframe_to_candles(df)


@pytest.fixture
def mixed_candles():
    """Mixed trend TSE data."""
    df = create_realistic_tse_data(num_samples=200, trend='mixed', seed=456)
    return dataframe_to_candles(df)


def create_realistic_tse_data(num_samples: int = 200, trend: str = 'mixed', seed: int = 42) -> pd.DataFrame:
    """Create realistic TSE market data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=num_samples, freq='1d')

    base_price = 15000.0
    volatility = 0.025
    base_volume = 500_000
    volume_variability = 0.5

    prices = [base_price]
    volumes = []

    for i in range(1, num_samples):
        if trend == 'uptrend':
            drift = 0.0015
        elif trend == 'downtrend':
            drift = -0.0015
        else:  # mixed
            if i < num_samples // 3:
                drift = 0.002
            elif i < 2 * num_samples // 3:
                drift = -0.001
            else:
                drift = 0.0005

        change = drift + rng.normal(0, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 100))

        volume = base_volume * (1 + rng.normal(0, volume_variability))
        volumes.append(int(max(volume, 1000)))

    volumes.insert(0, base_volume)

    df = pd.DataFrame({
        'timestamp': dates,
        'close': prices
    })

    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    open_prices = df['open'].to_numpy()
    close_prices = df['close'].to_numpy()
    body_highs = np.maximum(open_prices, close_prices)
    body_lows = np.minimum(open_prices, close_prices)
    wick_ranges = np.abs(close_prices) * 0.008
    wick_above = rng.uniform(0, wick_ranges)
    wick_below = rng.uniform(0, wick_ranges)

    df['high'] = body_highs + wick_above
    df['low'] = body_lows - wick_below
    df['volume'] = volumes

    return df


def dataframe_to_candles(df: pd.DataFrame) -> list[Candle]:
    """Convert DataFrame to Candle list."""
    candles = []
    for _, row in df.iterrows():
        candles.append(Candle(
            timestamp=row['timestamp'],
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=int(row['volume'])
        ))
    return candles


class TestThreeScenarioAnalysis:
    """Test suite for ThreeScenarioAnalysis class."""

    @pytest.fixture
    def scenario_analyzer(self, real_tse_candles):
        """Fixture for ScenarioAnalyzer instance."""
        analyzer = ScenarioAnalyzer()
        return analyzer, real_tse_candles

    def test_initialization(self, scenario_analyzer):
        """Test initialization of ScenarioAnalyzer."""
        analyzer, candles = scenario_analyzer
        assert isinstance(analyzer, ScenarioAnalyzer)
        assert len(candles) > 0

    def test_analyze_three_scenarios_basic(self, scenario_analyzer):
        """Test basic three scenario analysis."""
        analyzer, candles = scenario_analyzer
        result = analyzer.analyze("TSE_TEST", candles)

        assert isinstance(result, ThreeScenarioAnalysis)
        assert result.symbol == "TSE_TEST"
        assert isinstance(result.timestamp, datetime)
        assert result.current_price > 0
        assert len(result.optimistic.key_signals) > 0
        assert len(result.neutral.key_signals) > 0
        assert isinstance(result.pessimistic.key_signals, list)

    def test_analyze_three_scenarios_uptrend(self, real_tse_candles):
        """Test three scenario analysis with uptrend data."""
        analyzer = ScenarioAnalyzer()
        current_price = real_tse_candles[-1].close
        result = analyzer.analyze("TSE_UPTREND", real_tse_candles, current_price)

        # In uptrend, optimistic scenario should have higher score
        assert result.optimistic.score >= result.neutral.score
        assert result.optimistic.probability >= result.pessimistic.probability

    def test_analyze_three_scenarios_downtrend(self, downtrend_candles):
        """Test three scenario analysis with downtrend data."""
        analyzer = ScenarioAnalyzer()
        current_price = downtrend_candles[-1].close
        result = analyzer.analyze("TSE_DOWNTREND", downtrend_candles, current_price)

        # In downtrend, pessimistic scenario probability should dominate and recommend caution
        assert result.pessimistic.probability >= result.optimistic.probability
        assert result.pessimistic.recommendation in {"SELL", "STRONG_SELL", "AVOID"}

    def test_analyze_three_scenarios_mixed(self, mixed_candles):
        """Test three scenario analysis with mixed trend data."""
        analyzer = ScenarioAnalyzer()
        current_price = mixed_candles[-1].close
        result = analyzer.analyze("TSE_MIXED", mixed_candles, current_price)

        # Mixed should have balanced scenarios
        scores = [result.optimistic.score, result.neutral.score, result.pessimistic.score]
        assert max(scores) - min(scores) < 50  # Not too spread out

    def test_analyze_optimistic_scenario_targets_gain(self, scenario_analyzer):
        """Optimistic scenario should target higher price."""
        analyzer, candles = scenario_analyzer
        base = analyzer._base_technical_analysis(candles)
        current_price = candles[-1].close
        scenario = analyzer._analyze_optimistic_scenario(candles, current_price, 5.0, base)

        assert isinstance(scenario, ScenarioResult)
        assert scenario.target_price > current_price
        assert scenario.stop_loss < current_price
        assert scenario.recommendation in {"STRONG_BUY", "BUY", "HOLD"}

    def test_analyze_pessimistic_scenario_targets_loss(self, scenario_analyzer):
        """Pessimistic scenario should emphasize downside protection."""
        analyzer, candles = scenario_analyzer
        base = analyzer._base_technical_analysis(candles)
        current_price = candles[-1].close
        scenario = analyzer._analyze_pessimistic_scenario(candles, current_price, 5.0, base)

        assert scenario.scenario_type == "pessimistic"
        assert scenario.stop_loss < current_price
        assert scenario.risk_reward_ratio <= 1.0
        assert scenario.recommendation in {"STRONG_SELL", "SELL", "AVOID"}

    def test_base_analysis_contains_expected_metrics(self, scenario_analyzer):
        """Base technical analysis should include core metrics."""
        analyzer, candles = scenario_analyzer
        base = analyzer._base_technical_analysis(candles)

        for key in ["sma_20", "sma_50", "sma_200", "rsi", "macd_line", "signal_line", "patterns", "trend"]:
            assert key in base

    def test_signal_identification_consistency(self, scenario_analyzer):
        """Bullish/bearish/neutral signals should be lists of strings."""
        analyzer, candles = scenario_analyzer
        base = analyzer._base_technical_analysis(candles)

        signals = analyzer._identify_bullish_signals(base)
        assert all(isinstance(sig, str) for sig in signals)

        neutral_signals = analyzer._identify_neutral_signals(base)
        assert any("trend" in sig or "macd" in sig for sig in neutral_signals)

        bearish_signals = analyzer._identify_bearish_signals(base)
        assert isinstance(bearish_signals, list)

    def test_analyze_three_scenarios_empty_candles(self):
        """Test analysis with empty candles."""
        analyzer = ScenarioAnalyzer()

        with pytest.raises(ValueError):
            analyzer.analyze("EMPTY", [], 15000.0)

    def test_analyze_three_scenarios_single_candle(self):
        """Test analysis with single candle."""
        single_candle = [Candle(
            timestamp=pd.Timestamp.now(),
            open=15000.0,
            high=15100.0,
            low=14900.0,
            close=15050.0,
            volume=100000
        )]

        analyzer = ScenarioAnalyzer()
        result = analyzer.analyze("SINGLE", single_candle, 15050.0)

        assert isinstance(result, ThreeScenarioAnalysis)
        # Should handle minimal data

    def test_scenario_result_creation(self):
        """Test ScenarioResult creation."""
        result = ScenarioResult(
            scenario_type="test",
            score=75.0,
            probability=80.0,
            target_price=15500.0,
            stop_loss=14500.0,
            risk_reward_ratio=2.0,
            key_signals=["signal1", "signal2"],
            recommendation="BUY",
            confidence="HIGH",
            timeframe_days=30
        )

        assert result.scenario_type == "test"
        assert result.score == 75.0
        assert result.recommendation == "BUY"

    def test_analyze_three_scenarios_different_symbols(self, real_tse_candles):
        """Test analysis with different symbols."""
        symbols = ["TSE1", "TSE2", "TSE3"]
        analyzer = ScenarioAnalyzer()

        for symbol in symbols:
            result = analyzer.analyze(symbol, real_tse_candles)
            assert result.optimistic.scenario_type == "optimistic"
            assert result.pessimistic.scenario_type == "pessimistic"

    def test_calculate_expected_values_outputs_metrics(self, scenario_analyzer):
        """Expected value calculation should return coherent metrics."""
        analyzer, _ = scenario_analyzer
        optimistic = ScenarioResult("optimistic", 80, 60, 120.0, 100.0, 2.0, ["bullish"], "BUY", "HIGH", 30)
        neutral = ScenarioResult("neutral", 55, 30, 110.0, 100.0, 1.0, ["balanced"], "HOLD", "MEDIUM", 60)
        pessimistic = ScenarioResult("pessimistic", 25, 10, 90.0, 100.0, 0.5, ["bearish"], "SELL", "LOW", 90)

        expected_return, expected_risk, sharpe = analyzer._calculate_expected_values(optimistic, neutral, pessimistic)

        assert expected_risk > 0
        assert sharpe >= 0
        assert expected_return != 0

    def test_determine_recommended_scenario_prefers_confident_extremes(self, scenario_analyzer):
        """Recommendation should reflect dominant scenario."""
        analyzer, _ = scenario_analyzer
        optimistic = ScenarioResult("optimistic", 75, 55, 120.0, 100.0, 2.0, [], "BUY", "HIGH", 30)
        neutral = ScenarioResult("neutral", 50, 35, 110.0, 100.0, 1.0, [], "HOLD", "MEDIUM", 60)
        pessimistic = ScenarioResult("pessimistic", 20, 10, 90.0, 100.0, 0.5, [], "SELL", "LOW", 90)

        rec = analyzer._determine_recommended_scenario(optimistic, neutral, pessimistic)
        assert rec == "optimistic"

        low_conf_opt = ScenarioResult("optimistic", 60, 45, 118.0, 100.0, 1.5, [], "BUY", "MEDIUM", 45)
        bearish = ScenarioResult("pessimistic", 30, 55, 80.0, 95.0, 0.3, [], "SELL", "HIGH", 45)
        rec_bear = analyzer._determine_recommended_scenario(low_conf_opt, neutral, bearish)
        assert rec_bear == "pessimistic"

    def test_calculate_overall_confidence_thresholds(self, scenario_analyzer):
        """Overall confidence should follow probability ranges."""
        analyzer, _ = scenario_analyzer
        optimistic_high = ScenarioResult("optimistic", 75, 65, 120.0, 100.0, 2.0, [], "BUY", "HIGH", 30)
        neutral = ScenarioResult("neutral", 45, 25, 110.0, 100.0, 1.0, [], "HOLD", "MEDIUM", 60)
        pessimistic = ScenarioResult("pessimistic", 15, 10, 90.0, 100.0, 0.5, [], "SELL", "LOW", 90)

        assert analyzer._calculate_overall_confidence(optimistic_high, neutral, pessimistic) == "HIGH"

        optimistic_medium = ScenarioResult("optimistic", 60, 50, 118.0, 100.0, 1.5, [], "BUY", "MEDIUM", 45)
        neutral_prob = ScenarioResult("neutral", 50, 45, 110.0, 100.0, 1.0, [], "HOLD", "MEDIUM", 60)
        assert analyzer._calculate_overall_confidence(optimistic_medium, neutral_prob, pessimistic) == "MEDIUM"

    def test_calculate_atr_handles_short_and_long_series(self, scenario_analyzer):
        """ATR should be zero for short series and positive for long ones."""
        analyzer, candles = scenario_analyzer
        short_value = analyzer._calculate_atr(candles[:5])
        assert short_value == 0.0

        long_value = analyzer._calculate_atr(candles)
        assert long_value >= 0.0

    def test_calculate_rsi_handles_flat_series(self, scenario_analyzer):
        """RSI should return default extremes for flat data."""
        analyzer, _ = scenario_analyzer
        flat_prices = np.full(30, 100.0)
        rsi_value = analyzer._calculate_rsi(flat_prices)
        assert rsi_value == 100.0

        short_prices = np.array([100.0, 100.5])
        assert analyzer._calculate_rsi(short_prices) == 50.0

    def test_calculate_macd_handles_series_lengths(self, scenario_analyzer):
        """MACD should degrade gracefully with insufficient data."""
        analyzer, _ = scenario_analyzer
        short_series = np.linspace(1, 5, 5)
        macd_short, signal_short = analyzer._calculate_macd(short_series)
        assert macd_short == 0.0
        assert signal_short == 0.0

        long_series = np.linspace(1, 100, 200)
        macd_long, signal_long = analyzer._calculate_macd(long_series)
        assert isinstance(macd_long, float)
        assert isinstance(signal_long, float)
