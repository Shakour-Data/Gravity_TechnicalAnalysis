"""
Unit tests for Trend indicators (src/core/indicators/trend.py)

Tests cover:
- All 10 trend indicators (SMA, EMA, WMA, DEMA, TEMA, MACD, ADX, Donchian Channels, Aroon, Vortex Indicator, McGinley Dynamic)
- Signal generation and confidence levels
- Trend calculations and moving average logic
- Edge cases and error handling
- Backward compatibility methods
- Helper functions
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from gravity_tech.core.indicators.trend import (
    TrendIndicators
)
from gravity_tech.core.domain.entities import (
    Candle,
    IndicatorResult,
    CoreSignalStrength as SignalStrength,
    IndicatorCategory
)


class TestTrendIndicators:
    """Test TrendIndicators class methods"""

    @pytest.fixture
    def sample_candles(self):
        """Create sample candles for testing"""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        prices = []
        for i in range(60):
            # Upward trend with some noise
            base_price = 100 + i * 0.8
            price = base_price + 3 * np.sin(2 * np.pi * i / 15)  # Add some oscillation
            prices.append(price)

        candles = []
        for i, price in enumerate(prices):
            candle = Candle(
                timestamp=base_time,
                open=price - 0.5,
                high=price + 2.0,
                low=price - 2.0,
                close=price,
                volume=1000 + i * 10
            )
            candles.append(candle)

        return candles

    @pytest.fixture
    def minimal_candles(self):
        """Create minimal candles for edge case testing"""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        return [
            Candle(timestamp=base_time, open=100, high=101, low=99, close=100.5, volume=1000),
            Candle(timestamp=base_time, open=100.5, high=102, low=100, close=101.5, volume=1100),
            Candle(timestamp=base_time, open=101.5, high=103, low=101, close=102.5, volume=1200),
        ]

    def test_sma_normal_operation(self, sample_candles):
        """Test SMA with normal data"""
        result = TrendIndicators.sma(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.TREND
        assert result.indicator_name == "SMA(20)"
        assert isinstance(result.signal, SignalStrength)
        assert 0 <= result.confidence <= 1
        assert isinstance(result.description, str)
        assert "SMA" in result.description

    def test_sma_calculation_accuracy(self, sample_candles):
        """Test SMA calculation accuracy"""
        result = TrendIndicators.sma(sample_candles, period=20)

        closes = np.array([c.close for c in sample_candles])
        import pandas as pd
        expected_sma = pd.Series(closes).rolling(window=20).mean().iloc[-1]
        assert abs(result.value - expected_sma) < 0.01

    def test_sma_signal_generation(self, sample_candles):
        """Test SMA signal generation based on price position"""
        result = TrendIndicators.sma(sample_candles)

        # Signal should be based on price vs SMA and slope
        assert result.signal in [
            SignalStrength.VERY_BULLISH, SignalStrength.BULLISH,
            SignalStrength.BULLISH_BROKEN, SignalStrength.NEUTRAL,
            SignalStrength.BEARISH_BROKEN, SignalStrength.BEARISH, SignalStrength.VERY_BEARISH
        ]

    def test_ema_normal_operation(self, sample_candles):
        """Test EMA with normal data"""
        result = TrendIndicators.ema(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.TREND
        assert result.indicator_name == "EMA(20)"
        assert isinstance(result.signal, SignalStrength)

    def test_ema_calculation_accuracy(self, sample_candles):
        """Test EMA calculation accuracy"""
        result = TrendIndicators.ema(sample_candles, period=20)

        closes = np.array([c.close for c in sample_candles])
        expected_ema = pd.Series(closes).ewm(span=20, adjust=False).mean()
        expected_ema = expected_ema.iloc[-1]

        assert abs(result.value - expected_ema) < 0.01

    def test_wma_normal_operation(self, sample_candles):
        """Test WMA with normal data"""
        result = TrendIndicators.wma(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.TREND
        assert result.indicator_name == "WMA(20)"
        assert isinstance(result.signal, SignalStrength)

    def test_wma_calculation_accuracy(self, sample_candles):
        """Test WMA calculation accuracy"""
        result = TrendIndicators.wma(sample_candles, period=20)

        closes = np.array([c.close for c in sample_candles])
        weights = np.arange(1, 21)  # 1 to 20
        window = closes[-20:]
        expected_wma = np.dot(window, weights) / weights.sum()

        assert abs(result.value - expected_wma) < 0.01

    def test_dema_normal_operation(self, sample_candles):
        """Test DEMA with normal data"""
        result = TrendIndicators.dema(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.TREND
        assert result.indicator_name == "DEMA(20)"
        assert isinstance(result.signal, SignalStrength)

    def test_dema_calculation_accuracy(self, sample_candles):
        """Test DEMA calculation accuracy"""
        result = TrendIndicators.dema(sample_candles, period=20)

        closes = pd.Series([c.close for c in sample_candles])
        ema1 = closes.ewm(span=20, adjust=False).mean()
        ema2 = ema1.ewm(span=20, adjust=False).mean()
        expected_dema = 2 * ema1 - ema2
        expected_dema = expected_dema.iloc[-1]

        assert abs(result.value - expected_dema) < 0.01

    def test_tema_normal_operation(self, sample_candles):
        """Test TEMA with normal data"""
        result = TrendIndicators.tema(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.TREND
        assert result.indicator_name == "TEMA(20)"
        assert isinstance(result.signal, SignalStrength)

    def test_tema_calculation_accuracy(self, sample_candles):
        """Test TEMA calculation accuracy"""
        result = TrendIndicators.tema(sample_candles, period=20)

        closes = pd.Series([c.close for c in sample_candles])
        ema1 = closes.ewm(span=20, adjust=False).mean()
        ema2 = ema1.ewm(span=20, adjust=False).mean()
        ema3 = ema2.ewm(span=20, adjust=False).mean()
        expected_tema = 3 * ema1 - 3 * ema2 + ema3
        expected_tema = expected_tema.iloc[-1]

        assert abs(result.value - expected_tema) < 0.01

    def test_macd_normal_operation(self, sample_candles):
        """Test MACD with normal data"""
        result = TrendIndicators.macd(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.TREND
        assert result.indicator_name == "MACD"
        assert isinstance(result.signal, SignalStrength)

        # Check additional values
        assert "signal" in result.additional_values
        assert "histogram" in result.additional_values

    def test_macd_calculation_accuracy(self, sample_candles):
        """Test MACD calculation accuracy"""
        result = TrendIndicators.macd(sample_candles, fast=12, slow=26, signal_period=9)

        closes = pd.Series([c.close for c in sample_candles])
        ema_fast = closes.ewm(span=12, adjust=False).mean()
        ema_slow = closes.ewm(span=26, adjust=False).mean()
        expected_macd = ema_fast - ema_slow
        expected_signal = expected_macd.ewm(span=9, adjust=False).mean()
        expected_hist = expected_macd - expected_signal

        assert abs(result.value - expected_macd.iloc[-1]) < 0.01
        assert abs(result.additional_values["signal"] - expected_signal.iloc[-1]) < 0.01
        assert abs(result.additional_values["histogram"] - expected_hist.iloc[-1]) < 0.01

    def test_adx_normal_operation(self, sample_candles):
        """Test ADX with normal data"""
        result = TrendIndicators.adx(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.TREND
        assert result.indicator_name == "ADX(14)"
        assert isinstance(result.signal, SignalStrength)

        # Check additional values
        assert "+DI" in result.additional_values
        assert "-DI" in result.additional_values

    def test_adx_calculation_accuracy(self, sample_candles):
        """Test ADX calculation accuracy"""
        result = TrendIndicators.adx(sample_candles, period=14)

        df = pd.DataFrame({
            'high': [c.high for c in sample_candles],
            'low': [c.low for c in sample_candles],
            'close': [c.close for c in sample_candles]
        })

        # Calculate +DM and -DM
        df['high_diff'] = df['high'].diff()
        df['low_diff'] = -df['low'].diff()

        df['+DM'] = np.where(
            (df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0),
            df['high_diff'],
            0
        )
        df['-DM'] = np.where(
            (df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0),
            df['low_diff'],
            0
        )

        # True Range
        df['TR'] = df.apply(
            lambda row: max(
                row['high'] - row['low'],
                abs(row['high'] - df['close'].shift(1).loc[row.name]) if row.name > 0 else 0,
                abs(row['low'] - df['close'].shift(1).loc[row.name]) if row.name > 0 else 0
            ),
            axis=1
        )

        # Smooth values
        atr = df['TR'].rolling(window=14).mean()
        plus_di = 100 * (df['+DM'].rolling(window=14).mean() / atr)
        minus_di = 100 * (df['-DM'].rolling(window=14).mean() / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        expected_adx = dx.rolling(window=14).mean()

        assert abs(result.value - expected_adx.iloc[-1]) < 0.01

    def test_donchian_channels_normal_operation(self, sample_candles):
        """Test Donchian Channels with normal data"""
        result = TrendIndicators.donchian_channels(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.TREND
        assert "Donchian Channels" in result.indicator_name
        assert isinstance(result.signal, SignalStrength)

        # Check additional values
        assert "upper_band" in result.additional_values
        assert "middle_band" in result.additional_values
        assert "lower_band" in result.additional_values

    def test_aroon_normal_operation(self, sample_candles):
        """Test Aroon with normal data"""
        result = TrendIndicators.aroon(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.TREND
        assert "Aroon" in result.indicator_name
        assert isinstance(result.signal, SignalStrength)

        # Check additional values
        assert "aroon_up" in result.additional_values
        assert "aroon_down" in result.additional_values
        assert "aroon_oscillator" in result.additional_values

    def test_vortex_indicator_normal_operation(self, sample_candles):
        """Test Vortex Indicator with normal data"""
        result = TrendIndicators.vortex_indicator(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.TREND
        assert "Vortex Indicator" in result.indicator_name
        assert isinstance(result.signal, SignalStrength)

        # Check additional values
        assert "vi_plus" in result.additional_values
        assert "vi_minus" in result.additional_values
        assert "vi_difference" in result.additional_values

    def test_mcginley_dynamic_normal_operation(self, sample_candles):
        """Test McGinley Dynamic with normal data"""
        result = TrendIndicators.mcginley_dynamic(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.TREND
        assert "McGinley Dynamic" in result.indicator_name
        assert isinstance(result.signal, SignalStrength)

        # Check additional values
        assert "md_value" in result.additional_values
        assert "current_price" in result.additional_values
        assert "deviation_pct" in result.additional_values
        assert "slope_pct" in result.additional_values

    def test_calculate_all_normal(self, sample_candles):
        """Test calculate_all method"""
        results = TrendIndicators.calculate_all(sample_candles)

        assert isinstance(results, list)
        assert len(results) >= 8  # Should return multiple indicators

        for result in results:
            assert isinstance(result, IndicatorResult)
            assert result.category == IndicatorCategory.TREND

    def test_calculate_all_insufficient_data(self, minimal_candles):
        """Test calculate_all with insufficient data"""
        results = TrendIndicators.calculate_all(minimal_candles)

        # Should return basic indicators that work with minimal data
        assert isinstance(results, list)
        # May return fewer indicators with minimal data

    # Edge cases and error handling
    def test_all_indicators_with_empty_candles(self):
        """Test all indicators with empty candle list"""
        empty_candles = []

        # Should handle gracefully or raise appropriate errors
        with pytest.raises((IndexError, ZeroDivisionError, ValueError)):
            TrendIndicators.sma(empty_candles)

        with pytest.raises((IndexError, ZeroDivisionError, ValueError)):
            TrendIndicators.ema(empty_candles)

    def test_all_indicators_with_single_candle(self):
        """Test all indicators with single candle"""
        single_candle = [Candle(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            open=100, high=101, low=99, close=100.5, volume=1000
        )]

        # همه اندیکاتورها باید ValueError بدهند
        with pytest.raises(ValueError):
            TrendIndicators.sma(single_candle)
        with pytest.raises(ValueError):
            TrendIndicators.ema(single_candle)
        with pytest.raises(ValueError):
            TrendIndicators.wma(single_candle)
        with pytest.raises(ValueError):
            TrendIndicators.dema(single_candle)
        with pytest.raises(ValueError):
            TrendIndicators.tema(single_candle)
        with pytest.raises(ValueError):
            TrendIndicators.macd(single_candle)
        with pytest.raises(ValueError):
            TrendIndicators.adx(single_candle)
        with pytest.raises(ValueError):
            TrendIndicators.donchian_channels(single_candle)
        with pytest.raises(ValueError):
            TrendIndicators.aroon(single_candle)
        with pytest.raises(ValueError):
            TrendIndicators.vortex_indicator(single_candle)
        with pytest.raises(ValueError):
            TrendIndicators.mcginley_dynamic(single_candle)

    def test_indicators_with_zero_prices(self, sample_candles):
        """Test indicators with zero or negative prices"""
        # Create candle with zero close
        zero_candle = sample_candles[-1]._replace(close=0)
        test_candles = sample_candles[:-1] + [zero_candle]

        # Should handle gracefully
        result_sma = TrendIndicators.sma(test_candles)
        assert isinstance(result_sma, IndicatorResult)

        result_ema = TrendIndicators.ema(test_candles)
        assert isinstance(result_ema, IndicatorResult)

    def test_macd_with_extreme_values(self, sample_candles):
        """Test MACD with extreme price movements"""
        # Create large price jump
        extreme_candle = sample_candles[-1]._replace(close=sample_candles[-1].close * 2)
        test_candles = sample_candles[:-1] + [extreme_candle]

        result = TrendIndicators.macd(test_candles)
        assert isinstance(result, IndicatorResult)

    def test_adx_with_flat_market(self, sample_candles):
        """Test ADX with flat market conditions"""
        # Create flat candles
        flat_candles = []
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        for i in range(30):
            candle = Candle(
                timestamp=base_time,
                open=100, high=100.5, low=99.5, close=100,
                volume=1000
            )
            flat_candles.append(candle)

        result = TrendIndicators.adx(flat_candles)
        assert isinstance(result, IndicatorResult)

    def test_donchian_channels_breakout(self, sample_candles):
        """Test Donchian Channels breakout detection"""
        # Test breakout above upper channel
        breakout_candle = sample_candles[-1]._replace(close=sample_candles[-1].high + 10)
        test_candles = sample_candles[:-1] + [breakout_candle]

        result = TrendIndicators.donchian_channels(test_candles)
        assert result.signal == SignalStrength.VERY_BULLISH

        # Test breakout below lower channel
        breakout_candle = sample_candles[-1]._replace(close=sample_candles[-1].low - 10)
        test_candles = sample_candles[:-1] + [breakout_candle]

        result = TrendIndicators.donchian_channels(test_candles)
        # سیگنال جدید: نزولی شکسته شده یا نزولی
        assert result.signal.name in ["BEARISH_BROKEN", "BEARISH"]

    def test_aroon_extremes(self, sample_candles):
        """Test Aroon with extreme values"""
        result = TrendIndicators.aroon(sample_candles)

        aroon_up = result.additional_values["aroon_up"]
        aroon_down = result.additional_values["aroon_down"]

        assert 0 <= aroon_up <= 100
        assert 0 <= aroon_down <= 100

    def test_vortex_indicator_calculation(self, sample_candles):
        """Test Vortex Indicator calculation accuracy"""
        result = TrendIndicators.vortex_indicator(sample_candles, period=14)

        highs = np.array([c.high for c in sample_candles])
        lows = np.array([c.low for c in sample_candles])

        # Manual calculation
        vortex_plus = np.abs(highs[1:] - lows[:-1])
        vortex_minus = np.abs(lows[1:] - highs[:-1])

        vi_plus = pd.Series(vortex_plus).rolling(window=14).sum() / pd.Series(highs[1:] - lows[1:]).rolling(window=14).sum()
        vi_minus = pd.Series(vortex_minus).rolling(window=14).sum() / pd.Series(highs[1:] - lows[1:]).rolling(window=14).sum()

        expected_diff = vi_plus.iloc[-1] - vi_minus.iloc[-1]

        assert abs(result.value - expected_diff) < 0.01

    def test_mcginley_dynamic_adaptive(self, sample_candles):
        """Test McGinley Dynamic adaptive behavior"""
        result = TrendIndicators.mcginley_dynamic(sample_candles, period=20, k_factor=0.6)

        # Should adapt to price changes
        assert isinstance(result, IndicatorResult)
        assert result.additional_values["md_value"] > 0

    def test_signal_confidence_levels(self, sample_candles):
        """Test that confidence levels are appropriate for signals"""
        indicators = [
            lambda: TrendIndicators.sma(sample_candles),
            lambda: TrendIndicators.ema(sample_candles),
            lambda: TrendIndicators.macd(sample_candles),
            lambda: TrendIndicators.adx(sample_candles),
        ]

        for indicator_func in indicators:
            result = indicator_func()
            assert 0 <= result.confidence <= 1

            # Strong signals should have reasonable confidence
            if result.signal in [SignalStrength.VERY_BULLISH, SignalStrength.VERY_BEARISH]:
                assert result.confidence >= 0.7

    def test_indicator_names_formatting(self, sample_candles):
        """Test that indicator names are properly formatted"""
        assert TrendIndicators.sma(sample_candles, 25).indicator_name == "SMA(25)"
        assert TrendIndicators.ema(sample_candles, 25).indicator_name == "EMA(25)"
        assert TrendIndicators.wma(sample_candles, 25).indicator_name == "WMA(25)"
        assert TrendIndicators.dema(sample_candles, 25).indicator_name == "DEMA(25)"
        assert TrendIndicators.tema(sample_candles, 25).indicator_name == "TEMA(25)"
        assert TrendIndicators.macd(sample_candles).indicator_name == "MACD"
        assert TrendIndicators.adx(sample_candles, 21).indicator_name == "ADX(21)"
        assert "Donchian Channels" in TrendIndicators.donchian_channels(sample_candles, 25).indicator_name
        assert "Aroon" in TrendIndicators.aroon(sample_candles, 30).indicator_name
        assert "Vortex Indicator" in TrendIndicators.vortex_indicator(sample_candles, 21).indicator_name
        assert "McGinley Dynamic" in TrendIndicators.mcginley_dynamic(sample_candles, 25).indicator_name

    def test_description_content(self, sample_candles):
        """Test that descriptions contain relevant information"""
        result_sma = TrendIndicators.sma(sample_candles)
        assert "SMA" in result_sma.description and "%" in result_sma.description

        result_macd = TrendIndicators.macd(sample_candles)
        # متن فارسی: انتظار داریم 'MACD' و 'خط سیگنال' یا 'سیگنال' در توضیح باشد
        assert "MACD" in result_macd.description and ("خط سیگنال" in result_macd.description or "سیگنال" in result_macd.description)

        result_adx = TrendIndicators.adx(sample_candles)
        # متن فارسی: انتظار داریم 'قدرت روند' در توضیح باشد و 'ADX' نباشد
        assert ("قدرت روند" in result_adx.description) and ("ADX" not in result_adx.description)

    def test_additional_values_completeness(self, sample_candles):
        """Test that additional_values contain all expected keys when applicable"""
        result_macd = TrendIndicators.macd(sample_candles)
        required_macd_keys = ["signal", "histogram"]
        for key in required_macd_keys:
            assert key in result_macd.additional_values

        result_adx = TrendIndicators.adx(sample_candles)
        required_adx_keys = ["+DI", "-DI"]
        for key in required_adx_keys:
            assert key in result_adx.additional_values

        result_donchian = TrendIndicators.donchian_channels(sample_candles)
        required_donchian_keys = ["upper_band", "middle_band", "lower_band", "channel_width_pct", "price_position_pct"]
        for key in required_donchian_keys:
            assert key in result_donchian.additional_values

    def test_trend_indicators_range_bounds(self, sample_candles):
        """Test that trend indicators stay within expected ranges"""
        result_sma = TrendIndicators.sma(sample_candles)
        assert result_sma.value > 0

        result_macd = TrendIndicators.macd(sample_candles)
        # MACD can be any value

        result_adx = TrendIndicators.adx(sample_candles)
        assert 0 <= result_adx.value <= 100

        result_aroon = TrendIndicators.aroon(sample_candles)
        assert -100 <= result_aroon.value <= 100

    def test_moving_average_convergence(self, sample_candles):
        """Test moving average convergence behavior"""
        # Test with converging prices
        converging_candles = []
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        for i in range(40):
            # Converging to a price
            price = 100 + 50 * np.exp(-i * 0.1)
            candle = Candle(
                timestamp=base_time,
                open=price - 0.5,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1000
            )
            converging_candles.append(candle)

        result_sma = TrendIndicators.sma(converging_candles)
        assert isinstance(result_sma, IndicatorResult)

        result_ema = TrendIndicators.ema(converging_candles)
        assert isinstance(result_ema, IndicatorResult)

    def test_trend_strength_indicators(self, sample_candles):
        """Test trend strength indicators like ADX"""
        result_adx = TrendIndicators.adx(sample_candles)

        # ADX should indicate trend strength
        adx_value = result_adx.value
        if adx_value > 25:
            assert result_adx.signal in [SignalStrength.VERY_BULLISH, SignalStrength.BULLISH, SignalStrength.VERY_BEARISH, SignalStrength.BEARISH]
        else:
            assert result_adx.signal in [SignalStrength.BULLISH_BROKEN, SignalStrength.BEARISH_BROKEN, SignalStrength.NEUTRAL]

    def test_breakout_detection(self, sample_candles):
        """Test breakout detection in trend indicators"""
        # Test Donchian breakout
        result_donchian = TrendIndicators.donchian_channels(sample_candles)

        upper = result_donchian.additional_values["upper_band"]
        lower = result_donchian.additional_values["lower_band"]
        current_price = sample_candles[-1].close

        if current_price >= upper:
            assert "شکست سقف" in result_donchian.description
        elif current_price <= lower:
            assert "شکست کف" in result_donchian.description

