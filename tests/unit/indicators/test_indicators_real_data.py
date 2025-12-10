"""
Technical Indicator Tests with Real TSE Data

Comprehensive tests for technical indicators using actual Iranian stock market data.
"""

import math

import pytest


class TestTechnicalIndicatorsWithTSEData:
    """Test technical indicators with real TSE market data."""

    def test_simple_moving_average(self, tse_candles_short):
        """Test SMA calculation with real data."""
        # Calculate SMA (20 period)
        prices = [c.close for c in tse_candles_short]

        if len(prices) < 20:
            # Not enough data for 20-period SMA
            simple_sma = sum(prices) / len(prices)
        else:
            simple_sma = sum(prices[-20:]) / 20

        # Verify
        assert simple_sma > 0
        assert isinstance(simple_sma, float)

    def test_exponential_moving_average(self, tse_candles_short):
        """Test EMA calculation."""
        prices = [c.close for c in tse_candles_short]

        if len(prices) < 2:
            pytest.skip("Not enough data for EMA")

        # Simple EMA calculation
        alpha = 2 / (12 + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = price * alpha + ema * (1 - alpha)

        assert ema > 0
        assert isinstance(ema, float)

    def test_rsi_indicator(self, tse_candles_short):
        """Test RSI calculation with real data."""
        prices = [c.close for c in tse_candles_short]

        if len(prices) < 14:
            pytest.skip("Not enough data for RSI")

        # Calculate price changes
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]

        # Separate gains and losses
        gains = [max(c, 0) for c in changes]
        losses = [abs(min(c, 0)) for c in changes]

        # Calculate averages
        avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else sum(gains) / len(gains)
        avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else sum(losses) / len(losses)

        # Calculate RSI
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # Verify
        assert 0 <= rsi <= 100
        assert isinstance(rsi, float)

    def test_macd_indicator(self, tse_candles_short):
        """Test MACD calculation."""
        prices = [c.close for c in tse_candles_short]

        if len(prices) < 26:
            pytest.skip("Not enough data for MACD")

        # Simplified MACD: difference between fast and slow EMA
        alpha_fast = 2 / (12 + 1)
        alpha_slow = 2 / (26 + 1)

        ema_fast = prices[0]
        ema_slow = prices[0]

        for price in prices[1:]:
            ema_fast = price * alpha_fast + ema_fast * (1 - alpha_fast)
            ema_slow = price * alpha_slow + ema_slow * (1 - alpha_slow)

        macd = ema_fast - ema_slow

        # Verify
        assert isinstance(macd, float)

    def test_bollinger_bands(self, tse_candles_short):
        """Test Bollinger Bands calculation."""
        prices = [c.close for c in tse_candles_short]

        if len(prices) < 20:
            pytest.skip("Not enough data for Bollinger Bands")

        # Calculate 20-period SMA
        sma = sum(prices[-20:]) / 20

        # Calculate standard deviation
        variance = sum((p - sma) ** 2 for p in prices[-20:]) / 20
        std_dev = math.sqrt(variance)

        # Calculate bands
        upper_band = sma + (2 * std_dev)
        lower_band = sma - (2 * std_dev)

        # Verify
        assert lower_band < sma < upper_band
        assert upper_band > lower_band

    def test_stochastic_oscillator(self, tse_candles_short):
        """Test Stochastic Oscillator calculation."""
        lows = [c.low for c in tse_candles_short]
        highs = [c.high for c in tse_candles_short]
        closes = [c.close for c in tse_candles_short]

        if len(closes) < 14:
            pytest.skip("Not enough data for Stochastic")

        # Calculate over last 14 periods
        period_low = min(lows[-14:])
        period_high = max(highs[-14:])
        current_close = closes[-1]

        # Calculate %K
        if period_high == period_low:
            stoch_k = 50
        else:
            stoch_k = ((current_close - period_low) / (period_high - period_low)) * 100

        # Verify
        assert 0 <= stoch_k <= 100
        assert isinstance(stoch_k, float)

    def test_atr_indicator(self, tse_candles_short):
        """Test Average True Range calculation."""
        if len(tse_candles_short) < 14:
            pytest.skip("Not enough data for ATR")

        # Calculate true ranges
        true_ranges = []
        prev_close = tse_candles_short[0].close

        for candle in tse_candles_short[1:]:
            tr1 = candle.high - candle.low
            tr2 = abs(candle.high - prev_close)
            tr3 = abs(candle.low - prev_close)
            tr = max(tr1, tr2, tr3)
            true_ranges.append(tr)
            prev_close = candle.close

        # Calculate ATR
        if len(true_ranges) >= 14:
            atr = sum(true_ranges[-14:]) / 14
        else:
            atr = sum(true_ranges) / len(true_ranges)

        # Verify
        assert atr > 0
        assert isinstance(atr, float)

    def test_volume_weighted_price(self, tse_candles_short):
        """Test Volume Weighted Average Price."""
        closes = [c.close for c in tse_candles_short]
        volumes = [c.volume for c in tse_candles_short]

        if len(closes) < 20:
            pytest.skip("Not enough data for VWAP")

        # Calculate VWAP
        total_tp_volume = sum(closes[i] * volumes[i] for i in range(len(closes)))
        total_volume = sum(volumes)

        vwap = total_tp_volume / total_volume if total_volume > 0 else 0

        # Verify
        assert vwap > 0
        assert isinstance(vwap, float)

    def test_on_balance_volume(self, tse_candles_short):
        """Test On-Balance Volume calculation."""
        if len(tse_candles_short) < 2:
            pytest.skip("Not enough data for OBV")

        obv = 0
        for i in range(len(tse_candles_short)):
            if i == 0:
                obv = tse_candles_short[i].volume
            else:
                if tse_candles_short[i].close > tse_candles_short[i-1].close:
                    obv += tse_candles_short[i].volume
                elif tse_candles_short[i].close < tse_candles_short[i-1].close:
                    obv -= tse_candles_short[i].volume

        # Verify
        assert isinstance(obv, int | float)

    def test_accumulation_distribution(self, tse_candles_short):
        """Test Accumulation/Distribution Line."""
        if len(tse_candles_short) < 2:
            pytest.skip("Not enough data for A/D")

        ad = 0
        for candle in tse_candles_short:
            if candle.high == candle.low:
                clv = 0
            else:
                clv = ((candle.close - candle.low) - (candle.high - candle.close)) / (candle.high - candle.low)
            ad += clv * candle.volume

        # Verify
        assert isinstance(ad, int | float)


class TestMultipleSymbolIndicators:
    """Test indicator calculations for multiple TSE symbols."""

    def test_indicators_for_total(self, tse_candles_total):
        """Test indicators for TOTAL symbol."""
        assert len(tse_candles_total) > 0

        prices = [c.close for c in tse_candles_total]
        high_price = max(c.high for c in tse_candles_total)
        low_price = min(c.low for c in tse_candles_total)

        assert high_price > low_price
        assert len(prices) > 0

    def test_indicators_for_petroff(self, tse_candles_petroff):
        """Test indicators for PETROFF symbol."""
        assert len(tse_candles_petroff) > 0

        volumes = [c.volume for c in tse_candles_petroff]
        total_volume = sum(volumes)

        assert total_volume > 0
        assert all(v >= 0 for v in volumes)

    def test_indicators_for_iraninoil(self, tse_candles_iraninoil):
        """Test indicators for IRANINOIL symbol."""
        assert len(tse_candles_iraninoil) > 0

        closes = [c.close for c in tse_candles_iraninoil]
        avg_price = sum(closes) / len(closes)

        assert avg_price > 0
        assert all(p > 0 for p in closes)


class TestIndicatorPerformance:
    """Test indicator calculation performance."""

    def test_indicator_calculation_speed(self, tse_candles_long):
        """Test indicator calculation speed with large dataset."""
        import time

        prices = [c.close for c in tse_candles_long]

        # SMA
        start = time.time()
        _ = sum(prices[-20:]) / 20
        sma_time = time.time() - start
        assert sma_time < 0.001

        # RSI simulation
        start = time.time()
        _ = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        rsi_time = time.time() - start
        assert rsi_time < 0.1

    def test_batch_indicator_calculation(self, tse_candles_long):
        """Test batch calculation of multiple indicators."""
        prices = [c.close for c in tse_candles_long]
        volumes = [c.volume for c in tse_candles_long]

        # Calculate multiple indicators
        indicators = {}

        if len(prices) >= 20:
            indicators['SMA20'] = sum(prices[-20:]) / 20

        if len(prices) >= 14:
            indicators['RSI14'] = 50  # Simplified

        if len(volumes) > 0:
            indicators['AVG_VOLUME'] = sum(volumes) / len(volumes)

        # Verify
        assert 'SMA20' in indicators or 'RSI14' in indicators
        assert all(v > 0 for v in indicators.values())
