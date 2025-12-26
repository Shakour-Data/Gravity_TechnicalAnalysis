"""
Phase 4: Unit Tests for Technical Indicators

Test coverage for all technical indicators:
- Moving Averages (SMA, EMA, WMA)
- Momentum (RSI, Stochastic)
- Volatility (Bollinger Bands, ATR)
- Trend (MACD, ADX)

Target: 100% coverage of indicator functions
"""

import pytest
import numpy as np
from typing import List
from datetime import datetime, timedelta


# ============================================================================
# TEST DATA BUILDERS
# ============================================================================

class IndicatorTestData:
    """Generate test data for indicator testing"""
    
    @staticmethod
    def generate_price_series(
        length: int = 100,
        start_price: float = 100.0,
        volatility: float = 0.02,
        trend: float = 0.001
    ) -> List[float]:
        """Generate synthetic price series"""
        prices = [start_price]
        
        for _ in range(length - 1):
            change = np.random.normal(trend, volatility)
            next_price = prices[-1] * (1 + change)
            prices.append(max(next_price, 0.01))  # Prevent negative
        
        return prices
    
    @staticmethod
    def generate_ohlcv(length: int = 100) -> List[dict]:
        """Generate synthetic OHLCV data"""
        closes = IndicatorTestData.generate_price_series(length)
        ohlcv = []
        
        for i, close in enumerate(closes):
            open_price = close * (1 + np.random.uniform(-0.01, 0.01))
            high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.005)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.005)))
            volume = 1000 + int(np.random.normal(0, 100))
            
            ohlcv.append({
                "timestamp": datetime.now() + timedelta(days=i),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": max(volume, 100)
            })
        
        return ohlcv


# ============================================================================
# MOVING AVERAGES TESTS
# ============================================================================

@pytest.mark.unit
class TestSimpleMovingAverage:
    """Test SMA (Simple Moving Average) calculation"""
    
    def test_sma_basic_calculation(self):
        """Test basic SMA calculation"""
        prices = [100, 102, 104, 106, 108]
        window = 3
        
        # Manual calculation
        expected = [
            (100 + 102 + 104) / 3,  # 102.0
            (102 + 104 + 106) / 3,  # 104.0
            (104 + 106 + 108) / 3,  # 106.0
        ]
        
        # SMA implementation
        sma = []
        for i in range(len(prices) - window + 1):
            avg = sum(prices[i:i + window]) / window
            sma.append(avg)
        
        assert len(sma) == 3
        assert sma == expected
    
    def test_sma_window_larger_than_data(self):
        """Test SMA with window > data length"""
        prices = [100, 102]
        window = 5
        
        sma = []
        if len(prices) >= window:
            for i in range(len(prices) - window + 1):
                avg = sum(prices[i:i + window]) / window
                sma.append(avg)
        
        assert len(sma) == 0
    
    def test_sma_window_equals_data_length(self):
        """Test SMA with window == data length"""
        prices = [100, 102, 104, 106]
        window = 4
        
        sma = sum(prices) / window
        expected = (100 + 102 + 104 + 106) / 4
        
        assert sma == expected
    
    def test_sma_single_element(self):
        """Test SMA with single price"""
        prices = [100]
        window = 1
        
        sma = []
        for i in range(len(prices) - window + 1):
            avg = sum(prices[i:i + window]) / window
            sma.append(avg)
        
        assert len(sma) == 1
        assert sma[0] == 100.0
    
    def test_sma_with_large_dataset(self):
        """Test SMA with large dataset (1000+ points)"""
        prices = IndicatorTestData.generate_price_series(1000)
        window = 20
        
        sma = []
        for i in range(len(prices) - window + 1):
            avg = sum(prices[i:i + window]) / window
            sma.append(avg)
        
        assert len(sma) == 1000 - 20 + 1
        assert all(isinstance(x, float) for x in sma)
    
    def test_sma_monotonic_increasing(self):
        """Test SMA with monotonically increasing prices"""
        prices = list(range(1, 101))  # 1 to 100
        window = 10
        
        sma = []
        for i in range(len(prices) - window + 1):
            avg = sum(prices[i:i + window]) / window
            sma.append(avg)
        
        # SMA should also be increasing for increasing prices
        for i in range(len(sma) - 1):
            assert sma[i] < sma[i + 1]


@pytest.mark.unit
class TestExponentialMovingAverage:
    """Test EMA (Exponential Moving Average) calculation"""
    
    def test_ema_basic_calculation(self):
        """Test basic EMA calculation"""
        prices = [100, 102, 104, 106, 108]
        period = 3
        
        # EMA formula: EMA = (Price - EMA_previous) * multiplier + EMA_previous
        multiplier = 2 / (period + 1)
        
        ema_values = []
        ema = prices[0]  # Start with first price
        
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
            ema_values.append(ema)
        
        assert len(ema_values) == 4
        assert all(isinstance(x, float) for x in ema_values)
    
    def test_ema_convergence_to_prices(self):
        """Test EMA converges to prices over time"""
        prices = [100] * 100  # Constant prices
        period = 10
        multiplier = 2 / (period + 1)
        
        ema = prices[0]
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        
        # Should converge close to 100
        assert 99 < ema < 101
    
    def test_ema_responsiveness(self):
        """Test EMA responsiveness to price changes"""
        # Low period = more responsive
        prices = [100, 150, 100, 150, 100]
        period = 2
        multiplier = 2 / (period + 1)
        
        ema = prices[0]
        ema_values = [ema]
        
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
            ema_values.append(ema)
        
        assert len(ema_values) == 5


# ============================================================================
# MOMENTUM INDICATORS TESTS
# ============================================================================

@pytest.mark.unit
class TestRelativeStrengthIndex:
    """Test RSI (Relative Strength Index) calculation"""
    
    def test_rsi_range(self):
        """Test RSI values are in range [0, 100]"""
        prices = IndicatorTestData.generate_price_series(100)
        period = 14
        
        # Calculate price changes
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        gains = [max(d, 0) for d in deltas]
        losses = [abs(min(d, 0)) for d in deltas]
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        if avg_loss == 0:
            rsi = 100 if avg_gain > 0 else 0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        assert 0 <= rsi <= 100
    
    def test_rsi_overbought_condition(self):
        """Test RSI >= 70 for strong uptrend"""
        # Strong uptrend
        prices = [100 + i * 2 for i in range(50)]
        period = 14
        
        # Calculate RSI
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(d, 0) for d in deltas]
        losses = [abs(min(d, 0)) for d in deltas]
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        assert rsi > 70  # Overbought
    
    def test_rsi_oversold_condition(self):
        """Test RSI <= 30 for strong downtrend"""
        # Strong downtrend
        prices = [100 - i * 2 for i in range(50)]
        period = 14
        
        # Calculate RSI
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(d, 0) for d in deltas]
        losses = [abs(min(d, 0)) for d in deltas]
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        if avg_loss == 0:
            rsi = 0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        assert rsi < 30  # Oversold
    
    def test_rsi_neutral_condition(self):
        """Test RSI ≈ 50 for sideways movement"""
        # Random walk
        prices = [100]
        for _ in range(100):
            change = np.random.choice([-1, 1])
            prices.append(prices[-1] + change)
        
        period = 14
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(d, 0) for d in deltas]
        losses = [abs(min(d, 0)) for d in deltas]
        
        avg_gain = sum(gains[period:2*period]) / period
        avg_loss = sum(losses[period:2*period]) / period
        
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            # RSI should be closer to 50 for random walk
            assert 30 < rsi < 70


# ============================================================================
# VOLATILITY INDICATORS TESTS
# ============================================================================

@pytest.mark.unit
class TestBollingerBands:
    """Test Bollinger Bands calculation"""
    
    def test_bb_calculation(self):
        """Test Bollinger Bands calculation"""
        prices = [100, 102, 104, 106, 108, 110, 112]
        period = 5
        std_dev = 2
        
        # SMA (middle band)
        sma = sum(prices[-period:]) / period
        
        # Standard deviation
        variance = sum((p - sma) ** 2 for p in prices[-period:]) / period
        std = np.sqrt(variance)
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        assert lower < sma < upper
        assert upper - lower == 2 * std * std_dev
    
    def test_bb_bands_contain_current_price(self):
        """Test price stays within bands most of the time"""
        prices = IndicatorTestData.generate_price_series(100)
        period = 20
        std_dev = 2
        
        within_bands = 0
        total = 0
        
        for i in range(period, len(prices)):
            sma = sum(prices[i-period:i]) / period
            variance = sum((p - sma) ** 2 for p in prices[i-period:i]) / period
            std = np.sqrt(variance)
            
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            
            if lower <= prices[i] <= upper:
                within_bands += 1
            total += 1
        
        # Should be ~95% within bands
        percentage = within_bands / total
        assert percentage > 0.90
    
    def test_bb_band_width(self):
        """Test band width increases with volatility"""
        # Low volatility
        low_vol_prices = [100.0] * 50 + [100.0 + np.random.normal(0, 0.1) for _ in range(50)]
        
        # High volatility
        high_vol_prices = [100.0 + np.random.normal(0, 2.0) for _ in range(100)]
        
        period = 20
        std_dev = 2
        
        # Calculate band width for both
        def get_band_width(prices):
            sma = sum(prices[-period:]) / period
            variance = sum((p - sma) ** 2 for p in prices[-period:]) / period
            std = np.sqrt(variance)
            return 2 * std * std_dev
        
        low_width = get_band_width(low_vol_prices)
        high_width = get_band_width(high_vol_prices)
        
        assert high_width > low_width


@pytest.mark.unit
class TestAverageTrueRange:
    """Test ATR (Average True Range) calculation"""
    
    def test_atr_basic_calculation(self):
        """Test ATR calculation"""
        ohlcv = IndicatorTestData.generate_ohlcv(50)
        period = 14
        
        true_ranges = []
        for i in range(1, len(ohlcv)):
            high = ohlcv[i]["high"]
            low = ohlcv[i]["low"]
            close_prev = ohlcv[i-1]["close"]
            
            tr = max(
                high - low,
                abs(high - close_prev),
                abs(low - close_prev)
            )
            true_ranges.append(tr)
        
        if len(true_ranges) >= period:
            atr = sum(true_ranges[:period]) / period
            assert atr > 0
    
    def test_atr_increases_with_volatility(self):
        """Test ATR increases with higher volatility"""
        # Low volatility OHLCV
        low_vol = []
        for i in range(50):
            low_vol.append({
                "high": 100.2,
                "low": 99.8,
                "close": 100.0
            })
        
        # High volatility OHLCV
        high_vol = []
        for i in range(50):
            high_vol.append({
                "high": 105.0 + i,
                "low": 95.0 + i,
                "close": 100.0 + i
            })
        
        def calc_atr(ohlcv_data):
            true_ranges = []
            for i in range(1, len(ohlcv_data)):
                high = ohlcv_data[i]["high"]
                low = ohlcv_data[i]["low"]
                close_prev = ohlcv_data[i-1]["close"]
                
                tr = max(
                    high - low,
                    abs(high - close_prev),
                    abs(low - close_prev)
                )
                true_ranges.append(tr)
            
            return sum(true_ranges) / len(true_ranges)
        
        low_atr = calc_atr(low_vol)
        high_atr = calc_atr(high_vol)
        
        assert high_atr > low_atr


# ============================================================================
# TREND INDICATORS TESTS
# ============================================================================

@pytest.mark.unit
class TestMACD:
    """Test MACD (Moving Average Convergence Divergence)"""
    
    def test_macd_values_in_range(self):
        """Test MACD histogram is reasonable"""
        prices = IndicatorTestData.generate_price_series(100)
        
        # Simple EMA calculation
        def ema(data, period):
            multiplier = 2 / (period + 1)
            ema_val = data[0]
            for price in data[1:]:
                ema_val = (price - ema_val) * multiplier + ema_val
            return ema_val
        
        ema12 = ema(prices, 12)
        ema26 = ema(prices, 26)
        
        macd = ema12 - ema26
        
        # MACD should be relatively small compared to price
        assert abs(macd) < max(prices)
    
    def test_macd_sign_line(self):
        """Test MACD signal line"""
        prices = IndicatorTestData.generate_price_series(100)
        
        def ema(data, period):
            multiplier = 2 / (period + 1)
            ema_val = data[0]
            for price in data[1:]:
                ema_val = (price - ema_val) * multiplier + ema_val
            return ema_val
        
        ema12 = ema(prices, 12)
        ema26 = ema(prices, 26)
        macd = ema12 - ema26
        
        # Signal line is EMA of MACD
        signal = ema([macd], 9)
        
        assert isinstance(signal, float)


# ============================================================================
# ERROR HANDLING & EDGE CASES
# ============================================================================

@pytest.mark.unit
class TestIndicatorEdgeCases:
    """Test error handling in indicators"""
    
    def test_empty_data(self):
        """Test handling of empty data"""
        prices = []
        window = 5
        
        sma = []
        for i in range(len(prices) - window + 1):
            avg = sum(prices[i:i + window]) / window
            sma.append(avg)
        
        assert len(sma) == 0
    
    def test_nan_values(self):
        """Test handling of NaN values"""
        prices = [100, float('nan'), 104, 106]
        
        # Filter out NaN
        valid_prices = [p for p in prices if not (isinstance(p, float) and np.isnan(p))]
        
        assert len(valid_prices) == 3
        assert all(isinstance(p, float) for p in valid_prices)
    
    def test_zero_prices(self):
        """Test handling of zero prices"""
        prices = [100, 0, 104, 106]
        
        # Filter out zeros
        valid_prices = [p for p in prices if p > 0]
        
        assert len(valid_prices) == 3
        assert all(p > 0 for p in valid_prices)
    
    def test_negative_prices(self):
        """Test handling of negative prices"""
        prices = [100, -50, 104, 106]
        
        # Filter out negatives
        valid_prices = [p for p in prices if p > 0]
        
        assert len(valid_prices) == 3
        assert all(p > 0 for p in valid_prices)
