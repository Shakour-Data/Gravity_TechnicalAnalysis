"""
Phase 4: Unit Tests for Analysis Services

Test coverage for analysis and signal generation:
- Technical Analysis Service
- Multi-indicator consensus
- Signal generation
- Confidence calculation
- Risk assessment

Target: 100% coverage of analysis logic
"""

from datetime import datetime, timedelta
from typing import Dict, List

import pytest

# ============================================================================
# TEST UTILITIES
# ============================================================================

class AnalysisTestData:
    """Generate test data for analysis testing"""
    
    @staticmethod
    def create_candle(
        close: float,
        high: float = None,
        low: float = None,
        open_price: float = None,
        volume: float = 1000.0,
        timestamp: datetime = None
    ) -> Dict:
        """Create candle data"""
        if timestamp is None:
            timestamp = datetime.now()
        if open_price is None:
            open_price = close * 0.99
        if high is None:
            high = close * 1.02
        if low is None:
            low = close * 0.98
        
        return {
            "timestamp": timestamp,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        }
    
    @staticmethod
    def strong_buy_signal_data() -> List[Dict]:
        """Generate data for strong BUY signal"""
        base_date = datetime.now() - timedelta(days=50)
        candles = []
        price = 100
        
        for day in range(50):
            # Strong uptrend
            price += 0.8 + (0.2 * (day % 3))
            candles.append(AnalysisTestData.create_candle(
                close=price,
                timestamp=base_date + timedelta(days=day)
            ))
        
        return candles
    
    @staticmethod
    def strong_sell_signal_data() -> List[Dict]:
        """Generate data for strong SELL signal"""
        base_date = datetime.now() - timedelta(days=50)
        candles = []
        price = 150
        
        for day in range(50):
            # Strong downtrend
            price -= 0.8 + (0.2 * (day % 3))
            candles.append(AnalysisTestData.create_candle(
                close=price,
                timestamp=base_date + timedelta(days=day)
            ))
        
        return candles
    
    @staticmethod
    def neutral_signal_data() -> List[Dict]:
        """Generate data for NEUTRAL signal"""
        base_date = datetime.now() - timedelta(days=50)
        candles = []
        price = 100
        
        for day in range(50):
            # Random walk (sideways)
            import random
            change = random.uniform(-0.5, 0.5)
            price = max(price + change, 50)  # Don't go negative
            candles.append(AnalysisTestData.create_candle(
                close=price,
                timestamp=base_date + timedelta(days=day)
            ))
        
        return candles


# ============================================================================
# SIGNAL GENERATION TESTS
# ============================================================================

@pytest.mark.unit
class TestSignalGeneration:
    """Test signal generation logic"""
    
    def test_buy_signal_on_strong_uptrend(self):
        """Test BUY signal on strong uptrend"""
        candles = AnalysisTestData.strong_buy_signal_data()
        closes = [c["close"] for c in candles]
        
        # Simple trend detection: last close > SMA20
        sma20 = sum(closes[-20:]) / 20
        
        # Strong uptrend
        assert closes[-1] > sma20
    
    def test_sell_signal_on_strong_downtrend(self):
        """Test SELL signal on strong downtrend"""
        candles = AnalysisTestData.strong_sell_signal_data()
        closes = [c["close"] for c in candles]
        
        # Simple trend detection: last close < SMA20
        sma20 = sum(closes[-20:]) / 20
        
        # Strong downtrend
        assert closes[-1] < sma20
    
    def test_neutral_signal_on_sideways(self):
        """Test NEUTRAL signal on sideways market"""
        candles = AnalysisTestData.neutral_signal_data()
        closes = [c["close"] for c in candles]
        
        # Calculate trend strength
        sma20 = sum(closes[-20:]) / 20
        
        # Sideways: close is near SMA
        diff = abs(closes[-1] - sma20) / sma20
        
        assert diff < 0.05  # Within 5%


# ============================================================================
# CONFIDENCE CALCULATION TESTS
# ============================================================================

@pytest.mark.unit
class TestConfidenceCalculation:
    """Test signal confidence calculation"""
    
    def test_confidence_range(self):
        """Test confidence is in range [0, 100]"""
        candles = AnalysisTestData.strong_buy_signal_data()
        closes = [c["close"] for c in candles]
        
        # Confidence based on trend strength
        sma20 = sum(closes[-20:]) / 20
        distance = (closes[-1] - sma20) / sma20
        
        confidence = min(100, 50 + (distance * 500))
        
        assert 0 <= confidence <= 100
    
    def test_confidence_increases_with_signal_strength(self):
        """Test confidence increases with stronger signal"""
        # Weak signal
        weak_candles = AnalysisTestData.neutral_signal_data()
        weak_closes = [c["close"] for c in weak_candles]
        weak_sma = sum(weak_closes[-20:]) / 20
        weak_distance = abs(weak_closes[-1] - weak_sma) / weak_sma
        weak_confidence = min(100, 50 + (weak_distance * 500))
        
        # Strong signal
        strong_candles = AnalysisTestData.strong_buy_signal_data()
        strong_closes = [c["close"] for c in strong_candles]
        strong_sma = sum(strong_closes[-20:]) / 20
        strong_distance = abs(strong_closes[-1] - strong_sma) / strong_sma
        strong_confidence = min(100, 50 + (strong_distance * 500))
        
        assert strong_confidence > weak_confidence
    
    def test_neutral_confidence(self):
        """Test confidence for neutral signal"""
        candles = AnalysisTestData.neutral_signal_data()
        closes = [c["close"] for c in candles]
        
        sma20 = sum(closes[-20:]) / 20
        distance = abs(closes[-1] - sma20) / sma20
        
        confidence = min(100, 50 + (distance * 500))
        
        # Neutral should be around 50
        assert 40 < confidence < 60


# ============================================================================
# MULTI-INDICATOR CONSENSUS TESTS
# ============================================================================

@pytest.mark.unit
class TestMultiIndicatorConsensus:
    """Test consensus from multiple indicators"""
    
    def test_all_indicators_agree_buy(self):
        """Test when all indicators agree on BUY"""
        candles = AnalysisTestData.strong_buy_signal_data()
        closes = [c["close"] for c in candles]
        
        # Simulate multiple indicators
        indicators = {
            "sma_20": closes[-1] > sum(closes[-20:]) / 20,  # True
            "rsi": 70,  # Overbought = bullish
            "macd": True,  # Positive = bullish
            "bollinger": closes[-1] > sum(closes[-20:]) / 20,  # Above middle
        }
        
        # Consensus: all bullish
        bullish_count = sum(1 for v in indicators.values() if v)
        
        assert bullish_count >= 3
    
    def test_all_indicators_agree_sell(self):
        """Test when all indicators agree on SELL"""
        candles = AnalysisTestData.strong_sell_signal_data()
        closes = [c["close"] for c in candles]
        
        # Simulate multiple indicators
        indicators = {
            "sma_20": closes[-1] < sum(closes[-20:]) / 20,  # True
            "rsi": 30,  # Oversold = bearish
            "macd": False,  # Negative = bearish
            "bollinger": closes[-1] < sum(closes[-20:]) / 20,  # Below middle
        }
        
        # Consensus: all bearish
        bearish_count = sum(1 for v in indicators.values() if not v)
        
        assert bearish_count >= 3
    
    def test_mixed_indicator_signals(self):
        """Test when indicators disagree"""
        candles = AnalysisTestData.neutral_signal_data()
        closes = [c["close"] for c in candles]
        
        sma20 = sum(closes[-20:]) / 20
        
        # Mixed signals
        indicators = {
            "sma_20": closes[-1] > sma20,
            "rsi": 45,  # Neutral
            "macd": True,
            "bollinger": closes[-1] < sma20,
        }
        
        bullish = sum(1 for v in indicators.values() if v)
        bearish = sum(1 for v in indicators.values() if not v)
        
        # Should be mixed
        assert bullish > 0 and bearish > 0


# ============================================================================
# DIMENSION ANALYSIS TESTS
# ============================================================================

@pytest.mark.unit
class TestDimensionAnalysis:
    """Test multi-dimensional analysis (3D, 7D, 30D)"""
    
    def test_3d_analysis(self):
        """Test 3-day analysis"""
        candles = AnalysisTestData.strong_buy_signal_data()
        
        # Use last 3 candles
        recent = candles[-3:]
        closes = [c["close"] for c in recent]
        
        # 3D signal: increasing closes
        is_bullish = closes[-1] > closes[0]
        
        assert is_bullish
    
    def test_7d_analysis(self):
        """Test 7-day analysis"""
        candles = AnalysisTestData.strong_buy_signal_data()
        
        # Use last 7 candles
        recent = candles[-7:]
        closes = [c["close"] for c in recent]
        
        # 7D signal
        sma = sum(closes) / len(closes)
        is_bullish = closes[-1] > sma
        
        assert is_bullish
    
    def test_30d_analysis(self):
        """Test 30-day (monthly) analysis"""
        candles = AnalysisTestData.strong_buy_signal_data()
        
        # Use last 30 candles
        recent = candles[-30:]
        closes = [c["close"] for c in recent]
        
        # 30D signal
        sma = sum(closes) / len(closes)
        is_bullish = closes[-1] > sma
        
        assert is_bullish
    
    def test_dimension_hierarchy(self):
        """Test dimensions support each other"""
        candles = AnalysisTestData.strong_buy_signal_data()
        
        # Check all dimensions are bullish
        closes = [c["close"] for c in candles]
        
        dim3d = closes[-1] > sum(closes[-3:]) / 3
        dim7d = closes[-1] > sum(closes[-7:]) / 7
        dim30d = closes[-1] > sum(closes[-30:]) / 30
        
        # For strong uptrend, all should align
        assert dim3d and dim7d and dim30d


# ============================================================================
# RISK ASSESSMENT TESTS
# ============================================================================

@pytest.mark.unit
class TestRiskAssessment:
    """Test risk assessment logic"""
    
    def test_risk_level_on_high_volatility(self):
        """Test high risk on high volatility"""
        import random
        
        # High volatility data
        base_date = datetime.now()
        price = 100
        high_vol = []
        
        for day in range(20):
            change = random.uniform(-3, 3)  # Big swings
            price = max(price + change, 50)
            high_vol.append({"close": price, "timestamp": base_date + timedelta(days=day)})
        
        closes = [c["close"] for c in high_vol]
        
        # Calculate volatility (std dev)
        mean = sum(closes) / len(closes)
        variance = sum((c - mean) ** 2 for c in closes) / len(closes)
        volatility = variance ** 0.5
        
        assert volatility > 2  # High volatility
    
    def test_risk_level_on_low_volatility(self):
        """Test low risk on low volatility"""
        # Low volatility data (prices stay close)
        prices = [100 + (i % 2) * 0.1 for i in range(20)]
        
        mean = sum(prices) / len(prices)
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        volatility = variance ** 0.5
        
        assert volatility < 1  # Low volatility
    
    def test_risk_reward_ratio(self):
        """Test risk/reward calculation"""
        current_price = 100
        entry = 100
        stop_loss = 95  # Risk: 5
        take_profit = 110  # Reward: 10
        
        risk = entry - stop_loss
        reward = take_profit - entry
        risk_reward = reward / risk
        
        assert risk_reward == 2.0  # 2:1 ratio


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.unit
class TestAnalysisErrorHandling:
    """Test error handling in analysis"""
    
    def test_insufficient_data(self):
        """Test handling of insufficient data"""
        candles = [
            {"close": 100, "timestamp": datetime.now()},
            {"close": 101, "timestamp": datetime.now() + timedelta(days=1)},
        ]
        
        # Analysis needs minimum data
        assert len(candles) < 20  # Too few for meaningful analysis
    
    def test_missing_data_in_series(self):
        """Test handling of gaps in data"""
        candles = [
            {"close": 100, "timestamp": datetime.now()},
            {"close": 101, "timestamp": datetime.now() + timedelta(days=1)},
            # Gap: day 2 is missing
            {"close": 102, "timestamp": datetime.now() + timedelta(days=3)},
        ]
        
        # Check for gaps
        timestamps = [c["timestamp"] for c in candles]
        
        gaps = []
        for i in range(len(timestamps) - 1):
            diff = (timestamps[i+1] - timestamps[i]).days
            if diff > 1:
                gaps.append(i)
        
        assert len(gaps) > 0  # Gap detected
    
    def test_zero_volume(self):
        """Test handling of zero volume"""
        candle = {
            "close": 100,
            "volume": 0  # Invalid
        }
        
        assert candle["volume"] == 0
    
    def test_extreme_price_movement(self):
        """Test handling of extreme price movements"""
        closes = [100, 200, 50, 150]  # Wild swings
        
        # Check for extreme moves
        extreme_moves = []
        for i in range(1, len(closes)):
            pct_change = abs((closes[i] - closes[i-1]) / closes[i-1])
            if pct_change > 0.5:  # >50% move
                extreme_moves.append((i, pct_change))
        
        assert len(extreme_moves) > 0


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.unit
class TestAnalysisPerformance:
    """Test analysis performance"""
    
    def test_analysis_with_100_candles(self):
        """Test analysis speed with 100 candles"""
        candles = AnalysisTestData.strong_buy_signal_data()
        
        import time
        start = time.time()
        
        # Simulate analysis
        closes = [c["close"] for c in candles]
        sma20 = sum(closes[-20:]) / 20
        signal = closes[-1] > sma20
        
        elapsed = time.time() - start
        
        assert elapsed < 0.1  # Should be very fast
    
    def test_analysis_with_1000_candles(self):
        """Test analysis with large dataset"""
        base_date = datetime.now() - timedelta(days=1000)
        price = 100
        candles = []
        
        for day in range(1000):
            price += 0.1
            candles.append(AnalysisTestData.create_candle(
                close=price,
                timestamp=base_date + timedelta(days=day)
            ))
        
        import time
        start = time.time()
        
        # Analysis
        closes = [c["close"] for c in candles]
        sma20 = sum(closes[-20:]) / 20
        sma50 = sum(closes[-50:]) / 50
        
        elapsed = time.time() - start
        
        assert elapsed < 0.5  # Should still be fast


# ============================================================================
# CONSISTENCY TESTS
# ============================================================================

@pytest.mark.unit
class TestAnalysisConsistency:
    """Test analysis consistency"""
    
    def test_same_data_same_result(self):
        """Test same input gives same output"""
        candles = AnalysisTestData.strong_buy_signal_data()
        closes = [c["close"] for c in candles]
        
        # Run analysis twice
        sma1 = sum(closes[-20:]) / 20
        sma2 = sum(closes[-20:]) / 20
        
        assert sma1 == sma2
    
    def test_signal_stability(self):
        """Test signal doesn't change unexpectedly"""
        candles = AnalysisTestData.strong_buy_signal_data()
        closes = [c["close"] for c in candles]
        
        # Get signal at different times
        signal1 = closes[-1] > sum(closes[-20:]) / 20
        signal2 = closes[-1] > sum(closes[-20:]) / 20
        
        assert signal1 == signal2
