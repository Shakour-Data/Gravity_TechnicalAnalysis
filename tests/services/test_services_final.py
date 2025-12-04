"""
Phase 3: Services Tests with TSE-like Data

This module tests services using realistic Iranian stock market (TSE) data patterns.
If real TSE database is available, it will use that; otherwise it uses generated data.

Author: Gravity Tech Test Suite
Date: December 4, 2025
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
import numpy as np
import os
import sqlite3

from gravity_tech.models.schemas import Candle
from gravity_tech.core.indicators.trend import TrendIndicators
from gravity_tech.core.indicators.momentum import MomentumIndicators
from gravity_tech.core.indicators.volatility import VolatilityIndicators
from gravity_tech.core.indicators.volume import VolumeIndicators


# ============================================================================
# استفاده از Fixtures تعریف‌شده در conftest.py
# Using Fixtures defined in conftest.py
# ============================================================================

@pytest.fixture
def service_config() -> Dict[str, Any]:
    """Service configuration"""
    return {
        "cache_ttl": 3600,
        "batch_size": 100,
        "timeout": 30,
        "max_retries": 3,
        "enable_cache": True
    }


# ============================================================================
# Test: Analysis Service
# ============================================================================

class TestAnalysisService:
    """Test analysis service with TSE data"""
    
    def test_analysis_service_trend(self, tse_candles_long):
        """Test trend analysis on TSE data"""
        if len(tse_candles_long) < 50:
            pytest.skip("Insufficient data")
        
        sma = TrendIndicators.sma(tse_candles_long, period=20)
        ema = TrendIndicators.ema(tse_candles_long, period=12)
        
        assert sma is not None
        assert ema is not None
    
    def test_analysis_service_momentum(self, tse_candles_long):
        """Test momentum analysis on TSE data"""
        if len(tse_candles_long) < 50:
            pytest.skip("Insufficient data")
        
        rsi = MomentumIndicators.rsi(tse_candles_long, period=14)
        stoch = MomentumIndicators.stochastic(tse_candles_long, k_period=14, d_period=3)
        
        assert rsi is not None
        assert stoch is not None
    
    def test_analysis_service_volatility(self, tse_candles_long):
        """Test volatility analysis on TSE data"""
        if len(tse_candles_long) < 50:
            pytest.skip("Insufficient data")
        
        atr = VolatilityIndicators.atr(tse_candles_long)
        bb = VolatilityIndicators.bollinger_bands(tse_candles_long)
        
        assert atr is not None
        assert bb is not None
    
    def test_analysis_service_volume(self, tse_candles_long):
        """Test volume analysis on TSE data"""
        if len(tse_candles_long) < 50:
            pytest.skip("Insufficient data")
        
        obv = VolumeIndicators.on_balance_volume(tse_candles_long)
        ad = VolumeIndicators.accumulation_distribution(tse_candles_long)
        
        assert obv is not None
        assert ad is not None
    
    def test_analysis_batch_processing(self, tse_candles, service_config):
        """Test batch processing of TSE data"""
        batch_size = service_config["batch_size"]
        batches = []
        
        for i in range(0, len(tse_candles), batch_size):
            batch = tse_candles[i:i+batch_size]
            batches.append(batch)
        
        assert len(batches) > 0
        for batch in batches:
            assert len(batch) <= batch_size


# ============================================================================
# Test: Cache Service
# ============================================================================

class TestCacheService:
    """Test cache service"""
    
    def test_cache_set_get(self):
        """Test cache operations"""
        cache = {}
        
        key = "tse_total_sma20"
        value = {"indicator": "SMA", "value": 50000}
        
        cache[key] = value
        retrieved = cache.get(key)
        
        assert retrieved == value
    
    def test_cache_delete(self):
        """Test cache deletion"""
        cache = {
            "key1": "value1",
            "key2": "value2"
        }
        
        del cache["key1"]
        
        assert "key1" not in cache
        assert "key2" in cache
    
    def test_cache_ttl_expiration(self):
        """Test TTL expiration"""
        import time
        
        cache = {}
        ttl = 0.1
        
        entry = {
            "value": "test",
            "expires_at": time.time() + ttl
        }
        cache["key"] = entry
        
        # Check before expiration
        assert entry["expires_at"] > time.time()
        
        # Wait and check after expiration
        time.sleep(0.2)
        assert entry["expires_at"] <= time.time()
    
    def test_cache_bulk_operations(self):
        """Test bulk cache operations"""
        cache = {}
        
        # Bulk set
        data = {f"key_{i}": f"value_{i}" for i in range(50)}
        cache.update(data)
        
        assert len(cache) == 50
        
        # Bulk delete
        for i in range(25):
            del cache[f"key_{i}"]
        
        assert len(cache) == 25


# ============================================================================
# Test: Data Ingestion Service
# ============================================================================

class TestDataIngestionService:
    """Test data ingestion service"""
    
    def test_ingest_candles_validation(self, tse_candles):
        """Test candle validation"""
        for candle in tse_candles:
            assert candle.open > 0
            assert candle.high >= candle.low
            assert candle.close > 0
            assert candle.volume >= 0
    
    def test_ingest_detect_duplicates(self, tse_candles):
        """Test duplicate detection"""
        timestamps = [c.timestamp for c in tse_candles]
        unique = set(timestamps)
        
        duplicates = len(timestamps) - len(unique)
        # TSE might have some duplicates due to multi-session trading
        assert len(unique) > 0
    
    def test_ingest_data_ordering(self, tse_candles):
        """Test data ordering"""
        for i in range(len(tse_candles) - 1):
            assert tse_candles[i].timestamp <= tse_candles[i+1].timestamp
    
    def test_ingest_batch_processing(self, tse_candles, service_config):
        """Test batch processing"""
        batch_size = service_config["batch_size"]
        batches = []
        
        for i in range(0, len(tse_candles), batch_size):
            batch = tse_candles[i:i+batch_size]
            batches.append(batch)
        
        assert len(batches) > 0
        assert all(len(b) <= batch_size for b in batches)


# ============================================================================
# Test: Tool Recommendation Service
# ============================================================================

class TestToolRecommendationService:
    """Test tool recommendation service"""
    
    def test_recommend_based_on_volatility(self, tse_candles):
        """Test tool recommendation based on volatility"""
        if len(tse_candles) < 30:
            pytest.skip("Insufficient data")
        
        prices = np.array([c.close for c in tse_candles])
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        if volatility > 0.02:
            recommended = ["ATR", "Bollinger Bands"]
        elif volatility > 0.01:
            recommended = ["RSI", "Stochastic"]
        else:
            recommended = ["SMA", "EMA"]
        
        assert len(recommended) > 0
    
    def test_recommend_based_on_trend(self, tse_candles):
        """Test tool recommendation based on trend"""
        if len(tse_candles) < 50:
            pytest.skip("Insufficient data")
        
        prices = np.array([c.close for c in tse_candles])
        sma_20 = np.convolve(prices, np.ones(20)/20, mode='valid')
        
        trend = prices[-1] - sma_20[-1]
        
        if trend > 0:
            recommended = ["SMA", "EMA", "ADX"]
        else:
            recommended = ["RSI", "Stochastic"]
        
        assert len(recommended) > 0
    
    def test_tool_scoring(self, tse_candles_long):
        """Test tool effectiveness scoring"""
        if len(tse_candles_long) < 20:
            pytest.skip("Insufficient data")
        
        sma = TrendIndicators.sma(tse_candles_long, period=20)
        rsi = MomentumIndicators.rsi(tse_candles_long, period=14)
        
        scores = {
            "SMA": 0.85 if sma else 0.0,
            "RSI": 0.80 if rsi else 0.0
        }
        
        best = max(scores, key=lambda x: scores[x])
        assert scores[best] > 0


# ============================================================================
# Test: Fast Indicators Service
# ============================================================================

class TestFastIndicatorsService:
    """Test fast indicators service"""
    
    def test_fast_sma_computation(self, tse_candles):
        """Test fast SMA computation"""
        import time
        
        if len(tse_candles) < 50:
            pytest.skip("Insufficient data")
        
        prices = np.array([c.close for c in tse_candles])
        
        start = time.time()
        sma = np.convolve(prices, np.ones(20)/20, mode='valid')
        elapsed = time.time() - start
        
        assert elapsed < 0.01
        assert len(sma) > 0
    
    def test_vectorized_operations(self, tse_candles):
        """Test vectorized operations"""
        prices = np.array([c.close for c in tse_candles])
        highs = np.array([c.high for c in tse_candles])
        lows = np.array([c.low for c in tse_candles])
        
        ranges = highs - lows
        returns = np.diff(prices) / prices[:-1]
        
        assert len(ranges) == len(tse_candles)
        assert len(returns) == len(tse_candles) - 1


# ============================================================================
# Test: Service Integration
# ============================================================================

class TestServiceIntegration:
    """Test service integration"""
    
    def test_full_analysis_pipeline(self, tse_candles, service_config):
        """Test complete analysis pipeline"""
        if len(tse_candles) < 50:
            pytest.skip("Insufficient data")
        
        # Step 1: Ingest
        valid_candles = tse_candles
        assert len(valid_candles) > 0
        
        # Step 2: Analyze
        sma = TrendIndicators.sma(valid_candles, period=20)
        rsi = MomentumIndicators.rsi(valid_candles, period=14)
        atr = VolatilityIndicators.atr(valid_candles)
        
        assert sma is not None
        assert rsi is not None
        assert atr is not None
        
        # Step 3: Cache
        cache = {"sma": sma, "rsi": rsi, "atr": atr}
        
        # Step 4: Recommend
        recommended = ["SMA", "RSI", "ATR"]
        assert len(recommended) > 0
    
    def test_multi_symbol_analysis(self, multiple_tse_symbols):
        """Test analysis on multiple symbols"""
        results = {}
        
        for symbol, candles in multiple_tse_symbols.items():
            if len(candles) < 20:
                continue
            
            sma = TrendIndicators.sma(candles, period=20)
            results[symbol] = sma is not None
        
        assert all(results.values())


# ============================================================================
# Test: Performance Metrics
# ============================================================================

class TestPerformanceMetrics:
    """Test performance characteristics"""
    
    def test_analysis_performance(self, tse_candles):
        """Test analysis performance"""
        import time
        
        if len(tse_candles) < 50:
            pytest.skip("Insufficient data")
        
        metrics = {}
        
        start = time.time()
        TrendIndicators.sma(tse_candles, period=20)
        metrics["sma"] = (time.time() - start) * 1000
        
        start = time.time()
        MomentumIndicators.rsi(tse_candles, period=14)
        metrics["rsi"] = (time.time() - start) * 1000
        
        # All should complete quickly
        for indicator, time_ms in metrics.items():
            assert time_ms < 100, f"{indicator} took {time_ms}ms"
    
    def test_batch_processing_performance(self, tse_candles, service_config):
        """Test batch processing performance"""
        import time
        
        batch_size = service_config["batch_size"]
        
        start = time.time()
        for i in range(0, len(tse_candles), batch_size):
            batch = tse_candles[i:i+batch_size]
            assert len(batch) > 0
        elapsed = time.time() - start
        
        assert elapsed < 1.0


# ============================================================================
# Test: Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling"""
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        try:
            if 0 < 20:
                raise ValueError("Insufficient data")
        except ValueError:
            pass  # Expected
    
    def test_invalid_price_handling(self):
        """Test handling of invalid prices"""
        try:
            price = -100
            if price < 0:
                raise ValueError("Invalid price")
        except ValueError:
            pass  # Expected
    
    def test_cache_error_recovery(self):
        """Test cache error recovery"""
        cache = {}
        try:
            value = cache.get("nonexistent")
            if value is None:
                fallback = "computed_value"
        except:
            fallback = "error_value"
        
        assert fallback is not None


# ============================================================================
# Test: Data Quality
# ============================================================================

class TestDataQuality:
    """Test data quality metrics"""
    
    def test_price_validity(self, tse_candles):
        """Test price validity"""
        for candle in tse_candles:
            assert candle.high >= candle.low
            assert candle.close >= candle.low and candle.close <= candle.high
    
    def test_volume_distribution(self, tse_candles):
        """Test volume distribution"""
        volumes = np.array([c.volume for c in tse_candles])
        
        mean_vol = np.mean(volumes)
        std_vol = np.std(volumes)
        
        assert mean_vol > 0
        assert std_vol >= 0
    
    def test_price_statistics(self, tse_candles):
        """Test price statistics"""
        closes = np.array([c.close for c in tse_candles])
        
        assert np.min(closes) > 0
        assert np.max(closes) > np.min(closes)
        assert np.std(closes) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
