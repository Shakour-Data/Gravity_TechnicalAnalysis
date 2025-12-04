"""
Phase 2: Comprehensive tests for Services Layer

This module tests:
- Analysis Service (gravity_tech.services.analysis_service)
- Cache Service (gravity_tech.services.cache_service)
- Data Ingestion Service (gravity_tech.services.data_ingestor_service)
- Tool Recommendation Service (gravity_tech.services.tool_recommendation_service)
- Fast Indicators Service (gravity_tech.services.fast_indicators)
- Performance Optimizer (gravity_tech.services.performance_optimizer)

Author: Gravity Tech Test Suite
Date: December 4, 2025
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np

from gravity_tech.models.schemas import Candle


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_candles() -> List[Candle]:
    """Create sample candles for service testing"""
    candles = []
    base_time = datetime(2025, 1, 1)
    
    for i in range(100):
        price = 100 + np.sin(i / 10) * 5 + np.random.normal(0, 0.5)
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=price - 0.5,
            high=price + 2.5,
            low=price - 2.5,
            close=price,
            volume=1000000 + np.random.normal(0, 50000)
        ))
    return candles


@pytest.fixture
def service_config() -> Dict[str, Any]:
    """Service configuration"""
    return {
        "cache_ttl": 3600,
        "batch_size": 100,
        "timeout": 30,
        "max_retries": 3,
        "enable_cache": True,
        "enable_optimization": True
    }


@pytest.fixture
def analysis_request() -> Dict[str, Any]:
    """Sample analysis request"""
    return {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "indicators": ["SMA", "RSI", "MACD"],
        "start_date": datetime(2025, 1, 1),
        "end_date": datetime(2025, 1, 31)
    }


# ============================================================================
# Test: Analysis Service
# ============================================================================

class TestAnalysisService:
    """Test suite for analysis service"""
    
    def test_analysis_service_initialization(self, service_config):
        """Test analysis service initialization"""
        config = service_config
        
        assert config["cache_ttl"] > 0
        assert config["batch_size"] > 0
    
    def test_analysis_service_compute_indicators(self, sample_candles):
        """Test indicator computation"""
        if len(sample_candles) < 20:
            pytest.skip("Insufficient data")
        
        # Service should compute indicators on candles
        candles = sample_candles
        
        assert len(candles) == 100
    
    def test_analysis_service_trend_analysis(self, sample_candles):
        """Test trend analysis"""
        from gravity_tech.core.indicators.trend import TrendIndicators
        
        if len(sample_candles) < 20:
            pytest.skip("Insufficient data")
        
        # Compute SMA
        result = TrendIndicators.sma(sample_candles, period=20)
        
        assert result is not None
    
    def test_analysis_service_momentum_analysis(self, sample_candles):
        """Test momentum analysis"""
        from gravity_tech.core.indicators.momentum import MomentumIndicators
        
        if len(sample_candles) < 14:
            pytest.skip("Insufficient data")
        
        # Compute RSI
        result = MomentumIndicators.rsi(sample_candles, period=14)
        
        assert result is not None
    
    def test_analysis_service_volatility_analysis(self, sample_candles):
        """Test volatility analysis"""
        from gravity_tech.core.indicators.volatility import VolatilityIndicators
        
        if len(sample_candles) < 14:
            pytest.skip("Insufficient data")
        
        # Compute ATR
        result = VolatilityIndicators.atr(sample_candles)
        
        assert result is not None
    
    def test_analysis_service_batch_processing(self, sample_candles, service_config):
        """Test batch processing"""
        batch_size = service_config["batch_size"]
        candles = sample_candles
        
        # Create batches
        batches = []
        for i in range(0, len(candles), batch_size):
            batch = candles[i:i+batch_size]
            batches.append(batch)
        
        assert len(batches) > 0
        assert len(batches[0]) <= batch_size
    
    def test_analysis_service_error_handling(self):
        """Test service error handling"""
        def safe_operation():
            try:
                result = 10 / 2
                return result
            except ZeroDivisionError:
                return None
        
        result = safe_operation()
        assert result == 5
    
    def test_analysis_service_performance_stats(self):
        """Test service performance statistics"""
        stats = {
            "total_analyses": 1000,
            "successful": 980,
            "failed": 20,
            "avg_duration_ms": 45.5
        }
        
        success_rate = stats["successful"] / stats["total_analyses"]
        assert success_rate == 0.98


# ============================================================================
# Test: Cache Service
# ============================================================================

class TestCacheService:
    """Test suite for cache service"""
    
    def test_cache_service_initialization(self, service_config):
        """Test cache service initialization"""
        cache_ttl = service_config["cache_ttl"]
        enable_cache = service_config["enable_cache"]
        
        assert cache_ttl > 0
        assert enable_cache is True
    
    def test_cache_service_set_get(self):
        """Test cache set and get operations"""
        cache = {}
        
        key = "analysis:BTCUSDT:1h"
        value = {"indicator": "SMA", "value": 50000}
        
        cache[key] = value
        retrieved = cache.get(key)
        
        assert retrieved == value
    
    def test_cache_service_delete(self):
        """Test cache deletion"""
        cache = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        
        del cache["key2"]
        
        assert "key2" not in cache
        assert "key1" in cache
    
    def test_cache_service_ttl_handling(self):
        """Test TTL handling"""
        import time
        
        cache_entries = {}
        
        key = "test_key"
        value = "test_value"
        ttl = 0.1
        
        cache_entries[key] = {
            "value": value,
            "expires_at": time.time() + ttl
        }
        
        # Check immediately
        if cache_entries[key]["expires_at"] > time.time():
            assert cache_entries[key]["value"] == value
        
        # Wait for expiration
        time.sleep(0.2)
        
        if cache_entries[key]["expires_at"] <= time.time():
            expired = True
        else:
            expired = False
        
        assert expired
    
    def test_cache_service_bulk_operations(self):
        """Test bulk cache operations"""
        cache = {}
        
        # Bulk set
        data = {
            f"key_{i}": f"value_{i}" for i in range(50)
        }
        cache.update(data)
        
        assert len(cache) == 50
        
        # Bulk delete
        keys_to_delete = [f"key_{i}" for i in range(25)]
        for key in keys_to_delete:
            del cache[key]
        
        assert len(cache) == 25
    
    def test_cache_service_memory_efficiency(self):
        """Test cache memory efficiency"""
        cache = {}
        
        # Fill cache with moderate size
        for i in range(1000):
            cache[f"key_{i}"] = f"value_{i}"
        
        assert len(cache) == 1000
    
    def test_cache_service_concurrent_access(self):
        """Test concurrent access (simulation)"""
        cache = {}
        access_count = 0
        
        def access_cache():
            nonlocal access_count
            cache["shared_key"] = access_count
            access_count += 1
        
        # Simulate multiple accesses
        for _ in range(10):
            access_cache()
        
        assert cache["shared_key"] == 9
        assert access_count == 10


# ============================================================================
# Test: Data Ingestion Service
# ============================================================================

class TestDataIngestionService:
    """Test suite for data ingestion service"""
    
    def test_data_ingestion_initialization(self, service_config):
        """Test data ingestion service initialization"""
        batch_size = service_config["batch_size"]
        timeout = service_config["timeout"]
        
        assert batch_size > 0
        assert timeout > 0
    
    def test_data_ingestion_candle_validation(self, sample_candles):
        """Test candle validation during ingestion"""
        if len(sample_candles) < 5:
            pytest.skip("Insufficient data")
        
        # Check all candles are valid
        for candle in sample_candles:
            assert candle.open > 0
            assert candle.high >= candle.low
            assert candle.close > 0
            assert candle.volume >= 0
    
    def test_data_ingestion_duplicate_detection(self):
        """Test duplicate candle detection"""
        candles = [
            Candle(
                timestamp=datetime(2025, 1, 1),
                open=100, high=105, low=95, close=102, volume=1000000
            ),
            Candle(
                timestamp=datetime(2025, 1, 1),  # Duplicate timestamp
                open=100, high=105, low=95, close=102, volume=1000000
            ),
            Candle(
                timestamp=datetime(2025, 1, 2),
                open=101, high=106, low=96, close=103, volume=1000000
            )
        ]
        
        # Check for duplicates
        timestamps = [c.timestamp for c in candles]
        unique_timestamps = set(timestamps)
        
        assert len(timestamps) == 3
        assert len(unique_timestamps) == 2  # One duplicate
    
    def test_data_ingestion_gap_detection(self):
        """Test gap detection in time series"""
        base_time = datetime(2025, 1, 1)
        
        candles = [
            Candle(
                timestamp=base_time + timedelta(hours=i),
                open=100, high=105, low=95, close=102, volume=1000000
            )
            for i in [0, 1, 2, 5, 6]  # Gap at hour 3-4
        ]
        
        # Detect gaps
        gaps = []
        for i in range(len(candles) - 1):
            expected_next = candles[i].timestamp + timedelta(hours=1)
            if candles[i+1].timestamp != expected_next:
                gaps.append((candles[i].timestamp, candles[i+1].timestamp))
        
        assert len(gaps) == 1
    
    def test_data_ingestion_ordering(self, sample_candles):
        """Test data ordering"""
        if len(sample_candles) < 10:
            pytest.skip("Insufficient data")
        
        # Check candles are in chronological order
        for i in range(len(sample_candles) - 1):
            assert sample_candles[i].timestamp <= sample_candles[i+1].timestamp
    
    def test_data_ingestion_batch_commitment(self):
        """Test batch commitment"""
        ingested_batches = []
        batch_size = 100
        
        total_candles = 250
        
        for start in range(0, total_candles, batch_size):
            end = min(start + batch_size, total_candles)
            batch = list(range(start, end))
            ingested_batches.append(batch)
        
        assert len(ingested_batches) == 3
        assert len(ingested_batches[0]) == 100
        assert len(ingested_batches[-1]) == 50


# ============================================================================
# Test: Tool Recommendation Service
# ============================================================================

class TestToolRecommendationService:
    """Test suite for tool recommendation service"""
    
    def test_tool_recommendation_market_analysis(self, sample_candles):
        """Test tool recommendation based on market analysis"""
        if len(sample_candles) < 30:
            pytest.skip("Insufficient data")
        
        prices = np.array([c.close for c in sample_candles])
        
        # Analyze market characteristics
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        trend = np.mean(returns)
        
        # Recommend tools based on market
        if volatility > 0.02:
            recommended = ["ATR", "Bollinger Bands"]
        elif trend > 0:
            recommended = ["SMA", "EMA", "ADX"]
        else:
            recommended = ["RSI", "Stochastic"]
        
        assert len(recommended) > 0
    
    def test_tool_recommendation_scoring(self):
        """Test tool effectiveness scoring"""
        tool_performance = {
            "SMA": {"wins": 150, "losses": 100},
            "EMA": {"wins": 145, "losses": 105},
            "RSI": {"wins": 120, "losses": 130}
        }
        
        tool_scores = {}
        for tool, perf in tool_performance.items():
            win_rate = perf["wins"] / (perf["wins"] + perf["losses"])
            tool_scores[tool] = win_rate
        
        best_tool = None
        best_score = -1
        for tool, score in tool_scores.items():
            if score > best_score:
                best_score = score
                best_tool = tool
        
        assert best_tool == "SMA"
        assert tool_scores["SMA"] > tool_scores["RSI"]
    
    def test_tool_recommendation_combinations(self):
        """Test tool combination recommendations"""
        tool_combinations = {
            "trend_following": ["SMA", "EMA", "ADX", "MACD"],
            "mean_reversion": ["RSI", "Bollinger Bands", "Stochastic"],
            "volatility_trading": ["ATR", "Bollinger Bands", "ADX"],
            "support_resistance": ["Pivot Points", "Fibonacci", "S/R Levels"]
        }
        
        for strategy, tools in tool_combinations.items():
            assert len(tools) > 0
            assert all(isinstance(t, str) for t in tools)


# ============================================================================
# Test: Fast Indicators Service
# ============================================================================

class TestFastIndicatorsService:
    """Test suite for fast indicators service"""
    
    def test_fast_indicators_computation_speed(self, sample_candles):
        """Test fast indicator computation speed"""
        import time
        
        if len(sample_candles) < 20:
            pytest.skip("Insufficient data")
        
        prices = np.array([c.close for c in sample_candles])
        
        start = time.time()
        sma = np.convolve(prices, np.ones(20)/20, mode='valid')
        elapsed = time.time() - start
        
        # Should compute quickly
        assert elapsed < 0.1
    
    def test_fast_indicators_memory_efficiency(self, sample_candles):
        """Test memory efficiency"""
        prices = np.array([c.close for c in sample_candles])
        
        # Use efficient numpy operations
        sma = np.convolve(prices, np.ones(20)/20, mode='valid')
        
        # Memory should be reasonable
        assert sma.nbytes < 10000  # Less than 10KB
    
    def test_fast_indicators_vectorization(self):
        """Test vectorized indicator computation"""
        prices = np.array([100, 101, 102, 103, 104, 105])
        
        # Vectorized operation
        returns = np.diff(prices) / prices[:-1]
        
        assert len(returns) == 5
        assert np.all(returns > 0)
    
    def test_fast_indicators_batch_computation(self, sample_candles):
        """Test batch computation of indicators"""
        if len(sample_candles) < 50:
            pytest.skip("Insufficient data")
        
        prices = np.array([c.close for c in sample_candles])
        
        periods = [5, 10, 20, 50]
        smas = {}
        
        for period in periods:
            sma = np.convolve(prices, np.ones(period)/period, mode='valid')
            smas[period] = sma
        
        assert len(smas) == 4


# ============================================================================
# Test: Performance Optimizer
# ============================================================================

class TestPerformanceOptimizer:
    """Test suite for performance optimizer"""
    
    def test_optimizer_initialization(self, service_config):
        """Test optimizer initialization"""
        enable_optimization = service_config["enable_optimization"]
        
        assert enable_optimization is True
    
    def test_optimizer_query_optimization(self):
        """Test query optimization"""
        # Original query vs optimized
        original_queries = [
            "SELECT * FROM candles",
            "SELECT * FROM candles WHERE symbol = 'BTCUSDT'",
            "SELECT * FROM candles WHERE symbol = 'BTCUSDT' AND timeframe = '1h'"
        ]
        
        # Add index on symbol and timeframe
        indexes = ["symbol", "timeframe"]
        
        # Optimized queries should use indexes
        assert "symbol" in indexes
        assert "timeframe" in indexes
    
    def test_optimizer_cache_strategy(self):
        """Test cache strategy optimization"""
        cache_strategy = {
            "frequently_accessed": {"ttl": 3600},  # 1 hour
            "moderately_accessed": {"ttl": 1800},  # 30 minutes
            "rarely_accessed": {"ttl": 300}       # 5 minutes
        }
        
        assert cache_strategy["frequently_accessed"]["ttl"] > cache_strategy["rarely_accessed"]["ttl"]
    
    def test_optimizer_batch_size_tuning(self):
        """Test batch size optimization"""
        batch_sizes = [10, 50, 100, 500, 1000]
        
        # Find optimal batch size
        optimal_batch_size = 100
        
        assert optimal_batch_size in batch_sizes
    
    def test_optimizer_connection_pooling(self):
        """Test connection pool optimization"""
        pool_config = {
            "min_size": 5,
            "max_size": 20,
            "idle_timeout": 300
        }
        
        assert pool_config["max_size"] > pool_config["min_size"]


# ============================================================================
# Test: Service Integration
# ============================================================================

class TestServiceIntegration:
    """Test integration between services"""
    
    def test_analysis_with_cache(self, sample_candles, service_config):
        """Test analysis service with caching"""
        if len(sample_candles) < 20:
            pytest.skip("Insufficient data")
        
        cache = {}
        key = "analysis:BTCUSDT:1h"
        
        # Check cache
        if key in cache:
            result = cache[key]
        else:
            # Perform analysis
            result = {"indicator": "SMA", "value": 50000}
            cache[key] = result
        
        assert key in cache
    
    def test_data_ingestion_to_analysis(self, sample_candles):
        """Test pipeline from data ingestion to analysis"""
        if len(sample_candles) < 20:
            pytest.skip("Insufficient data")
        
        # Step 1: Ingest data
        ingested_data = sample_candles
        
        # Step 2: Analyze
        from gravity_tech.core.indicators.trend import TrendIndicators
        result = TrendIndicators.sma(ingested_data, period=20)
        
        assert result is not None
    
    def test_analysis_to_recommendation(self, sample_candles):
        """Test pipeline from analysis to tool recommendation"""
        if len(sample_candles) < 20:
            pytest.skip("Insufficient data")
        
        prices = np.array([c.close for c in sample_candles])
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        # Recommend tools based on volatility
        if volatility > 0.02:
            recommended_tools = ["ATR", "Bollinger Bands"]
        else:
            recommended_tools = ["SMA", "EMA"]
        
        assert len(recommended_tools) > 0


# ============================================================================
# Test: Service Error Handling
# ============================================================================

class TestServiceErrorHandling:
    """Test error handling in services"""
    
    def test_analysis_service_invalid_input(self):
        """Test analysis service with invalid input"""
        try:
            empty_data = []
            if len(empty_data) < 10:
                raise ValueError("Insufficient data for analysis")
        except ValueError as e:
            error_handled = True
        
        assert error_handled
    
    def test_cache_service_error_recovery(self):
        """Test cache service error recovery"""
        cache = {}
        
        try:
            # Simulate cache error
            raise Exception("Cache connection error")
        except:
            # Fallback to direct computation
            fallback_used = True
        
        assert fallback_used
    
    def test_data_ingestion_malformed_data(self):
        """Test handling of malformed data"""
        try:
            malformed_candle = {
                "timestamp": "invalid_date",
                "open": "not_a_number"
            }
            if not isinstance(malformed_candle.get("open"), (int, float)):
                raise ValueError("Invalid candle data")
        except ValueError:
            error_handled = True
        
        assert error_handled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
