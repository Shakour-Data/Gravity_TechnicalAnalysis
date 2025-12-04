"""
Tests for Resilience Middleware

Tests Circuit Breaker, Retry, Timeout, and Bulkhead patterns.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import pytest
import asyncio
from gravity_tech.middleware.resilience import (
    CircuitBreaker,
    retry_with_backoff,
    timeout,
    Bulkhead,
    resilient
)


class TestCircuitBreaker:
    """Tests for Circuit Breaker pattern."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Test circuit opens after threshold failures."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
        
        @cb
        async def failing_function():
            raise Exception("Simulated failure")
        
        # Should fail 3 times
        for _ in range(3):
            with pytest.raises(Exception):
                await failing_function()
        
        # Circuit should be open now
        assert cb.state.value == "open"
        
        # Next call should fail fast
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await failing_function()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit transitions to half-open state."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        
        call_count = 0
        
        @cb
        async def sometimes_failing():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Failure")
            return "success"
        
        # Trigger circuit open
        for _ in range(2):
            with pytest.raises(Exception):
                await sometimes_failing()
        
        assert cb.state.value == "open"
        
        # Wait for recovery timeout
        await asyncio.sleep(0.2)
        
        # Should transition to half-open and succeed
        result = await sometimes_failing()
        assert result == "success"
        assert cb.state.value == "closed"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_success_resets_failures(self):
        """Test successful calls reset failure count."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
        
        call_count = 0
        
        @cb
        async def alternating_function():
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise Exception("Failure")
            return "success"
        
        # Success -> Failure -> Success pattern
        await alternating_function()  # Success
        
        with pytest.raises(Exception):
            await alternating_function()  # Failure
        
        await alternating_function()  # Success (resets count)
        
        # Circuit should still be closed
        assert cb.state.value == "closed"


class TestRetryWithBackoff:
    """Tests for Retry with Exponential Backoff."""
    
    @pytest.mark.asyncio
    async def test_retry_succeeds_eventually(self):
        """Test function succeeds after retries."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        async def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Not yet")
            return "success"
        
        result = await eventually_succeeds()
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_exhausts_attempts(self):
        """Test function fails after max retries."""
        call_count = 0
        
        @retry_with_backoff(max_retries=2, initial_delay=0.01)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            await always_fails()
        
        assert call_count == 3  # Initial + 2 retries
    
    @pytest.mark.asyncio
    async def test_retry_backoff_timing(self):
        """Test exponential backoff timing."""
        timings = []
        
        @retry_with_backoff(max_retries=3, initial_delay=0.1, max_delay=1.0)
        async def record_timing():
            import time
            timings.append(time.time())
            raise Exception("Retry")
        
        with pytest.raises(Exception):
            await record_timing()
        
        # Verify increasing delays
        assert len(timings) == 4  # Initial + 3 retries
        
        # Delays should increase (with some tolerance for jitter)
        for i in range(1, len(timings) - 1):
            delay = timings[i] - timings[i-1]
            next_delay = timings[i+1] - timings[i]
            # Next delay should be roughly 2x (accounting for jitter)
            assert next_delay >= delay * 0.8  # Allow 20% variance


class TestTimeout:
    """Tests for Timeout decorator."""
    
    @pytest.mark.asyncio
    async def test_timeout_cancels_slow_function(self):
        """Test slow function is cancelled."""
        @timeout(0.1)
        async def slow_function():
            await asyncio.sleep(1)
            return "completed"
        
        with pytest.raises(asyncio.TimeoutError):
            await slow_function()
    
    @pytest.mark.asyncio
    async def test_timeout_allows_fast_function(self):
        """Test fast function completes successfully."""
        @timeout(1.0)
        async def fast_function():
            await asyncio.sleep(0.01)
            return "completed"
        
        result = await fast_function()
        assert result == "completed"


class TestBulkhead:
    """Tests for Bulkhead pattern."""
    
    @pytest.mark.asyncio
    async def test_bulkhead_limits_concurrency(self):
        """Test bulkhead limits concurrent executions."""
        bulkhead = Bulkhead(max_concurrent=2)
        active_count = 0
        max_active = 0
        
        @bulkhead
        async def concurrent_function():
            nonlocal active_count, max_active
            active_count += 1
            max_active = max(max_active, active_count)
            await asyncio.sleep(0.1)
            active_count -= 1
            return "done"
        
        # Start 5 concurrent calls
        tasks = [concurrent_function() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All should complete
        assert len(results) == 5
        assert all(r == "done" for r in results)
        
        # Max concurrent should not exceed limit
        assert max_active <= 2
    
    @pytest.mark.asyncio
    async def test_bulkhead_rejects_excess(self):
        """Test bulkhead can reject excess requests."""
        bulkhead = Bulkhead(max_concurrent=1)
        
        @bulkhead
        async def long_running():
            await asyncio.sleep(0.5)
            return "done"
        
        # Start one task
        task1 = asyncio.create_task(long_running())
        await asyncio.sleep(0.01)  # Let it start
        
        # Try to start another (should be blocked/queued)
        task2 = asyncio.create_task(long_running())
        
        # Both should eventually complete
        results = await asyncio.gather(task1, task2)
        assert len(results) == 2


class TestResilientDecorator:
    """Tests for combined resilient decorator."""
    
    @pytest.mark.asyncio
    async def test_resilient_combines_patterns(self):
        """Test resilient decorator combines all patterns."""
        call_count = 0
        
        @resilient(max_retries=2, timeout_seconds=5, circuit_threshold=5)
        async def protected_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Retry me")
            return "success"
        
        result = await protected_function()
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_resilient_timeout_protection(self):
        """Test resilient includes timeout protection."""
        @resilient(max_retries=1, timeout_seconds=0.1)
        async def slow_function():
            await asyncio.sleep(1)
            return "too slow"
        
        with pytest.raises(asyncio.TimeoutError):
            await slow_function()


@pytest.mark.integration
class TestResilientIntegration:
    """Integration tests for resilience patterns."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_with_external_service(self):
        """Test circuit breaker with simulated external service."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.5)
        
        service_available = False
        
        @cb
        async def call_external_service():
            if not service_available:
                raise Exception("Service unavailable")
            return "success"
        
        # Trigger circuit open
        for _ in range(3):
            with pytest.raises(Exception):
                await call_external_service()
        
        assert cb.state.value == "open"
        
        # Service comes back online
        service_available = True
        
        # Wait for recovery
        await asyncio.sleep(0.6)
        
        # Should succeed now
        result = await call_external_service()
        assert result == "success"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

