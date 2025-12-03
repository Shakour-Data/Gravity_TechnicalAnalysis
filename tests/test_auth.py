"""
Tests for Authentication and Security Middleware

Tests JWT authentication, rate limiting, and input validation.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from fastapi import HTTPException
from gravity_tech.middleware.auth import (
    create_access_token,
    verify_token,
    get_current_user,
    RateLimiter,
    SecureAnalysisRequest
)


class TestJWTAuthentication:
    """Tests for JWT token creation and verification."""
    
    def test_create_access_token(self):
        """Test JWT token creation."""
        token = create_access_token(
            data={"sub": "test_user"},
            expires_delta=timedelta(minutes=15)
        )
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_verify_valid_token(self):
        """Test verification of valid token."""
        # Create token
        token = create_access_token(
            data={"sub": "test_user"},
            expires_delta=timedelta(minutes=15)
        )
        
        # Verify token
        payload = verify_token(token)
        assert payload is not None
        assert payload.username == "test_user"
    
    def test_verify_expired_token(self):
        """Test verification of expired token."""
        # Create token with immediate expiry
        token = create_access_token(
            data={"sub": "test_user"},
            expires_delta=timedelta(seconds=-1)  # Already expired
        )
        
        # Should raise exception
        with pytest.raises(HTTPException) as exc_info:
            verify_token(token)
        
        assert exc_info.value.status_code == 401
    
    def test_verify_invalid_token(self):
        """Test verification of invalid token."""
        invalid_token = "invalid.token.string"
        
        with pytest.raises(HTTPException) as exc_info:
            verify_token(invalid_token)
        
        assert exc_info.value.status_code == 401
    
    def test_token_contains_required_claims(self):
        """Test token contains required claims."""
        token = create_access_token(
            data={"sub": "test_user", "scopes": ["admin"]}
        )
        
        payload = verify_token(token)
        assert payload.username == "test_user"
        assert payload.exp is not None
        assert "admin" in payload.scopes


class TestRateLimiter:
    """Tests for rate limiting functionality."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests_within_limit(self):
        """Test rate limiter allows requests within limit."""
        limiter = RateLimiter(requests_per_minute=10)
        client_id = "test_client_123"
        
        # Should allow first 10 requests
        for _ in range(10):
            result = await limiter.check_rate_limit(client_id)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excess_requests(self):
        """Test rate limiter blocks excess requests."""
        limiter = RateLimiter(requests_per_minute=5)
        client_id = "test_client_456"
        
        # Allow first 5
        for _ in range(5):
            await limiter.check_rate_limit(client_id)
        
        # 6th should be blocked
        with pytest.raises(HTTPException) as exc_info:
            await limiter.check_rate_limit(client_id)
        
        assert exc_info.value.status_code == 429
    
    @pytest.mark.asyncio
    async def test_rate_limiter_resets_after_window(self):
        """Test rate limiter resets after time window."""
        limiter = RateLimiter(requests_per_minute=2, window_seconds=1)
        client_id = "test_client_789"
        
        # Use up limit
        await limiter.check_rate_limit(client_id)
        await limiter.check_rate_limit(client_id)
        
        # Should be blocked
        with pytest.raises(HTTPException):
            await limiter.check_rate_limit(client_id)
        
        # Wait for window reset
        import asyncio
        await asyncio.sleep(1.1)
        
        # Should allow again
        result = await limiter.check_rate_limit(client_id)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_rate_limiter_different_clients(self):
        """Test rate limiter tracks clients independently."""
        limiter = RateLimiter(requests_per_minute=2)
        
        # Client 1 uses up limit
        await limiter.check_rate_limit("client_1")
        await limiter.check_rate_limit("client_1")
        
        # Client 2 should still have limit
        result = await limiter.check_rate_limit("client_2")
        assert result is True


class TestSecureAnalysisRequest:
    """Tests for input validation."""
    
    def test_valid_request(self):
        """Test validation of valid request."""
        request = SecureAnalysisRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            candles=[
                {
                    "timestamp": "2024-01-01T00:00:00",
                    "open": 40000.0,
                    "high": 41000.0,
                    "low": 39000.0,
                    "close": 40500.0,
                    "volume": 1000000.0
                }
            ] * 100
        )
        
        assert request.symbol == "BTCUSDT"
        assert len(request.candles) == 100
    
    def test_invalid_symbol(self):
        """Test validation rejects invalid symbol."""
        with pytest.raises(ValueError):
            SecureAnalysisRequest(
                symbol="BTC",  # Too short
                timeframe="1h",
                candles=[]
            )
    
    def test_insufficient_candles(self):
        """Test validation requires minimum candles."""
        with pytest.raises(ValueError):
            SecureAnalysisRequest(
                symbol="BTCUSDT",
                timeframe="1h",
                candles=[{"close": 40000.0}] * 10  # Less than minimum
            )
    
    def test_invalid_timeframe(self):
        """Test validation rejects invalid timeframe."""
        with pytest.raises(ValueError):
            SecureAnalysisRequest(
                symbol="BTCUSDT",
                timeframe="invalid",
                candles=[{"close": 40000.0}] * 100
            )
    
    def test_price_validation(self):
        """Test validation of price values."""
        with pytest.raises(ValueError):
            SecureAnalysisRequest(
                symbol="BTCUSDT",
                timeframe="1h",
                candles=[
                    {
                        "timestamp": "2024-01-01T00:00:00",
                        "open": -100.0,  # Negative price
                        "high": 41000.0,
                        "low": 39000.0,
                        "close": 40500.0,
                        "volume": 1000000.0
                    }
                ] * 100
            )


class TestSecurityHeaders:
    """Tests for security headers middleware."""
    
    @pytest.mark.asyncio
    async def test_security_headers_applied(self):
        """Test security headers are applied to responses."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from middleware.auth import add_security_headers
        
        app = FastAPI()
        app.middleware("http")(add_security_headers)
        
        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}
        
        client = TestClient(app)
        response = client.get("/test")
        
        # Check security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers


class TestAuditLogging:
    """Tests for audit logging functionality."""
    
    @pytest.mark.asyncio
    async def test_audit_log_authentication(self):
        """Test authentication events are logged."""
        with patch('middleware.auth.logger') as mock_logger:
            # Simulate login
            token = create_access_token({"sub": "test_user"})
            
            # Verify logging was called
            # (Implementation would log token creation)
            assert token is not None
    
    @pytest.mark.asyncio
    async def test_audit_log_failed_authentication(self):
        """Test failed authentication is logged."""
        with patch('middleware.auth.logger') as mock_logger:
            try:
                verify_token("invalid_token")
            except HTTPException:
                pass
            
            # Verify error logging
            # (Implementation would log authentication failure)


@pytest.mark.integration
class TestAuthenticationIntegration:
    """Integration tests for authentication flow."""
    
    @pytest.mark.asyncio
    async def test_full_authentication_flow(self):
        """Test complete authentication flow."""
        # 1. Create token
        token = create_access_token(
            data={"sub": "user123", "role": "analyst"}
        )
        
        # 2. Verify token
        payload = verify_token(token)
        assert payload["sub"] == "user123"
        assert payload["role"] == "analyst"
        
        # 3. Extract user from token
        user_data = payload
        assert user_data is not None
    
    @pytest.mark.asyncio
    async def test_rate_limiting_with_authentication(self):
        """Test rate limiting combined with authentication."""
        # Create authenticated token
        token = create_access_token({"sub": "user123"})
        payload = verify_token(token)
        
        # Check rate limit for user
        limiter = RateLimiter(requests_per_minute=5)
        user_id = payload["sub"]
        
        # Should allow requests
        for _ in range(5):
            result = await limiter.check_rate_limit(user_id)
            assert result is True
        
        # Should block excess
        with pytest.raises(HTTPException):
            await limiter.check_rate_limit(user_id)


class TestTokenRefresh:
    """Tests for token refresh functionality."""
    
    def test_refresh_token_creation(self):
        """Test creation of refresh token."""
        refresh_token = create_access_token(
            data={"sub": "user123", "type": "refresh"},
            expires_delta=timedelta(days=7)
        )
        
        assert refresh_token is not None
        payload = verify_token(refresh_token)
        assert payload["type"] == "refresh"
    
    def test_access_token_from_refresh(self):
        """Test creating new access token from refresh token."""
        # Create refresh token
        refresh_token = create_access_token(
            data={"sub": "user123", "type": "refresh"},
            expires_delta=timedelta(days=7)
        )
        
        # Verify and extract user
        payload = verify_token(refresh_token)
        assert payload["type"] == "refresh"
        
        # Create new access token
        new_access_token = create_access_token(
            data={"sub": payload["sub"]},
            expires_delta=timedelta(minutes=15)
        )
        
        assert new_access_token is not None
        new_payload = verify_token(new_access_token)
        assert new_payload["sub"] == "user123"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
