"""
Comprehensive Middleware Authentication & Authorization Tests

Tests for JWT authentication, rate limiting, and security features:
- Token creation and validation
- JWT expiration handling
- Scope-based authorization
- Rate limiting with Token Bucket algorithm
- Security headers
- Input validation for analysis requests

Author: Gravity Tech Team
Date: December 5, 2025
Version: 1.0.0
License: MIT
"""

import time
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException, status
from gravity_tech.config.settings import settings
from gravity_tech.middleware.auth import (
    RateLimiter,
    SecureAnalysisRequest,
    TokenData,
    create_access_token,
    get_current_user,
    rate_limiter,
    verify_token,
)


class TestTokenDataModel:
    """Test TokenData model functionality"""

    def test_token_data_creation(self):
        """Test TokenData model initialization"""
        token_data = TokenData(
            username="user123",
            scopes=["read", "write"]
        )
        assert token_data.username == "user123"
        assert token_data.scopes == ["read", "write"]

    def test_token_data_default_scopes(self):
        """Test TokenData with default empty scopes"""
        token_data = TokenData(username="user123")
        assert token_data.username == "user123"
        assert token_data.scopes == []

    def test_token_data_getitem_access_sub(self):
        """Test dictionary-like access for 'sub' key"""
        token_data = TokenData(username="user123")
        assert token_data["sub"] == "user123"

    def test_token_data_getitem_access_scopes(self):
        """Test dictionary-like access for 'scopes' key"""
        token_data = TokenData(username="user123", scopes=["read", "admin"])
        assert token_data["scopes"] == ["read", "admin"]

    def test_token_data_getitem_raises_keyerror(self):
        """Test KeyError for invalid token data access"""
        token_data = TokenData(username="user123")
        with pytest.raises(KeyError):
            _ = token_data["invalid_key"]

    def test_token_data_contains_operator(self):
        """Test 'in' operator for TokenData"""
        token_data = TokenData(username="user123")
        assert "sub" in token_data
        assert "scopes" in token_data
        assert "exp" in token_data
        assert "invalid" not in token_data

    def test_token_data_with_expiration(self):
        """Test TokenData with expiration time"""
        exp_time = datetime.now(UTC) + timedelta(hours=1)

        token_data = TokenData(
            username="user123",
            exp=exp_time
        )
        assert token_data.exp == exp_time


class TestTokenCreation:
    """Test JWT token creation functionality"""

    def test_create_basic_token(self):
        """Test basic token creation"""
        data = {"sub": "user123"}
        token = create_access_token(data)

        assert isinstance(token, str)
        assert len(token) > 0
        assert token.count(".") == 2  # JWT has 3 parts

    def test_create_token_with_scopes(self):
        """Test token creation with scopes"""
        data = {
            "sub": "user123",
            "scopes": ["read", "write", "admin"]
        }
        token = create_access_token(data)

        # Decode and verify
        import jwt
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        assert payload["scopes"] == ["read", "write", "admin"]

    def test_create_token_with_custom_expiration(self):
        """Test token creation with custom expiration"""
        data = {"sub": "user123"}
        expires_delta = timedelta(hours=2)
        token = create_access_token(data, expires_delta)

        import jwt
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm]
        )

        exp_time = datetime.fromtimestamp(payload["exp"], tz=UTC)
        now = datetime.now(UTC)
        time_diff_seconds = (exp_time - now).total_seconds()

        # Should be approximately 2 hours (allow 2 minute buffer)
        assert 6900 < time_diff_seconds < 7320  # ~1h56min to ~2h2min

    def test_create_token_default_expiration(self):
        """Test token creation uses default settings expiration"""
        data = {"sub": "user123"}
        token = create_access_token(data)

        import jwt
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm]
        )

        exp_time = datetime.fromtimestamp(payload["exp"], tz=UTC)
        now = datetime.now(UTC)
        minutes_to_expiry = (exp_time - now).total_seconds() / 60

        # Should expire within the settings window
        assert minutes_to_expiry <= settings.jwt_expiration_minutes + 1

    def test_create_token_preserves_additional_claims(self):
        """Test that custom claims are preserved in token"""
        data = {
            "sub": "user123",
            "email": "user@example.com",
            "role": "analyst",
            "level": 5
        }
        token = create_access_token(data)

        import jwt
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm]
        )

        assert payload["email"] == "user@example.com"
        assert payload["role"] == "analyst"
        assert payload["level"] == 5

    def test_create_multiple_tokens_are_different(self):
        """Test that creating multiple tokens produces different results"""
        data = {"sub": "user123"}
        token1 = create_access_token(data)
        time.sleep(1)  # Longer delay to ensure different timestamps
        token2 = create_access_token(data)

        # Tokens should be different due to exp timestamp
        assert token1 != token2


class TestTokenVerification:
    """Test JWT token verification and validation"""

    def test_verify_valid_token(self):
        """Test verification of valid token"""
        data = {"sub": "user123", "scopes": ["read"]}
        token = create_access_token(data)

        token_data = verify_token(token)
        assert token_data.username == "user123"
        assert token_data.scopes == ["read"]

    def test_verify_token_invalid_signature(self):
        """Test verification fails with modified signature"""
        data = {"sub": "user123"}
        token = create_access_token(data)

        # Corrupt the signature
        corrupted = token[:-10] + "0000000000"

        with pytest.raises(HTTPException) as exc_info:
            verify_token(corrupted)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_verify_token_expired(self):
        """Test verification fails for expired token"""
        data = {"sub": "user123"}
        # Create expired token (expires 1 second ago)
        token = create_access_token(data, expires_delta=timedelta(seconds=-1))

        with pytest.raises(HTTPException) as exc_info:
            verify_token(token)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_verify_token_malformed(self):
        """Test verification fails for malformed token"""
        with pytest.raises(HTTPException) as exc_info:
            verify_token("not.a.token")

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_verify_token_empty(self):
        """Test verification fails for empty token"""
        with pytest.raises(HTTPException) as exc_info:
            verify_token("")

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_verify_token_missing_sub(self):
        """Test verification fails when 'sub' claim missing"""
        import jwt
        # Create token without 'sub'
        payload = {
            "exp": datetime.now(UTC) + timedelta(hours=1)
        }
        token = jwt.encode(
            payload,
            settings.secret_key,
            algorithm=settings.jwt_algorithm
        )

        with pytest.raises(HTTPException) as exc_info:
            verify_token(token)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_verify_token_extracts_scopes(self):
        """Test that scopes are properly extracted from token"""
        data = {
            "sub": "user123",
            "scopes": ["read", "write", "delete"]
        }
        token = create_access_token(data)

        token_data = verify_token(token)
        assert token_data.scopes == ["read", "write", "delete"]

    def test_verify_token_no_scopes(self):
        """Test that missing scopes default to empty list"""
        import jwt
        payload = {
            "sub": "user123",
            "exp": datetime.now(UTC) + timedelta(hours=1)
        }
        token = jwt.encode(
            payload,
            settings.secret_key,
            algorithm=settings.jwt_algorithm
        )

        token_data = verify_token(token)
        assert token_data.scopes == []


class TestRateLimiterFunctionality:
    """Test rate limiting with Token Bucket algorithm"""

    def test_rate_limiter_initialization(self):
        """Test RateLimiter initialization"""
        limiter = RateLimiter(requests_per_minute=60, burst=10, window_seconds=60)
        assert limiter.rate == 1.0  # 60 requests per 60 seconds = 1 per second
        assert limiter.burst == 10

    def test_rate_limiter_allows_requests_within_limit(self):
        """Test that requests within limit are allowed"""
        limiter = RateLimiter(requests_per_minute=100, burst=10, window_seconds=60)

        # Should allow up to burst count
        for _ in range(10):
            assert limiter.is_allowed("client1") is True

    def test_rate_limiter_denies_requests_over_burst(self):
        """Test that requests exceeding burst are denied"""
        limiter = RateLimiter(requests_per_minute=60, burst=5, window_seconds=60)

        # Use up burst quota
        for _ in range(5):
            assert limiter.is_allowed("client1") is True

        # Should be rate limited
        assert limiter.is_allowed("client1") is False

    def test_rate_limiter_different_clients_independent(self):
        """Test that rate limits are per-client"""
        limiter = RateLimiter(requests_per_minute=60, burst=3, window_seconds=60)

        # Client 1: use up quota
        for _ in range(3):
            assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is False

        # Client 2: should have fresh quota
        for _ in range(3):
            assert limiter.is_allowed("client2") is True
        assert limiter.is_allowed("client2") is False

    @patch('time.time')
    def test_rate_limiter_refills_over_time(self, mock_time):
        """Test that tokens refill as time passes"""
        mock_time.return_value = 0
        limiter = RateLimiter(requests_per_minute=60, burst=5, window_seconds=60)

        # Use up burst
        for _ in range(5):
            limiter.is_allowed("client1")

        # Should be denied
        assert limiter.is_allowed("client1") is False

        # Move time forward 10 seconds (10 tokens should refill)
        mock_time.return_value = 10

        # Should now allow a request
        assert limiter.is_allowed("client1") is True

    def test_rate_limiter_get_retry_after(self):
        """Test getting retry-after time"""
        limiter = RateLimiter(requests_per_minute=60, burst=2, window_seconds=60)

        # Use up quota
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")

        # Should return time to wait
        retry_after = limiter.get_retry_after("client1")
        assert retry_after > 0

    def test_global_rate_limiter_instance(self):
        """Test that global rate limiter instance exists"""
        assert rate_limiter is not None
        assert isinstance(rate_limiter, RateLimiter)


class TestSecureAnalysisRequest:
    """Test SecureAnalysisRequest model validation"""

    def test_valid_analysis_request(self):
        """Test creating valid analysis request"""
        request = SecureAnalysisRequest(
            symbol="TOTAL",
            timeframe="1h"
        )
        assert request.symbol == "TOTAL"
        assert request.timeframe == "1h"

    def test_symbol_validation_uppercase(self):
        """Test that symbols are converted to uppercase"""
        request = SecureAnalysisRequest(
            symbol="total",
            timeframe="1h"
        )
        assert request.symbol == "TOTAL"

    def test_symbol_validation_length(self):
        """Test symbol length validation"""
        with pytest.raises(ValueError) as exc:
            SecureAnalysisRequest(
                symbol="",
                timeframe="1h"
            )
        assert "between 1 and 20" in str(exc.value)

    def test_symbol_validation_too_long(self):
        """Test symbol too long rejected"""
        with pytest.raises(ValueError):
            SecureAnalysisRequest(
                symbol="X" * 21,
                timeframe="1h"
            )

    def test_symbol_validation_invalid_characters(self):
        """Test symbol with invalid characters rejected"""
        with pytest.raises(ValueError) as exc:
            SecureAnalysisRequest(
                symbol="TOTAL@",
                timeframe="1h"
            )
        assert "invalid characters" in str(exc.value)

    def test_timeframe_validation_valid(self):
        """Test valid timeframe values"""
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
        for tf in valid_timeframes:
            request = SecureAnalysisRequest(
                symbol="TOTAL",
                timeframe=tf
            )
            assert request.timeframe == tf

    def test_timeframe_validation_invalid(self):
        """Test invalid timeframe rejected"""
        with pytest.raises(ValueError) as exc:
            SecureAnalysisRequest(
                symbol="TOTAL",
                timeframe="2h"
            )
        assert "Invalid timeframe" in str(exc.value)

    def test_max_candles_validation(self):
        """Test max_candles validation"""
        # Valid: 10 to 1000
        request = SecureAnalysisRequest(
            symbol="TOTAL",
            timeframe="1h",
            max_candles=100
        )
        assert request.max_candles == 100

    def test_max_candles_too_small(self):
        """Test max_candles too small rejected"""
        with pytest.raises(ValueError) as exc:
            SecureAnalysisRequest(
                symbol="TOTAL",
                timeframe="1h",
                max_candles=5
            )
        assert "between 10 and 1000" in str(exc.value)

    def test_max_candles_too_large(self):
        """Test max_candles too large rejected"""
        with pytest.raises(ValueError):
            SecureAnalysisRequest(
                symbol="TOTAL",
                timeframe="1h",
                max_candles=2000
            )

    def test_candles_validation_structure(self):
        """Test candles validation for structure"""
        candles = [
            {"open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000},
            {"open": 105, "high": 115, "low": 100, "close": 110, "volume": 1100},
            {"open": 110, "high": 120, "low": 105, "close": 115, "volume": 1200},
            {"open": 115, "high": 125, "low": 110, "close": 120, "volume": 1300},
            {"open": 120, "high": 130, "low": 115, "close": 125, "volume": 1400},
            {"open": 125, "high": 135, "low": 120, "close": 130, "volume": 1500},
            {"open": 130, "high": 140, "low": 125, "close": 135, "volume": 1600},
            {"open": 135, "high": 145, "low": 130, "close": 140, "volume": 1700},
            {"open": 140, "high": 150, "low": 135, "close": 145, "volume": 1800},
            {"open": 145, "high": 155, "low": 140, "close": 150, "volume": 1900}
        ]
        request = SecureAnalysisRequest(
            symbol="TOTAL",
            timeframe="1h",
            candles=candles
        )
        if request.candles is not None:
            assert len(request.candles) == 10

    def test_candles_validation_minimum_count(self):
        """Test candles minimum count requirement"""
        candles = [
            {"open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000}
        ]
        with pytest.raises(ValueError) as exc:
            SecureAnalysisRequest(
                symbol="TOTAL",
                timeframe="1h",
                candles=candles
            )
        assert "at least 10" in str(exc.value)

    def test_candles_validation_negative_values(self):
        """Test that negative candle prices are rejected"""
        candles = [
            {"open": -100, "high": 110, "low": 90, "close": 105, "volume": 1000}
        ] + [
            {"open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000}
            for _ in range(9)
        ]
        with pytest.raises(ValueError) as exc:
            SecureAnalysisRequest(
                symbol="TOTAL",
                timeframe="1h",
                candles=candles
            )
        assert "non-negative" in str(exc.value)


@pytest.mark.asyncio
class TestCurrentUserDependency:
    """Test get_current_user dependency"""

    async def test_get_current_user_with_valid_credentials(self):
        """Test get_current_user with valid HTTP credentials"""
        data = {"sub": "user123"}
        token = create_access_token(data)

        credentials = Mock()
        credentials.credentials = token

        user = await get_current_user(credentials)
        assert user is not None
        assert user.username == "user123"

    async def test_get_current_user_with_none_credentials(self):
        """Test get_current_user returns None for no credentials"""
        user = await get_current_user(None)
        assert user is None

    async def test_get_current_user_invalid_token(self):
        """Test get_current_user with invalid token"""
        credentials = Mock()
        credentials.credentials = "invalid.token"

        with pytest.raises(HTTPException):
            await get_current_user(credentials)


class TestIntegrationFlows:
    """Integration tests for complete authentication flows"""

    def test_complete_auth_flow_create_and_verify(self):
        """Test complete flow: create token -> verify token"""
        # Create token
        user_data = {"sub": "user123", "scopes": ["read", "write"]}
        token = create_access_token(user_data)

        # Verify token
        token_data = verify_token(token)
        assert token_data.username == "user123"
        assert token_data.scopes == ["read", "write"]

    def test_auth_with_rate_limiting(self):
        """Test authentication combined with rate limiting"""
        limiter = RateLimiter(requests_per_minute=100, burst=5, window_seconds=60)
        user_data = {"sub": "user123"}
        token = create_access_token(user_data)

        # Simulate requests with rate limiting
        for _ in range(5):
            assert limiter.is_allowed("user123") is True
            _ = verify_token(token)

        # 6th request should be rate limited
        assert limiter.is_allowed("user123") is False

    def test_secure_request_validation_flow(self):
        """Test complete flow with request validation"""
        # Create token
        user_data = {"sub": "analyst1", "scopes": ["read", "write"]}
        token = create_access_token(user_data)

        # Verify token
        verify_token(token)

        # Validate secure request
        request = SecureAnalysisRequest(
            symbol="PETROFF",
            timeframe="1h"
        )
        assert request.symbol == "PETROFF"


class TestEdgeCases:
    """Edge case and security tests"""

    def test_token_with_special_usernames(self):
        """Test tokens with various special characters in username"""
        usernames = [
            "user@example.com",
            "user-name",
            "user_name",
            "user.name",
            "123user",
            "سلام"  # Persian text
        ]

        for username in usernames:
            data = {"sub": username}
            token = create_access_token(data)
            token_data = verify_token(token)
            assert token_data.username == username

    def test_very_long_scope_list(self):
        """Test handling large number of scopes"""
        scopes = [f"scope_{i}" for i in range(100)]
        data = {"sub": "user123", "scopes": scopes}
        token = create_access_token(data)

        token_data = verify_token(token)
        assert len(token_data.scopes) == 100

    def test_rate_limiter_with_many_clients(self):
        """Test rate limiter with many simultaneous clients"""
        limiter = RateLimiter(requests_per_minute=60, burst=3, window_seconds=60)

        # Simulate 100 clients
        for client_id in range(100):
            for _request_num in range(3):
                assert limiter.is_allowed(f"client_{client_id}") is True

    def test_token_with_unicode_claims(self):
        """Test token with Unicode in claims"""
        data = {
            "sub": "user123",
            "name": "علی احمد",  # Persian name
            "city": "تهران"  # Tehran
        }
        token = create_access_token(data)

        import jwt
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        assert payload["name"] == "علی احمد"
        assert payload["city"] == "تهران"

    def test_repeated_token_verification(self):
        """Test verifying same token multiple times"""
        data = {"sub": "user123"}
        token = create_access_token(data)

        for _ in range(10):
            verify_token(token)


    def test_concurrent_rate_limit_behavior(self):
        """Test rate limiter behavior with conceptual concurrency"""
        limiter = RateLimiter(requests_per_minute=100, burst=5, window_seconds=60)

        # Simulate rapid concurrent requests from same client
        results = []
        for _ in range(10):
            results.append(limiter.is_allowed("concurrent_client"))

        # First 5 succeed, rest fail
        assert results.count(True) == 5
        assert results.count(False) == 5
