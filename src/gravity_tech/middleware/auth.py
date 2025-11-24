"""
Enhanced Security Middleware

امنیت پیشرفته: Authentication, Authorization, Rate Limiting, Input Validation

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from typing import Optional, List
import time
from collections import defaultdict
import jwt
from datetime import datetime, timedelta
import structlog
from pydantic import BaseModel, field_validator

from gravity_tech.config.settings import settings

logger = structlog.get_logger()

security = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


# ═══════════════════════════════════════════════════════════════
# JWT Authentication
# ═══════════════════════════════════════════════════════════════

class TokenData(BaseModel):
    """داده‌های توکن"""
    username: str
    scopes: List[str] = []
    exp: Optional[datetime] = None


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    ایجاد JWT token
    
    Args:
        data: داده‌های توکن (username, scopes, etc.)
        expires_delta: مدت زمان اعتبار توکن
    
    Returns:
        JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expiration_minutes)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.jwt_algorithm
    )
    
    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """
    تایید JWT token
    
    Args:
        token: JWT token string
    
    Returns:
        TokenData با اطلاعات کاربر
    
    Raises:
        HTTPException: در صورت نامعتبر بودن توکن
    """
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        token_data = TokenData(
            username=username,
            scopes=payload.get("scopes", []),
            exp=datetime.fromtimestamp(payload.get("exp"))
        )
        
        return token_data
    
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[TokenData]:
    """
    دریافت کاربر فعلی از توکن
    
    استفاده به صورت Dependency در endpoint:
        @app.get("/protected")
        async def protected_route(user: TokenData = Depends(get_current_user)):
            return {"user": user.username}
    """
    if credentials is None:
        return None
    
    return verify_token(credentials.credentials)


# ═══════════════════════════════════════════════════════════════
# Rate Limiting
# ═══════════════════════════════════════════════════════════════

class RateLimiter:
    """
    Rate Limiter با الگوریتم Token Bucket
    
    محدودیت تعداد درخواست‌ها برای جلوگیری از abuse
    
    Args:
        requests_per_minute: تعداد درخواست مجاز در دقیقه
        burst: حداکثر تعداد درخواست‌های burst
    """
    
    def __init__(self, requests_per_minute: int = 60, burst: int = 10):
        self.rate = requests_per_minute / 60.0  # requests per second
        self.burst = burst
        self.clients = defaultdict(lambda: {"tokens": burst, "last_update": time.time()})
    
    def _refill_tokens(self, client_id: str):
        """شارژ مجدد توکن‌ها بر اساس زمان گذشته"""
        current = time.time()
        client = self.clients[client_id]
        
        time_passed = current - client["last_update"]
        client["tokens"] = min(
            self.burst,
            client["tokens"] + time_passed * self.rate
        )
        client["last_update"] = current
    
    def is_allowed(self, client_id: str) -> bool:
        """
        بررسی اینکه آیا درخواست مجاز است
        
        Args:
            client_id: شناسه کلاینت (IP یا user ID)
        
        Returns:
            True اگر مجاز باشد، False در غیر این صورت
        """
        self._refill_tokens(client_id)
        
        client = self.clients[client_id]
        
        if client["tokens"] >= 1:
            client["tokens"] -= 1
            return True
        
        return False
    
    def get_retry_after(self, client_id: str) -> int:
        """زمان انتظار تا درخواست بعدی (ثانیه)"""
        self._refill_tokens(client_id)
        client = self.clients[client_id]
        
        if client["tokens"] >= 1:
            return 0
        
        tokens_needed = 1 - client["tokens"]
        return int(tokens_needed / self.rate) + 1


# Global rate limiter instance
rate_limiter = RateLimiter(requests_per_minute=100, burst=20)


async def check_rate_limit(request: Request):
    """
    Dependency برای بررسی rate limit
    
    استفاده:
        @app.get("/api/endpoint", dependencies=[Depends(check_rate_limit)])
        async def endpoint():
            return {"data": "..."}
    """
    client_id = request.client.host if request.client else "unknown"
    
    if not rate_limiter.is_allowed(client_id):
        retry_after = rate_limiter.get_retry_after(client_id)
        
        logger.warning(
            "rate_limit_exceeded",
            client=client_id,
            path=request.url.path,
            retry_after=retry_after
        )
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)}
        )


# ═══════════════════════════════════════════════════════════════
# Input Validation
# ═══════════════════════════════════════════════════════════════

class SecureAnalysisRequest(BaseModel):
    """
    مدل validated برای درخواست تحلیل
    
    شامل validatorهای امنیتی برای جلوگیری از حملات
    """
    symbol: str
    timeframe: str
    max_candles: Optional[int] = 100
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        """Validation برای symbol"""
        if not v or len(v) > 20:
            raise ValueError('Symbol must be between 1 and 20 characters')
        
        # فقط حروف، اعداد، و /-
        if not all(c.isalnum() or c in ['/', '-', '_'] for c in v):
            raise ValueError('Symbol contains invalid characters')
        
        return v.upper()
    
    @field_validator('timeframe')
    @classmethod
    def validate_timeframe(cls, v):
        """Validation برای timeframe"""
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
        
        if v not in valid_timeframes:
            raise ValueError(f'Invalid timeframe. Must be one of: {valid_timeframes}')
        
        return v
    
    @field_validator('max_candles')
    @classmethod
    def validate_max_candles(cls, v):
        """Validation برای تعداد کندل‌ها"""
        if v is not None:
            if v < 10 or v > 1000:
                raise ValueError('max_candles must be between 10 and 1000')
        
        return v


# ═══════════════════════════════════════════════════════════════
# Security Headers Middleware
# ═══════════════════════════════════════════════════════════════

def setup_security(app: FastAPI):
    """
    راه‌اندازی امنیت پیشرفته
    
    شامل:
    - Security headers
    - CORS configuration
    - Rate limiting
    - Request logging
    """
    
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        """اضافه کردن security headers به تمام پاسخ‌ها"""
        response = await call_next(request)
        
        # Security Headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # API Version
        response.headers["X-API-Version"] = "1.0.0"
        
        return response
    
    @app.middleware("http")
    async def log_security_events(request: Request, call_next):
        """لاگ کردن رویدادهای امنیتی"""
        
        # لاگ درخواست‌های مشکوک
        if request.method in ["POST", "PUT", "DELETE"]:
            logger.info(
                "security_audit",
                method=request.method,
                path=request.url.path,
                client=request.client.host if request.client else "unknown",
                user_agent=request.headers.get("user-agent")
            )
        
        response = await call_next(request)
        return response
    
    logger.info("security_middleware_enabled")
