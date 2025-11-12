"""
Security middleware setup
"""

from fastapi import FastAPI
from fastapi.security import HTTPBearer
from config.settings import settings


security = HTTPBearer()


def setup_security(app: FastAPI):
    """Configure security settings"""
    
    # Add security headers middleware
    @app.middleware("http")
    async def add_security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response
