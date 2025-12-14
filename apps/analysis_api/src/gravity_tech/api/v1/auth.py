"""
Authentication API endpoints

Simple login endpoint for frontend authentication.

Author: Gravity Tech Team
Date: December 10, 2025
Version: 1.0.0
License: MIT
"""

from datetime import UTC, datetime, timedelta

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from gravity_tech.config.settings import settings
from pydantic import BaseModel

router = APIRouter()


class Token(BaseModel):
    access_token: str
    token_type: str


class User(BaseModel):
    username: str


@router.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Simple login endpoint.
    For demo purposes, accepts any username/password.
    In production, implement proper authentication.
    """
    # Simple demo auth - accept any credentials
    if not form_data or not form_data.username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create token
    access_token_expires = datetime.now(UTC) + timedelta(minutes=settings.jwt_expiration_minutes)
    access_token = jwt.encode(
        {
            "sub": form_data.username,
            "exp": access_token_expires,
            "scopes": ["read", "write"]
        },
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )

    return Token(access_token=access_token, token_type="bearer")
