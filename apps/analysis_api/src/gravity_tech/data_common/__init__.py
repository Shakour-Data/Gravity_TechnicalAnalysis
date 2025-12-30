"""
Shared helpers for DB DSN resolution and symbol standardization.
"""

from __future__ import annotations

import os


def resolve_dsn(default: str | None = None) -> str:
    """
    Resolve DATABASE_URL from environment, falling back to provided default.
    """
    dsn = os.getenv("DATABASE_URL") or os.getenv("PG_DSN") or default
    if not dsn:
        raise RuntimeError("DATABASE_URL/PG_DSN not set and no default provided")
    # Normalize legacy postgres+psycopg2 URL to standard postgres://
    if dsn.startswith("postgresql+psycopg2://"):
        dsn = dsn.replace("postgresql+psycopg2://", "postgresql://", 1)
    return dsn


def resolve_tse_dsn(default: str | None = None) -> str:
    """
    Resolve TSE data source DSN/path.
    """
    dsn = (
        os.getenv("TSE_DATABASE_URL")
        or os.getenv("TSE_DB_URL")
        or os.getenv("TSE_POSTGRES_URL")
        or default
    )
    if not dsn:
        raise RuntimeError("TSE DSN not set")
    if dsn.startswith("postgresql+psycopg2://"):
        dsn = dsn.replace("postgresql+psycopg2://", "postgresql://", 1)
    return dsn


def standardize_symbol(symbol: str) -> str:
    """
    Basic normalization hook for symbols; extend as needed.
    """
    return symbol.strip()
