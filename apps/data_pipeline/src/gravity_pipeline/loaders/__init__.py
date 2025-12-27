"""Data loaders for persisting data to databases"""

from .base import Loader
from .postgres_loader import PostgreSQLLoader
from .sqlite_loader import SQLiteLoader

__all__ = [
    "Loader",
    "SQLiteLoader",
    "PostgreSQLLoader",
]
