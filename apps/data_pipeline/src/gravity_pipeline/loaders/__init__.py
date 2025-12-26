"""Data loaders for persisting data to databases"""

from .base import Loader
from .sqlite_loader import SQLiteLoader
from .postgres_loader import PostgreSQLLoader

__all__ = [
    "Loader",
    "SQLiteLoader",
    "PostgreSQLLoader",
]
