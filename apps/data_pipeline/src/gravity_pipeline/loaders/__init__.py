"""Data loaders for persisting data to databases"""

from .base import Loader
from .sqlite_loader import SQLiteLoader

try:
    from .postgres_loader import PostgreSQLLoader
except ImportError:
    PostgreSQLLoader = None

__all__ = [
    "Loader",
    "SQLiteLoader",
]

if PostgreSQLLoader is not None:
    __all__.append("PostgreSQLLoader")
