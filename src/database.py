"""
Legacy shim kept for backward compatibility.

All consumers should migrate to `gravity_tech.database.tse_data_source`.
"""

from gravity_tech.database.tse_data_source import (  # noqa: F401
    TSEDatabaseConnector,
    init_price_data,
    tse_data_source,
)

__all__ = ["TSEDatabaseConnector", "tse_data_source", "init_price_data"]
