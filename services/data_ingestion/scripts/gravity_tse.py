"""
Thin compatibility wrapper around the third-party ``finpy_tse`` package.

The original Gravity ingestion pipeline expected a local ``gravity_tse`` module
that exposed a handful of helper functions (``Get_Price_History``, sector/index
fetchers, USD/Rial history, ...).  That module is not published publicly, but
``finpy_tse`` already provides the same functionality with identical function
names.  To keep this project self-contained, we expose the expected API surface
and delegate the heavy lifting to ``finpy_tse`` while adding a tiny bit of
error-handling/retry logic.
"""

from __future__ import annotations

import os
import ssl
from collections.abc import Callable
from functools import wraps
from typing import Any

import pandas as pd

try:
    import finpy_tse as _fpy
except ImportError:  # type: ignore
    _fpy = None

# Disable SSL verification if the environment requires custom root CAs (common on
# the legacy TSETMC endpoints).  This mirrors what the original gravity_tse
# script did and keeps compatibility with tightly controlled networks.
os.environ.setdefault("PYTHONHTTPSVERIFY", "0")
try:  # pragma: no cover - platform dependent
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except Exception:
    pass

DataFrame = pd.DataFrame


def _ensure_dataframe(result: Any) -> DataFrame:
    """Normalize finpy_tse responses to pandas.DataFrame."""
    if result is None:
        return pd.DataFrame()
    if isinstance(result, pd.DataFrame):
        return result
    try:
        return pd.DataFrame(result)
    except Exception:  # pragma: no cover - defensive
        return pd.DataFrame()


def _wrap(name: str) -> Callable[..., DataFrame]:
    if _fpy is None:

        def _missing(*args: Any, **kwargs: Any) -> DataFrame:
            raise ModuleNotFoundError("finpy_tse is required for gravity_tse data fetching")

        return _missing

    finpy_func: Callable[..., Any] = getattr(_fpy, name)

    @wraps(finpy_func)
    def _inner(*args: Any, retries: int = 3, **kwargs: Any) -> DataFrame:
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                return _ensure_dataframe(finpy_func(*args, **kwargs))
            except Exception as exc:  # pragma: no cover - finpy raises requests errors
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        return pd.DataFrame()

    return _inner


# Price history for individual instruments
Get_Price_History = _wrap("Get_Price_History")

# Sector indices
Get_SectorIndex_History = _wrap("Get_SectorIndex_History")

# Market-wide indices
Get_CWI_History = _wrap("Get_CWI_History")
Get_EWI_History = _wrap("Get_EWI_History")
Get_CWPI_History = _wrap("Get_CWPI_History")
Get_EWPI_History = _wrap("Get_EWPI_History")
Get_FFI_History = _wrap("Get_FFI_History")
Get_MKT1I_History = _wrap("Get_MKT1I_History")
Get_MKT2I_History = _wrap("Get_MKT2I_History")
Get_INDI_History = _wrap("Get_INDI_History")
Get_ACT50_History = _wrap("Get_ACT50_History")
Get_LCI30_History = _wrap("Get_LCI30_History")
Get_RI_History = _wrap("Get_RI_History")

# Currency series
Get_USD_RIAL = _wrap("Get_USD_RIAL")

__all__ = [
    "Get_Price_History",
    "Get_SectorIndex_History",
    "Get_CWI_History",
    "Get_EWI_History",
    "Get_CWPI_History",
    "Get_EWPI_History",
    "Get_FFI_History",
    "Get_MKT1I_History",
    "Get_MKT2I_History",
    "Get_INDI_History",
    "Get_ACT50_History",
    "Get_LCI30_History",
    "Get_RI_History",
    "Get_USD_RIAL",
]
