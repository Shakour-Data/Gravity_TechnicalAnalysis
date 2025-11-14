"""
Unit Tests for Momentum Indicators

Tests for TSI, Schaff Trend Cycle, and Connors RSI momentum indicators.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import numpy as np
import pytest

from src.core.indicators.momentum import tsi, schaff_trend_cycle, connors_rsi


@pytest.fixture
def uptrend_prices():
    # simple linear uptrend
    return np.linspace(100.0, 150.0, 120)


@pytest.fixture
def downtrend_prices():
    return np.linspace(150.0, 100.0, 120)


@pytest.fixture
def sideways_prices():
    rng = np.random.RandomState(42)
    base = 120.0
    return base + rng.normal(scale=0.5, size=120)


def test_tsi_uptrend(uptrend_prices):
    res = tsi(uptrend_prices)
    assert 'values' in res
    vals = res['values']
    assert vals.shape[0] == uptrend_prices.shape[0]
    # TSI should be positive for persistent uptrend
    assert res['signal'] == 'BUY'
    assert 0.0 <= res['confidence'] <= 1.0


def test_stc_behaviour(uptrend_prices, sideways_prices):
    res_up = schaff_trend_cycle(uptrend_prices)
    res_side = schaff_trend_cycle(sideways_prices)
    assert res_up['values'].shape[0] == uptrend_prices.shape[0]
    # STC should be generally higher in uptrend than sideways
    assert np.nanmean(res_up['values']) >= np.nanmean(res_side['values'])


def test_connors_rsi_values(uptrend_prices, downtrend_prices):
    cr_up = connors_rsi(uptrend_prices)
    cr_down = connors_rsi(downtrend_prices)
    assert cr_up['values'].shape[0] == uptrend_prices.shape[0]
    # Uptrend should yield higher CRSI than downtrend
    assert np.nanmean(cr_up['values']) > np.nanmean(cr_down['values'])
    assert 0 <= cr_up['confidence'] <= 1
