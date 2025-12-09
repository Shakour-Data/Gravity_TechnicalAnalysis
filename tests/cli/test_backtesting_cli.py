import pytest

from gravity_tech.ml.backtesting import run_backtest_with_synthetic_data
from gravity_tech.ml.backtest_optimizer import suggest_params


@pytest.mark.unit
def test_backtesting_synthetic_runs():
    backtester = run_backtest_with_synthetic_data(n_bars=400)
    # Should produce a backtester instance and a trades list (may be empty depending on patterns found)
    assert backtester.trades is not None
    metrics = backtester.calculate_metrics()
    assert isinstance(metrics, dict)


def test_suggest_params_handles_empty_history():
    suggestion = suggest_params(symbol="UNKNOWN", interval="1d")
    assert suggestion.min_confidence == 0.6
    assert suggestion.limit is None
