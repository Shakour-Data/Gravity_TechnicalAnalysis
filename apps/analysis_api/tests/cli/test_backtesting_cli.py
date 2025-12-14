import pytest
from gravity_tech.ml.backtest_optimizer import suggest_params
from gravity_tech.ml.backtesting import run_backtest_with_synthetic_data


@pytest.mark.unit
def test_backtesting_synthetic_runs():
    backtester = run_backtest_with_synthetic_data(n_bars=400)
    # Should produce a backtester instance and a trades list (may be empty depending on patterns found)
    assert backtester.trades is not None
    metrics = backtester.calculate_metrics()
    assert isinstance(metrics, dict)


def test_suggest_params_handles_empty_history():
    from unittest.mock import patch
    with patch('gravity_tech.ml.backtest_optimizer.DatabaseManager') as mock_db:
        mock_manager = mock_db.return_value
        mock_manager.db_type = 'JSON_FILE'
        mock_manager.json_data = {"backtest_runs": []}  # Empty history
        suggestion = suggest_params(symbol="UNKNOWN", interval="1d", db_manager=mock_manager)
        assert suggestion.min_confidence == 0.6
        assert suggestion.limit is None
