"""
Walk-Forward Backtesting Analysis

This module implements walk-forward optimization for backtesting:
- Rolling window optimization
- Out-of-sample testing
- Multiple time periods analysis
- Overfitting prevention
- Realistic performance evaluation

Author: Gravity Tech Team
Date: December 4, 2025
Version: 1.0.0
License: MIT
"""

import logging
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from gravity_tech.models.schemas import Candle

logger = logging.getLogger(__name__)


class Trade(NamedTuple):
    entry_time: Any
    exit_time: Any
    entry_price: float
    exit_price: float
    quantity: int
    profit_loss: float
    commission: float

class BacktestResult(NamedTuple):
    final_capital: float
    total_return: float
    total_trades: int
    winning_trades: int
    max_drawdown: float
    sharpe_ratio: float
    trades: list

class WalkForwardResult(NamedTuple):
    num_windows: int
    optimization_window: int
    testing_window: int
    step_size: int
    average_oos_return: float
    median_oos_return: float
    oos_return_std: float
    oos_sharpe_ratio: float
    consistency_ratio: float
    max_drawdown_avg: float
    parameter_stability: float
    walk_forward_results: list
    description: str

logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """Walk-forward optimization for backtesting trading strategies"""

    def __init__(self, optimization_window: int = 252,  # 1 year
                 testing_window: int = 63,     # 3 months
                 step_size: int = 21):         # 1 month
        self.optimization_window = optimization_window
        self.testing_window = testing_window
        self.step_size = step_size

    def run_walk_forward_analysis(self, historical_data: list[Candle],
                                 strategy_optimizer: Callable,
                                 parameter_ranges: dict[str, list[Any]],
                                 initial_capital: float = 10000.0,
                                 commission: float = 0.001) -> WalkForwardResult:
        """
        Run walk-forward optimization analysis

        Args:
            historical_data: Historical price data
            strategy_optimizer: Function that optimizes strategy parameters
            parameter_ranges: Dictionary of parameter ranges to test
            initial_capital: Starting capital
            commission: Trading commission

        Returns:
            WalkForwardResult with walk-forward analysis
        """
        logger.info("Starting walk-forward analysis")

        if len(historical_data) < self.optimization_window + self.testing_window:
            raise ValueError("Insufficient data for walk-forward analysis")

        # Convert to DataFrame
        df = self._candles_to_dataframe(historical_data)

        # Generate walk-forward windows
        windows = self._generate_walk_forward_windows(len(df))

        logger.info(f"Generated {len(windows)} walk-forward windows")

        # Run walk-forward optimization
        walk_forward_results = []

        for i, (opt_start, opt_end, test_start, test_end) in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)}: Opt {opt_start}-{opt_end}, Test {test_start}-{test_end}")

            # Optimization phase
            optimization_data = df.iloc[opt_start:opt_end]
            opt_candles = self._dataframe_to_candles(optimization_data)

            # Find optimal parameters
            optimal_params = strategy_optimizer(opt_candles, parameter_ranges)

            # Testing phase
            testing_data = df.iloc[test_start:test_end]
            test_candles = self._dataframe_to_candles(testing_data)

            # Run strategy with optimal parameters
            test_result = self._run_strategy_with_params(test_candles, optimal_params,
                                                        initial_capital, commission)

            window_result = {
                'window_id': i,
                'optimization_period': {
                    'start': df.index[opt_start],
                    'end': df.index[opt_end-1]
                },
                'testing_period': {
                    'start': df.index[test_start],
                    'end': df.index[test_end-1]
                },
                'optimal_parameters': optimal_params,
                'test_result': test_result,
                'out_of_sample_return': test_result.total_return
            }

            walk_forward_results.append(window_result)

        # Analyze overall results
        analysis = self._analyze_walk_forward_results(walk_forward_results)

        logger.info(f"Walk-forward analysis completed. Average OOS return: {analysis['average_oos_return']:.2%}")

        return WalkForwardResult(
            num_windows=len(walk_forward_results),
            optimization_window=self.optimization_window,
            testing_window=self.testing_window,
            step_size=self.step_size,
            average_oos_return=analysis['average_oos_return'],
            median_oos_return=analysis['median_oos_return'],
            oos_return_std=analysis['oos_return_std'],
            oos_sharpe_ratio=analysis['oos_sharpe_ratio'],
            consistency_ratio=analysis['consistency_ratio'],
            max_drawdown_avg=analysis['max_drawdown_avg'],
            parameter_stability=analysis['parameter_stability'],
            walk_forward_results=walk_forward_results,
            description=f"Walk-forward analysis with {len(walk_forward_results)} windows"
        )

    def _generate_walk_forward_windows(self, data_length: int) -> list[tuple[int, int, int, int]]:
        """Generate walk-forward window indices"""
        windows = []

        start_idx = 0

        while start_idx + self.optimization_window + self.testing_window <= data_length:
            opt_start = start_idx
            opt_end = start_idx + self.optimization_window
            test_start = opt_end
            test_end = test_start + self.testing_window

            if test_end > data_length:
                break

            windows.append((opt_start, opt_end, test_start, test_end))
            start_idx += self.step_size

        return windows

    def _run_strategy_with_params(self, candles: list[Candle], params: dict[str, Any],
                                 initial_capital: float, commission: float) -> BacktestResult:
        """Run strategy with specific parameters"""
        # This is a placeholder - in practice, you'd implement your specific strategy
        # For now, we'll create a simple moving average crossover strategy

        try:
            # Extract parameters
            fast_period = params.get('fast_period', 10)
            slow_period = params.get('slow_period', 20)
            stop_loss = params.get('stop_loss', 0.02)
            take_profit = params.get('take_profit', 0.05)

            # Simple MA crossover strategy
            capital = initial_capital
            position = 0  # 0 = no position, 1 = long
            trades = []
            entry_price = 0.0
            max_capital = initial_capital
            max_drawdown = 0.0

            closes = [c.close for c in candles]

            # Calculate moving averages
            fast_ma = self._calculate_sma(closes, fast_period)
            slow_ma = self._calculate_sma(closes, slow_period)

            for i in range(max(fast_period, slow_period), len(candles)):
                current_price = closes[i]

                # Update max capital and drawdown
                current_value = capital + (position * (current_price - entry_price))
                max_capital = max(max_capital, current_value)
                current_drawdown = (max_capital - current_value) / max_capital
                max_drawdown = max(max_drawdown, current_drawdown)

                # Trading logic
                if position == 0:
                    # Look for entry
                    if (fast_ma[i] > slow_ma[i] and
                        fast_ma[i-1] <= slow_ma[i-1]):  # Bullish crossover
                        position = 1
                        entry_price = current_price * (1 + commission)
                        capital -= entry_price * commission  # Commission

                elif position == 1:
                    # Check exit conditions
                    exit_signal = False

                    # Stop loss
                    if current_price <= entry_price * (1 - stop_loss):
                        exit_price = current_price * (1 - commission)
                        exit_signal = True

                    # Take profit
                    elif current_price >= entry_price * (1 + take_profit):
                        exit_price = current_price * (1 - commission)
                        exit_signal = True

                    # Bearish crossover
                    elif (fast_ma[i] < slow_ma[i] and
                          fast_ma[i-1] >= slow_ma[i-1]):
                        exit_price = current_price * (1 - commission)
                        exit_signal = True

                    if exit_signal:
                        profit_loss = (exit_price - entry_price)
                        capital += profit_loss

                        trade = Trade(
                            entry_time=candles[i].timestamp,
                            exit_time=candles[i].timestamp,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            quantity=1,
                            profit_loss=profit_loss,
                            commission=commission * entry_price
                        )
                        trades.append(trade)

                        position = 0
                        entry_price = 0.0

            # Calculate final metrics
            total_return = (capital - initial_capital) / initial_capital
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.profit_loss > 0])

            # Calculate Sharpe ratio (simplified)
            if trades:
                returns = [t.profit_loss / initial_capital for t in trades]
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0.0

            return BacktestResult(
                final_capital=capital,
                total_return=total_return,
                total_trades=total_trades,
                winning_trades=winning_trades,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                trades=trades
            )

        except Exception as e:
            logger.error(f"Error running strategy: {e}")
            return BacktestResult(
                final_capital=initial_capital,
                total_return=0.0,
                total_trades=0,
                winning_trades=0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                trades=[]
            )

    def _analyze_walk_forward_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze walk-forward results"""
        oos_returns = [r['out_of_sample_return'] for r in results]

        # Basic statistics
        average_oos_return = np.mean(oos_returns)
        median_oos_return = np.median(oos_returns)
        oos_return_std = np.std(oos_returns)

        # Sharpe ratio for OOS returns
        if oos_return_std > 0:
            oos_sharpe_ratio = average_oos_return / oos_return_std * np.sqrt(4)  # Quarterly
        else:
            oos_sharpe_ratio = 0.0

        # Consistency ratio (percentage of positive periods)
        consistency_ratio = np.mean([1 if r > 0 else 0 for r in oos_returns])

        # Average maximum drawdown
        max_drawdowns = [r['test_result'].max_drawdown for r in results]
        max_drawdown_avg = np.mean(max_drawdowns)

        # Parameter stability (how much parameters change between windows)
        parameters_list = [r['optimal_parameters'] for r in results]
        if parameters_list:
            param_keys = parameters_list[0].keys()
            stability_scores = {}

            for key in param_keys:
                values = [p.get(key, 0) for p in parameters_list]
                if len(values) > 1:
                    # Coefficient of variation as stability measure
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    cv = std_val / (abs(mean_val) + 1e-8)
                    stability_scores[key] = 1.0 / (1.0 + cv)  # Higher is more stable
                else:
                    stability_scores[key] = 1.0

            parameter_stability = np.mean(list(stability_scores.values()))
        else:
            parameter_stability = 0.0

        return {
            'average_oos_return': average_oos_return,
            'median_oos_return': median_oos_return,
            'oos_return_std': oos_return_std,
            'oos_sharpe_ratio': oos_sharpe_ratio,
            'consistency_ratio': consistency_ratio,
            'max_drawdown_avg': max_drawdown_avg,
            'parameter_stability': parameter_stability
        }

    def _calculate_sma(self, data: list[float], period: int) -> list[float]:
        """Calculate Simple Moving Average"""
        sma = []
        for i in range(len(data)):
            if i < period - 1:
                sma.append(np.nan)
            else:
                sma.append(np.mean(data[i-period+1:i+1]))
        return sma

    def _candles_to_dataframe(self, candles: list[Candle]) -> pd.DataFrame:
        """Convert list of candles to DataFrame"""
        data = {
            'timestamp': [c.timestamp for c in candles],
            'open': [c.open for c in candles],
            'high': [c.high for c in candles],
            'low': [c.low for c in candles],
            'close': [c.close for c in candles],
            'volume': [c.volume for c in candles]
        }
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    def _dataframe_to_candles(self, df: pd.DataFrame) -> list[Candle]:
        """Convert DataFrame back to list of candles"""
        candles = []
        for idx, row in df.iterrows():
            candle = Candle(
                timestamp=pd.to_datetime(str(idx)),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume'])
            )
            candles.append(candle)
        return candles


def create_walk_forward_backtester(optimization_window: int = 252,
                                 testing_window: int = 63) -> WalkForwardBacktester:
    """
    Factory function to create walk-forward backtester

    Args:
        optimization_window: Size of optimization window (in periods)
        testing_window: Size of testing window (in periods)

    Returns:
        Configured WalkForwardBacktester instance
    """
    return WalkForwardBacktester(
        optimization_window=optimization_window,
        testing_window=testing_window
    )


def run_walk_forward_backtest(historical_data: list[Candle],
                             strategy_optimizer: Callable,
                             parameter_ranges: dict[str, list[Any]],
                             optimization_window: int = 252) -> WalkForwardResult:
    """
    Convenience function to run walk-forward backtest

    Args:
        historical_data: Historical price data
        strategy_optimizer: Parameter optimization function
        parameter_ranges: Parameter ranges to test
        optimization_window: Optimization window size

    Returns:
        Walk-forward analysis results
    """
    backtester = WalkForwardBacktester(optimization_window=optimization_window)
    return backtester.run_walk_forward_analysis(
        historical_data, strategy_optimizer, parameter_ranges
    )
