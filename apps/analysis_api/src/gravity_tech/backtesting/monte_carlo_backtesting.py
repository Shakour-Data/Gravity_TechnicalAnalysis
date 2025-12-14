"""
================================================================================
Monte Carlo Backtesting Implementation

Advanced backtesting using Monte Carlo simulation for:
- Strategy robustness testing
- Risk assessment
- Performance distribution analysis
- Confidence interval calculation

Last Updated: 2025-11-07 (Phase 2.1 - Task 1.4)
================================================================================
"""

import logging
import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo simulation."""

    total_simulations: int
    successful_simulations: int
    success_rate: float
    average_return: float
    median_return: float
    std_return: float
    max_return: float
    min_return: float
    sharpe_ratio_avg: float
    max_drawdown_avg: float
    confidence_intervals: dict[str, tuple[float, float]]
    return_distribution: list[float]
    simulation_details: list[dict[str, Any]]


@dataclass
class MonteCarloBacktester:
    """
    Monte Carlo backtesting for strategy validation.

    Uses Monte Carlo simulation to test strategy robustness under
    various market conditions and parameter variations.
    """

    def __init__(
        self,
        strategy_function: Callable,
        historical_data: pd.DataFrame,
        num_simulations: int = 1000,
        test_window_days: int = 252,  # 1 year
        confidence_levels: list[float] | None = None
    ):
        """
        Initialize Monte Carlo backtester.

        Args:
            strategy_function: Function that takes data and returns trades/signals
            historical_data: Historical price data
            num_simulations: Number of Monte Carlo simulations
            test_window_days: Length of each test window in days
            confidence_levels: Confidence levels for intervals (default: [0.95, 0.99])
        """
        self.strategy_function = strategy_function
        self.historical_data = historical_data.copy()
        self.num_simulations = num_simulations
        self.test_window_days = test_window_days
        self.confidence_levels = confidence_levels or [0.95, 0.99]

        # Validate data
        self._validate_data()

    def _validate_data(self):
        """Validate input data format."""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.historical_data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if len(self.historical_data) < self.test_window_days:
            raise ValueError(f"Insufficient data: need at least {self.test_window_days} periods")

    def run_monte_carlo_analysis(
        self,
        parallel: bool = True,
        max_workers: int | None = None
    ) -> MonteCarloResult:
        """
        Run Monte Carlo analysis.

        Args:
            parallel: Whether to run simulations in parallel
            max_workers: Maximum number of parallel workers

        Returns:
            MonteCarloResult with analysis results
        """
        logger.info(f"Starting Monte Carlo analysis with {self.num_simulations} simulations")

        if parallel and mp.cpu_count() > 1:
            return self._run_parallel_simulations(max_workers)
        else:
            return self._run_sequential_simulations()

    def _run_parallel_simulations(self, max_workers: int | None) -> MonteCarloResult:
        """Run simulations in parallel."""
        workers = max_workers or min(mp.cpu_count(), self.num_simulations // 10)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self._run_single_simulation, i)
                for i in range(self.num_simulations)
            ]

            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Simulation failed: {e}")

        return self._analyze_simulation_results(results)

    def _run_sequential_simulations(self) -> MonteCarloResult:
        """Run simulations sequentially."""
        results = []
        for i in range(self.num_simulations):
            try:
                result = self._run_single_simulation(i)
                results.append(result)
            except Exception as e:
                logger.error(f"Simulation {i} failed: {e}")

        return self._analyze_simulation_results(results)

    def _run_single_simulation(self, simulation_id: int) -> dict[str, Any]:
        """
        Run a single Monte Carlo simulation.

        Args:
            simulation_id: Unique simulation identifier

        Returns:
            Dictionary with simulation results
        """
        try:
            # Sample random test window
            test_data = self._sample_random_window()

            # Add noise to simulate different market conditions
            noisy_data = self._add_market_noise(test_data)

            # Run strategy on modified data
            strategy_result = self.strategy_function(noisy_data)

            # Calculate performance metrics
            performance = self._calculate_performance_metrics(strategy_result, noisy_data)

            return {
                'simulation_id': simulation_id,
                'success': performance['total_return'] > 0,
                'total_return': performance['total_return'],
                'sharpe_ratio': performance['sharpe_ratio'],
                'max_drawdown': performance['max_drawdown'],
                'win_rate': performance['win_rate'],
                'profit_factor': performance['profit_factor'],
                'details': performance
            }

        except Exception as e:
            logger.error(f"Error in simulation {simulation_id}: {e}")
            return {
                'simulation_id': simulation_id,
                'success': False,
                'total_return': -1.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 1.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'details': {'error': str(e)}
            }

    def _sample_random_window(self) -> pd.DataFrame:
        """
        Sample a random test window from historical data.

        Returns:
            Random subset of historical data
        """
        max_start_idx = len(self.historical_data) - self.test_window_days

        if max_start_idx <= 0:
            return self.historical_data.copy()

        start_idx = np.random.randint(0, max_start_idx)
        end_idx = start_idx + self.test_window_days

        return self.historical_data.iloc[start_idx:end_idx].copy()

    def _add_market_noise(
        self,
        data: pd.DataFrame,
        volatility_factor: float = 0.1,
        trend_bias: float = 0.0
    ) -> pd.DataFrame:
        """
        Add realistic market noise to data.

        Args:
            data: Original price data
            volatility_factor: Factor to adjust volatility
            trend_bias: Trend bias to add

        Returns:
            Modified data with added noise
        """
        modified_data = data.copy()

        # Calculate returns
        returns = modified_data['close'].pct_change().fillna(0)

        # Add volatility noise
        volatility_noise = np.random.normal(0, volatility_factor, len(returns))
        noisy_returns = returns + volatility_noise

        # Add trend bias
        trend_noise = np.linspace(0, trend_bias, len(returns))
        noisy_returns += trend_noise

        # Reconstruct prices
        modified_data['close'] = modified_data['close'].iloc[0] * (1 + noisy_returns).cumprod()

        # Adjust OHLC accordingly
        for i in range(1, len(modified_data)):
            base_price = modified_data['close'].iloc[i]
            volatility = abs(returns.iloc[i]) * (1 + volatility_factor)

            # Generate realistic OHLC
            high_mult = 1 + np.random.uniform(0, volatility)
            low_mult = 1 - np.random.uniform(0, volatility)

            modified_data.loc[modified_data.index[i], 'high'] = base_price * high_mult
            modified_data.loc[modified_data.index[i], 'low'] = base_price * low_mult
            modified_data.loc[modified_data.index[i], 'open'] = modified_data['close'].iloc[i-1]

        return modified_data

    def _calculate_performance_metrics(
        self,
        strategy_result: dict[str, Any],
        data: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Calculate performance metrics for a strategy result.

        Args:
            strategy_result: Result from strategy function
            data: Price data used

        Returns:
            Dictionary with performance metrics
        """
        # Extract trades/signals from strategy result
        trades = strategy_result.get('trades', [])
        signals = strategy_result.get('signals', [])

        if not trades and not signals:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }

        # Calculate returns
        returns = self._calculate_strategy_returns(trades or signals, data)

        if not returns:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }

        # Calculate metrics
        total_return = (1 + pd.Series(returns)).prod() - 1  # type: ignore
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(returns)
        win_rate = self._calculate_win_rate(returns)
        profit_factor = self._calculate_profit_factor(returns)

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }

    def _calculate_strategy_returns(
        self,
        trades_signals: list[dict[str, Any]],
        data: pd.DataFrame
    ) -> list[float]:
        """
        Calculate strategy returns from trades/signals.

        Args:
            trades_signals: List of trades or signals
            data: Price data

        Returns:
            List of periodic returns
        """
        returns = []
        position = 0  # 1 for long, -1 for short, 0 for neutral

        for i, item in enumerate(trades_signals):
            if 'signal' in item:
                # Signal-based
                signal = item['signal']
                if signal > 0 and position <= 0:
                    position = 1
                elif signal < 0 and position >= 0:
                    position = -1
                elif signal == 0:
                    position = 0
            elif 'action' in item:
                # Trade-based
                action = item['action']
                if action == 'buy' and position <= 0:
                    position = 1
                elif action == 'sell' and position >= 0:
                    position = -1

            # Calculate return for this period
            if i > 0:
                price_return = data['close'].iloc[i] / data['close'].iloc[int(i-1)] - 1
                strategy_return = position * price_return
                returns.append(strategy_return)

        return returns

    def _calculate_sharpe_ratio(self, returns: list[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if not returns:
            return 0.0

        returns_series = pd.Series(returns)
        excess_returns = returns_series - risk_free_rate / 252  # Daily risk-free rate

        if excess_returns.std() == 0:
            return 0.0

        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    def _calculate_max_drawdown(self, returns: list[float]) -> float:
        """Calculate maximum drawdown."""
        if not returns:
            return 0.0

        cumulative = (1 + pd.Series(returns)).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max

        return abs(drawdowns.min())

    def _calculate_win_rate(self, returns: list[float]) -> float:
        """Calculate win rate."""
        if not returns:
            return 0.0

        winning_trades = sum(1 for r in returns if r > 0)
        return winning_trades / len(returns)

    def _calculate_profit_factor(self, returns: list[float]) -> float:
        """Calculate profit factor."""
        if not returns:
            return 0.0

        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def _analyze_simulation_results(self, results: list[dict[str, Any]]) -> MonteCarloResult:
        """
        Analyze results from all simulations.

        Args:
            results: List of simulation results

        Returns:
            MonteCarloResult with aggregated analysis
        """
        if not results:
            raise ValueError("No simulation results to analyze")

        # Extract metrics
        returns = [r['total_return'] for r in results]
        successful = [r for r in results if r['success']]

        # Calculate statistics
        success_rate = len(successful) / len(results)
        avg_return = float(np.mean(returns))
        median_return = float(np.median(returns))
        std_return = float(np.std(returns))
        max_return = float(max(returns))
        min_return = float(min(returns))

        # Calculate confidence intervals
        confidence_intervals = {}
        for level in self.confidence_levels:
            lower_percentile = (1 - level) / 2 * 100
            upper_percentile = (1 + level) / 2 * 100

            confidence_intervals[f"{int(level*100)}%"] = (
                float(np.percentile(returns, lower_percentile)),  # type: ignore
                float(np.percentile(returns, upper_percentile))   # type: ignore
            )

        # Additional metrics from successful simulations
        if successful:
            sharpe_ratios = [r['sharpe_ratio'] for r in successful]
            max_drawdowns = [r['max_drawdown'] for r in successful]

            sharpe_avg = float(np.mean(sharpe_ratios))
            max_dd_avg = float(np.mean(max_drawdowns))
        else:
            sharpe_avg = 0.0
            max_dd_avg = 1.0

        return MonteCarloResult(
            total_simulations=len(results),
            successful_simulations=len(successful),
            success_rate=success_rate,
            average_return=avg_return,
            median_return=median_return,
            std_return=std_return,
            max_return=max_return,
            min_return=min_return,
            sharpe_ratio_avg=sharpe_avg,
            max_drawdown_avg=max_dd_avg,
            confidence_intervals=confidence_intervals,
            return_distribution=returns,
            simulation_details=results
        )
