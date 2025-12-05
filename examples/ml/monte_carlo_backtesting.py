"""
Monte Carlo Backtesting Analysis

This module implements Monte Carlo simulation for backtesting:
- Multiple random scenarios generation
- Statistical analysis of trading strategies
- Risk assessment through simulation
- Confidence intervals for performance metrics

Author: Gravity Tech Team
Date: December 4, 2025
Version: 1.0.0
License: MIT
"""

import logging
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from gravity_tech.models.schemas import Candle

logger = logging.getLogger(__name__)


# Define result classes locally for the example
class MonteCarloResult:
    """Result of Monte Carlo simulation."""

    def __init__(self, num_simulations: int, success_rate: float, average_return: float,
                 median_return: float, std_return: float, max_return: float, min_return: float,
                 confidence_interval_95: tuple[float, float], sharpe_ratio: float,
                 max_drawdown_avg: float, win_rate: float, profit_factor: float,
                 simulation_results: list[dict[str, Any]], description: str):
        self.num_simulations = num_simulations
        self.success_rate = success_rate
        self.average_return = average_return
        self.median_return = median_return
        self.std_return = std_return
        self.max_return = max_return
        self.min_return = min_return
        self.confidence_interval_95 = confidence_interval_95
        self.sharpe_ratio = sharpe_ratio
        self.max_drawdown_avg = max_drawdown_avg
        self.win_rate = win_rate
        self.profit_factor = profit_factor
        self.simulation_results = simulation_results
        self.description = description


class MonteCarloBacktester:
    """Monte Carlo simulation for backtesting trading strategies"""

    def __init__(self, num_simulations: int = 1000, confidence_level: float = 0.95):
        self.num_simulations = num_simulations
        self.confidence_level = confidence_level
        self.random_seed = 42
        np.random.seed(self.random_seed)

    def run_monte_carlo_analysis(self, historical_data: list[Candle],
                                strategy_function: Callable[[list[Candle], float, float], Any],
                                initial_capital: float = 10000.0,
                                commission: float = 0.001) -> MonteCarloResult:
        """
        Run Monte Carlo simulation analysis

        Args:
            historical_data: Historical price data
            strategy_function: Function that implements the trading strategy
            initial_capital: Starting capital for each simulation
            commission: Trading commission per trade

        Returns:
            MonteCarloResult with simulation statistics
        """
        logger.info(f"Starting Monte Carlo analysis with {self.num_simulations} simulations")

        # Convert to DataFrame for easier manipulation
        df = self._candles_to_dataframe(historical_data)

        # Run simulations in parallel
        num_cores = min(multiprocessing.cpu_count(), 8)  # Limit to 8 cores max

        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = []
            for i in range(self.num_simulations):
                future = executor.submit(
                    self._run_single_simulation,
                    df.copy(),
                    strategy_function,
                    initial_capital,
                    commission,
                    i
                )
                futures.append(future)

            # Collect results
            simulation_results = []
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    simulation_results.append(result)
                except Exception as e:
                    logger.error(f"Simulation failed: {e}")
                    continue

        if not simulation_results:
            raise ValueError("All simulations failed")

        # Analyze results
        analysis = self._analyze_simulation_results(simulation_results, initial_capital)

        logger.info(f"Monte Carlo analysis completed. Success rate: {analysis['success_rate']:.2%}")

        return MonteCarloResult(
            num_simulations=len(simulation_results),
            success_rate=analysis['success_rate'],
            average_return=analysis['average_return'],
            median_return=analysis['median_return'],
            std_return=analysis['std_return'],
            max_return=analysis['max_return'],
            min_return=analysis['min_return'],
            confidence_interval_95=analysis['confidence_interval_95'],
            sharpe_ratio=analysis['sharpe_ratio'],
            max_drawdown_avg=analysis['max_drawdown_avg'],
            win_rate=analysis['win_rate'],
            profit_factor=analysis['profit_factor'],
            simulation_results=simulation_results,
            description=f"Monte Carlo analysis with {len(simulation_results)} successful simulations"
        )

    def _run_single_simulation(self, df: pd.DataFrame, strategy_function: Callable[[list[Candle], float, float], Any],
                             initial_capital: float, commission: float, sim_id: int) -> dict[str, Any]:
        """Run a single Monte Carlo simulation"""
        try:
            # Create bootstrapped sample (sample with replacement)
            sample_size = len(df)
            indices = np.random.choice(sample_size, size=sample_size, replace=True)
            bootstrapped_data = df.iloc[indices].copy()

            # Sort by date to maintain temporal order
            bootstrapped_data = bootstrapped_data.sort_index()

            # Add some noise to prices to simulate different market conditions
            price_noise = np.random.normal(0, 0.02, len(bootstrapped_data))  # 2% volatility
            bootstrapped_data['close'] *= (1 + price_noise)
            bootstrapped_data['high'] = bootstrapped_data[['high', 'close']].max(axis=1)
            bootstrapped_data['low'] = bootstrapped_data[['low', 'close']].min(axis=1)

            # Convert back to candles
            candles = self._dataframe_to_candles(bootstrapped_data)

            # Run strategy on bootstrapped data
            result = strategy_function(candles, initial_capital, commission)

            return {
                'simulation_id': sim_id,
                'final_capital': result.final_capital,
                'total_return': result.total_return,
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'trades': result.trades
            }

        except Exception as e:
            logger.error(f"Simulation {sim_id} failed: {e}")
            return {
                'simulation_id': sim_id,
                'final_capital': initial_capital,
                'total_return': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'trades': []
            }

    def _analyze_simulation_results(self, results: list[dict[str, Any]],
                                  initial_capital: float) -> dict[str, Any]:
        """Analyze Monte Carlo simulation results"""
        returns = [r['total_return'] for r in results]

        # Basic statistics
        average_return = np.mean(returns)
        median_return = np.median(returns)
        std_return = np.std(returns)
        max_return = np.max(returns)
        min_return = np.min(returns)

        # Confidence interval
        sorted_returns = np.sort(returns)
        lower_idx = int((1 - self.confidence_level) / 2 * len(sorted_returns))
        upper_idx = int((1 + self.confidence_level) / 2 * len(sorted_returns))
        confidence_interval_95 = (sorted_returns[lower_idx], sorted_returns[upper_idx])

        # Success rate (profitable simulations)
        success_rate = np.mean([1 if r > 0 else 0 for r in returns])

        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        if std_return > 0:
            sharpe_ratio = (average_return - risk_free_rate) / std_return
        else:
            sharpe_ratio = 0.0

        # Maximum drawdown analysis
        max_drawdowns = [r['max_drawdown'] for r in results]
        max_drawdown_avg = np.mean(max_drawdowns)

        # Win rate
        win_rates = []
        profit_factors = []

        for r in results:
            if r['total_trades'] > 0:
                win_rate = r['winning_trades'] / r['total_trades']
                win_rates.append(win_rate)

                # Calculate profit factor
                winning_trades = [t for t in r['trades'] if t.profit_loss > 0]
                losing_trades = [t for t in r['trades'] if t.profit_loss < 0]

                gross_profit = sum(t.profit_loss for t in winning_trades)
                gross_loss = abs(sum(t.profit_loss for t in losing_trades))

                if gross_loss > 0:
                    profit_factor = gross_profit / gross_loss
                else:
                    profit_factor = float('inf') if gross_profit > 0 else 1.0

                profit_factors.append(profit_factor)

        win_rate = np.mean(win_rates) if win_rates else 0.0
        profit_factor = np.mean(profit_factors) if profit_factors else 1.0

        return {
            'success_rate': success_rate,
            'average_return': average_return,
            'median_return': median_return,
            'std_return': std_return,
            'max_return': max_return,
            'min_return': min_return,
            'confidence_interval_95': confidence_interval_95,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_avg': max_drawdown_avg,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }

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
            # Ensure timestamp is datetime
            if hasattr(idx, 'to_pydatetime'):
                ts = idx.to_pydatetime()  # type: ignore
            elif isinstance(idx, str):
                ts = datetime.fromisoformat(idx)
            else:
                ts = pd.to_datetime(idx).to_pydatetime()  # type: ignore

            candle = Candle(
                timestamp=ts,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume'])
            )
            candles.append(candle)
        return candles

    def generate_probability_distribution(self, results: MonteCarloResult) -> dict[str, Any]:
        """
        Generate probability distribution analysis

        Args:
            results: Monte Carlo results

        Returns:
            Probability distribution statistics
        """
        returns = [r['total_return'] for r in results.simulation_results]

        # Create histogram bins
        hist, bin_edges = np.histogram(returns, bins=50, density=True)

        # Calculate percentiles
        percentiles = {
            '1%': np.percentile(returns, 1),
            '5%': np.percentile(returns, 5),
            '10%': np.percentile(returns, 10),
            '25%': np.percentile(returns, 25),
            '50%': np.percentile(returns, 50),
            '75%': np.percentile(returns, 75),
            '90%': np.percentile(returns, 90),
            '95%': np.percentile(returns, 95),
            '99%': np.percentile(returns, 99)
        }

        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)  # 5th percentile for 95% VaR
        var_99 = np.percentile(returns, 1)  # 1st percentile for 99% VaR

        # Expected Shortfall (CVaR)
        cvar_95 = np.mean([r for r in returns if r <= var_95])
        cvar_99 = np.mean([r for r in returns if r <= var_99])

        return {
            'histogram': {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            },
            'percentiles': percentiles,
            'value_at_risk': {
                'var_95': var_95,
                'var_99': var_99
            },
            'conditional_var': {
                'cvar_95': cvar_95,
                'cvar_99': cvar_99
            },
            'distribution_stats': {
                'skewness': float(pd.Series(returns).skew()),  # type: ignore
                'kurtosis': float(pd.Series(returns).kurtosis())  # type: ignore
            }
        }


def create_monte_carlo_backtester(num_simulations: int = 1000) -> MonteCarloBacktester:
    """
    Factory function to create Monte Carlo backtester

    Args:
        num_simulations: Number of simulations to run

    Returns:
        Configured MonteCarloBacktester instance
    """
    return MonteCarloBacktester(num_simulations=num_simulations)


def run_monte_carlo_backtest(historical_data: list[Candle], strategy_function: Callable[[list[Candle], float, float], Any],
                           num_simulations: int = 1000) -> MonteCarloResult:
    """
    Convenience function to run Monte Carlo backtest

    Args:
        historical_data: Historical price data
        strategy_function: Trading strategy function
        num_simulations: Number of simulations

    Returns:
        Monte Carlo analysis results
    """
    backtester = MonteCarloBacktester(num_simulations=num_simulations)
    return backtester.run_monte_carlo_analysis(historical_data, strategy_function)
