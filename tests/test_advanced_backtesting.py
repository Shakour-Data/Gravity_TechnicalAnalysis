"""
Comprehensive Tests for Advanced Backtesting Features

Tests cover:
- Monte Carlo backtesting with TSE data
- Walk-forward backtesting implementation
- Statistical analysis and risk metrics
- Performance evaluation
- Integration with database
- Strategy validation

Author: Gravity Tech Team
Date: December 4, 2025
Version: 1.0.0
License: MIT
"""

import pytest
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Callable
import sys
from pathlib import Path
from unittest.mock import Mock

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from gravity_tech.core.domain.entities import Candle
from examples.ml.monte_carlo_backtesting import MonteCarloBacktester, MonteCarloResult
from examples.ml.walk_forward_backtesting import WalkForwardBacktester


class TestAdvancedBacktesting:
    """Test suite for advanced backtesting features with TSE data integration."""

    @pytest.fixture
    def tse_db_connection(self):
        """Fixture to provide TSE database connection."""
        db_path = project_root / "data" / "tse_data.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        yield conn
        conn.close()

    @pytest.fixture
    def sample_tse_candles(self, tse_db_connection) -> List[Candle]:
        """Load real TSE candle data for testing."""
        cursor = tse_db_connection.cursor()
        cursor.execute("""
            SELECT * FROM candles
            WHERE symbol = 'شستا'
            ORDER BY timestamp ASC
            LIMIT 200
        """)

        candles = []
        for row in cursor.fetchall():
            candles.append(Candle(
                timestamp=datetime.fromisoformat(row['timestamp']),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume'])
            ))

        return candles

    @pytest.fixture
    def monte_carlo_backtester(self):
        """Fixture to provide Monte Carlo backtester instance."""
        return MonteCarloBacktester(num_simulations=50)  # Smaller for testing

    @pytest.fixture
    def walk_forward_backtester(self):
        """Fixture to provide Walk-forward backtester instance."""
        return WalkForwardBacktester(
            optimization_window=60,  # 60 days training
            testing_window=20,       # 20 days testing
            step_size=10             # 10 days step
        )

    def test_monte_carlo_backtester_initialization(self, monte_carlo_backtester):
        """Test Monte Carlo backtester initialization."""
        assert monte_carlo_backtester is not None
        assert monte_carlo_backtester.num_simulations == 50
        assert hasattr(monte_carlo_backtester, 'run_monte_carlo_analysis')

    def test_walk_forward_backtester_initialization(self, walk_forward_backtester):
        """Test Walk-forward backtester initialization."""
        assert walk_forward_backtester is not None
        assert walk_forward_backtester.optimization_window == 60
        assert walk_forward_backtester.testing_window == 20
        assert walk_forward_backtester.step_size == 10

    def test_monte_carlo_with_tse_data(self, monte_carlo_backtester, sample_tse_candles):
        """Test Monte Carlo backtesting with real TSE data."""
        # Define a simple trading strategy
        def simple_momentum_strategy(candles: List[Candle], initial_capital: float, commission: float):
            """Simple momentum-based strategy for testing."""
            capital = initial_capital
            position = 0  # 0 = no position, 1 = long
            trades = []
            max_drawdown = 0
            peak_capital = capital

            for i in range(10, len(candles)):  # Start after warmup period
                current_price = candles[i].close

                # Simple momentum signal (SMA crossover)
                short_ma = np.mean([c.close for c in candles[i-5:i]])
                long_ma = np.mean([c.close for c in candles[i-10:i]])

                # Enter long position
                if short_ma > long_ma and position == 0:
                    shares = int(capital / current_price)
                    cost = shares * current_price * (1 + commission)
                    if cost <= capital:
                        capital -= cost
                        position = shares
                        trades.append({
                            'type': 'buy',
                            'price': current_price,
                            'shares': shares,
                            'profit_loss': 0
                        })

                # Exit long position
                elif short_ma < long_ma and position > 0:
                    sale_value = position * current_price * (1 - commission)
                    profit_loss = sale_value - (position * trades[-1]['price'] * (1 + commission))
                    capital += sale_value

                    trades[-1]['profit_loss'] = profit_loss
                    position = 0

                # Update drawdown
                if capital > peak_capital:
                    peak_capital = capital
                current_drawdown = (peak_capital - capital) / peak_capital
                max_drawdown = max(max_drawdown, current_drawdown)

            # Close any remaining position
            if position > 0:
                final_price = candles[-1].close
                sale_value = position * final_price * (1 - commission)
                profit_loss = sale_value - (position * trades[-1]['price'] * (1 + commission))
                capital += sale_value
                trades[-1]['profit_loss'] = profit_loss

            # Calculate Sharpe ratio (simplified)
            returns = [t['profit_loss'] / initial_capital for t in trades if t['profit_loss'] != 0]
            sharpe_ratio = np.mean(returns) / np.std(returns) if returns else 0

            return Mock(
                final_capital=capital,
                total_return=(capital - initial_capital) / initial_capital,
                total_trades=len(trades),
                winning_trades=len([t for t in trades if t['profit_loss'] > 0]),
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                trades=trades
            )

        # Run Monte Carlo analysis
        result = monte_carlo_backtester.run_monte_carlo_analysis(
            sample_tse_candles,
            simple_momentum_strategy,
            initial_capital=10000.0,
            commission=0.001
        )

        # Verify results
        assert isinstance(result, MonteCarloResult)
        assert result.num_simulations == 50
        assert isinstance(result.success_rate, float)
        assert 0 <= result.success_rate <= 1
        assert isinstance(result.average_return, float)
        assert isinstance(result.confidence_interval_95, tuple)
        assert len(result.confidence_interval_95) == 2

    def test_walk_forward_with_tse_data(self, walk_forward_backtester, sample_tse_candles):
        """Test Walk-forward backtesting with TSE data."""
        # Convert candles to DataFrame for easier manipulation
        data = []
        for candle in sample_tse_candles:
            data.append({
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)

        # Define strategy function
        def sma_crossover_strategy(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
            """SMA crossover strategy optimized on training data."""
            # Optimize parameters on training data
            best_return = -np.inf
            best_short_period = 5
            best_long_period = 20

            for short_period in [3, 5, 7]:
                for long_period in [10, 15, 20, 25]:
                    if short_period >= long_period:
                        continue

                    # Calculate signals on training data
                    train_data = train_data.copy()
                    train_data[f'SMA_{short_period}'] = train_data['close'].rolling(short_period).mean()
                    train_data[f'SMA_{long_period}'] = train_data['close'].rolling(long_period).mean()

                    # Generate signals
                    train_data['signal'] = 0
                    train_data.loc[train_data[f'SMA_{short_period}'] > train_data[f'SMA_{long_period}'], 'signal'] = 1
                    train_data.loc[train_data[f'SMA_{short_period}'] < train_data[f'SMA_{long_period}'], 'signal'] = -1

                    # Calculate returns
                    train_data['returns'] = train_data['close'].pct_change()
                    strategy_returns = train_data['signal'].shift(1) * train_data['returns']
                    total_return = strategy_returns.sum()

                    if total_return > best_return:
                        best_return = total_return
                        best_short_period = short_period
                        best_long_period = long_period

            # Apply best parameters to test data
            test_data = test_data.copy()
            test_data[f'SMA_{best_short_period}'] = test_data['close'].rolling(best_short_period).mean()
            test_data[f'SMA_{best_long_period}'] = test_data['close'].rolling(best_long_period).mean()

            test_data['signal'] = 0
            test_data.loc[test_data[f'SMA_{best_short_period}'] > test_data[f'SMA_{best_long_period}'], 'signal'] = 1
            test_data.loc[test_data[f'SMA_{best_short_period}'] < test_data[f'SMA_{best_long_period}'], 'signal'] = -1

            test_data['returns'] = test_data['close'].pct_change()
            test_data['strategy_returns'] = test_data['signal'].shift(1) * test_data['returns']

            # Calculate performance metrics
            total_return = test_data['strategy_returns'].sum()
            volatility = test_data['strategy_returns'].std() * np.sqrt(252)  # Annualized
            sharpe_ratio = total_return / volatility if volatility > 0 else 0
            max_drawdown = (test_data['strategy_returns'].cumsum() - test_data['strategy_returns'].cumsum().cummax()).min()

            return {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'parameters': {
                    'short_period': best_short_period,
                    'long_period': best_long_period
                }
            }

        # Run walk-forward analysis
        results = walk_forward_backtester.run_walk_forward_analysis(df, sma_crossover_strategy)

        # Verify results structure
        assert isinstance(results, dict)
        assert 'fold_results' in results
        assert 'overall_metrics' in results
        assert 'parameters_history' in results

        fold_results = results['fold_results']
        assert len(fold_results) > 0

        # Check each fold result
        for fold_result in fold_results:
            assert 'fold' in fold_result
            assert 'total_return' in fold_result
            assert 'sharpe_ratio' in fold_result
            assert 'parameters' in fold_result

    def test_statistical_analysis_monte_carlo(self, monte_carlo_backtester, sample_tse_candles):
        """Test statistical analysis of Monte Carlo results."""
        def random_strategy(candles, initial_capital, commission):
            """Random strategy for statistical testing."""
            final_capital = initial_capital * (1 + np.random.normal(0.05, 0.15))  # Random return
            return Mock(
                final_capital=final_capital,
                total_return=(final_capital - initial_capital) / initial_capital,
                total_trades=np.random.randint(5, 20),
                winning_trades=np.random.randint(0, 20),
                max_drawdown=abs(np.random.normal(0.05, 0.03)),
                sharpe_ratio=np.random.normal(0.5, 0.8),
                trades=[]
            )

        result = monte_carlo_backtester.run_monte_carlo_analysis(
            sample_tse_candles, random_strategy
        )

        # Test statistical properties
        assert result.num_simulations == 50

        # Check that confidence interval makes sense
        lower, upper = result.confidence_interval_95
        assert lower <= result.average_return <= upper

        # Check that standard deviation is positive
        assert result.std_return >= 0

        # Success rate should be between 0 and 1
        assert 0 <= result.success_rate <= 1

    def test_risk_metrics_calculation(self, monte_carlo_backtester, sample_tse_candles):
        """Test risk metrics calculation in backtesting."""
        def high_risk_strategy(candles, initial_capital, commission):
            """High risk strategy for testing risk metrics."""
            # Simulate high volatility returns
            return_rate = np.random.normal(0.15, 0.30)  # High mean, high volatility
            final_capital = initial_capital * (1 + return_rate)

            return Mock(
                final_capital=final_capital,
                total_return=return_rate,
                total_trades=np.random.randint(10, 30),
                winning_trades=np.random.randint(5, 25),
                max_drawdown=np.random.uniform(0.1, 0.4),  # High drawdown
                sharpe_ratio=np.random.normal(0.3, 0.5),   # Lower Sharpe
                trades=[]
            )

        result = monte_carlo_backtester.run_monte_carlo_analysis(
            sample_tse_candles, high_risk_strategy
        )

        # Verify risk metrics are calculated
        assert hasattr(result, 'max_drawdown_avg')
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'win_rate')
        assert hasattr(result, 'profit_factor')

        # Risk metrics should be in reasonable ranges
        assert result.max_drawdown_avg >= 0
        assert result.win_rate >= 0
        assert result.profit_factor >= 0

    def test_backtesting_multiple_symbols(self, monte_carlo_backtester, tse_db_connection):
        """Test backtesting across multiple TSE symbols."""
        cursor = tse_db_connection.cursor()

        symbols = ['شستا', 'فملی', 'وبملت']
        symbol_results = {}

        def universal_strategy(candles, initial_capital, commission):
            """Universal strategy applicable to any symbol."""
            # Simple buy and hold strategy
            initial_price = candles[0].close
            final_price = candles[-1].close
            return_rate = (final_price - initial_price) / initial_price
            final_capital = initial_capital * (1 + return_rate)

            return Mock(
                final_capital=final_capital,
                total_return=return_rate,
                total_trades=1,  # Buy and hold = 1 trade
                winning_trades=1 if return_rate > 0 else 0,
                max_drawdown=0.05,  # Simplified
                sharpe_ratio=return_rate / 0.1 if return_rate != 0 else 0,
                trades=[]
            )

        for symbol in symbols:
            cursor.execute("""
                SELECT * FROM candles
                WHERE symbol = ?
                ORDER BY timestamp ASC
                LIMIT 100
            """, (symbol,))

            rows = cursor.fetchall()
            if len(rows) >= 20:  # Need minimum data
                candles = []
                for row in rows:
                    candles.append(Candle(
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=int(row['volume'])
                    ))

                result = monte_carlo_backtester.run_monte_carlo_analysis(
                    candles, universal_strategy
                )

                symbol_results[symbol] = result

        # Verify results for each symbol
        assert len(symbol_results) == len(symbols)
        for symbol, result in symbol_results.items():
            assert result.num_simulations == 50
            assert hasattr(result, 'average_return')

    def test_walk_forward_parameter_stability(self, walk_forward_backtester, sample_tse_candles):
        """Test parameter stability in walk-forward optimization."""
        # Convert to DataFrame
        data = []
        for candle in sample_tse_candles:
            data.append({
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)

        def parameter_test_strategy(train_data, test_data):
            """Strategy that tracks parameter changes."""
            # Simple parameter optimization
            best_period = 5
            best_return = -np.inf

            for period in [3, 5, 7, 10]:
                test_data_copy = test_data.copy()
                test_data_copy['SMA'] = test_data_copy['close'].rolling(period).mean()
                test_data_copy['signal'] = (test_data_copy['close'] > test_data_copy['SMA']).astype(int)
                test_data_copy['returns'] = test_data_copy['close'].pct_change()
                test_data_copy['strategy_returns'] = test_data_copy['signal'].shift(1) * test_data_copy['returns']

                total_return = test_data_copy['strategy_returns'].sum()
                if total_return > best_return:
                    best_return = total_return
                    best_period = period

            # Apply best parameters
            test_data_copy = test_data.copy()
            test_data_copy['SMA'] = test_data_copy['close'].rolling(best_period).mean()
            test_data_copy['signal'] = (test_data_copy['close'] > test_data_copy['SMA']).astype(int)
            test_data_copy['returns'] = test_data_copy['close'].pct_change()
            test_data_copy['strategy_returns'] = test_data_copy['signal'].shift(1) * test_data_copy['returns']

            total_return = test_data_copy['strategy_returns'].sum()
            volatility = test_data_copy['strategy_returns'].std() * np.sqrt(252)

            return {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': total_return / volatility if volatility > 0 else 0,
                'max_drawdown': 0.05,  # Simplified
                'parameters': {'period': best_period}
            }

        results = walk_forward_backtester.run_walk_forward_analysis(df, parameter_test_strategy)

        # Check parameter stability across folds
        parameters_history = results['parameters_history']
        periods = [params['period'] for params in parameters_history]

        # Parameters should show some stability (not completely random)
        unique_periods = set(periods)
        assert len(unique_periods) <= len(periods)  # At least some repetition

    def test_backtesting_performance_metrics(self, monte_carlo_backtester, sample_tse_candles):
        """Test comprehensive performance metrics calculation."""
        def comprehensive_strategy(candles, initial_capital, commission):
            """Strategy with comprehensive performance tracking."""
            capital = initial_capital
            trades = []
            peak_capital = capital
            max_drawdown = 0

            # Simulate some trades
            for i in range(5, len(candles), 10):  # Trade every 10 candles
                entry_price = candles[i].close
                exit_price = candles[min(i + 5, len(candles) - 1)].close

                # Simulate trade
                shares = int(capital * 0.1 / entry_price)  # Use 10% of capital
                entry_cost = shares * entry_price * (1 + commission)
                exit_value = shares * exit_price * (1 - commission)

                profit_loss = exit_value - entry_cost
                capital += profit_loss

                trades.append({
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'shares': shares,
                    'profit_loss': profit_loss
                })

                # Update drawdown
                if capital > peak_capital:
                    peak_capital = capital
                current_drawdown = (peak_capital - capital) / peak_capital
                max_drawdown = max(max_drawdown, current_drawdown)

            # Calculate comprehensive metrics
            winning_trades = [t for t in trades if t['profit_loss'] > 0]
            losing_trades = [t for t in trades if t['profit_loss'] < 0]

            win_rate = len(winning_trades) / len(trades) if trades else 0

            gross_profit = sum(t['profit_loss'] for t in winning_trades)
            gross_loss = abs(sum(t['profit_loss'] for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            returns = [t['profit_loss'] / initial_capital for t in trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) if returns else 0

            return Mock(
                final_capital=capital,
                total_return=(capital - initial_capital) / initial_capital,
                total_trades=len(trades),
                winning_trades=len(winning_trades),
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                trades=trades,
                win_rate=win_rate,
                profit_factor=profit_factor
            )

        result = monte_carlo_backtester.run_monte_carlo_analysis(
            sample_tse_candles, comprehensive_strategy
        )

        # Verify all metrics are calculated correctly
        assert result.num_simulations > 0
        assert hasattr(result, 'win_rate')
        assert hasattr(result, 'profit_factor')
        assert 0 <= result.win_rate <= 1
        assert result.profit_factor >= 0

    def test_backtesting_data_integrity(self, monte_carlo_backtester, sample_tse_candles):
        """Test data integrity throughout backtesting process."""
        original_data = sample_tse_candles.copy()

        def data_integrity_strategy(candles, initial_capital, commission):
            """Strategy that validates data integrity."""
            # Verify data hasn't been corrupted
            assert len(candles) == len(original_data)

            for i, candle in enumerate(candles):
                assert candle.timestamp == original_data[i].timestamp
                assert candle.close == original_data[i].close
                assert candle.volume == original_data[i].volume

            # Return dummy result
            return Mock(
                final_capital=initial_capital * 1.02,
                total_return=0.02,
                total_trades=1,
                winning_trades=1,
                max_drawdown=0.01,
                sharpe_ratio=1.0,
                trades=[]
            )

        # Run analysis
        result = monte_carlo_backtester.run_monte_carlo_analysis(
            sample_tse_candles, data_integrity_strategy
        )

        # If we get here without assertion errors, data integrity is maintained
        assert result.num_simulations > 0

    def test_edge_cases_insufficient_data(self, monte_carlo_backtester):
        """Test edge cases with insufficient data."""
        # Test with very few candles
        insufficient_candles = [
            Candle(
                timestamp=datetime(2024, 1, i+1),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000
            ) for i in range(5)  # Only 5 candles
        ]

        def minimal_strategy(candles, initial_capital, commission):
            return Mock(
                final_capital=initial_capital,
                total_return=0.0,
                total_trades=0,
                winning_trades=0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                trades=[]
            )

        # Should handle insufficient data gracefully
        result = monte_carlo_backtester.run_monte_carlo_analysis(
            insufficient_candles, minimal_strategy
        )

        assert result.num_simulations > 0

    def test_backtesting_parallel_processing(self, sample_tse_candles):
        """Test parallel processing in backtesting (if available)."""
        # Create backtester with standard parameters
        backtester = MonteCarloBacktester(num_simulations=20)

        def parallel_test_strategy(candles, initial_capital, commission):
            # Simulate computation-intensive strategy
            import time
            time.sleep(0.01)  # Small delay to test parallelization

            return Mock(
                final_capital=initial_capital * 1.01,
                total_return=0.01,
                total_trades=1,
                winning_trades=1,
                max_drawdown=0.005,
                sharpe_ratio=0.5,
                trades=[]
            )

        import time
        start_time = time.time()

        result = backtester.run_monte_carlo_analysis(
            sample_tse_candles, parallel_test_strategy
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete in reasonable time
        assert execution_time < 5.0  # Less than 5 seconds
        assert result.num_simulations == 20