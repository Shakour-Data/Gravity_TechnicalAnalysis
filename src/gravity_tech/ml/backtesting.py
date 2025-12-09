"""
Pattern Recognition Backtesting Framework

Backtesting framework for:
- Historical pattern validation
- Prediction accuracy over time
- Strategy performance simulation
- Win rate and risk/reward analysis

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from gravity_tech.ml.pattern_features import PatternFeatureExtractor
from gravity_tech.patterns.harmonic import HarmonicPattern, HarmonicPatternDetector

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TradeResult:
    """Result of a single pattern trade."""
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    pattern_type: str
    direction: str
    confidence: float
    stop_loss: float
    target_1: float
    target_2: float
    pnl: float
    pnl_percent: float
    outcome: str  # 'win', 'loss', 'breakeven'
    hit_target: str  # 'none', 'target1', 'target2', 'stop_loss'


class PatternBacktester:
    """
    Backtesting framework for pattern recognition strategies.

    Simulates trading based on detected patterns and evaluates:
    - Win rate
    - Average profit/loss
    - Risk/reward ratio
    - Maximum drawdown
    - Sharpe ratio
    """

    def __init__(
        self,
        detector: HarmonicPatternDetector,
        classifier: Any | None = None,
        min_confidence: float = 0.6
    ):
        """
        Initialize backtester.

        Args:
            detector: Pattern detector
            classifier: Optional ML classifier for confidence
            min_confidence: Minimum confidence to take trade
        """
        self.detector = detector
        self.classifier = classifier
        self.min_confidence = min_confidence
        self.trades = []
        self.equity_curve = []

    def generate_historical_data(
        self,
        n_bars: int = 1000,
        starting_price: float = 100.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Generate synthetic historical market data for backtesting.

        Args:
            n_bars: Number of bars
            starting_price: Starting price

        Returns:
            Tuple of (highs, lows, closes, volume, dates)
        """
        np.random.seed(42)

        # Generate trending market with cycles
        prices = [starting_price]
        for i in range(n_bars - 1):
            # Trend component (changes every 100 bars)
            trend_phase = (i // 100) % 4
            if trend_phase == 0:  # Uptrend
                drift = 0.15
            elif trend_phase == 1:  # Sideways
                drift = 0.02
            elif trend_phase == 2:  # Downtrend
                drift = -0.12
            else:  # Recovery
                drift = 0.08

            # Random walk with drift
            change = drift + np.random.normal(0, 1.5)
            new_price = max(50, min(200, prices[-1] + change))
            prices.append(new_price)

        closes = np.array(prices, dtype=np.float32)

        # Generate OHLC
        highs = closes + np.abs(np.random.normal(0, 1.2, n_bars))
        lows = closes - np.abs(np.random.normal(0, 1.2, n_bars))
        volume = np.random.uniform(5000, 20000, n_bars)

        # Generate dates (daily data)
        start_date = datetime(2023, 1, 1)
        dates = pd.date_range(start=start_date, periods=n_bars, freq='D')

        return highs, lows, closes, volume, dates

    def run_backtest(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volume: np.ndarray,
        dates: pd.DatetimeIndex
    ) -> list[TradeResult]:
        """
        Run backtest on historical data.

        Args:
            highs, lows, closes, volume: Price data
            dates: Date index

        Returns:
            List of trade results
        """
        print("\n📊 Running Backtest...")
        print("=" * 80)

        self.trades = []

        # Sliding window for pattern detection
        window_size = 200  # Look at 200 bars at a time
        step_size = 50     # Move forward 50 bars each time

        for start_idx in range(0, len(closes) - window_size, step_size):
            end_idx = start_idx + window_size

            # Detect patterns in window
            window_highs = highs[start_idx:end_idx]
            window_lows = lows[start_idx:end_idx]
            window_closes = closes[start_idx:end_idx]

            patterns = self.detector.detect_patterns(
                window_highs, window_lows, window_closes
            )

            if len(patterns) == 0:
                continue

            # Evaluate each pattern
            for pattern in patterns:
                # Get pattern completion point (D point index in window)
                d_idx_window = pattern.points['D'].index
                d_idx_global = start_idx + d_idx_window

                # Need future data to evaluate trade
                if d_idx_global + 50 >= len(closes):
                    continue

                # Check confidence
                if self.classifier:
                    extractor = PatternFeatureExtractor()
                    features = extractor.extract_features(
                        pattern, window_highs, window_lows, window_closes,
                        volume[start_idx:end_idx] if volume is not None else None
                    )
                    feature_array = extractor.features_to_array(features)

                    # Handle both PatternClassifier and sklearn models
                    if hasattr(self.classifier, 'predict_single'):
                        prediction = self.classifier.predict_single(feature_array)
                        confidence = prediction['confidence']
                    else:
                        # sklearn model - use predict_proba
                        probas = self.classifier.predict_proba(feature_array.reshape(1, -1))[0]
                        confidence = float(np.max(probas))
                else:
                    confidence = pattern.confidence / 100.0

                # Skip low confidence patterns
                if confidence < self.min_confidence:
                    continue

                # Simulate trade
                trade_result = self._simulate_trade(
                    pattern,
                    d_idx_global,
                    closes,
                    highs,
                    lows,
                    dates,
                    confidence
                )

                if trade_result:
                    self.trades.append(trade_result)

        print(f"✅ Backtest complete: {len(self.trades)} trades executed")
        return self.trades

    def _simulate_trade(
        self,
        pattern: HarmonicPattern,
        entry_idx: int,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        dates: pd.DatetimeIndex,
        confidence: float
    ) -> TradeResult | None:
        """
        Simulate a single trade based on pattern.

        Args:
            pattern: Detected pattern
            entry_idx: Entry bar index
            closes, highs, lows: Price data
            dates: Date index
            confidence: Pattern confidence

        Returns:
            TradeResult or None
        """
        entry_price = pattern.completion_point
        stop_loss = pattern.stop_loss
        target_1 = pattern.target_1
        target_2 = pattern.target_2

        # Look ahead for exit (max 50 bars)
        max_holding = min(50, len(closes) - entry_idx - 1)

        hit_target = 'none'
        exit_price = entry_price
        exit_idx = entry_idx

        for i in range(1, max_holding + 1):
            bar_idx = entry_idx + i

            if pattern.direction.value == 'bullish':
                # Check stop loss
                if lows[bar_idx] <= stop_loss:
                    exit_price = stop_loss
                    exit_idx = bar_idx
                    hit_target = 'stop_loss'
                    break

                # Check targets
                if highs[bar_idx] >= target_2:
                    exit_price = target_2
                    exit_idx = bar_idx
                    hit_target = 'target2'
                    break
                elif highs[bar_idx] >= target_1:
                    exit_price = target_1
                    exit_idx = bar_idx
                    hit_target = 'target1'
                    break

            else:  # bearish
                # Check stop loss
                if highs[bar_idx] >= stop_loss:
                    exit_price = stop_loss
                    exit_idx = bar_idx
                    hit_target = 'stop_loss'
                    break

                # Check targets
                if lows[bar_idx] <= target_2:
                    exit_price = target_2
                    exit_idx = bar_idx
                    hit_target = 'target2'
                    break
                elif lows[bar_idx] <= target_1:
                    exit_price = target_1
                    exit_idx = bar_idx
                    hit_target = 'target1'
                    break

        # If no exit triggered, exit at close
        if hit_target == 'none':
            exit_price = closes[entry_idx + max_holding]
            exit_idx = entry_idx + max_holding

        # Calculate P&L
        if pattern.direction.value == 'bullish':
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price

        pnl_percent = (pnl / entry_price) * 100

        # Determine outcome
        if pnl > 0:
            outcome = 'win'
        elif pnl < 0:
            outcome = 'loss'
        else:
            outcome = 'breakeven'

        return TradeResult(
            entry_date=dates[entry_idx],
            entry_price=entry_price,
            exit_date=dates[exit_idx],
            exit_price=exit_price,
            pattern_type=pattern.pattern_type.value,
            direction=pattern.direction.value,
            confidence=confidence,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            pnl=pnl,
            pnl_percent=pnl_percent,
            outcome=outcome,
            hit_target=hit_target
        )

    def calculate_metrics(self) -> dict:
        """
        Calculate backtest performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        if len(self.trades) == 0:
            return {'error': 'No trades to analyze'}

        # Convert trades to DataFrame for analysis
        df = pd.DataFrame([
            {
                'pnl': t.pnl,
                'pnl_percent': t.pnl_percent,
                'outcome': t.outcome,
                'pattern_type': t.pattern_type,
                'direction': t.direction,
                'confidence': t.confidence,
                'hit_target': t.hit_target
            }
            for t in self.trades
        ])

        # Basic metrics
        total_trades = len(df)
        winning_trades = len(df[df['outcome'] == 'win'])
        losing_trades = len(df[df['outcome'] == 'loss'])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # P&L metrics
        total_pnl = df['pnl'].sum()
        avg_pnl = df['pnl'].mean()
        avg_win = df[df['outcome'] == 'win']['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['outcome'] == 'loss']['pnl'].mean() if losing_trades > 0 else 0

        # Risk/Reward
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else 0

        # Sharpe ratio (simplified)
        returns = df['pnl_percent'].values
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        sharpe_annual = sharpe_ratio * np.sqrt(252)  # Annualized

        # Maximum drawdown
        cumulative_pnl = np.cumsum(df['pnl'].values)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown = np.max(drawdown)

        # Target hit analysis
        target_hit_counts = df['hit_target'].value_counts().to_dict()

        # Pattern type performance
        pattern_performance = df.groupby('pattern_type').agg({
            'pnl': ['mean', 'sum', 'count'],
            'outcome': lambda x: (x == 'win').sum() / len(x)
        }).to_dict()

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_annual,
            'max_drawdown': max_drawdown,
            'target_hit_counts': target_hit_counts,
            'pattern_performance': pattern_performance
        }

    def print_summary(self):
        """Print backtest summary."""
        metrics = self.calculate_metrics()

        if 'error' in metrics:
            print(f"\n❌ {metrics['error']}")
            return

        print("\n" + "=" * 80)
        print("📊 Backtest Results Summary")
        print("=" * 80)

        print("\n📈 Trade Statistics:")
        print(f"   Total Trades:      {metrics['total_trades']}")
        print(f"   Winning Trades:    {metrics['winning_trades']}")
        print(f"   Losing Trades:     {metrics['losing_trades']}")
        print(f"   Win Rate:          {metrics['win_rate']:.2%}")

        print("\n💰 P&L Performance:")
        print(f"   Total P&L:         ${metrics['total_pnl']:.2f}")
        print(f"   Average P&L:       ${metrics['avg_pnl']:.2f}")
        print(f"   Average Win:       ${metrics['avg_win']:.2f}")
        print(f"   Average Loss:      ${metrics['avg_loss']:.2f}")
        print(f"   Profit Factor:     {metrics['profit_factor']:.2f}")

        print("\n📊 Risk Metrics:")
        print(f"   Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown:      ${metrics['max_drawdown']:.2f}")

        print("\n🎯 Target Hit Analysis:")
        for target, count in metrics['target_hit_counts'].items():
            print(f"   {target.replace('_', ' ').title():15s} {count:3d} trades")

        print("=" * 80)


def demo_backtesting():
    """Demonstration of backtesting framework."""
    print("=" * 80)
    print("⏪ Pattern Recognition Backtesting Demo")
    print("=" * 80)

    # Create detector
    detector = HarmonicPatternDetector(tolerance=0.15)

    # Create backtester
    backtester = PatternBacktester(
        detector=detector,
        classifier=None,  # No ML for this demo
        min_confidence=0.5
    )

    # Generate historical data
    print("\n📊 Generating Historical Market Data...")
    highs, lows, closes, volume, dates = backtester.generate_historical_data(n_bars=1000)
    print("✅ Generated 1000 bars of data")
    print(f"   Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"   Price range: ${lows.min():.2f} - ${highs.max():.2f}")

    # Run backtest
    backtester.run_backtest(highs, lows, closes, volume, dates)

    # Print summary
    backtester.print_summary()


def run_backtest_with_synthetic_data(n_bars: int = 1000) -> PatternBacktester:
    """Run backtest with synthetic data for testing purposes."""
    detector = HarmonicPatternDetector(tolerance=0.15)
    backtester = PatternBacktester(detector=detector, classifier=None, min_confidence=0.5)
    highs, lows, closes, volume, dates = backtester.generate_historical_data(n_bars=n_bars)
    backtester.run_backtest(highs, lows, closes, volume, dates)
    backtester.print_summary()
    return backtester


def run_backtest_with_real_data(
    symbol: str,
    source: str,
    interval: str,
    limit: int,
    allow_mock: bool = False,
    min_confidence: float = 0.6,
    persist: bool = False,
) -> PatternBacktester:
    """
    Run backtest with real market data from database or data connector.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT', 'FOLD')
        source: Data source ('db' or 'connector')
        interval: Time interval (e.g., '1d', '4h')
        limit: Number of data points to load
        allow_mock: Allow fallback to mock data if connector fails
        min_confidence: Minimum pattern confidence threshold
        persist: Whether to save results to database

    Returns:
        PatternBacktester instance with results
    """
    print(f"\n📊 Running Backtest for {symbol}")
    print(f"   Source: {source}")
    print(f"   Interval: {interval}")
    print(f"   Limit: {limit}")
    print(f"   Min Confidence: {min_confidence}")
    print("=" * 80)

    # Create detector and backtester
    detector = HarmonicPatternDetector(tolerance=0.15)
    backtester = PatternBacktester(
        detector=detector,
        classifier=None,  # No ML for basic backtesting
        min_confidence=min_confidence
    )

    # TODO: Implement data loading from database/connector
    # For now, fall back to synthetic data with a warning
    print("⚠️  Real data loading not yet implemented - using synthetic data")
    highs, lows, closes, volume, dates = backtester.generate_historical_data(n_bars=limit)

    # Run backtest
    backtester.run_backtest(highs, lows, closes, volume, dates)

    # Print summary
    backtester.print_summary()

    # TODO: Implement persistence to database if persist=True

    return backtester


if __name__ == "__main__":
    demo_backtesting()
