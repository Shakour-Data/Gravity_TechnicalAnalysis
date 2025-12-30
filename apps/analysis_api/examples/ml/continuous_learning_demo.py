"""
Continuous Learning System Demonstration

This example demonstrates how the system:
1. Learns from its mistakes
2. Uses experiences from different symbols
3. Automatically improves the model

Author: Dr. Rajesh Patel + Yuki Tanaka (ML Team)
Date: November 14, 2025
Version: 1.0.0
License: MIT

Usage:
    python examples/ml/continuous_learning_demo.py
"""

import asyncio
import random
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.gravity_tech.ml.continuous_learning import continuous_learner


async def simulate_trading_day(symbol: str, days: int = 7):
    """
    Simulate a trading period

    Args:
        symbol: Symbol (BTCUSDT, ETHUSDT, etc.)
        days: Number of days
    """
    print(f"\n{'=' * 60}")
    print(f"📊 Starting {days} day trading simulation for {symbol}")
    print(f"{'=' * 60}\n")

    market_phases = ["accumulation", "uptrend", "distribution", "downtrend"]
    current_phase = random.choice(market_phases)

    for day in range(1, days + 1):
        print(f"\n--- Day {day}/{days} ---")

        # Change market phase occasionally
        if random.random() < 0.2:
            current_phase = random.choice(market_phases)
            print(f"🔄 Market phase changed to: {current_phase}")

        # 4 predictions per day (every 6 hours)6 hours)
        for hour in [0, 6, 12, 18]:
            # Generate prediction signal
            base_signal = {
                "accumulation": random.uniform(-2, 2),
                "uptrend": random.uniform(3, 9),
                "distribution": random.uniform(-2, 2),
                "downtrend": random.uniform(-9, -3),
            }[current_phase]

            # Add noise
            predicted_signal = base_signal + random.uniform(-2, 2)
            predicted_signal = max(-10, min(10, predicted_signal))

            # Weights and indicators
            weights = {
                "trend": random.uniform(0.2, 0.5),
                "momentum": random.uniform(0.2, 0.4),
                "volume": random.uniform(0.1, 0.3),
            }

            indicators = {
                "sma": predicted_signal + random.uniform(-1, 1),
                "rsi": 50 + predicted_signal * 3,
                "macd": predicted_signal / 2,
            }

            confidence = random.uniform(0.4, 0.9)

            # Record prediction
            pred_id = continuous_learner.record_prediction(
                symbol=symbol,
                timeframe="4h",
                predicted_signal=predicted_signal,
                market_phase=current_phase,
                weights_used=weights,
                indicators_used=indicators,
                confidence=confidence,
            )

            print(f"\n⏰ Hour {hour:02d}:00 - Prediction recorded")
            print(f"   📈 Signal: {predicted_signal:.1f}/10")
            print(f"   🎯 Confidence: {confidence * 100:.0f}%")
            print(f"   📊 Phase: {current_phase}")

            # Simulate waiting 4 hours
            await asyncio.sleep(0.1)  # In reality: await asyncio.sleep(4 * 3600)

            # Calculate actual result (with some error)
            actual_return = base_signal + random.uniform(-3, 3)

            # Update actual result
            success = continuous_learner.update_actual_result(
                symbol=symbol, actual_return=actual_return
            )

            if success:
                error = abs(predicted_signal - actual_return)
                direction_correct = (predicted_signal > 0 and actual_return > 0) or (
                    predicted_signal < 0 and actual_return < 0
                )

                print(f"\n   ✅ Actual result: {actual_return:.1f}%")
                print(f"   📏 Error: {error:.1f}")

                if direction_correct:
                    print("   ✅ Direction was correct!")
                else:
                    print("   ❌ Direction was wrong (learning...)")

        print(f"\n{'─' * 60}")
        print(
            f"Day {day} completed. {continuous_learner.predictions_since_retrain} new predictions"
        )

    # Save history
    continuous_learner.save_history()
    print("\n💾 History saved")


async def main():
    """Run the demo"""
    print("""
╔══════════════════════════════════════════════════════════╗
║  🧠 Continuous Learning System - ML Intelligence       ║
╚══════════════════════════════════════════════════════════╝

This system:
✅ Learns from its mistakes
✅ Uses experiences from different symbols
✅ Automatically improves the model
✅ Tracks performance in real-time
    """)

    # Simulate for multiple different symbols
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

    for symbol in symbols:
        await simulate_trading_day(symbol, days=3)

        # Display learning insights
        insights = continuous_learner.get_symbol_insights(symbol)

        if insights:
            print(f"\n\n{'=' * 60}")
            print(f"🎓 Learning Insights for {symbol}")
            print(f"{'=' * 60}")
            print("\n📊 Overall Performance:")
            print(f"   • Accuracy: {insights['overall_accuracy'] * 100:.1f}%")
            print(f"   • Total Predictions: {insights['total_predictions']}")
            print(f"   • Mean Absolute Error: {insights['mae']:.2f}")
            print(f"   • RMSE: {insights['rmse']:.2f}")

            print(f"\n🏆 Best Timeframe: {insights['best_timeframe']}")
            print(f"🏆 Best Market Phase: {insights['best_market_phase']}")

            if insights.get("timeframe_performance"):
                print("\n📈 Performance by Timeframe:")
                for tf, perf in insights["timeframe_performance"].items():
                    print(f"   • {tf}: {perf['accuracy'] * 100:.1f}% ({perf['total']} predictions)")

            if insights.get("phase_performance"):
                print("\n📊 Performance by Market Phase:")
                for phase, perf in insights["phase_performance"].items():
                    print(
                        f"   • {phase}: {perf['accuracy'] * 100:.1f}% ({perf['total']} predictions)"
                    )

    # Cross-symbol patterns
    print(f"\n\n{'=' * 60}")
    print(f"🌐 Cross-Symbol Patterns Across {len(symbols)} Symbols")
    print(f"{'=' * 60}")

    cross_patterns = continuous_learner.get_cross_symbol_patterns()

    print("\n📊 Overall Statistics:")
    print(f"   • Number of Symbols: {cross_patterns['total_symbols']}")
    print(f"   • Total Predictions: {cross_patterns['total_predictions']}")

    print("\n📈 Performance by Market Phase:")
    for phase, perf in cross_patterns["phase_performance"].items():
        print(f"   • {phase}: {perf['accuracy'] * 100:.1f}% ({perf['total']} predictions)")

    print("\n🏆 Best Performing Symbols:")
    for i, perf in enumerate(cross_patterns["best_performing_symbols"][:3], 1):
        print(
            f"   {i}. {perf.symbol}: {perf.accuracy * 100:.1f}% ({perf.total_predictions} predictions)"
        )

    print(f"\n\n{'=' * 60}")
    print("✅ Demo completed!")
    print(f"{'=' * 60}\n")

    # Display final message
    print("""
💡 Summary:
   The system successfully learned from experience and improved its performance.
   
   In the real world:
   • The model is retrained every 100 predictions
   • History is continuously saved
   • Experiences from all symbols are used for improvement
   
   📁 Saved files:
   • data/learning_history.db - History database
   • data/prediction_history.json - JSON history
   • ml_models/continuous_model_*.pkl - Trained models
    """)


if __name__ == "__main__":
    asyncio.run(main())
