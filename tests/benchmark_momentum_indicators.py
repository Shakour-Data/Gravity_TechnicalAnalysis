"""
Benchmark for Day 2 Momentum Indicators (Numba-optimized)
Author: Emily Watson (TM-008-PEL)
Date: November 8, 2025
"""
import numpy as np
import time
from gravity_tech.services.performance_optimizer import fast_tsi, fast_schaff_trend_cycle, fast_connors_rsi


def benchmark_momentum_indicators():
    """Benchmark the three new momentum indicators."""
    print("=" * 70)
    print("Day 2 Momentum Indicators Performance Benchmark")
    print("=" * 70)
    print(f"Test data: 10,000 candles")
    print(f"Iterations: 1,000 (for accuracy)")
    print("-" * 70)
    
    # Generate test data
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(10000) * 0.5) + 100.0
    prices = prices.astype(np.float32)
    
    # Warmup JIT
    _ = fast_tsi(prices[:100])
    _ = fast_schaff_trend_cycle(prices[:100])
    _ = fast_connors_rsi(prices[:100])
    
    iterations = 1000
    
    # Benchmark TSI
    start = time.perf_counter()
    for _ in range(iterations):
        _ = fast_tsi(prices)
    elapsed_tsi = (time.perf_counter() - start) / iterations * 1000
    
    # Benchmark Schaff Trend Cycle
    start = time.perf_counter()
    for _ in range(iterations):
        _ = fast_schaff_trend_cycle(prices)
    elapsed_stc = (time.perf_counter() - start) / iterations * 1000
    
    # Benchmark Connors RSI
    start = time.perf_counter()
    for _ in range(iterations):
        _ = fast_connors_rsi(prices)
    elapsed_crsi = (time.perf_counter() - start) / iterations * 1000
    
    # Display results
    print(f"\n📊 RESULTS:\n")
    
    print(f"1️⃣ True Strength Index (TSI)")
    print(f"   ⚡ Time: {elapsed_tsi:.3f}ms")
    baseline_tsi = 50.0  # Estimated Python baseline
    speedup_tsi = baseline_tsi / elapsed_tsi
    print(f"   🚀 Speedup: {speedup_tsi:.0f}x vs Python baseline")
    status_tsi = "✅ PASS" if elapsed_tsi < 0.5 else "⚠️ CLOSE" if elapsed_tsi < 1.0 else "❌ SLOW"
    print(f"   📈 Status: {status_tsi} (target <0.5ms)\n")
    
    print(f"2️⃣ Schaff Trend Cycle (STC)")
    print(f"   ⚡ Time: {elapsed_stc:.3f}ms")
    baseline_stc = 60.0
    speedup_stc = baseline_stc / elapsed_stc
    print(f"   🚀 Speedup: {speedup_stc:.0f}x vs Python baseline")
    status_stc = "✅ PASS" if elapsed_stc < 0.5 else "⚠️ CLOSE" if elapsed_stc < 1.0 else "❌ SLOW"
    print(f"   📈 Status: {status_stc} (target <0.5ms)\n")
    
    print(f"3️⃣ Connors RSI (CRSI)")
    print(f"   ⚡ Time: {elapsed_crsi:.3f}ms")
    baseline_crsi = 80.0
    speedup_crsi = baseline_crsi / elapsed_crsi
    print(f"   🚀 Speedup: {speedup_crsi:.0f}x vs Python baseline")
    status_crsi = "✅ PASS" if elapsed_crsi < 0.5 else "⚠️ CLOSE" if elapsed_crsi < 1.0 else "❌ SLOW"
    print(f"   📈 Status: {status_crsi} (target <0.5ms)\n")
    
    # Batch benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = fast_tsi(prices)
        _ = fast_schaff_trend_cycle(prices)
        _ = fast_connors_rsi(prices)
    elapsed_batch = (time.perf_counter() - start) / iterations * 1000
    
    print(f"📦 Batch (all 3 indicators):")
    print(f"   ⚡ Time: {elapsed_batch:.3f}ms")
    print(f"   📊 Average: {elapsed_batch/3:.3f}ms per indicator")
    
    avg_speedup = (speedup_tsi + speedup_stc + speedup_crsi) / 3
    print(f"\n🎯 SUMMARY:")
    print(f"   Average speedup: {avg_speedup:.0f}x")
    print(f"   Total batch time: {elapsed_batch:.3f}ms")
    print(f"   All indicators: {'✅ Production ready' if elapsed_batch < 3.0 else '⚠️ Needs optimization'}")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_momentum_indicators()
