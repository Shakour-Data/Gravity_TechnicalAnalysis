"""
Training Pipeline برای Multi-Horizon Cycle System

آموزش مدل سیکل برای سه افق مستقل
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd
from gravity_tech.core.domain.entities import Candle
from gravity_tech.ml.multi_horizon_cycle_features import MultiHorizonCycleFeatureExtractor
from gravity_tech.ml.multi_horizon_weights import MultiHorizonWeightLearner


def create_realistic_cycle_data(
    num_samples: int = 2000,
    cycle_regime: str = 'mixed'  # 'fast', 'slow', 'mixed', 'range'
) -> list[Candle]:
    """
    ایجاد داده‌های واقعی با سیکل‌های مختلف

    Args:
        num_samples: تعداد کندل‌ها
        cycle_regime: رژیم سیکل
            - 'fast': سیکل‌های سریع (8-15 کندل)
            - 'slow': سیکل‌های کند (30-50 کندل)
            - 'mixed': ترکیب سیکل‌های مختلف
            - 'range': رنج بدون روند (سیکل واضح)
    """
    np.random.seed(42)

    dates = pd.date_range(end=pd.Timestamp.now(), periods=num_samples, freq='1h')

    base_price = 50000
    candles = []

    for i in range(num_samples):
        # تعیین cycle period بر اساس regime
        if cycle_regime == 'fast':
            cycle_period = 12  # 12 کندل
            amplitude = base_price * 0.02  # 2%
        elif cycle_regime == 'slow':
            cycle_period = 40  # 40 کندل
            amplitude = base_price * 0.05  # 5%
        elif cycle_regime == 'range':
            cycle_period = 20
            amplitude = base_price * 0.03  # 3%
        else:  # mixed
            # تغییر دوره سیکل در طول زمان
            if i % 400 < 200:
                cycle_period = 15  # fast
                amplitude = base_price * 0.02
            else:
                cycle_period = 35  # slow
                amplitude = base_price * 0.04

        # محاسبه موقعیت در سیکل (phase)
        phase = (i % cycle_period) / cycle_period * 2 * np.pi

        # موج سینوسی برای سیکل
        cycle_component = amplitude * np.sin(phase)

        # trend (کم یا صفر برای range)
        if cycle_regime == 'range':
            trend_component = 0
        else:
            # trend خیلی ملایم
            trend_component = base_price * 0.0001 * i

        # noise
        noise = np.random.normal(0, base_price * 0.005)

        # قیمت پایانی
        close_price = base_price + trend_component + cycle_component + noise

        # قیمت باز شدن (از close قبلی)
        if i == 0:
            open_price = base_price
        else:
            open_price = candles[-1].close

        # High/Low با توجه به volatility داخل کندل
        intracandle_volatility = amplitude * 0.3
        high_change = abs(np.random.normal(0, intracandle_volatility))
        low_change = abs(np.random.normal(0, intracandle_volatility))

        high_price = max(open_price, close_price) + high_change
        low_price = min(open_price, close_price) - low_change

        # Volume (بیشتر در turning points)
        # Volume بالاتر در فازهای 0-90 و 180-270 (turning points)
        phase_deg = (phase * 180 / np.pi) % 360
        if 315 <= phase_deg or phase_deg < 45 or 135 <= phase_deg < 225:
            # نزدیک کف یا سقف
            volume_multiplier = 1.5
        else:
            volume_multiplier = 1.0

        base_volume = 1000000
        volume = base_volume * volume_multiplier * (1 + np.random.normal(0, 0.2))
        volume = max(volume, 100000)

        # ایجاد Candle
        candle = Candle(
            timestamp=dates[i],
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume
        )
        candles.append(candle)

    return candles




def _phase_to_target(phase: float) -> float:
    """Map averaged phase (degrees) to a supervisory target."""
    normalized = phase % 360
    if 45 <= normalized < 135:
        return 1.0
    if 135 <= normalized < 225:
        return -0.5
    if 225 <= normalized < 315:
        return -1.0
    return 0.5


def build_cycle_dataset(
    candles: Sequence[Candle],
    lookback_period: int,
    horizons: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    extractor = MultiHorizonCycleFeatureExtractor(lookback_period=lookback_period)
    horizon_steps = {h: int(h.replace('d', '')) for h in horizons}
    max_horizon = max(horizon_steps.values())

    feature_rows: list[dict[str, float]] = []
    target_rows: list[dict[str, float]] = []

    for idx in range(lookback_period, len(candles) - max_horizon):
        window = candles[idx - lookback_period: idx]
        features = extractor.extract_cycle_features(list(window))
        targets: dict[str, float] = {}

        for horizon, steps in horizon_steps.items():
            future_end = idx + steps
            future_window = candles[future_end - lookback_period:future_end]
            future_features = extractor.extract_cycle_features(list(future_window))
            future_phase = future_features.get('cycle_avg_phase', 0.0)
            targets[f'return_{horizon}'] = _phase_to_target(future_phase)

        feature_rows.append(features)
        target_rows.append(targets)

    X = pd.DataFrame(feature_rows).replace([np.inf, -np.inf], 0).fillna(0)
    Y = pd.DataFrame(target_rows).replace([np.inf, -np.inf], 0).fillna(0)
    return X, Y


def train_cycle_model(
    candles: Sequence[Candle],
    horizons: Sequence[str] | None = None,
    lookback_period: int = 100,
    test_size: float = 0.2,
    output_dir: str = 'models/cycle',
    verbose: bool = True,
) -> MultiHorizonWeightLearner:
    horizons = list(horizons or ['3d', '7d', '30d'])

    if verbose:
        print("=" * 70)
        print("🚀 TRAINING MULTI-HORIZON CYCLE MODEL")
        print("=" * 70)
        print(f"\n📈 Dataset candles: {len(candles)}")
        print(f"🕒 Horizons: {horizons}")
        print(f"🔍 Lookback: {lookback_period}")

    X, Y = build_cycle_dataset(candles, lookback_period, horizons)

    if verbose:
        print(f"\n✅ Prepared {len(X)} samples with {X.shape[1]} features.")

    learner = MultiHorizonWeightLearner(
        horizons=horizons,
        test_size=test_size,
        random_state=42,
        lgbm_params={
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'n_estimators': 150,
            'learning_rate': 0.05,
            'num_leaves': 63,
            'max_depth': 6,
        },
    )
    learner.train(X, Y, verbose=verbose)

    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, 'cycle_weights.json')
    model_path = os.path.join(output_dir, 'cycle_weights.pkl')
    learner.save_weights(weights_path)
    learner.save_model_state(model_path)

    if verbose:
        print(f"\n💾 Weights saved to {weights_path}")
        print(f"💾 Model state saved to {model_path}")
        for horizon in horizons:
            hw = learner.get_horizon_weights(horizon)
            if hw:
                print(f"\n[{horizon.upper()}] R² Test: {hw.metrics.get('r2_test', 0):.3f}")
                print(f"[{horizon.upper()}] MAE Test: {hw.metrics.get('mae_test', 0):.4f}")
                print(f"[{horizon.upper()}] Confidence: {hw.confidence:.2f}")

    return learner


def main() -> MultiHorizonWeightLearner:
    print("=" * 70)
    print("Multi-Horizon Cycle Training Pipeline")
    print("=" * 70)

    training_candles: list[Candle] = []
    regimes = ['fast', 'slow', 'range', 'mixed']
    for regime in regimes:
        print(f"🔁 Generating {regime} regime samples...")
        training_candles.extend(create_realistic_cycle_data(600, regime))

    learner = train_cycle_model(
        candles=training_candles,
        horizons=['3d', '7d', '30d'],
        lookback_period=100,
        output_dir='models/cycle',
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("✅ Cycle training completed successfully.")
    print("=" * 70)
    return learner


if __name__ == '__main__':
    main()
