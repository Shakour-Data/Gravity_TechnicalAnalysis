"""
Training Pipeline برای Multi-Horizon Cycle System

آموزش مدل سیکل برای سه افق مستقل
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from config import TSE_DB_FILE

from gravity_tech.core.domain.entities import Candle
from gravity_tech.database.tse_data_source import tse_data_source
from gravity_tech.ml.multi_horizon_cycle_features import MultiHorizonCycleFeatureExtractor
from gravity_tech.ml.multi_horizon_weights import MultiHorizonWeightLearner


def create_realistic_cycle_data(
    num_samples: int = 2000,
    cycle_regime: str = "mixed",  # 'fast', 'slow', 'mixed', 'range'
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

    dates = pd.date_range(end=pd.Timestamp.now(), periods=num_samples, freq="1h")

    base_price = 50000
    candles = []

    for i in range(num_samples):
        # تعیین cycle period بر اساس regime
        if cycle_regime == "fast":
            cycle_period = 12  # 12 کندل
            amplitude = base_price * 0.02  # 2%
        elif cycle_regime == "slow":
            cycle_period = 40  # 40 کندل
            amplitude = base_price * 0.05  # 5%
        elif cycle_regime == "range":
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
        if cycle_regime == "range":
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
            volume=volume,
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


def _rows_to_candles(
    rows: Sequence[dict[str, Any]], symbol: str, timeframe: str = "1d"
) -> list[Candle]:
    """Convert raw dictionaries from DB fetchers to Candle objects."""
    candles: list[Candle] = []
    for row in rows:
        ts = row.get("timestamp")
        if ts is None:
            continue
        candles.append(
            Candle(
                timestamp=ts,
                open=float(row.get("open", 0.0) or 0.0),
                high=float(row.get("high", 0.0) or 0.0),
                low=float(row.get("low", 0.0) or 0.0),
                close=float(row.get("close", 0.0) or 0.0),
                volume=float(row.get("volume", 0.0) or 0.0),
                symbol=str(symbol),
                timeframe=timeframe,
            )
        )
    return candles


def load_tse_instruments(
    symbols: Sequence[str] | None = None,
    market_indices: Sequence[str] | None = None,
    sector_indices: Sequence[str] | None = None,
    *,
    min_candles: int,
    max_symbols: int = 8,
    max_market_indices: int = 4,
    max_sector_indices: int = 6,
    timeframe: str = "1d",
    verbose: bool = True,
) -> dict[str, list[Candle]]:
    """
    Load instruments from the main TSE SQLite database, including symbols,
    market indices, and sector indices.
    """
    instruments: dict[str, list[Candle]] = {}

    if not os.path.exists(TSE_DB_FILE):
        if verbose:
            print(f"⚠️  TSE database not found at {TSE_DB_FILE}")
        return instruments

    if tse_data_source is None:
        if verbose:
            print("⚠️  TSE data source is not configured; skipping real data load.")
        return instruments

    resolved_symbols = list(symbols or [])
    resolved_market_indices = list(market_indices or [])
    resolved_sector_indices = list(sector_indices or [])

    if not resolved_symbols and hasattr(tse_data_source, "list_symbols"):
        try:
            resolved_symbols = tse_data_source.list_symbols(limit=max_symbols)
        except Exception as exc:
            if verbose:
                print(f"⚠️  Unable to list symbols from DB: {exc}")

    if not resolved_market_indices and hasattr(tse_data_source, "list_market_indices"):
        try:
            resolved_market_indices = tse_data_source.list_market_indices(limit=max_market_indices)
        except Exception as exc:
            if verbose:
                print(f"⚠️  Unable to list market indices: {exc}")

    if not resolved_sector_indices and hasattr(tse_data_source, "list_sector_indices"):
        try:
            resolved_sector_indices = tse_data_source.list_sector_indices(limit=max_sector_indices)
        except Exception as exc:
            if verbose:
                print(f"⚠️  Unable to list sector indices: {exc}")

    def _load_group(label: str, codes: Sequence[str], fetcher) -> None:
        for code in codes:
            try:
                raw_rows = fetcher(code)
            except Exception as exc:
                if verbose:
                    print(f"⚠️  Failed to load {label} '{code}': {exc}")
                continue

            candles = _rows_to_candles(raw_rows, str(code), timeframe)
            if len(candles) < min_candles:
                if verbose:
                    print(f"• Skipping {label} '{code}': {len(candles)} < {min_candles} candles")
                continue

            instruments[str(code)] = candles
            if verbose:
                print(f"• Loaded {label} '{code}' with {len(candles)} candles")

    _load_group("symbol", resolved_symbols, tse_data_source.fetch_price_data)
    _load_group("market index", resolved_market_indices, tse_data_source.fetch_market_index)
    if hasattr(tse_data_source, "fetch_sector_index"):
        _load_group("sector index", resolved_sector_indices, tse_data_source.fetch_sector_index)

    if verbose:
        print(f"✅ Real TSE sequences ready: {len(instruments)}")
    return instruments


def build_cycle_dataset(
    candles: Sequence[Candle],
    lookback_period: int,
    horizons: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    extractor = MultiHorizonCycleFeatureExtractor(lookback_period=lookback_period)
    horizon_steps = {h: int(h.replace("d", "")) for h in horizons}
    max_horizon = max(horizon_steps.values())

    feature_rows: list[dict[str, float]] = []
    target_rows: list[dict[str, float]] = []

    for idx in range(lookback_period, len(candles) - max_horizon):
        window = candles[idx - lookback_period : idx]
        features = extractor.extract_cycle_features(list(window))
        targets: dict[str, float] = {}

        for horizon, steps in horizon_steps.items():
            future_end = idx + steps
            future_window = candles[future_end - lookback_period : future_end]
            future_features = extractor.extract_cycle_features(list(future_window))
            future_phase = future_features.get("cycle_avg_phase", 0.0)
            targets[f"return_{horizon}"] = _phase_to_target(future_phase)

        feature_rows.append(features)
        target_rows.append(targets)

    X = pd.DataFrame(feature_rows).replace([np.inf, -np.inf], 0).fillna(0)
    Y = pd.DataFrame(target_rows).replace([np.inf, -np.inf], 0).fillna(0)
    return X, Y


def train_cycle_model(
    candles: Sequence[Candle] | None = None,
    horizons: Sequence[str] | None = None,
    lookback_period: int = 100,
    test_size: float = 0.2,
    output_dir: str = "models/cycle",
    verbose: bool = True,
    instrument_candles: Mapping[str, Sequence[Candle]] | None = None,
) -> MultiHorizonWeightLearner:
    horizons = list(horizons or ["3d", "7d", "30d"])
    horizon_steps = {h: int(h.replace("d", "")) for h in horizons}
    max_horizon = max(horizon_steps.values())
    min_required = lookback_period + max_horizon

    if verbose:
        print("=" * 70)
        print("🚀 TRAINING MULTI-HORIZON CYCLE MODEL")
        print("=" * 70)
        if instrument_candles:
            print(f"\n📊 Instruments: {len(instrument_candles)} sequences")
            print(f"🔁 Min candles per instrument: {min_required}")
        elif candles is not None:
            print(f"\n📈 Dataset candles: {len(candles)}")
        print(f"🕒 Horizons: {horizons}")
        print(f"🔍 Lookback: {lookback_period}")

    if instrument_candles:
        feature_parts: list[pd.DataFrame] = []
        target_parts: list[pd.DataFrame] = []
        for name, series in instrument_candles.items():
            if len(series) < min_required:
                if verbose:
                    print(f"- Skipping {name}: requires {min_required} candles, has {len(series)}")
                continue
            Xi, Yi = build_cycle_dataset(series, lookback_period, horizons)
            if not Xi.empty and not Yi.empty:
                feature_parts.append(Xi)
                target_parts.append(Yi)
                if verbose:
                    print(f"- {name}: added {len(Xi)} samples")

        if not feature_parts or not target_parts:
            raise ValueError("No instruments had enough data to build the training set.")

        X = pd.concat(feature_parts, ignore_index=True).replace([np.inf, -np.inf], 0).fillna(0)
        Y = pd.concat(target_parts, ignore_index=True).replace([np.inf, -np.inf], 0).fillna(0)
    else:
        if candles is None:
            raise ValueError("Candles are required when instrument_candles is not provided.")
        if len(candles) < min_required:
            raise ValueError(
                f"Not enough candles for training. Need {min_required}, got {len(candles)}."
            )
        X, Y = build_cycle_dataset(candles, lookback_period, horizons)

    if verbose:
        print(f"\n✅ Prepared {len(X)} samples with {X.shape[1]} features.")

    learner = MultiHorizonWeightLearner(
        horizons=horizons,
        test_size=test_size,
        random_state=42,
        lgbm_params={
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "n_estimators": 150,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 6,
        },
    )
    learner.train(X, Y, verbose=verbose)

    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "cycle_weights.json")
    model_path = os.path.join(output_dir, "cycle_weights.pkl")
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

    horizons = ["3d", "7d", "30d"]
    lookback_period = 100
    max_horizon = max(int(h.replace("d", "")) for h in horizons)
    min_candles = lookback_period + max_horizon

    print(f"\U0001f4e5 Trying to load real TSE data from {TSE_DB_FILE} ...")
    instrument_sets = load_tse_instruments(
        min_candles=min_candles,
        timeframe="1d",
        verbose=True,
    )

    if instrument_sets:
        learner = train_cycle_model(
            candles=None,
            instrument_candles=instrument_sets,
            horizons=horizons,
            lookback_period=lookback_period,
            output_dir="models/cycle",
            verbose=True,
        )
    else:
        print("\u26a0\ufe0f  Falling back to synthetic cycle generation.")
        training_candles: list[Candle] = []
        regimes = ["fast", "slow", "range", "mixed"]
        for regime in regimes:
            print(f"\U0001f6a7 Generating {regime} regime samples...")
            training_candles.extend(create_realistic_cycle_data(600, regime))

        learner = train_cycle_model(
            candles=training_candles,
            horizons=horizons,
            lookback_period=lookback_period,
            output_dir="models/cycle",
            verbose=True,
        )

    print("\n" + "=" * 70)
    print("\u2705 Cycle training completed successfully.")
    print("=" * 70)
    return learner


if __name__ == "__main__":
    main()
