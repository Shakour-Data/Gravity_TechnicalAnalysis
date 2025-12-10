import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from gravity_tech.core.domain.entities import Candle
from gravity_tech.ml.pipeline_factory import (
    build_pipeline_from_weights,
    load_cycle_analyzer,
    load_support_resistance_analyzer,
)
from gravity_tech.ml.multi_horizon_weights import MultiHorizonWeightLearner
from datetime import timezone


class DummyFeatureCache:
    def __init__(self):
        features = {"feat_a": 0.6, "feat_b": 0.4}
        self.trend_features = dict(features)
        self.momentum_features = dict(features)
        self.volatility_features = dict(features)


class StubScore(SimpleNamespace):
    @property
    def accuracy(self):
        return self.confidence


class StubCycleAnalyzer:
    def analyze(self, candles):
        def _score(horizon_label: str):
            return StubScore(
                horizon=horizon_label,
                score=0.25,
                confidence=0.8,
                signal=SimpleNamespace(value="BULLISH"),
                phase=45.0,
                cycle_period=20.0,
            )

        return SimpleNamespace(
            cycle_3d=_score("3d"),
            cycle_7d=_score("7d"),
            cycle_30d=_score("30d"),
        )


class StubSupportResistanceAnalyzer:
    def analyze(self, candles):
        def _score(horizon_label: str):
            return StubScore(
                horizon=horizon_label,
                score=0.6,
                confidence=0.7,
                bounce_probability=0.65,
                breakout_probability=0.2,
                nearest_support=95.0,
                nearest_resistance=105.0,
                support_strength=0.7,
                resistance_strength=0.4,
                sr_position=0.2,
                distance_to_key_level=0.05,
                signal="NEAR_SUPPORT",
                recommendation="HOLD",
                nearest_level_type="Support",
            )

        return SimpleNamespace(
            score_3d=_score("3d"),
            score_7d=_score("7d"),
            score_30d=_score("30d"),
            overall_sr_score=0.6,
            overall_confidence=0.7,
            horizons_agreement="aligned",
            alignment="aligned",
            critical_support=95.0,
            critical_resistance=105.0,
            overall_signal="NEAR_SUPPORT",
            overall_recommendation="BUY",
        )


def test_build_pipeline_from_weight_files(tmp_path):
    feature_names = ["feat_a", "feat_b"]
    weights_paths = [
        _write_weights(tmp_path / f"{name}.json", feature_names)
        for name in ("trend_weights", "momentum_weights", "volatility_weights")
    ]

    learner = MultiHorizonWeightLearner()
    learner.load_weights(str(weights_paths[0]))
    preds = learner.predict_multi_horizon(pd.DataFrame([{"feat_a": 0.5, "feat_b": 0.3}]))
    assert list(preds.columns) == ["pred_3d", "pred_7d", "pred_30d"]

    candles = _make_candles(150)

    pipeline = build_pipeline_from_weights(
        candles=candles,
        trend_weights_path=str(weights_paths[0]),
        momentum_weights_path=str(weights_paths[1]),
        volatility_weights_path=str(weights_paths[2]),
        cycle_analyzer=StubCycleAnalyzer(),
        sr_analyzer=StubSupportResistanceAnalyzer(),
        feature_cache=DummyFeatureCache(),
        verbose=False,
        use_volume_matrix=False,
    )

    result = pipeline.analyze()

    assert result.trend_score is not None
    assert result.momentum_score is not None
    assert result.volatility_score is not None
    assert result.decision.final_signal is not None


def test_cycle_analyzer_loads_weights(tmp_path):
    weights_path = _write_weights(tmp_path / "cycle_weights.json", ["cycle_avg_signal"])
    analyzer = load_cycle_analyzer(str(weights_path))

    class StubExtractor:
        def extract_cycle_features(self, candles):
            return {
                "cycle_avg_signal": 0.4,
                "cycle_avg_phase": 90.0,
                "cycle_avg_period": 18.0,
            }

    analyzer.feature_extractor = StubExtractor()
    candles = _make_candles(130)
    analysis = analyzer.analyze(candles)
    assert analysis.cycle_3d.score == pytest.approx(0.4, rel=1e-6)


def test_support_resistance_analyzer_uses_model_state(tmp_path):
    weights_path, model_path = _write_sr_assets(tmp_path)
    analyzer = load_support_resistance_analyzer(str(weights_path), str(model_path))

    base_features = {
        'nearest_resistance_dist': 1.0,
        'resistance_strength': 0.0,
        'nearest_support_dist': 0.0,
        'support_strength': 0.0,
        'sr_position': 0.2,
        'sr_bias': 0.0,
        'level_density': 0.1,
        'fib_signal': 0.0,
        'camarilla_signal': 0.0,
        'nearest_level_dist': 1.0,
        'support_count': 1.0,
        'resistance_count': 1.0,
    }

    feature_payload: dict[str, float] = {}
    for horizon in ['3d', '7d', '30d']:
        prefix = f"{horizon}_"
        for key, value in base_features.items():
            feature_payload[f"{prefix}{key}"] = value

    analyzer.feature_extractor = SimpleNamespace(extract_all_horizons=lambda _: feature_payload)
    candles = _make_candles(80)
    analysis = analyzer.analyze(candles)
    assert analysis.score_3d.score == pytest.approx(-0.25, rel=1e-6)


def _write_weights(path: Path, feature_names: list[str]) -> Path:
    horizons = ["3d", "7d", "30d"]
    weights_payload = {
        "horizons": horizons,
        "feature_names": feature_names,
        "weights": {
            horizon: {
                "horizon": horizon,
                "weights": {name: 0.5 for name in feature_names},
                "metrics": {
                    "r2_train": 0.1,
                    "r2_test": 0.1,
                    "mae_train": 0.05,
                    "mae_test": 0.05,
                },
                "confidence": 0.6,
            }
            for horizon in horizons
        },
    }
    path.write_text(json.dumps(weights_payload), encoding="utf-8")
    return path


def _make_candles(count: int) -> list[Candle]:
    base_time = datetime.now(timezone.utc) - timedelta(days=count)
    candles: list[Candle] = []
    price = 100.0

    for idx in range(count):
        open_price = price + (idx * 0.01)
        close_price = open_price + 0.5
        high_price = close_price + 0.3
        low_price = open_price - 0.3
        candles.append(
            Candle(
                timestamp=base_time + timedelta(days=idx),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=1_000 + idx,
            )
        )
        price = close_price

    return candles


def _write_sr_assets(tmp_path: Path) -> tuple[Path, Path]:
    weights = {
        horizon: {
            'nearest_resistance_dist': -0.2,
            'resistance_strength': -0.1,
            'nearest_support_dist': 0.2,
            'support_strength': 0.1,
            'sr_position': -0.3,
            'sr_bias': 0.15,
            'level_density': 0.1,
            'fib_signal': 0.1,
            'camarilla_signal': 0.1,
        }
        for horizon in ['3d', '7d', '30d']
    }
    weights_path = tmp_path / "sr_weights.json"
    weights_path.write_text(json.dumps(weights, indent=2), encoding="utf-8")

    model_payload = {
        horizon: {
            'feature_names': list(values.keys()),
            'weights': list(values.values()),
            'intercept': 0.0,
        }
        for horizon, values in weights.items()
    }
    model_path = tmp_path / "sr_weights.pkl"
    model_path.write_bytes(pickle.dumps(model_payload))
    return weights_path, model_path
