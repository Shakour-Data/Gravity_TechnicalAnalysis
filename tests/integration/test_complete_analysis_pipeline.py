import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from gravity_tech.core.domain.entities import Candle
from gravity_tech.ml.pipeline_factory import build_pipeline_from_weights
from gravity_tech.ml.multi_horizon_cycle_analysis import CycleScore
from gravity_tech.ml.multi_horizon_support_resistance_analysis import SupportResistanceScore
from gravity_tech.models.schemas import SignalStrength

pytestmark = pytest.mark.integration


class StubCycleAnalyzer:
    def analyze(self, candles):
        def _score(horizon: str) -> CycleScore:
            return CycleScore(
                horizon=horizon,
                score=0.2,
                confidence=0.75,
                signal=SignalStrength.BULLISH,
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
        def _score(horizon: str) -> SupportResistanceScore:
            return SupportResistanceScore(
                horizon=horizon,
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


def test_complete_pipeline_runs_with_cached_features(tmp_path):
    candles = _make_candles(180)

    trend_weights = _write_weights(
        tmp_path / "trend_weights.json",
        ["sma_signal", "ema_signal", "macd_signal"],
    )
    momentum_weights = _write_weights(
        tmp_path / "momentum_weights.json",
        ["rsi_signal", "stochastic_signal", "cci_signal"],
    )
    volatility_weights = _write_weights(
        tmp_path / "volatility_weights.json",
        ["atr_signal", "bollinger_bands_signal", "keltner_channel_signal"],
    )

    pipeline = build_pipeline_from_weights(
        candles=candles,
        trend_weights_path=str(trend_weights),
        momentum_weights_path=str(momentum_weights),
        volatility_weights_path=str(volatility_weights),
        cycle_analyzer=StubCycleAnalyzer(),
        sr_analyzer=StubSupportResistanceAnalyzer(),
        verbose=False,
        use_volume_matrix=False,
    )

    result = pipeline.analyze()

    assert result.trend_score is not None
    assert result.momentum_score is not None
    assert result.volatility_score is not None
    assert result.decision.final_signal is not None
    assert result.decision.final_confidence > 0


def _write_weights(path: Path, feature_names: list[str]) -> Path:
    horizons = ["3d", "7d", "30d"]
    payload = {
        "horizons": horizons,
        "feature_names": feature_names,
        "weights": {
            horizon: {
                "horizon": horizon,
                "weights": {feature: 0.5 for feature in feature_names},
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
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_candles(count: int) -> list[Candle]:
    base_time = datetime.utcnow() - timedelta(days=count)
    candles: list[Candle] = []
    price = 100.0

    for idx in range(count):
        open_price = price + (idx * 0.02)
        close_price = open_price + 0.4
        high_price = close_price + 0.3
        low_price = open_price - 0.3
        candles.append(
            Candle(
                timestamp=base_time + timedelta(days=idx),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=10_000 + idx * 5,
            )
        )
        price = close_price

    return candles
