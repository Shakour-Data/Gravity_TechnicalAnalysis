"""
Helpers for constructing multi-horizon analyzers and complete pipelines from
saved weight artifacts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from gravity_tech.core.domain.entities import Candle
from gravity_tech.ml.complete_analysis_pipeline import CompleteAnalysisPipeline
from gravity_tech.ml.multi_horizon_analysis import MultiHorizonAnalyzer
from gravity_tech.ml.multi_horizon_cycle_analysis import MultiHorizonCycleAnalyzer
from gravity_tech.ml.multi_horizon_momentum_analysis import MultiHorizonMomentumAnalyzer
from gravity_tech.ml.multi_horizon_support_resistance_analysis import (
    MultiHorizonSupportResistanceAnalyzer,
)
from gravity_tech.ml.multi_horizon_volatility_analysis import MultiHorizonVolatilityAnalyzer
from gravity_tech.ml.multi_horizon_weights import MultiHorizonWeightLearner

logger = logging.getLogger(__name__)


@dataclass
class AnalyzerBundle:
    trend: Optional[MultiHorizonAnalyzer]
    momentum: Optional[MultiHorizonMomentumAnalyzer]
    volatility: Optional[MultiHorizonVolatilityAnalyzer]


def load_trend_analyzer(weights_path: str, model_path: Optional[str] = None) -> MultiHorizonAnalyzer:
    learner = _load_weight_learner(weights_path, model_path)
    return MultiHorizonAnalyzer(learner)


def load_momentum_analyzer(weights_path: str, model_path: Optional[str] = None) -> MultiHorizonMomentumAnalyzer:
    learner = _load_weight_learner(weights_path, model_path)
    return MultiHorizonMomentumAnalyzer(learner)


def load_volatility_analyzer(weights_path: str, model_path: Optional[str] = None) -> MultiHorizonVolatilityAnalyzer:
    learner = _load_weight_learner(weights_path, model_path)
    return MultiHorizonVolatilityAnalyzer(learner)


def load_cycle_analyzer(weights_path: str, model_path: Optional[str] = None, *, lookback_period: int = 100) -> MultiHorizonCycleAnalyzer:
    learner = _load_weight_learner(weights_path, model_path)
    return MultiHorizonCycleAnalyzer(weight_learner=learner, lookback_period=lookback_period)


def load_support_resistance_analyzer(
    weights_path: str,
    model_path: Optional[str] = None,
) -> MultiHorizonSupportResistanceAnalyzer:
    return MultiHorizonSupportResistanceAnalyzer(weights_path=weights_path, model_path=model_path)


def build_pipeline_from_weights(
    candles: list[Candle],
    *,
    trend_weights_path: str,
    momentum_weights_path: str,
    volatility_weights_path: str,
    trend_model_path: Optional[str] = None,
    momentum_model_path: Optional[str] = None,
    volatility_model_path: Optional[str] = None,
    cycle_weights_path: Optional[str] = None,
    cycle_model_path: Optional[str] = None,
    sr_weights_path: Optional[str] = None,
    sr_model_path: Optional[str] = None,
    cycle_analyzer: Optional[MultiHorizonCycleAnalyzer] = None,
    sr_analyzer: Optional[MultiHorizonSupportResistanceAnalyzer] = None,
    feature_cache=None,
    **pipeline_kwargs,
) -> CompleteAnalysisPipeline:
    """
    Build a CompleteAnalysisPipeline instance using saved analyzer artifacts.
    """
    trend_analyzer = load_trend_analyzer(trend_weights_path, trend_model_path)
    momentum_analyzer = load_momentum_analyzer(momentum_weights_path, momentum_model_path)
    volatility_analyzer = load_volatility_analyzer(volatility_weights_path, volatility_model_path)
    if cycle_analyzer is None and cycle_weights_path:
        cycle_analyzer = load_cycle_analyzer(cycle_weights_path, cycle_model_path)
    if sr_analyzer is None and sr_weights_path:
        sr_analyzer = load_support_resistance_analyzer(sr_weights_path, sr_model_path)

    return CompleteAnalysisPipeline(
        candles=candles,
        trend_analyzer=trend_analyzer,
        momentum_analyzer=momentum_analyzer,
        volatility_analyzer=volatility_analyzer,
        cycle_analyzer=cycle_analyzer,
        sr_analyzer=sr_analyzer,
        feature_cache=feature_cache,
        **pipeline_kwargs,
    )


def load_trend_analyzer_from_config(config_path: str) -> MultiHorizonAnalyzer:
    """
    Convenience helper that consumes the generated config JSON produced by the training pipeline.
    """
    config = _read_config(config_path)
    level1 = config.get("level1", {})
    weights_file = level1.get("weights_file")
    if not weights_file:
        raise ValueError(f"No Level 1 weights file declared in {config_path}")

    weights_path = _resolve_relative_path(config_path, weights_file)
    model_file = level1.get("model_file")
    if model_file:
        model_path = _resolve_relative_path(config_path, model_file)
    else:
        model_path = Path(weights_path).with_suffix(".pkl")

    return load_trend_analyzer(
        str(weights_path),
        str(model_path) if model_path.exists() else None,
    )


def _load_weight_learner(weights_path: str, model_path: Optional[str]) -> MultiHorizonWeightLearner:
    learner = MultiHorizonWeightLearner()
    learner.load_weights(weights_path)

    candidate_model = Path(model_path) if model_path else Path(weights_path).with_suffix(".pkl")
    if candidate_model.exists():
        learner.load_model_state(str(candidate_model))
    else:
        logger.info(
            "multi_horizon.model_state_missing",
            extra={"weights_path": weights_path, "model_path": str(candidate_model)},
        )

    return learner


def _read_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_relative_path(config_path: str, declared_path: str) -> Path:
    base_dir = Path(config_path).parent
    candidate = Path(declared_path)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()
