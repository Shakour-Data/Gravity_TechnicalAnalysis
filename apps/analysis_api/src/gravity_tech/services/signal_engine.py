"""
Signal aggregation logic extracted from the Pydantic contract.

This keeps business rules close to the service layer and prevents
`TechnicalAnalysisResult` from carrying behavioral logic.
"""

from __future__ import annotations

import numpy as np

from gravity_tech.core.contracts.analysis import TechnicalAnalysisResult
from gravity_tech.core.domain.entities import IndicatorResult, SignalStrength


def _calc_category_score_and_accuracy(indicators: list[IndicatorResult]) -> tuple[float, float]:
    """Calculate category score and average accuracy."""

    if not indicators:
        return 0.0, 0.0

    weighted_sum = sum(ind.signal.get_score() * ind.confidence for ind in indicators)
    total_weight = sum(ind.confidence for ind in indicators)
    avg_accuracy = total_weight / len(indicators) if indicators else 0.0
    score = weighted_sum / total_weight if total_weight > 0 else 0.0
    return score, avg_accuracy


def compute_overall_signals(result: TechnicalAnalysisResult) -> TechnicalAnalysisResult:
    """
    Compute overall signal and confidence for a TechnicalAnalysisResult.

    Mutates and returns the same instance for convenience.
    """

    trend_score, trend_accuracy = _calc_category_score_and_accuracy(result.trend_indicators)
    momentum_score, momentum_accuracy = _calc_category_score_and_accuracy(result.momentum_indicators)
    cycle_score, cycle_accuracy = _calc_category_score_and_accuracy(result.cycle_indicators)
    volume_score, volume_accuracy = _calc_category_score_and_accuracy(result.volume_indicators)

    base_weights = {
        "trend": 0.30,
        "momentum": 0.25,
        "cycle": 0.25,
        "volume": 0.20,
    }

    accuracies = {
        "trend": trend_accuracy,
        "momentum": momentum_accuracy,
        "cycle": cycle_accuracy,
        "volume": volume_accuracy,
    }

    total_weighted_accuracy = sum(base_weights[cat] * accuracies[cat] for cat in base_weights.keys())

    if total_weighted_accuracy > 0:
        adjusted_weights = {
            cat: (base_weights[cat] * accuracies[cat]) / total_weighted_accuracy for cat in base_weights.keys()
        }
    else:
        adjusted_weights = base_weights

    overall_score = (
        (trend_score * adjusted_weights["trend"])
        + (momentum_score * adjusted_weights["momentum"])
        + (cycle_score * adjusted_weights["cycle"])
    )

    volume_weight = adjusted_weights["volume"]
    volume_confirmation = abs(volume_score) * volume_weight

    if overall_score * volume_score > 0:
        overall_score *= 1 + volume_confirmation
    else:
        overall_score *= 1 - volume_confirmation

    overall_score = max(-2.0, min(2.0, overall_score))
    normalized_score = overall_score / 2.0

    result.overall_trend_signal = SignalStrength.from_value(trend_score / 2.0)
    result.overall_momentum_signal = SignalStrength.from_value(momentum_score / 2.0)
    result.overall_cycle_signal = SignalStrength.from_value(cycle_score / 2.0)
    result.overall_signal = SignalStrength.from_value(normalized_score)

    all_scores = []
    all_confidences = []

    for indicators in [
        result.trend_indicators,
        result.momentum_indicators,
        result.cycle_indicators,
        result.volume_indicators,
    ]:
        all_scores.extend([ind.signal.get_score() for ind in indicators])
        all_confidences.extend([ind.confidence for ind in indicators])

    if all_scores and all_confidences:
        std_dev = np.std(all_scores)
        agreement_confidence = max(0.0, min(1.0, 1.0 - (std_dev / 4.0)))
        accuracy_confidence = float(np.mean(all_confidences))
        result.overall_confidence = float((agreement_confidence * 0.6) + (accuracy_confidence * 0.4))
    else:
        result.overall_confidence = 0.5

    return result
