"""
API Response Formatters

این ماژول توابعی برای فرمت کردن خروجی API فراهم می‌کند.

همه امتیازها به محدوده [-100, +100] و اعتماد به [0, 100] تبدیل می‌شوند.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

from typing import Any

from gravity_tech.utils.display_formatters import (
    confidence_to_display,
    get_confidence_label,
    get_signal_label,
    score_to_display,
)


def format_horizon_score(horizon_score, use_persian: bool = False) -> dict[str, Any]:
    """
    فرمت کردن یک HorizonScore برای API

    Args:
        horizon_score: شیء HorizonScore (از trend یا momentum)
        use_persian: استفاده از برچسب‌های فارسی

    Returns:
        دیکشنری فرمت شده برای API
    """
    return {
        "horizon": horizon_score.horizon,
        "score": score_to_display(horizon_score.score),
        "confidence": confidence_to_display(horizon_score.confidence),
        "signal": get_signal_label(horizon_score.score, use_persian),
        "confidence_quality": get_confidence_label(horizon_score.confidence, use_persian),
        "raw_score": round(horizon_score.score, 3),  # برای debugging
        "raw_confidence": round(horizon_score.confidence, 3)
    }


def format_trend_response(
    analysis_result,
    use_persian: bool = False,
    include_raw: bool = False
) -> dict[str, Any]:
    """
    فرمت کردن نتیجه تحلیل روند برای API

    Args:
        analysis_result: نتیجه از MultiHorizonTrendAnalyzer.analyze()
        use_persian: استفاده از برچسب‌های فارسی
        include_raw: شامل کردن مقادیر خام [-1,+1] برای debugging

    Returns:
        دیکشنری JSON-ready برای API response

    Example:
        ```python
        from gravity_tech.ml.multi_horizon_analysis import MultiHorizonTrendAnalyzer

        analyzer = MultiHorizonTrendAnalyzer.load("models/trend")
        result = analyzer.analyze(trend_features)

        api_response = format_trend_response(result, use_persian=False)
        # → {
        #     "analysis_type": "TREND",
        #     "horizons": {
        #         "3d": {"score": 85, "confidence": 82, "signal": "VERY_BULLISH", ...},
        #         "7d": {"score": 75, "confidence": 78, "signal": "BULLISH", ...},
        #         "30d": {"score": 60, "confidence": 75, "signal": "BULLISH", ...}
        #     },
        #     "overall": {
        #         "score": 73,
        #         "confidence": 78,
        #         "signal": "BULLISH",
        #         "recommendation": "BUY"
        #     }
        # }
        ```
    """
    response = {
        "analysis_type": "TREND" if not use_persian else "روند",
        "horizons": {}
    }

    # فرمت کردن هر horizon
    for horizon_score in analysis_result:
        horizon_key = f"{horizon_score.horizon}d"
        response["horizons"][horizon_key] = format_horizon_score(
            horizon_score,
            use_persian
        )

    # محاسبه overall (میانگین وزن‌دار)
    if len(analysis_result) > 0:
        total_weighted_score = sum(
            hs.score * hs.confidence for hs in analysis_result
        )
        total_confidence = sum(hs.confidence for hs in analysis_result)

        if total_confidence > 0:
            overall_score = total_weighted_score / total_confidence
            overall_confidence = total_confidence / len(analysis_result)

            response["overall"] = {
                "score": score_to_display(overall_score),
                "confidence": confidence_to_display(overall_confidence),
                "signal": get_signal_label(overall_score, use_persian),
                "confidence_quality": get_confidence_label(overall_confidence, use_persian),
                "recommendation": _get_recommendation(overall_score, use_persian)
            }

            if include_raw:
                response["overall"]["raw_score"] = round(overall_score, 3)
                response["overall"]["raw_confidence"] = round(overall_confidence, 3)

    return response


def format_momentum_response(
    analysis_result,
    use_persian: bool = False,
    include_raw: bool = False
) -> dict[str, Any]:
    """
    فرمت کردن نتیجه تحلیل مومنتوم برای API

    Args:
        analysis_result: نتیجه از MultiHorizonMomentumAnalyzer.analyze()
        use_persian: استفاده از برچسب‌های فارسی
        include_raw: شامل کردن مقادیر خام

    Returns:
        دیکشنری JSON-ready برای API response
    """
    response = {
        "analysis_type": "MOMENTUM" if not use_persian else "مومنتوم",
        "horizons": {}
    }

    # فرمت کردن هر horizon
    for momentum_score in analysis_result:
        horizon_key = f"{momentum_score.horizon}d"
        response["horizons"][horizon_key] = {
            "horizon": momentum_score.horizon,
            "score": score_to_display(momentum_score.score),
            "confidence": confidence_to_display(momentum_score.confidence),
            "signal": get_signal_label(momentum_score.score, use_persian),
            "confidence_quality": get_confidence_label(momentum_score.confidence, use_persian)
        }

        if include_raw:
            response["horizons"][horizon_key]["raw_score"] = round(momentum_score.score, 3)
            response["horizons"][horizon_key]["raw_confidence"] = round(momentum_score.confidence, 3)

    # محاسبه overall
    if len(analysis_result) > 0:
        total_weighted_score = sum(
            ms.score * ms.confidence for ms in analysis_result
        )
        total_confidence = sum(ms.confidence for ms in analysis_result)

        if total_confidence > 0:
            overall_score = total_weighted_score / total_confidence
            overall_confidence = total_confidence / len(analysis_result)

            response["overall"] = {
                "score": score_to_display(overall_score),
                "confidence": confidence_to_display(overall_confidence),
                "signal": get_signal_label(overall_score, use_persian),
                "confidence_quality": get_confidence_label(overall_confidence, use_persian),
                "recommendation": _get_momentum_recommendation(overall_score, use_persian)
            }

            if include_raw:
                response["overall"]["raw_score"] = round(overall_score, 3)
                response["overall"]["raw_confidence"] = round(overall_confidence, 3)

    return response


def format_combined_response(
    combined_analysis,
    trend_analysis,
    momentum_analysis,
    use_persian: bool = False
) -> dict[str, Any]:
    """
    فرمت کردن نتیجه ترکیبی برای API

    Args:
        combined_analysis: نتیجه تحلیل ترکیبی
        trend_analysis: نتیجه تحلیل روند
        momentum_analysis: نتیجه تحلیل مومنتوم
        use_persian: استفاده از برچسب‌های فارسی

    Returns:
        دیکشنری فرمت شده برای API
    """
    response = {
        "type": "combined_analysis",
        "recommendation": {
            "action": combined_analysis.final_action,
            "confidence": combined_analysis.final_confidence,
            "scores": {
                "3d": combined_analysis.combined_score_3d,
                "7d": combined_analysis.combined_score_7d,
                "30d": combined_analysis.combined_score_30d
            }
        }
    }

    if trend_analysis:
        response["trend_analysis"] = format_trend_response(trend_analysis, use_persian)

    if momentum_analysis:
        response["momentum_analysis"] = format_momentum_response(momentum_analysis, use_persian)

    return response


# ═══════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════

def _get_recommendation(score: float, use_persian: bool = False) -> str:
    """دریافت توصیه بر اساس امتیاز روند"""
    if use_persian:
        if score >= 0.7:
            return "خرید قوی"
        elif score >= 0.3:
            return "خرید"
        elif score >= 0.1:
            return "انباشت"
        elif score >= -0.1:
            return "نگهداری"
        elif score >= -0.3:
            return "سودگیری"
        elif score >= -0.7:
            return "فروش"
        else:
            return "فروش قوی"
    else:
        if score >= 0.7:
            return "STRONG_BUY"
        elif score >= 0.3:
            return "BUY"
        elif score >= 0.1:
            return "ACCUMULATE"
        elif score >= -0.1:
            return "HOLD"
        elif score >= -0.3:
            return "TAKE_PROFIT"
        elif score >= -0.7:
            return "SELL"
        else:
            return "STRONG_SELL"


def _get_momentum_recommendation(score: float, use_persian: bool = False) -> str:
    """دریافت توصیه بر اساس امتیاز مومنتوم"""
    if use_persian:
        if score >= 0.7:
            return "ورود فوری"
        elif score >= 0.3:
            return "ورود"
        elif score >= 0.1:
            return "آماده باش"
        elif score >= -0.1:
            return "انتظار"
        elif score >= -0.3:
            return "کاهش پوزیشن"
        elif score >= -0.7:
            return "خروج"
        else:
            return "خروج فوری"
    else:
        if score >= 0.7:
            return "ENTER_NOW"
        elif score >= 0.3:
            return "ENTER"
        elif score >= 0.1:
            return "PREPARE"
        elif score >= -0.1:
            return "WAIT"
        elif score >= -0.3:
            return "REDUCE"
        elif score >= -0.7:
            return "EXIT"
        else:
            return "EXIT_NOW"


def _get_combined_action(
    trend_score: float,
    momentum_score: float,
    combined_score: float,
    use_persian: bool = False
) -> str:
    """دریافت اقدام نهایی بر اساس ترکیب روند و مومنتوم"""
    if use_persian:
        if combined_score >= 0.7:
            return "خرید قوی"
        elif combined_score >= 0.4:
            if trend_score > 0.5 and momentum_score > 0.3:
                return "خرید"
            else:
                return "انباشت محتاطانه"
        elif combined_score >= 0.1:
            return "انباشت"
        elif combined_score >= -0.1:
            return "نگهداری"
        elif combined_score >= -0.4:
            if trend_score < -0.3:
                return "کاهش پوزیشن"
            else:
                return "نگهداری محتاطانه"
        elif combined_score >= -0.7:
            return "فروش"
        else:
            return "فروش قوی"
    else:
        if combined_score >= 0.7:
            return "STRONG_BUY"
        elif combined_score >= 0.4:
            if trend_score > 0.5 and momentum_score > 0.3:
                return "BUY"
            else:
                return "CAUTIOUS_BUY"
        elif combined_score >= 0.1:
            return "ACCUMULATE"
        elif combined_score >= -0.1:
            return "HOLD"
        elif combined_score >= -0.4:
            if trend_score < -0.3:
                return "REDUCE"
            else:
                return "CAUTIOUS_HOLD"
        elif combined_score >= -0.7:
            return "SELL"
        else:
            return "STRONG_SELL"


def format_analysis_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """
    فرمت کردن خلاصه تحلیل برای API

    Args:
        summary: دیکشنری حاوی اطلاعات خلاصه تحلیل

    Returns:
        دیکشنری فرمت شده برای API
    """
    from datetime import datetime

    return {
        "type": "analysis_summary",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": summary
    }


def format_error_response(
    message: str,
    error_code: str = "INTERNAL_ERROR",
    details: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    فرمت کردن پاسخ خطا برای API

    Args:
        message: پیام خطا
        error_code: کد خطا
        details: جزئیات اضافی خطا

    Returns:
        دیکشنری فرمت شده برای API
    """
    from datetime import datetime

    error_response = {
        "type": "error",
        "timestamp": datetime.utcnow().isoformat(),
        "error": {
            "code": error_code,
            "message": message
        }
    }

    if details:
        error_response["error"]["details"] = details

    return error_response


# ═══════════════════════════════════════════════════════════════════
# مثال استفاده
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    from models.schemas import HorizonScore, SignalStrength

    print("=" * 70)
    print("API Response Formatters - Test Examples")
    print("=" * 70)

    # ساخت نتایج نمونه
    trend_scores = [
        HorizonScore(
            horizon=3,
            score=0.85,
            confidence=0.82,
            signal=SignalStrength.VERY_BULLISH
        ),
        HorizonScore(
            horizon=7,
            score=0.75,
            confidence=0.78,
            signal=SignalStrength.BULLISH
        ),
        HorizonScore(
            horizon=30,
            score=0.60,
            confidence=0.75,
            signal=SignalStrength.BULLISH
        )
    ]

    momentum_scores = [
        HorizonScore(
            horizon=3,
            score=-0.20,
            confidence=0.70,
            signal=SignalStrength.WEAK_BEARISH
        ),
        HorizonScore(
            horizon=7,
            score=0.30,
            confidence=0.72,
            signal=SignalStrength.WEAK_BULLISH
        ),
        HorizonScore(
            horizon=30,
            score=0.55,
            confidence=0.68,
            signal=SignalStrength.BULLISH
        )
    ]

    print("\n📊 TREND Analysis Response (English):")
    print("-" * 70)
    trend_response = format_trend_response(trend_scores, use_persian=False)
    print(json.dumps(trend_response, indent=2, ensure_ascii=False))

    print("\n📈 MOMENTUM Analysis Response (English):")
    print("-" * 70)
    momentum_response = format_momentum_response(momentum_scores, use_persian=False)
    print(json.dumps(momentum_response, indent=2, ensure_ascii=False))

    print("\n🔄 COMBINED Analysis Response (English):")
    print("-" * 70)
    combined_response = format_combined_response(
        trend_scores,
        momentum_scores,
        trend_weight=0.6,
        momentum_weight=0.4,
        use_persian=False
    )
    print(json.dumps(combined_response, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("✅ All formatter tests completed!")
    print("=" * 70)
