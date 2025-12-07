"""
Technical Analysis Service - Main Orchestrator

This service coordinates all indicators and pattern detection.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""


import asyncio

import structlog
from gravity_tech.analysis.market_phase import analyze_market_phase
from gravity_tech.config.settings import settings
from gravity_tech.core.contracts.analysis import (
    AnalysisRequest,
    MarketPhaseResult,
    TechnicalAnalysisResult,
)
from gravity_tech.core.domain.entities import Candle, IndicatorResult
from gravity_tech.core.indicators.cycle import CycleIndicators
from gravity_tech.core.indicators.momentum import MomentumIndicators
from gravity_tech.core.indicators.support_resistance import (
    SupportResistanceIndicators,
)
from gravity_tech.core.indicators.trend import TrendIndicators
from gravity_tech.core.indicators.volatility import (
    VolatilityIndicators,
    convert_volatility_to_indicator_result,
)
from gravity_tech.core.indicators.volume import VolumeIndicators
from gravity_tech.patterns.candlestick import CandlestickPatterns
from gravity_tech.patterns.elliott_wave import analyze_elliott_waves
from gravity_tech.services.fast_indicators import (
    FastBatchAnalyzer,
    FastMomentumIndicators,
    FastTrendIndicators,
    FastVolatilityIndicators,
)
from gravity_tech.services.signal_engine import compute_overall_signals

logger = structlog.get_logger()


class TechnicalAnalysisService:
    """Main service for technical analysis"""

    @staticmethod
    async def analyze(request: AnalysisRequest) -> TechnicalAnalysisResult:
        """
        Perform comprehensive technical analysis
        """
        logger.info(
            "starting_analysis",
            symbol=request.symbol,
            timeframe=request.timeframe,
            candle_count=len(request.candles),
        )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, TechnicalAnalysisService._analyze_sync, request
        )

    @staticmethod
    def _analyze_sync(request: AnalysisRequest) -> TechnicalAnalysisResult:
        """Synchronous body executed inside a thread pool."""
        result = TechnicalAnalysisResult(symbol=request.symbol, timeframe=request.timeframe)

        try:
            fast_names = TechnicalAnalysisService._prefill_fast_indicators(result, request.candles)

            TechnicalAnalysisService._extend_indicators(
                result.trend_indicators,
                TrendIndicators.calculate_all(request.candles),
                fast_names["trend_indicators"],
            )
            TechnicalAnalysisService._extend_indicators(
                result.momentum_indicators,
                MomentumIndicators.calculate_all(request.candles),
                fast_names["momentum_indicators"],
            )
            TechnicalAnalysisService._extend_indicators(
                result.cycle_indicators,
                CycleIndicators.calculate_all(request.candles),
                set(),
            )
            TechnicalAnalysisService._extend_indicators(
                result.volume_indicators,
                VolumeIndicators.calculate_all(request.candles),
                fast_names["volume_indicators"],
            )
            TechnicalAnalysisService._extend_indicators(
                result.volatility_indicators,
                VolatilityIndicators.calculate_all(request.candles),
                fast_names["volatility_indicators"],
            )
            # TODO: SupportResistanceIndicators doesn't have calculate_all method yet
            # TechnicalAnalysisService._extend_indicators(
            #     result.support_resistance_indicators,
            #     SupportResistanceIndicators.calculate_all(request.candles),
            #     set(),
            # )

            # Detect candlestick patterns
            result.candlestick_patterns = CandlestickPatterns.detect_patterns(request.candles)

            # Analyze Elliott Waves
            result.elliott_wave_analysis = analyze_elliott_waves(request.candles)

            # Analyze Market Phase (Dow Theory)
            phase_analysis = analyze_market_phase(request.candles)
            result.market_phase_analysis = MarketPhaseResult(
                market_phase=phase_analysis["market_phase"],
                phase_strength=phase_analysis["phase_strength"],
                description=phase_analysis["description"],
                overall_score=phase_analysis["detailed_analysis"]["overall_score"],
                trend_structure=phase_analysis["detailed_analysis"]["trend_structure"],
                volume_confirmation=phase_analysis["detailed_analysis"]["volume_behavior"].get("status") == "analyzed",
                recommendations=phase_analysis["recommendations"],
                detailed_scores=phase_analysis["detailed_analysis"]["scores"],
                dow_theory_compliance=phase_analysis["dow_theory_compliance"],
                timestamp=phase_analysis["timestamp"]
            )

            compute_overall_signals(result)

            logger.info(
                "analysis_completed",
                symbol=request.symbol,
                overall_signal=result.overall_signal.value if result.overall_signal else None,
                confidence=result.overall_confidence
            )

        except Exception as e:
            logger.error(
                "analysis_failed",
                symbol=request.symbol,
                error=str(e)
            )
            raise

        return result

    @staticmethod
    def _extend_indicators(
        target: list[IndicatorResult],
        indicators: list[IndicatorResult] | dict[str, IndicatorResult] | dict[str, any],
        skip_names: set[str],
    ):
        """Merge indicators while avoiding duplicates."""
        if isinstance(indicators, dict):
            # Handle dict case - convert keys and values
            for name, indicator in indicators.items():
                if name in skip_names:
                    continue

                # Convert to IndicatorResult if needed
                if isinstance(indicator, IndicatorResult):
                    target.append(indicator)
                elif hasattr(indicator, 'value') and hasattr(indicator, 'signal'):
                    # Convert VolatilityResult and similar types
                    converted = convert_volatility_to_indicator_result(indicator, name.upper())
                    target.append(converted)
        else:
            # Handle list case
            for indicator in indicators:
                if hasattr(indicator, 'indicator_name') and indicator.indicator_name in skip_names:
                    continue
                target.append(indicator)

    @staticmethod
    def _prefill_fast_indicators(
        result: TechnicalAnalysisResult, candles: list[Candle]
    ) -> dict[str, set[str]]:
        """Use the optimized fast indicator path when enabled."""

        fast_used: dict[str, set[str]] = {
            "trend_indicators": set(),
            "momentum_indicators": set(),
            "volume_indicators": set(),
            "volatility_indicators": set(),
        }

        if not settings.use_fast_indicators:
            return fast_used

        try:
            fast_results = FastBatchAnalyzer.analyze_all_indicators(candles)
        except Exception as exc:
            logger.warning("fast_indicator_batch_failed", error=str(exc))
            return fast_used

        def add(target_attr: str, key: str):
            indicator = fast_results.get(key)
            if indicator:
                getattr(result, target_attr).append(indicator)
                fast_used[target_attr].add(indicator.indicator_name)

        add("trend_indicators", "SMA_20")
        add("trend_indicators", "SMA_50")
        add("trend_indicators", "EMA_12")
        add("trend_indicators", "EMA_26")
        add("trend_indicators", "MACD")
        add("momentum_indicators", "RSI")
        add("volatility_indicators", "BB")
        add("volatility_indicators", "ATR")

        return fast_used

    @staticmethod
    async def analyze_specific_indicators(
        candles: list[Candle],
        indicator_names: list[str]
    ) -> list[IndicatorResult]:
        """
        Analyze specific indicators only

        Args:
            candles: List of candles
            indicator_names: Names of specific indicators to calculate

        Returns:
            List of indicator results
        """
        results = []

        use_fast = settings.use_fast_indicators

        fast_indicator_map = {
            "sma": lambda: FastTrendIndicators.fast_calculate_sma(candles, 20),
            "ema": lambda: FastTrendIndicators.fast_calculate_ema(candles, 20),
            "macd": lambda: FastTrendIndicators.fast_calculate_macd(candles),
            "rsi": lambda: FastMomentumIndicators.fast_calculate_rsi(candles, 14),
            "bollinger": lambda: FastVolatilityIndicators.fast_calculate_bollinger_bands(candles),
            "atr": lambda: FastVolatilityIndicators.fast_calculate_atr(candles, 14),
        }

        # Map indicator names to methods
        indicator_map = {
            # Trend
            "sma": lambda: TrendIndicators.sma(candles, 20),
            "ema": lambda: TrendIndicators.ema(candles, 20),
            "macd": lambda: TrendIndicators.macd(candles),
            "adx": lambda: TrendIndicators.adx(candles, 14),

            # Momentum
            "rsi": lambda: MomentumIndicators.rsi(candles, 14),
            "stochastic": lambda: MomentumIndicators.stochastic(candles, 14, 3),
            "cci": lambda: MomentumIndicators.cci(candles, 20),

            # Cycle
            "sine": lambda: CycleIndicators.sine_wave(candles, 20),
            "dpo": lambda: CycleIndicators.detrended_price_oscillator(candles, 20),
            "stc": lambda: CycleIndicators.schaff_trend_cycle(candles, 23, 50, 10),
            "cycle_phase": lambda: CycleIndicators.cycle_phase_index(candles, 20),

            # Volume
            "obv": lambda: VolumeIndicators.obv(candles),
            "cmf": lambda: VolumeIndicators.cmf(candles, 20),
            "vwap": lambda: VolumeIndicators.vwap(candles),

            # Volatility
            "bollinger": lambda: VolatilityIndicators.bollinger_bands(candles, 20, 2.0),
            "atr": lambda: VolatilityIndicators.atr(candles, 14),

            # Support/Resistance
            "pivot": lambda: SupportResistanceIndicators.pivot_points(candles),
            "fibonacci": lambda: SupportResistanceIndicators.fibonacci_retracement(candles, 50),
        }

        for name in indicator_names:
            if name.lower() in indicator_map:
                try:
                    if use_fast and name.lower() in fast_indicator_map:
                        result = fast_indicator_map[name.lower()]()
                    else:
                        result = indicator_map[name.lower()]()
                    results.append(result)
                except Exception as e:
                    logger.error("indicator_calculation_failed", indicator=name, error=str(e))

        return results
