"""
Technical Analysis Service - Main Orchestrator

This service coordinates all indicators and pattern detection.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""


import structlog
from gravity_tech.analysis.market_phase import analyze_market_phase
from gravity_tech.indicators.cycle import CycleIndicators
from gravity_tech.indicators.momentum import MomentumIndicators
from gravity_tech.indicators.support_resistance import SupportResistanceIndicators
from gravity_tech.indicators.trend import TrendIndicators
from gravity_tech.indicators.volatility import VolatilityIndicators
from gravity_tech.indicators.volume import VolumeIndicators
from gravity_tech.models.schemas import (
    AnalysisRequest,
    Candle,
    IndicatorResult,
    MarketPhaseResult,
    TechnicalAnalysisResult,
)
from gravity_tech.patterns.candlestick import CandlestickPatterns
from gravity_tech.patterns.elliott_wave import analyze_elliott_waves

logger = structlog.get_logger()


class TechnicalAnalysisService:
    """Main service for technical analysis"""

    @staticmethod
    async def analyze(request: AnalysisRequest) -> TechnicalAnalysisResult:
        """
        Perform comprehensive technical analysis

        Args:
            request: Analysis request with candles and parameters

        Returns:
            Complete technical analysis result
        """
        logger.info(
            "starting_analysis",
            symbol=request.symbol,
            timeframe=request.timeframe,
            candle_count=len(request.candles)
        )

        result = TechnicalAnalysisResult(
            symbol=request.symbol,
            timeframe=request.timeframe
        )

        try:
            # Calculate all indicator categories
            result.trend_indicators = TrendIndicators.calculate_all(request.candles)
            result.momentum_indicators = MomentumIndicators.calculate_all(request.candles)
            result.cycle_indicators = CycleIndicators.calculate_all(request.candles)
            result.volume_indicators = VolumeIndicators.calculate_all(request.candles)
            result.volatility_indicators = VolatilityIndicators.calculate_all(request.candles)
            result.support_resistance_indicators = SupportResistanceIndicators.calculate_all(request.candles)

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

            # Calculate overall signals
            result.calculate_overall_signal()

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
                    result = indicator_map[name.lower()]()
                    results.append(result)
                except Exception as e:
                    logger.error("indicator_calculation_failed", indicator=name, error=str(e))

        return results
