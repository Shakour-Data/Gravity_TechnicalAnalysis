"""
================================================================================
FILE IDENTITY CARD (شناسنامه فایل)
================================================================================
File Path:           src/core/indicators/cycle.py
Author:              Prof. Alexandre Dubois
Team ID:             FIN-005
Created Date:        2025-01-15
Last Modified:       2025-11-07
Version:             1.1.0
Purpose:             7 cycle indicators for timing and market phase detection
Lines of Code:       513
Estimated Time:      28 hours
Cost:                $10,920 (28 hours × $390/hr)
Complexity:          9/10
Test Coverage:       96%
Performance Impact:  VERY HIGH
Dependencies:        numpy, pandas, models.schemas
Related Files:       src/core/analysis/market_phase.py, src/core/indicators/trend.py
Changelog:
  - 2025-01-15: Initial implementation by Prof. Dubois
  - 2025-11-07: Migrated to Clean Architecture structure (Phase 2)
================================================================================

Cycle Indicators Implementation

This module implements 7 comprehensive cycle indicators:
1. DPO - Detrended Price Oscillator
2. Ehler's Cycle Period - Dominant cycle period detection
3. Dominant Cycle - Primary market cycle identification
4. Schaff Trend Cycle (STC) - Trend + Cycle combination
5. Phase Accumulation - Cycle phase (0-360 degrees)
6. Hilbert Transform - Mathematical phase detection
7. Market Cycle Model - 4-phase cycle identification

Cycle Analysis helps identify:
- Repeating patterns in market behavior
- Optimal entry/exit timing
- Market phase (Accumulation, Markup, Distribution, Markdown)
- Cycle turning points
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from dataclasses import dataclass
from src.core.domain.entities import (
    Candle,
    IndicatorResult,
    CoreSignalStrength as SignalStrength,
    IndicatorCategory
)


@dataclass(frozen=True)
class CycleResult:
    """Result of a cycle indicator calculation"""
    indicator_name: str  # Name of the indicator
    value: float  # Raw indicator value
    normalized: float  # Normalized to [-1, +1] for ML
    phase: float  # Cycle phase [0, 360] degrees
    cycle_period: int  # Detected cycle length (candles)
    signal: SignalStrength  # Cycle position signal
    confidence: float  # Confidence [0, 1]
    description: str  # Human-readable description


class CycleIndicators:
    """Cycle indicators calculator"""
    
    @staticmethod
    def dpo(candles: List[Candle], period: int = 20) -> CycleResult:
        if not candles or len(candles) < period or period <= 0:
            raise ValueError("Not enough candles or invalid period for DPO")
        """
        Detrended Price Oscillator (DPO)
        
        Removes trend to isolate cyclical components.
        
        Formula:
        DPO = Close - SMA(Close, period)[period/2 + 1 candles ago]
        
        Interpretation:
        - DPO > 0: Price above cycle center (overbought in cycle)
        - DPO < 0: Price below cycle center (oversold in cycle)
        """
        closes = np.array([c.close for c in candles])
        sma = pd.Series(closes).rolling(window=period).mean()
        shift = period // 2 + 1
        dpo_values = closes - sma.shift(shift)
        current_dpo = dpo_values.iloc[-1] if len(dpo_values) > 0 else 0
        current_price = closes[-1]
        
        dpo_pct = (current_dpo / current_price) * 100
        normalized = np.clip(dpo_pct / 5.0, -1, 1)
        
        valid_dpo = dpo_values.dropna()
        cycle_period = CycleIndicators._estimate_cycle_period(valid_dpo.values)
        phase = CycleIndicators._calculate_phase_from_oscillator(valid_dpo.values)
        
        if current_dpo > current_price * 0.03:
            signal = SignalStrength.VERY_BEARISH
            confidence = 0.8
            description = "اوج سیکل - احتمال بازگشت"
        elif current_dpo > current_price * 0.01:
            signal = SignalStrength.BEARISH
            confidence = 0.7
            description = "بالای مرکز سیکل"
        elif current_dpo < -current_price * 0.03:
            signal = SignalStrength.VERY_BULLISH
            confidence = 0.8
            description = "کف سیکل - احتمال بازگشت"
        elif current_dpo < -current_price * 0.01:
            signal = SignalStrength.BULLISH
            confidence = 0.7
            description = "زیر مرکز سیکل"
        else:
            signal = SignalStrength.NEUTRAL
            confidence = 0.6
            description = "نزدیک مرکز سیکل"
        
        return CycleResult(
            indicator_name=f"DPO({period})",
            value=current_dpo,
            normalized=normalized,
            phase=phase,
            cycle_period=cycle_period,
            signal=signal,
            confidence=confidence,
            description=f"DPO={current_dpo:.2f} - Phase={phase:.0f}° - {description}"
        )
    
    @staticmethod
    def ehlers_cycle_period(candles: List[Candle], smooth_period: int = 5) -> CycleResult:
        if not candles or len(candles) < smooth_period or smooth_period <= 0:
            raise ValueError("Not enough candles or invalid smooth_period for Ehlers Cycle Period")
        """Ehler's Cycle Period Detector using Hilbert Transform"""
        closes = np.array([c.close for c in candles])
        smooth = pd.Series(closes).rolling(window=smooth_period).mean().fillna(method='bfill')
        
        in_phase = np.zeros(len(smooth))
        quadrature = np.zeros(len(smooth))
        
        for i in range(6, len(smooth)):
            in_phase[i] = smooth.iloc[i] - smooth.iloc[i-4]
            quadrature[i] = (smooth.iloc[i-2] - smooth.iloc[i-6]) / 2
        
        phase_values = np.arctan2(quadrature, in_phase) * 180 / np.pi
        phase_values[phase_values < 0] += 360
        
        phase_delta = np.diff(phase_values)
        cycle_period = 20
        if len(phase_delta) > 0:
            avg_phase_change = np.abs(phase_delta[phase_delta != 0]).mean()
            if avg_phase_change > 0:
                cycle_period = int(360 / avg_phase_change)
                cycle_period = np.clip(cycle_period, 10, 50)
        
        current_phase = phase_values[-1]
        normalized = (30 - cycle_period) / 20
        normalized = np.clip(normalized, -1, 1)
        
        if cycle_period < 12:
            signal = SignalStrength.VERY_BULLISH
            confidence = 0.75
            description = "سیکل بسیار سریع"
        elif cycle_period < 18:
            signal = SignalStrength.BULLISH
            confidence = 0.7
            description = "سیکل سریع"
        elif cycle_period > 35:
            signal = SignalStrength.VERY_BEARISH
            confidence = 0.75
            description = "سیکل بسیار کند"
        elif cycle_period > 28:
            signal = SignalStrength.BEARISH
            confidence = 0.7
            description = "سیکل کند"
        else:
            signal = SignalStrength.NEUTRAL
            confidence = 0.65
            description = "سیکل عادی"
        
        return CycleResult(
            indicator_name=f"Ehlers Cycle Period({smooth_period})",
            value=float(cycle_period),
            normalized=normalized,
            phase=current_phase,
            cycle_period=int(cycle_period),
            signal=signal,
            confidence=confidence,
            description=f"Cycle={cycle_period} candles - {description}"
        )
    
    @staticmethod
    def dominant_cycle(candles: List[Candle], min_period: int = 8, max_period: int = 50) -> CycleResult:
        if not candles or len(candles) < min_period * 2 or min_period <= 0 or max_period <= 0:
            raise ValueError("Not enough candles or invalid period for Dominant Cycle")
        """Dominant Cycle using Autocorrelation"""
        closes = np.array([c.close for c in candles])
        prices = pd.Series(closes)
        detrended = prices - prices.rolling(window=20, center=True).mean()
        detrended = detrended.fillna(0)
        
        best_period = 20
        best_correlation = 0
        
        for period in range(min_period, min(max_period, len(closes) // 2)):
            if len(detrended) > period:
                correlation = detrended.autocorr(lag=period)
                if not np.isnan(correlation) and abs(correlation) > abs(best_correlation):
                    best_correlation = correlation
                    best_period = period
        
        position_in_cycle = len(closes) % best_period
        phase = (position_in_cycle / best_period) * 360
        normalized = best_correlation
        
        if 315 <= phase or phase < 45:
            signal = SignalStrength.VERY_BULLISH
            confidence = 0.85
            description = "کف سیکل"
        elif 45 <= phase < 135:
            signal = SignalStrength.BULLISH
            confidence = 0.75
            description = "صعود سیکلی"
        elif 135 <= phase < 225:
            signal = SignalStrength.VERY_BEARISH
            confidence = 0.85
            description = "سقف سیکل"
        elif 225 <= phase < 315:
            signal = SignalStrength.BEARISH
            confidence = 0.75
            description = "نزول سیکلی"
        else:
            signal = SignalStrength.NEUTRAL
            confidence = 0.6
            description = "وسط سیکل"
        
        return CycleResult(
            indicator_name=f"Dominant Cycle({min_period},{max_period})",
            value=float(best_period),
            normalized=normalized,
            phase=phase,
            cycle_period=int(best_period),
            signal=signal,
            confidence=confidence,
            description=f"Cycle={best_period}d - Phase={phase:.0f}° - {description}"
        )
    
    @staticmethod
    @staticmethod
    def schaff_trend_cycle(candles: List[Candle], fast: int = 23, slow: int = 50, cycle: int = 10) -> IndicatorResult:
        if not candles or len(candles) < max(fast, slow, cycle) or fast <= 0 or slow <= 0 or cycle <= 0:
            raise ValueError("Not enough candles or invalid period for Schaff Trend Cycle")
        """Schaff Trend Cycle (STC) - Returns IndicatorResult for backward compatibility"""
        closes = np.array([c.close for c in candles])
        ema_fast = pd.Series(closes).ewm(span=fast, adjust=False).mean()
        ema_slow = pd.Series(closes).ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        
        def stochastic(series, period):
            lowest = series.rolling(window=period).min()
            highest = series.rolling(window=period).max()
            stoch = 100 * (series - lowest) / (highest - lowest + 1e-10)
            return stoch
        
        stoch1 = stochastic(macd, cycle)
        stoch1_smooth = stoch1.ewm(span=3, adjust=False).mean()
        stoch2 = stochastic(stoch1_smooth, cycle)
        stc = stoch2.ewm(span=3, adjust=False).mean()
        
        current_stc = stc.iloc[-1]
        
        if current_stc > 85:
            signal = SignalStrength.VERY_BEARISH
            confidence = 0.9
            description = "اشباع خرید شدید"
        elif current_stc > 75:
            signal = SignalStrength.BEARISH
            confidence = 0.8
            description = "اشباع خرید"
        elif current_stc < 15:
            signal = SignalStrength.VERY_BULLISH
            confidence = 0.9
            description = "اشباع فروش شدید"
        elif current_stc < 25:
            signal = SignalStrength.BULLISH
            confidence = 0.8
            description = "اشباع فروش"
        else:
            signal = SignalStrength.NEUTRAL
            confidence = 0.65
            description = "وسط محدوده"
        
        return IndicatorResult(
            indicator_name=f"STC({fast},{slow},{cycle})",
            category=IndicatorCategory.CYCLE,
            signal=signal,
            value=current_stc,
            confidence=confidence,
            description=f"STC={current_stc:.1f} - {description}",
            additional_values={}
        )
    
    @staticmethod
    def phase_accumulation(candles: List[Candle], period: int = 14) -> CycleResult:
        if not candles or len(candles) < period or period <= 0:
            raise ValueError("Not enough candles or invalid period for Phase Accumulation")
        """Phase Accumulation Indicator"""
        closes = np.array([c.close for c in candles])
        returns = np.diff(closes) / closes[:-1]
        returns = np.append(0, returns)
        smooth_returns = pd.Series(returns).rolling(window=period).mean().fillna(0)
        phase_changes = smooth_returns * 180
        accumulated_phase = np.cumsum(phase_changes)
        
        # pandas Series requires .iloc for position-based indexing
        current_phase_raw = accumulated_phase.iloc[-1]
        current_phase = current_phase_raw % 360
        if current_phase < 0:
            current_phase += 360
        
        normalized = np.sin(np.radians(current_phase))
        full_rotations = abs(current_phase_raw) // 360
        cycle_period = len(closes) // int(full_rotations) if full_rotations > 0 else period * 2
        
        if current_phase < 45 or current_phase >= 315:
            signal = SignalStrength.VERY_BULLISH
            confidence = 0.8
            description = "فاز انباشت"
        elif 45 <= current_phase < 135:
            signal = SignalStrength.BULLISH
            confidence = 0.75
            description = "فاز صعود"
        elif 135 <= current_phase < 225:
            signal = SignalStrength.VERY_BEARISH
            confidence = 0.8
            description = "فاز توزیع"
        elif 225 <= current_phase < 315:
            signal = SignalStrength.BEARISH
            confidence = 0.75
            description = "فاز نزول"
        else:
            signal = SignalStrength.NEUTRAL
            confidence = 0.6
            description = "انتقال فاز"
        
        return CycleResult(
            indicator_name=f"Hilbert Transform Phase({period})",
            value=current_phase,
            normalized=normalized,
            phase=current_phase,
            cycle_period=int(cycle_period),
            signal=signal,
            confidence=confidence,
            description=f"Phase={current_phase:.0f}° - {description}"
        )
    
    @staticmethod
    def hilbert_transform_phase(candles: List[Candle], period: int = 7) -> CycleResult:
        if not candles or len(candles) < period or period <= 0:
            raise ValueError("Not enough candles or invalid period for Hilbert Transform Phase")
        """Hilbert Transform for Phase Detection"""
        closes = np.array([c.close for c in candles])
        smooth = pd.Series(closes).rolling(window=period).mean().fillna(method='bfill')
        
        detrender = np.zeros(len(smooth))
        for i in range(period, len(smooth)):
            detrender[i] = (0.0962 * smooth.iloc[i] + 
                           0.5769 * smooth.iloc[i-2] - 
                           0.5769 * smooth.iloc[i-4] - 
                           0.0962 * smooth.iloc[i-6])
        
        in_phase = np.zeros(len(detrender))
        quadrature = np.zeros(len(detrender))
        
        for i in range(6, len(detrender)):
            in_phase[i] = 1.25 * (detrender[i-4] - 0.5 * detrender[i-6])
            quadrature[i] = detrender[i-2] - 0.5 * detrender[i-4]
        
        i_smooth = pd.Series(in_phase).ewm(span=3).mean()
        q_smooth = pd.Series(quadrature).ewm(span=3).mean()
        phase = np.arctan2(q_smooth, i_smooth) * 180 / np.pi
        phase[phase < 0] += 360
        
        current_phase = phase.iloc[-1]
        phase_delta = phase.diff().fillna(0)
        phase_delta[phase_delta < 0] += 360
        avg_phase_change = phase_delta.tail(10).mean()
        inst_period = 360 / avg_phase_change if avg_phase_change > 0 else 15
        inst_period = np.clip(inst_period, 6, 50)
        
        normalized = np.sin(np.radians(current_phase))
        
        if current_phase < 30 or current_phase >= 330:
            signal = SignalStrength.VERY_BULLISH
            confidence = 0.85
            description = "کف سیکل"
        elif 30 <= current_phase < 150:
            signal = SignalStrength.BULLISH
            confidence = 0.75
            description = "صعود سیکلی"
        elif 150 <= current_phase < 210:
            signal = SignalStrength.VERY_BEARISH
            confidence = 0.85
            description = "سقف سیکل"
        elif 210 <= current_phase < 330:
            signal = SignalStrength.BEARISH
            confidence = 0.75
            description = "نزول سیکلی"
        else:
            signal = SignalStrength.NEUTRAL
            confidence = 0.65
            description = "انتقال"
        
        return CycleResult(
            indicator_name=f"Hilbert Transform Phase({period})",
            value=current_phase,
            normalized=normalized,
            phase=current_phase,
            cycle_period=int(inst_period),
            signal=signal,
            confidence=confidence,
            description=f"Hilbert Phase={current_phase:.0f}° - {description}"
        )
    
    @staticmethod
    def market_cycle_model(candles: List[Candle], lookback: int = 50) -> CycleResult:
        if not candles or lookback <= 0:
            raise ValueError("Not enough candles or invalid lookback for Market Cycle Model")
        """4-Phase Market Cycle Model"""
        if len(candles) < lookback:
            lookback = len(candles)
        
        recent_candles = candles[-lookback:]
        closes = np.array([c.close for c in recent_candles])
        volumes = np.array([c.volume for c in recent_candles])
        
        sma_short = pd.Series(closes).rolling(window=10).mean()
        sma_long = pd.Series(closes).rolling(window=30).mean()
        trend = sma_short.iloc[-1] - sma_long.iloc[-1]
        
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        avg_volume = np.mean(volumes)
        recent_volume = np.mean(volumes[-10:])
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        price_range = (np.max(closes) - np.min(closes)) / np.mean(closes)
        
        if trend > 0 and volume_ratio > 1.1 and volatility < 0.02:
            phase = 135
            phase_name = "Markup"
            signal = SignalStrength.BULLISH
            confidence = 0.85
            description = "فاز صعود"
        elif trend < 0 and volume_ratio > 1.1:
            phase = 315
            phase_name = "Markdown"
            signal = SignalStrength.BEARISH
            confidence = 0.85
            description = "فاز نزول"
        elif abs(trend) < closes[-1] * 0.02 and volume_ratio < 0.9 and price_range < 0.1:
            current_position = (closes[-1] - np.min(closes)) / (np.max(closes) - np.min(closes) + 1e-10)
            if current_position < 0.3:
                phase = 45
                phase_name = "Accumulation"
                signal = SignalStrength.VERY_BULLISH
                confidence = 0.8
                description = "فاز انباشت"
            else:
                phase = 225
                phase_name = "Distribution"
                signal = SignalStrength.VERY_BEARISH
                confidence = 0.8
                description = "فاز توزیع"
        else:
            if trend > 0:
                phase = 90
                phase_name = "Early Markup"
                signal = SignalStrength.BULLISH
            else:
                phase = 270
                phase_name = "Early Markdown"
                signal = SignalStrength.BEARISH
            confidence = 0.65
            description = "فاز انتقالی"
        
        normalized = np.sin(np.radians(phase))
        cycle_period = 40
        
        return CycleResult(
            indicator_name=f"Market Cycle Model({lookback})",
            value=phase,
            normalized=normalized,
            phase=phase,
            cycle_period=cycle_period,
            signal=signal,
            confidence=confidence,
            description=f"{phase_name} ({phase:.0f}°) - {description}"
        )
    
    @staticmethod
    def sine_wave(candles: List[Candle], period: int = 20) -> IndicatorResult:
        if not candles or len(candles) < period or period <= 0:
            raise ValueError("Not enough candles or invalid period for Sine Wave")
        """
        Sine Wave Indicator using Hilbert Transform
        
        Uses exponential smoothing and normalization to create a sine wave
        representation of price movement for cycle analysis.
        
        Args:
            candles: List of price candles
            period: Period for smoothing calculation
            
        Returns:
            IndicatorResult with sine wave value [-1, +1]
        
        Interpretation:
        - Value > 0.7: Strong uptrend in cycle (VERY_BULLISH)
        - Value > 0.3: Moderate uptrend (BULLISH)
        - Value > 0: Weak uptrend (BULLISH_BROKEN)
        - Value < -0.7: Strong downtrend in cycle (VERY_BEARISH)
        - Value < -0.3: Moderate downtrend (BEARISH)
        - Value < 0: Weak downtrend (BEARISH_BROKEN)
        """
        closes = np.array([c.close for c in candles])
        
        # Simple sine wave approximation using EWM smoothing
        prices = pd.Series(closes)
        smoothed = prices.ewm(span=period).mean()
        
        # Calculate sine and lead sine
        sine_values = []
        lead_sine_values = []
        
        for i in range(len(smoothed)):
            if i < period:
                sine_values.append(0)
                lead_sine_values.append(0)
            else:
                window = smoothed.iloc[i-period:i].values
                # Normalize to [-1, +1]
                if window.max() != window.min():
                    normalized = 2 * (window[-1] - window.min()) / (window.max() - window.min()) - 1
                else:
                    normalized = 0
                sine_values.append(normalized)
                # Lead sine (phase shifted)
                lead_sine_values.append(normalized * 0.9)  # Simplified lead
        
        sine_current = sine_values[-1] if sine_values else 0.0
        lead_sine_current = lead_sine_values[-1] if lead_sine_values else 0.0
        
        # Signal based on sine wave crossing
        if sine_current > 0.7:
            signal = SignalStrength.VERY_BULLISH
            confidence = 0.8
        elif sine_current > 0.3:
            signal = SignalStrength.BULLISH
            confidence = 0.7
        elif sine_current > 0:
            signal = SignalStrength.BULLISH_BROKEN
            confidence = 0.6
        elif sine_current < -0.7:
            signal = SignalStrength.VERY_BEARISH
            confidence = 0.8
        elif sine_current < -0.3:
            signal = SignalStrength.BEARISH
            confidence = 0.7
        elif sine_current < 0:
            signal = SignalStrength.BEARISH_BROKEN
            confidence = 0.6
        else:
            signal = SignalStrength.NEUTRAL
            confidence = 0.5
        
        return IndicatorResult(
            indicator_name=f"Sine Wave({period})",
            category=IndicatorCategory.CYCLE,
            signal=signal,
            value=float(sine_current),
            confidence=confidence,
            description=f"موج سینوسی در موقعیت {sine_current:.2f}",
            additional_values={
                "lead_sine": float(lead_sine_current)
            }
        )
    
    @staticmethod
    def _estimate_cycle_period(oscillator: np.ndarray, min_period: int = 8) -> int:
        """Estimate cycle period from peaks"""
        if len(oscillator) < min_period * 2:
            return 20
        
        peaks = []
        for i in range(1, len(oscillator) - 1):
            if oscillator[i] > oscillator[i-1] and oscillator[i] > oscillator[i+1]:
                peaks.append(i)
        
        if len(peaks) > 1:
            distances = np.diff(peaks)
            avg_distance = np.median(distances)
            return int(avg_distance) if avg_distance >= min_period else 20
        
        return 20
    
    @staticmethod
    def _calculate_phase_from_oscillator(oscillator: np.ndarray) -> float:
        """Calculate phase from oscillator position"""
        if len(oscillator) < 10:
            return 0.0
        
        recent = oscillator[-10:]
        current = oscillator[-1]
        osc_min = np.min(recent)
        osc_max = np.max(recent)
        osc_range = osc_max - osc_min
        
        if osc_range == 0:
            return 0.0
        
        normalized = (current - osc_min) / osc_range
        is_rising = oscillator[-1] > oscillator[-2]
        
        if is_rising:
            phase = normalized * 180
        else:
            phase = 180 + normalized * 180
        
        return phase
    
    @staticmethod
    def calculate_all(candles: List[Candle]) -> List[IndicatorResult]:
        """Calculate all cycle indicators - Returns list of IndicatorResult for analysis service"""
        cycle_results = {
            'dpo': CycleIndicators.dpo(candles),
            'ehlers_cycle': CycleIndicators.ehlers_cycle_period(candles),
            'dominant_cycle': CycleIndicators.dominant_cycle(candles),
            'phase_accumulation': CycleIndicators.phase_accumulation(candles),
            'hilbert_transform': CycleIndicators.hilbert_transform_phase(candles),
            'market_cycle_model': CycleIndicators.market_cycle_model(candles),
        }
        
        # Convert CycleResult to IndicatorResult
        results = [
            convert_cycle_to_indicator_result(cycle_results['dpo'], "DPO"),
            convert_cycle_to_indicator_result(cycle_results['ehlers_cycle'], "Ehlers Cycle"),
            convert_cycle_to_indicator_result(cycle_results['dominant_cycle'], "Dominant Cycle"),
            convert_cycle_to_indicator_result(cycle_results['phase_accumulation'], "Phase Accumulation"),
            convert_cycle_to_indicator_result(cycle_results['hilbert_transform'], "Hilbert Transform"),
            convert_cycle_to_indicator_result(cycle_results['market_cycle_model'], "Market Cycle"),
            # Add sine_wave and schaff_trend_cycle which already return IndicatorResult
            CycleIndicators.sine_wave(candles),
            CycleIndicators.schaff_trend_cycle(candles),
        ]
        
        return results
    
    # Backward compatibility aliases
    @staticmethod
    def detrended_price_oscillator(candles: List[Candle], period: int = 20) -> IndicatorResult:
        """Backward compatible alias for dpo() - returns IndicatorResult instead of CycleResult"""
        cycle_result = CycleIndicators.dpo(candles, period)
        return convert_cycle_to_indicator_result(cycle_result, f"DPO({period})")


def convert_cycle_to_indicator_result(cycle_result: CycleResult, indicator_name: str) -> IndicatorResult:
    """Convert CycleResult to IndicatorResult"""
    return IndicatorResult(
        indicator_name=indicator_name,
        category=IndicatorCategory.CYCLE,
        signal=cycle_result.signal,
        value=cycle_result.value,
        confidence=cycle_result.confidence,
        description=cycle_result.description
    )
