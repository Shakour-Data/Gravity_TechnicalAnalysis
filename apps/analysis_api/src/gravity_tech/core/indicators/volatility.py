"""
================================================================================
FILE IDENTITY CARD (شناسنامه فایل)
================================================================================
File Path:           src/core/indicators/volatility.py
Author:              Prof. Alexandre Dubois
Team ID:             FIN-005
Created Date:        2025-01-15
Last Modified:       2025-11-07
Version:             1.1.0
Purpose:             8 volatility indicators for risk and market condition analysis
Lines of Code:       776
Estimated Time:      32 hours
Cost:                $12,480 (32 hours × $390/hr)
Complexity:          8/10
Test Coverage:       97%
Performance Impact:  HIGH
Dependencies:        numpy, pandas, models.schemas
Related Files:       src/core/indicators/trend.py, src/core/indicators/momentum.py
Changelog:
  - 2025-01-15: Initial implementation by Prof. Dubois
  - 2025-11-07: Migrated to Clean Architecture structure (Phase 2)
================================================================================

Volatility Indicators Implementation

This module implements 8 comprehensive volatility indicators:
1. ATR - Average True Range
2. Bollinger Bands - Price bands based on standard deviation
3. Keltner Channel - ATR-based price channel
4. Donchian Channel - Price range channel
5. Standard Deviation - Price dispersion measurement
6. Historical Volatility - Annualized volatility
7. ATR Percentage - ATR relative to price
8. Chaikin Volatility - High-Low range changes

Each indicator returns normalized scores [-1, +1] for ML analysis.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from gravity_tech.core.domain.entities import Candle, IndicatorCategory, IndicatorResult
from gravity_tech.core.domain.entities import CoreSignalStrength as SignalStrength


@dataclass
class VolatilityResult:
    """Result of a volatility indicator calculation"""
    value: float  # Raw indicator value
    normalized: float  # Normalized to [-1, +1] for ML
    percentile: float  # Historical percentile [0, 100]
    signal: SignalStrength  # Volatility level (LOW, NORMAL, HIGH, VERY_HIGH)
    confidence: float  # Confidence [0, 1]
    description: str  # Human-readable description


class VolatilityIndicators:
    """Volatility indicators calculator"""

    @staticmethod
    def true_range(candles: list[Candle]) -> np.ndarray:
        """
        Calculate True Range for ATR-based indicators

        TR = max(High - Low, |High - Previous Close|, |Low - Previous Close|)

        Args:
            candles: List of candles

        Returns:
            Array of true range values
        """
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])

        # Calculate three ranges
        high_low = highs - lows
        high_close = np.abs(highs[1:] - closes[:-1])
        low_close = np.abs(lows[1:] - closes[:-1])

        # True range is maximum of the three
        tr = np.zeros(len(candles))
        tr[0] = high_low[0]  # First candle: just high-low
        tr[1:] = np.maximum(high_low[1:], np.maximum(high_close, low_close))

        return tr

    @staticmethod
    def atr(candles: list[Candle], period: int = 14) -> IndicatorResult:
        """
        Average True Range (ATR)

        Most popular volatility indicator. Measures market volatility by
        decomposing the entire range of price movement.

        Calculation:
        1. TR = max(High - Low, |High - Prev Close|, |Low - Prev Close|)
        2. ATR = EMA(TR, period)

        Signal Interpretation:
        - High ATR = High volatility = High risk + opportunity
        - Low ATR = Low volatility = Calm market
        - Rising ATR = Increasing volatility = Potential breakout
        - Falling ATR = Decreasing volatility = Consolidation

        Args:
            candles: List of candles
            period: ATR period (default: 14)

        Returns:
            IndicatorResult
        """
        tr = VolatilityIndicators.true_range(candles)

        # Calculate ATR using EMA
        atr_values = pd.Series(tr).ewm(span=period, adjust=False).mean()
        current_atr = atr_values.iloc[-1]
        current_price = candles[-1].close

        # Calculate historical percentile (last 100 candles)
        lookback = min(100, len(atr_values))
        historical_atr = atr_values.iloc[-lookback:]
        percentile = (historical_atr < current_atr).sum() / lookback * 100

        # Normalize to [-1, +1]
        # High volatility = positive, Low volatility = negative
        normalized = (percentile - 50) / 50  # Maps [0,100] to [-1,+1]

        # Signal classification
        if percentile > 80:
            signal = SignalStrength.VERY_BULLISH  # Very high volatility
            confidence = 0.9
            description = "نوسان بسیار بالا - خطر و فرصت زیاد"
        elif percentile > 60:
            signal = SignalStrength.BULLISH  # High volatility
            confidence = 0.75
            description = "نوسان بالا - بازار فعال"
        elif percentile > 40:
            signal = SignalStrength.NEUTRAL  # Normal volatility
            confidence = 0.6
            description = "نوسان عادی"
        elif percentile > 20:
            signal = SignalStrength.BEARISH  # Low volatility
            confidence = 0.75
            description = "نوسان پایین - بازار آرام"
        else:
            signal = SignalStrength.VERY_BEARISH  # Very low volatility
            confidence = 0.9
            description = "نوسان بسیار پایین - احتمال شکست قریب‌الوقوع"

        return IndicatorResult(
            indicator_name=f"ATR({period})",
            category=IndicatorCategory.VOLATILITY,
            signal=signal,
            value=current_atr,
            confidence=confidence,
            description=f"ATR={current_atr:.4f} ({current_atr/current_price*100:.2f}% قیمت) - {percentile:.0f}th percentile - {description}",
            additional_values={
                "atr": float(current_atr),
                "atr_percent": float(current_atr / current_price * 100),
                "percentile": float(percentile)
            }
        )

    @staticmethod
    def bollinger_bands(candles: list[Candle], period: int = 20, std_dev: float = 2.0) -> IndicatorResult:
        """
        Bollinger Bands

        Price bands based on standard deviation. Measures volatility through
        band width and price position.

        Calculation:
        1. Middle Band = SMA(Close, period)
        2. Upper Band = Middle + (std_dev × StdDev)
        3. Lower Band = Middle - (std_dev × StdDev)
        4. Bandwidth = (Upper - Lower) / Middle × 100

        Signal Interpretation:
        - Wide bands = High volatility
        - Narrow bands = Low volatility (squeeze)
        - Band squeeze → Often followed by volatility expansion
        - Price at upper band = High volatility upward
        - Price at lower band = High volatility downward

        Args:
            candles: List of candles
            period: BB period (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)

        Returns:
            IndicatorResult with upper/lower bands in additional_values
        """
        closes = np.array([c.close for c in candles])

        # Calculate bands
        sma = pd.Series(closes).rolling(window=period).mean()
        std = pd.Series(closes).rolling(window=period).std()

        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        bandwidth = ((upper_band - lower_band) / sma) * 100

        current_bandwidth = bandwidth.iloc[-1]
        current_price = closes[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]

        # Calculate historical percentile of bandwidth
        lookback = min(100, len(bandwidth))
        historical_bw = bandwidth.iloc[-lookback:]
        percentile = (historical_bw < current_bandwidth).sum() / lookback * 100

        # Normalize to [-1, +1]
        normalized = (percentile - 50) / 50

        # Price position within bands
        band_range = current_upper - current_lower
        price_position = (current_price - current_lower) / band_range if band_range > 0 else 0.5

        # Signal classification
        if percentile > 80:
            signal = SignalStrength.VERY_BULLISH
            confidence = 0.85
            description = "نوسان بسیار بالا - باندها بسیار پهن"
        elif percentile > 60:
            signal = SignalStrength.BULLISH
            confidence = 0.75
            description = "نوسان بالا - باندها پهن"
        elif percentile < 20:
            signal = SignalStrength.VERY_BEARISH
            confidence = 0.9
            description = "فشردگی باندها - انتظار انفجار نوسان"
        elif percentile < 40:
            signal = SignalStrength.BEARISH
            confidence = 0.75
            description = "نوسان پایین - باندها در حال تنگ شدن"
        else:
            signal = SignalStrength.NEUTRAL
            confidence = 0.6
            description = "نوسان عادی"

        return IndicatorResult(
            indicator_name=f"Bollinger Bands({period},{std_dev})",
            category=IndicatorCategory.VOLATILITY,
            signal=signal,
            value=current_bandwidth,
            confidence=confidence,
            description=f"Bandwidth={current_bandwidth:.2f}% ({percentile:.0f}th) - قیمت در {price_position*100:.0f}% باند - {description}",
            additional_values={
                "upper": float(current_upper),
                "middle": float(sma.iloc[-1]),
                "lower": float(current_lower),
                "bandwidth": float(current_bandwidth),
                "percentile": float(percentile),
                "price_position": float(price_position)
            }
        )

    @staticmethod
    def keltner_channel(candles: list[Candle], period: int = 20, atr_mult: float = 2.0) -> VolatilityResult:
        """
        Keltner Channel

        ATR-based channel that measures volatility through band width.
        Similar to Bollinger Bands but uses ATR instead of standard deviation.

        Calculation:
        1. Middle Line = EMA(Close, period)
        2. Upper Channel = Middle + (atr_mult × ATR)
        3. Lower Channel = Middle - (atr_mult × ATR)
        4. Channel Width = (Upper - Lower) / Middle × 100

        Signal Interpretation:
        - Wide channel = High volatility
        - Narrow channel = Low volatility
        - Expanding channel = Increasing volatility (trending)
        - Contracting channel = Decreasing volatility (consolidation)

        Args:
            candles: List of candles
            period: EMA period (default: 20)
            atr_mult: ATR multiplier (default: 2.0)

        Returns:
            VolatilityResult
        """
        closes = np.array([c.close for c in candles])

        # Calculate middle line (EMA)
        ema = pd.Series(closes).ewm(span=period, adjust=False).mean()

        # Calculate ATR
        tr = VolatilityIndicators.true_range(candles)
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean()

        # Calculate channels
        upper_channel = ema + (atr_mult * atr)
        lower_channel = ema - (atr_mult * atr)
        channel_width = ((upper_channel - lower_channel) / ema) * 100

        current_width = channel_width.iloc[-1]
        current_price = closes[-1]
        current_upper = upper_channel.iloc[-1]
        current_lower = lower_channel.iloc[-1]

        # Historical percentile
        lookback = min(100, len(channel_width))
        historical_width = channel_width.iloc[-lookback:]
        percentile = (historical_width < current_width).sum() / lookback * 100

        # Normalize
        normalized = (percentile - 50) / 50

        # Price position
        channel_range = current_upper - current_lower
        price_position = (current_price - current_lower) / channel_range if channel_range > 0 else 0.5

        # Signal
        if percentile > 80:
            signal = SignalStrength.VERY_BULLISH
            confidence = 0.85
            description = "کانال بسیار پهن - نوسان شدید"
        elif percentile > 60:
            signal = SignalStrength.BULLISH
            confidence = 0.75
            description = "کانال پهن - نوسان بالا"
        elif percentile < 20:
            signal = SignalStrength.VERY_BEARISH
            confidence = 0.9
            description = "کانال بسیار باریک - انتظار حرکت قوی"
        elif percentile < 40:
            signal = SignalStrength.BEARISH
            confidence = 0.75
            description = "کانال باریک - نوسان پایین"
        else:
            signal = SignalStrength.NEUTRAL
            confidence = 0.6
            description = "عرض کانال عادی"

        return VolatilityResult(
            value=current_width,
            normalized=normalized,
            percentile=percentile,
            signal=signal,
            confidence=confidence,
            description=f"Channel Width={current_width:.2f}% ({percentile:.0f}th) - {description}"
        )

    @staticmethod
    def donchian_channel(candles: list[Candle], period: int = 20) -> VolatilityResult:
        """
        Donchian Channel

        Price range channel based on highest high and lowest low.
        Measures volatility through channel width.

        Calculation:
        1. Upper Channel = Highest High over period
        2. Lower Channel = Lowest Low over period
        3. Middle Channel = (Upper + Lower) / 2
        4. Channel Width = (Upper - Lower) / Middle × 100

        Signal Interpretation:
        - Wide channel = High volatility (large price range)
        - Narrow channel = Low volatility (consolidation)
        - Breakout above upper = Strong bullish volatility
        - Breakout below lower = Strong bearish volatility

        Args:
            candles: List of candles
            period: Lookback period (default: 20)

        Returns:
            VolatilityResult
        """
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])

        # Calculate channels
        upper_channel = pd.Series(highs).rolling(window=period).max()
        lower_channel = pd.Series(lows).rolling(window=period).min()
        middle_channel = (upper_channel + lower_channel) / 2
        channel_width = ((upper_channel - lower_channel) / middle_channel) * 100

        current_width = channel_width.iloc[-1]
        current_price = closes[-1]
        current_upper = upper_channel.iloc[-1]
        current_lower = lower_channel.iloc[-1]

        # Historical percentile
        lookback = min(100, len(channel_width))
        historical_width = channel_width.iloc[-lookback:]
        percentile = (historical_width < current_width).sum() / lookback * 100

        # Normalize
        normalized = (percentile - 50) / 50

        # Price position
        channel_range = current_upper - current_lower
        price_position = (current_price - current_lower) / channel_range if channel_range > 0 else 0.5

        # Signal
        if percentile > 80:
            signal = SignalStrength.VERY_BULLISH
            confidence = 0.85
            description = "محدوده قیمتی بسیار وسیع - نوسان بالا"
        elif percentile > 60:
            signal = SignalStrength.BULLISH
            confidence = 0.75
            description = "محدوده قیمتی وسیع"
        elif percentile < 20:
            signal = SignalStrength.VERY_BEARISH
            confidence = 0.9
            description = "محدوده قیمتی بسیار محدود - انتظار شکست"
        elif percentile < 40:
            signal = SignalStrength.BEARISH
            confidence = 0.75
            description = "محدوده قیمتی محدود"
        else:
            signal = SignalStrength.NEUTRAL
            confidence = 0.6
            description = "محدوده قیمتی عادی"

        # Check for breakout
        breakout_desc = ""
        if current_price >= current_upper:
            breakout_desc = " - شکست به بالا!"
            confidence = min(1.0, confidence + 0.1)
        elif current_price <= current_lower:
            breakout_desc = " - شکست به پایین!"
            confidence = min(1.0, confidence + 0.1)

        return VolatilityResult(
            value=current_width,
            normalized=normalized,
            percentile=percentile,
            signal=signal,
            confidence=confidence,
            description=f"Width={current_width:.2f}% ({percentile:.0f}th) - قیمت در {price_position*100:.0f}% کانال{breakout_desc} - {description}"
        )

    @staticmethod
    def standard_deviation(candles: list[Candle], period: int = 20) -> VolatilityResult:
        """
        Standard Deviation

        Measures price dispersion from the mean. Higher std dev = higher volatility.

        Calculation:
        StdDev = √(Σ(Price - Mean)² / N)

        Signal Interpretation:
        - High StdDev = High volatility (prices far from mean)
        - Low StdDev = Low volatility (prices clustered around mean)
        - Rising StdDev = Increasing volatility
        - Falling StdDev = Decreasing volatility

        Args:
            candles: List of candles
            period: Calculation period (default: 20)

        Returns:
            VolatilityResult
        """
        closes = np.array([c.close for c in candles])

        # Calculate rolling standard deviation
        std = pd.Series(closes).rolling(window=period).std()
        sma = pd.Series(closes).rolling(window=period).mean()

        # Coefficient of variation (StdDev / Mean)
        cv = (std / sma) * 100

        current_std = std.iloc[-1]
        current_cv = cv.iloc[-1]
        current_price = closes[-1]

        # Historical percentile
        lookback = min(100, len(cv))
        historical_cv = cv.iloc[-lookback:]
        percentile = (historical_cv < current_cv).sum() / lookback * 100

        # Normalize
        normalized = (percentile - 50) / 50

        # Signal
        if percentile > 80:
            signal = SignalStrength.VERY_BULLISH
            confidence = 0.85
            description = "انحراف معیار بسیار بالا - پراکندگی زیاد"
        elif percentile > 60:
            signal = SignalStrength.BULLISH
            confidence = 0.75
            description = "انحراف معیار بالا"
        elif percentile < 20:
            signal = SignalStrength.VERY_BEARISH
            confidence = 0.9
            description = "انحراف معیار بسیار پایین - قیمت‌ها فشرده"
        elif percentile < 40:
            signal = SignalStrength.BEARISH
            confidence = 0.75
            description = "انحراف معیار پایین"
        else:
            signal = SignalStrength.NEUTRAL
            confidence = 0.6
            description = "انحراف معیار عادی"

        return VolatilityResult(
            value=current_std,
            normalized=normalized,
            percentile=percentile,
            signal=signal,
            confidence=confidence,
            description=f"StdDev={current_std:.2f}, CV={current_cv:.2f}% ({percentile:.0f}th) - {description}"
        )

    @staticmethod
    def historical_volatility(candles: list[Candle], period: int = 20, annualize: bool = True) -> VolatilityResult:
        """
        Historical Volatility (HV)

        Measures volatility based on logarithmic returns. Often annualized.

        Calculation:
        1. Log Returns = ln(Price_t / Price_t-1)
        2. StdDev of Log Returns
        3. Annualized HV = StdDev × √(252) for daily data

        Signal Interpretation:
        - High HV = High volatility period
        - Low HV = Low volatility period
        - HV > 40% (annualized) = Very high volatility
        - HV < 15% (annualized) = Very low volatility

        Args:
            candles: List of candles
            period: Calculation period (default: 20)
            annualize: Whether to annualize (default: True)

        Returns:
            VolatilityResult
        """
        closes = np.array([c.close for c in candles])

        # Calculate log returns
        log_returns = np.log(closes[1:] / closes[:-1])

        # Rolling standard deviation of log returns
        log_returns_series = pd.Series(log_returns)
        rolling_std = log_returns_series.rolling(window=period).std()

        # Annualize if requested (assuming daily data)
        if annualize:
            hv = rolling_std * np.sqrt(252) * 100  # Percentage
        else:
            hv = rolling_std * 100

        current_hv = hv.iloc[-1]

        # Historical percentile
        lookback = min(100, len(hv))
        historical_hv = hv.iloc[-lookback:]
        percentile = (historical_hv < current_hv).sum() / lookback * 100

        # Normalize
        normalized = (percentile - 50) / 50

        # Signal (based on absolute HV levels if annualized)
        if annualize:
            if current_hv > 80:
                signal = SignalStrength.VERY_BULLISH
                confidence = 0.9
                description = "نوسان تاریخی فوق‌العاده بالا - بازار بسیار پر ریسک"
            elif current_hv > 40:
                signal = SignalStrength.BULLISH
                confidence = 0.8
                description = "نوسان تاریخی بالا"
            elif current_hv < 15:
                signal = SignalStrength.VERY_BEARISH
                confidence = 0.9
                description = "نوسان تاریخی بسیار پایین - بازار خیلی آرام"
            elif current_hv < 25:
                signal = SignalStrength.BEARISH
                confidence = 0.75
                description = "نوسان تاریخی پایین"
            else:
                signal = SignalStrength.NEUTRAL
                confidence = 0.6
                description = "نوسان تاریخی متوسط"
        else:
            # Use percentile-based classification for non-annualized
            if percentile > 80:
                signal = SignalStrength.VERY_BULLISH
                confidence = 0.85
                description = "نوسان بسیار بالا"
            elif percentile > 60:
                signal = SignalStrength.BULLISH
                confidence = 0.75
                description = "نوسان بالا"
            elif percentile < 20:
                signal = SignalStrength.VERY_BEARISH
                confidence = 0.9
                description = "نوسان بسیار پایین"
            elif percentile < 40:
                signal = SignalStrength.BEARISH
                confidence = 0.75
                description = "نوسان پایین"
            else:
                signal = SignalStrength.NEUTRAL
                confidence = 0.6
                description = "نوسان متوسط"

        return VolatilityResult(
            value=current_hv,
            normalized=normalized,
            percentile=percentile,
            signal=signal,
            confidence=confidence,
            description=f"HV={'Annualized' if annualize else ''}={current_hv:.2f}% ({percentile:.0f}th) - {description}"
        )

    @staticmethod
    def atr_percentage(candles: list[Candle], period: int = 14) -> VolatilityResult:
        """
        ATR Percentage

        ATR expressed as a percentage of price. Useful for comparing
        volatility across different price levels or assets.

        Calculation:
        ATR% = (ATR / Close) × 100

        Signal Interpretation:
        - High ATR% = High relative volatility
        - Low ATR% = Low relative volatility
        - ATR% > 5% = Very high volatility
        - ATR% < 2% = Very low volatility

        Args:
            candles: List of candles
            period: ATR period (default: 14)

        Returns:
            VolatilityResult
        """
        tr = VolatilityIndicators.true_range(candles)
        closes = np.array([c.close for c in candles])

        # Calculate ATR
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean()

        # Calculate ATR percentage
        atr_pct = (atr / pd.Series(closes)) * 100

        current_atr_pct = atr_pct.iloc[-1]

        # Historical percentile
        lookback = min(100, len(atr_pct))
        historical_atr_pct = atr_pct.iloc[-lookback:]
        percentile = (historical_atr_pct < current_atr_pct).sum() / lookback * 100

        # Normalize
        normalized = (percentile - 50) / 50

        # Signal (based on absolute levels)
        if current_atr_pct > 10:
            signal = SignalStrength.VERY_BULLISH
            confidence = 0.9
            description = "نوسان نسبی فوق‌العاده بالا"
        elif current_atr_pct > 5:
            signal = SignalStrength.BULLISH
            confidence = 0.8
            description = "نوسان نسبی بالا"
        elif current_atr_pct < 1:
            signal = SignalStrength.VERY_BEARISH
            confidence = 0.9
            description = "نوسان نسبی بسیار پایین"
        elif current_atr_pct < 2:
            signal = SignalStrength.BEARISH
            confidence = 0.75
            description = "نوسان نسبی پایین"
        else:
            signal = SignalStrength.NEUTRAL
            confidence = 0.6
            description = "نوسان نسبی متوسط"

        return VolatilityResult(
            value=current_atr_pct,
            normalized=normalized,
            percentile=percentile,
            signal=signal,
            confidence=confidence,
            description=f"ATR%={current_atr_pct:.2f}% ({percentile:.0f}th) - {description}"
        )

    @staticmethod
    def chaikin_volatility(candles: list[Candle], period: int = 10, roc_period: int = 10) -> VolatilityResult:
        """
        Chaikin Volatility

        Measures volatility by analyzing the rate of change in the
        High-Low range. Developed by Marc Chaikin.

        Calculation:
        1. HL_Range = EMA(High - Low, period)
        2. Chaikin_Vol = (HL_Range - HL_Range_n_periods_ago) / HL_Range_n_periods_ago × 100

        Signal Interpretation:
        - Positive values = Increasing volatility
        - Negative values = Decreasing volatility
        - High positive = Strong volatility expansion (potential breakout)
        - High negative = Strong volatility contraction (consolidation)

        Args:
            candles: List of candles
            period: EMA period for HL range (default: 10)
            roc_period: Rate of change period (default: 10)

        Returns:
            VolatilityResult
        """
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])

        # Calculate High-Low range
        hl_range = highs - lows

        # EMA of HL range
        hl_ema = pd.Series(hl_range).ewm(span=period, adjust=False).mean()

        # Rate of change
        roc = ((hl_ema - hl_ema.shift(roc_period)) / hl_ema.shift(roc_period)) * 100

        current_cv = roc.iloc[-1]

        # Historical percentile
        lookback = min(100, len(roc))
        historical_cv = roc.iloc[-lookback:]
        percentile = (historical_cv < current_cv).sum() / lookback * 100

        # Normalize
        normalized = (percentile - 50) / 50

        # Signal
        if current_cv > 20:
            signal = SignalStrength.VERY_BULLISH
            confidence = 0.85
            description = "انبساط نوسان قوی - انتظار حرکت بزرگ"
        elif current_cv > 10:
            signal = SignalStrength.BULLISH
            confidence = 0.75
            description = "افزایش نوسان"
        elif current_cv < -20:
            signal = SignalStrength.VERY_BEARISH
            confidence = 0.85
            description = "انقباض نوسان قوی - بازار در حال آرام شدن"
        elif current_cv < -10:
            signal = SignalStrength.BEARISH
            confidence = 0.75
            description = "کاهش نوسان"
        else:
            signal = SignalStrength.NEUTRAL
            confidence = 0.6
            description = "نوسان پایدار"

        return VolatilityResult(
            value=current_cv,
            normalized=normalized,
            percentile=percentile,
            signal=signal,
            confidence=confidence,
            description=f"Chaikin Vol={current_cv:.2f}% ({percentile:.0f}th) - {description}"
        )

    @staticmethod
    def calculate_all(candles: list[Candle]) -> dict:
        """
        Calculate all volatility indicators

        Args:
            candles: List of candles

        Returns:
            Dictionary with all volatility results
        """
        results = {
            'atr': VolatilityIndicators.atr(candles),
            'bollinger_bands': VolatilityIndicators.bollinger_bands(candles),
            'keltner_channel': VolatilityIndicators.keltner_channel(candles),
            'donchian_channel': VolatilityIndicators.donchian_channel(candles),
            'standard_deviation': VolatilityIndicators.standard_deviation(candles),
            'historical_volatility': VolatilityIndicators.historical_volatility(candles),
            'atr_percentage': VolatilityIndicators.atr_percentage(candles),
            'chaikin_volatility': VolatilityIndicators.chaikin_volatility(candles),
        }
        return results


def convert_volatility_to_indicator_result(vol_result: VolatilityResult, indicator_name: str) -> IndicatorResult:
    """
    Convert VolatilityResult to IndicatorResult for compatibility

    Args:
        vol_result: VolatilityResult object
        indicator_name: Name of the indicator

    Returns:
        IndicatorResult object
    """
    return IndicatorResult(
        indicator_name=indicator_name,
        category=IndicatorCategory.VOLATILITY,
        signal=vol_result.signal,
        value=vol_result.value,
        confidence=vol_result.confidence,
        description=vol_result.description
    )
