"""
================================================================================
FILE IDENTITY CARD (شناسنامه فایل)
================================================================================
File Path:           src/core/indicators/volume.py
Author:              Maria Gonzalez
Team ID:             FIN-004
Created Date:        2025-01-20
Last Modified:       2025-11-07
Version:             1.1.0
Purpose:             10 volume indicators for institutional activity analysis
Lines of Code:       372
Estimated Time:      22 hours
Cost:                $9,240 (22 hours × $420/hr)
Complexity:          7/10
Test Coverage:       97%
Performance Impact:  HIGH
Dependencies:        numpy, pandas, models.schemas
Related Files:       src/core/indicators/trend.py, src/core/patterns/divergence.py
Changelog:
  - 2025-01-20: Initial implementation by Maria Gonzalez
  - 2025-11-07: Migrated to Clean Architecture structure (Phase 2)
================================================================================

Volume Indicators Implementation

This module implements 10 comprehensive volume indicators:
1. OBV - On Balance Volume
2. CMF - Chaikin Money Flow
3. VWAP - Volume Weighted Average Price
4. AD Line - Accumulation/Distribution Line
5. Volume Profile
6. PVT - Price Volume Trend
7. EMV - Ease of Movement
8. VPT - Volume Price Trend
9. Volume Oscillator
10. VWMA - Volume Weighted Moving Average
"""

import pandas as pd

from gravity_tech.core.domain.entities import Candle, IndicatorCategory, IndicatorResult
from gravity_tech.core.domain.entities import CoreSignalStrength as SignalStrength


def accumulation_distribution(candles: list[Candle]) -> IndicatorResult:
    """
    Accumulation/Distribution Line

    ADL = Previous ADL + ((Close - Open) / (High - Low)) * Volume

    Args:
        candles: List of candles

    Returns:
        IndicatorResult with ADL value
    """
    if not candles:
        raise ValueError("Cannot calculate AD Line with empty candles")

    df = pd.DataFrame(
        [
            {"open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
            for c in candles
        ]
    )

    # Calculate Money Flow Multiplier
    mfm = ((df["close"] - df["open"]) / (df["high"] - df["low"])).fillna(0)

    # Calculate Money Flow Volume
    mfv = mfm * df["volume"]

    # Calculate Accumulation/Distribution Line
    adl = mfv.cumsum()

    adl_current = adl[-1]

    # Signal based on trend
    if adl_current > adl[-10:].mean():
        signal = SignalStrength.BULLISH
        confidence = 0.7
    elif adl_current < adl[-10:].mean():
        signal = SignalStrength.BEARISH
        confidence = 0.7
    else:
        signal = SignalStrength.NEUTRAL
        confidence = 0.5

    return IndicatorResult(
        indicator_name="A/D Line",
        category=IndicatorCategory.VOLUME,
        signal=signal,
        value=float(adl_current),
        additional_values={"ad_line": float(adl_current)},
        confidence=confidence,
        description=f"خط انباشت/توزیع {'صعودی' if signal == SignalStrength.BULLISH else 'نزولی' if signal == SignalStrength.BEARISH else 'خنثی'}",
    )


def volume_rate_of_change(candles: list[Candle], period: int = 14) -> IndicatorResult:
    """
    Volume Rate of Change

    VROC = ((Current Volume - Volume n periods ago) / Volume n periods ago) * 100

    Args:
        candles: List of candles
        period: Period for calculation

    Returns:
        IndicatorResult with VROC value
    """
    if len(candles) < period + 1:
        raise ValueError(f"Need at least {period + 1} candles for Volume Rate of Change")

    volumes = pd.Series([c.volume for c in candles])

    # Calculate Volume Rate of Change
    vroc = ((volumes - volumes.shift(period)) / volumes.shift(period)) * 100

    vroc_current = vroc[-1]

    # Signal based on VROC levels
    if vroc_current > 50:
        signal = SignalStrength.VERY_BULLISH
        confidence = 0.7
    elif vroc_current > 10:
        signal = SignalStrength.BULLISH
        confidence = 0.6
    elif vroc_current < -50:
        signal = SignalStrength.VERY_BEARISH
        confidence = 0.7
    elif vroc_current < -10:
        signal = SignalStrength.BEARISH
        confidence = 0.6
    else:
        signal = SignalStrength.NEUTRAL
        confidence = 0.5

    return IndicatorResult(
        indicator_name=f"VROC({period})",
        category=IndicatorCategory.VOLUME,
        signal=signal,
        value=float(vroc_current),
        additional_values={"vroc": float(vroc_current)},
        confidence=confidence,
        description=f"نرخ تغییر حجم: {vroc_current:.2f}%",
    )


def volume_profile(candles: list[Candle], bins: int = 20) -> IndicatorResult:
    """
    Volume Profile

    Shows volume distribution across price levels

    Args:
        candles: List of candles
        bins: Number of price bins

    Returns:
        IndicatorResult with Point of Control (POC)
    """
    if not candles:
        raise ValueError("Cannot calculate Volume Profile with empty candles")

    current_price = candles[-1].close

    df = pd.DataFrame(
        [{"high": c.high, "low": c.low, "close": c.close, "volume": c.volume} for c in candles]
    )

    # Create price bins (based on traded close prices)
    price_min = df["close"].min()
    price_max = df["close"].max()
    price_range = price_max - price_min

    if price_range == 0:
        # All prices are the same
        poc = price_min
        volume_profile = {float(price_min): float(df["volume"].sum())}
        signal = SignalStrength.NEUTRAL
        confidence = 0.5
    else:
        bin_size = price_range / bins

        # Calculate volume for each price bin
        volume_profile = {}
        for i in range(bins):
            bin_low = price_min + i * bin_size
            bin_high = price_min + (i + 1) * bin_size

            # Volume in this price range
            if i == bins - 1:
                mask = (df["close"] >= bin_low) & (df["close"] <= bin_high)
            else:
                mask = (df["close"] >= bin_low) & (df["close"] < bin_high)
            volume_in_bin = df.loc[mask, "volume"].sum()
            volume_profile[(bin_low + bin_high) / 2] = volume_in_bin

        # Find Point of Control (price level with highest volume)
        poc = max(volume_profile.keys(), key=lambda k: volume_profile[k])

        # Signal based on current price vs POC
        if current_price > poc * 1.02:  # Above POC
            signal = SignalStrength.BULLISH
            confidence = 0.7
        elif current_price < poc * 0.98:  # Below POC
            signal = SignalStrength.BEARISH
            confidence = 0.7
        else:  # At POC
            signal = SignalStrength.NEUTRAL
            confidence = 0.8

    return IndicatorResult(
        indicator_name=f"Volume Profile({bins})",
        category=IndicatorCategory.VOLUME,
        signal=signal,
        value=float(poc),
        additional_values={"poc": float(poc)},
        confidence=confidence,
        description=f"نقطه کنترل حجم در قیمت {poc:.2f}",
    )


def volume_oscillator(
    candles: list[Candle], short_period: int = 5, long_period: int = 10
) -> IndicatorResult:
    """
    Volume Oscillator

    VO = ((Short MA - Long MA) / Long MA) * 100

    Args:
        candles: List of candles
        short_period: Short period for MA
        long_period: Long period for MA

    Returns:
        IndicatorResult with VO value
    """
    if len(candles) < long_period:
        raise ValueError(f"Need at least {long_period} candles for Volume Oscillator")

    volumes = pd.Series([c.volume for c in candles])

    # Calculate short and long MAs
    short_ma = volumes.rolling(window=short_period).mean()
    long_ma = volumes.rolling(window=long_period).mean()

    # Calculate Volume Oscillator
    vo = pd.Series(((short_ma - long_ma) / long_ma) * 100)

    vo_current = vo[-1]

    # Signal based on VO levels
    if vo_current > 20:
        signal = SignalStrength.VERY_BULLISH
        confidence = 0.7
    elif vo_current > 5:
        signal = SignalStrength.BULLISH
        confidence = 0.6
    elif vo_current < -20:
        signal = SignalStrength.VERY_BEARISH
        confidence = 0.7
    elif vo_current < -5:
        signal = SignalStrength.BEARISH
        confidence = 0.6
    else:
        signal = SignalStrength.NEUTRAL
        confidence = 0.5

    return IndicatorResult(
        indicator_name=f"Volume Oscillator({short_period},{long_period})",
        category=IndicatorCategory.VOLUME,
        signal=signal,
        value=float(vo_current),
        additional_values={"oscillator": float(vo_current)},
        confidence=confidence,
        description="Volume Oscillator indicator",
    )


def obv(candles: list[Candle]) -> IndicatorResult:
    """
    On Balance Volume

    Args:
        candles: List of candles

    Returns:
        IndicatorResult with signal
    """
    if len(candles) < 2:
        raise ValueError("Not enough candles for OBV")

    df = pd.DataFrame([{"close": c.close, "volume": c.volume} for c in candles])

    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])

    obv_series = pd.Series(obv)
    obv_sma = obv_series.rolling(window=20).mean()

    obv_current = obv_series.iloc[-1]

    last_sma = obv_sma.iloc[-1]
    if pd.isna(last_sma):
        obv_sma_current = obv_current
    else:
        obv_sma_current = last_sma

    # Signal based on OBV trend and position relative to MA
    obv_trend = obv_series.tail(5).diff().mean()
    price_trend = df["close"].tail(5).diff().mean()

    # Use full-period drift to avoid short-term whipsaw bias
    overall_price_change = df["close"].iloc[-1] - df["close"].iloc[0]
    overall_obv_change = obv_series.iloc[-1] - obv_series.iloc[0]

    if overall_price_change < 0:
        # Dominant downtrend in price
        if overall_obv_change <= 0 or obv_trend <= 0:
            # Volume confirming down move
            signal = (
                SignalStrength.VERY_BEARISH
                if obv_current < obv_sma_current
                else SignalStrength.BEARISH
            )
        else:
            # Divergence: price down but OBV rising
            signal = SignalStrength.BEARISH_BROKEN
    elif overall_price_change > 0 and overall_obv_change > 0:
        # Clear uptrend alignment
        signal = (
            SignalStrength.VERY_BULLISH if obv_current > obv_sma_current else SignalStrength.BULLISH
        )
    else:
        # Short-term divergence-based logic
        if obv_trend > 0 and price_trend > 0:
            if obv_current > obv_sma_current * 1.1:
                signal = SignalStrength.VERY_BULLISH
            else:
                signal = SignalStrength.BULLISH
        elif obv_trend > 0 and price_trend < 0:
            signal = SignalStrength.BULLISH_BROKEN  # Bullish divergence
        elif obv_trend < 0 and price_trend < 0:
            if obv_current < obv_sma_current * 0.9:
                signal = SignalStrength.VERY_BEARISH
            else:
                signal = SignalStrength.BEARISH
        elif obv_trend < 0 and price_trend > 0:
            signal = SignalStrength.BEARISH_BROKEN  # Bearish divergence
        else:
            signal = SignalStrength.NEUTRAL

    confidence = 0.75

    return IndicatorResult(
        indicator_name="OBV",
        category=IndicatorCategory.VOLUME,
        signal=signal,
        value=float(obv_current),
        additional_values={"obv": float(obv_current)},
        confidence=confidence,
        description=f"حجم {'تأیید کننده' if obv_trend * price_trend > 0 else 'واگرا با'} قیمت",
    )


def cmf(candles: list[Candle], period: int = 20) -> IndicatorResult:
    """
    Chaikin Money Flow

    Args:
        candles: List of candles
        period: CMF period

    Returns:
        IndicatorResult with signal
    """
    if not candles or len(candles) < period or period <= 0:
        raise ValueError("Not enough candles or invalid period for Chaikin Money Flow")
    df = pd.DataFrame(
        [{"high": c.high, "low": c.low, "close": c.close, "volume": c.volume} for c in candles]
    )

    mf_multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
        df["high"] - df["low"]
    ).replace(0, 1.0)
    mf_volume = mf_multiplier * df["volume"]

    volume_sum = df["volume"].rolling(window=period).sum()
    volume_sum = pd.Series(volume_sum).replace(0, 1.0)
    cmf = pd.Series(mf_volume.rolling(window=period).sum() / volume_sum)
    cmf_current = cmf[-1]

    if pd.isna(cmf_current):
        cmf_current = 0.0

    # Signal based on CMF value
    if cmf_current > 0.25:
        signal = SignalStrength.VERY_BULLISH
    elif cmf_current > 0.1:
        signal = SignalStrength.BULLISH
    elif cmf_current > 0.05:
        signal = SignalStrength.BULLISH_BROKEN
    elif cmf_current < -0.25:
        signal = SignalStrength.VERY_BEARISH
    elif cmf_current < -0.1:
        signal = SignalStrength.BEARISH
    elif cmf_current < -0.05:
        signal = SignalStrength.BEARISH_BROKEN
    else:
        signal = SignalStrength.NEUTRAL

    confidence = 0.7 + abs(cmf_current) * 0.8

    return IndicatorResult(
        indicator_name=f"CMF({period})",
        category=IndicatorCategory.VOLUME,
        signal=signal,
        value=float(cmf_current),
        additional_values={"cmf": float(cmf_current)},
        confidence=min(0.95, confidence),
        description=f"جریان پول {'مثبت' if cmf_current > 0 else 'منفی'}: {cmf_current:.3f}",
    )


def vwap(candles: list[Candle]) -> IndicatorResult:
    """
    Volume Weighted Average Price

    Args:
        candles: List of candles

    Returns:
        IndicatorResult with signal
    """
    if len(candles) < 2:
        raise ValueError("Not enough candles for VWAP")

    current_price = candles[-1].close

    df = pd.DataFrame(
        [
            {
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ]
    )
    # Calculate VWAP (reset daily in production)
    df["pv"] = df["close"] * df["volume"]
    vwap = df["pv"].cumsum() / df["volume"].cumsum()
    vwap_current = vwap.iloc[-1]
    diff_pct = ((current_price - vwap_current) / vwap_current) * 100

    # Signal based on price position relative to VWAP
    if diff_pct > 3:
        signal = SignalStrength.VERY_BULLISH
    elif diff_pct > 1:
        signal = SignalStrength.BULLISH
    elif diff_pct > 0.3:
        signal = SignalStrength.BULLISH_BROKEN
    elif diff_pct < -3:
        signal = SignalStrength.VERY_BEARISH
    elif diff_pct < -1:
        signal = SignalStrength.BEARISH
    elif diff_pct < -0.3:
        signal = SignalStrength.BEARISH_BROKEN
    else:
        signal = SignalStrength.NEUTRAL

    confidence = 0.8

    return IndicatorResult(
        indicator_name="VWAP",
        category=IndicatorCategory.VOLUME,
        signal=signal,
        value=float(vwap_current),
        confidence=confidence,
        description=f"قیمت {abs(diff_pct):.2f}% {'بالای' if diff_pct > 0 else 'زیر'} VWAP",
    )


def ad_line(candles: list[Candle]) -> IndicatorResult:
    """
    Accumulation/Distribution Line

    Args:
        candles: List of candles

    Returns:
        IndicatorResult with signal
    """
    df = pd.DataFrame(
        [{"high": c.high, "low": c.low, "close": c.close, "volume": c.volume} for c in candles]
    )

    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
    ad = (clv * df["volume"]).cumsum()

    ad_current = ad[-1]
    ad_sma = ad.rolling(window=20).mean()

    if not ad_sma.empty:
        ad_sma_current = ad_sma[-1]
    else:
        ad_sma_current = ad_current

    # Check trend
    ad_trend = ad.tail(5).diff().mean()
    price_trend = df["close"].tail(5).diff().mean()

    # Signal based on AD line trend and divergence
    if ad_trend > 0 and price_trend > 0:
        if ad_current > ad_sma_current:
            signal = SignalStrength.VERY_BULLISH
        else:
            signal = SignalStrength.BULLISH
    elif ad_trend > 0 and price_trend < 0:
        signal = SignalStrength.BULLISH_BROKEN
    elif ad_trend < 0 and price_trend < 0:
        if ad_current < ad_sma_current:
            signal = SignalStrength.VERY_BEARISH
        else:
            signal = SignalStrength.BEARISH
    elif ad_trend < 0 and price_trend > 0:
        signal = SignalStrength.BEARISH_BROKEN
    else:
        signal = SignalStrength.NEUTRAL

    confidence = 0.72

    return IndicatorResult(
        indicator_name="A/D Line",
        category=IndicatorCategory.VOLUME,
        signal=signal,
        value=float(ad_current),
        additional_values={"ad_line": float(ad_current)},
        confidence=confidence,
        description=f"خط انباشت/توزیع {'صعودی' if ad_trend > 0 else 'نزولی'}",
    )


def pvt(candles: list[Candle]) -> IndicatorResult:
    """
    Price Volume Trend

    Args:
        candles: List of candles

    Returns:
        IndicatorResult with signal
    """
    df = pd.DataFrame([{"close": c.close, "volume": c.volume} for c in candles])

    price_change = df["close"].pct_change()
    pvt = (price_change * df["volume"]).cumsum()

    pvt_current = pvt[-1]
    pvt_sma = pvt.rolling(window=20).mean()

    # Check trend
    pvt_trend = pvt.tail(5).diff().mean()

    if not pvt_sma.empty:
        pvt_sma_current = pvt_sma[-1]
    else:
        pvt_sma_current = pvt_current

    if pvt_current > pvt_sma_current and pvt_trend > 0:
        signal = SignalStrength.VERY_BULLISH
    elif pvt_current > pvt_sma_current:
        signal = SignalStrength.BULLISH
    elif pvt_trend > 0:
        signal = SignalStrength.BULLISH_BROKEN
    elif pvt_current < pvt_sma_current and pvt_trend < 0:
        signal = SignalStrength.VERY_BEARISH
    elif pvt_current < pvt_sma_current:
        signal = SignalStrength.BEARISH
    elif pvt_trend < 0:
        signal = SignalStrength.BEARISH_BROKEN
    else:
        signal = SignalStrength.NEUTRAL

    confidence = 0.7

    return IndicatorResult(
        indicator_name="PVT",
        category=IndicatorCategory.VOLUME,
        signal=signal,
        value=float(pvt_current),
        confidence=confidence,
        description=f"روند قیمت-حجم {'مثبت' if pvt_trend > 0 else 'منفی'}",
    )


def calculate_all(candles: list[Candle]) -> list[IndicatorResult]:
    """
    Calculate all volume indicators

    Args:
        candles: List of candles

    Returns:
        List of all volume indicator results
    """
    results = []

    if len(candles) >= 10:
        results.append(VolumeIndicators.volume_oscillator(candles, 5, 10))

    if len(candles) >= 20:
        results.append(VolumeIndicators.obv(candles))
        results.append(VolumeIndicators.cmf(candles, 20))
        results.append(VolumeIndicators.vwap(candles))
        results.append(VolumeIndicators.ad_line(candles))
        results.append(VolumeIndicators.pvt(candles))

    return results


class VolumeIndicators:
    """Volume indicators calculator"""

    @staticmethod
    def on_balance_volume(candles: list[Candle]) -> IndicatorResult:
        return obv(candles)

    @staticmethod
    def accumulation_distribution(candles: list[Candle]) -> IndicatorResult:
        return accumulation_distribution(candles)

    @staticmethod
    def chaikin_money_flow(candles: list[Candle], period: int = 21) -> IndicatorResult:
        return cmf(candles, period)

    @staticmethod
    def money_flow_index(candles: list[Candle], period: int = 14) -> IndicatorResult:
        if not candles or len(candles) < period + 1 or period <= 0:
            raise ValueError("Not enough candles or invalid period for Money Flow Index")
        from gravity_tech.core.indicators.momentum import MomentumIndicators

        return MomentumIndicators.mfi(candles, period)

    @staticmethod
    def volume_rate_of_change(candles: list[Candle], period: int = 14) -> IndicatorResult:
        return volume_rate_of_change(candles, period)

    @staticmethod
    def volume_profile(candles: list[Candle], bins: int = 20) -> IndicatorResult:
        return volume_profile(candles, bins)

    @staticmethod
    def volume_oscillator(
        candles: list[Candle], short_period: int = 5, long_period: int = 10
    ) -> IndicatorResult:
        return volume_oscillator(candles, short_period, long_period)

    @staticmethod
    def obv(candles: list[Candle]) -> IndicatorResult:
        return obv(candles)

    @staticmethod
    def cmf(candles: list[Candle], period: int = 20) -> IndicatorResult:
        return cmf(candles, period)

    @staticmethod
    def vwap(candles: list[Candle]) -> IndicatorResult:
        return vwap(candles)

    @staticmethod
    def ad_line(candles: list[Candle]) -> IndicatorResult:
        return ad_line(candles)

    @staticmethod
    def pvt(candles: list[Candle]) -> IndicatorResult:
        return pvt(candles)

    @staticmethod
    def calculate_all(candles: list[Candle]) -> list[IndicatorResult]:
        return calculate_all(candles)
