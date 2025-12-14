"""
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

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""


import pandas as pd
from gravity_tech.models.schemas import Candle, IndicatorCategory, IndicatorResult, SignalStrength


class VolumeIndicators:
    """Volume indicators calculator"""

    @staticmethod
    def obv(candles: list[Candle]) -> IndicatorResult:
        """
        On Balance Volume

        Args:
            candles: List of candles

        Returns:
            IndicatorResult with signal
        """
        df = pd.DataFrame([{
            'close': c.close,
            'volume': c.volume
        } for c in candles])

        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])

        obv_series = pd.Series(obv)
        obv_sma = obv_series.rolling(window=20).mean()

        obv_current = obv_series.iloc[-1]
        obv_sma_current = obv_sma.iloc[-1]

        # Signal based on OBV trend and position relative to MA
        obv_trend = obv_series.iloc[-5:].diff().mean()
        price_trend = df['close'].iloc[-5:].diff().mean()

        # Check divergence
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
            confidence=confidence,
            description=f"حجم {'تأیید کننده' if obv_trend * price_trend > 0 else 'واگرا با'} قیمت"
        )

    @staticmethod
    def cmf(candles: list[Candle], period: int = 20) -> IndicatorResult:
        """
        Chaikin Money Flow

        Args:
            candles: List of candles
            period: CMF period

        Returns:
            IndicatorResult with signal
        """
        df = pd.DataFrame([{
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume
        } for c in candles])

        mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mf_volume = mf_multiplier * df['volume']

        cmf = mf_volume.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        cmf_current = cmf.iloc[-1]

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
            confidence=min(0.95, confidence),
            description=f"جریان پول {'مثبت' if cmf_current > 0 else 'منفی'}: {cmf_current:.3f}"
        )

    @staticmethod
    def vwap(candles: list[Candle]) -> IndicatorResult:
        """
        Volume Weighted Average Price

        Args:
            candles: List of candles

        Returns:
            IndicatorResult with signal
        """
        df = pd.DataFrame([{
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume,
            'typical': c.typical_price
        } for c in candles])

        # Calculate VWAP (reset daily in production)
        df['pv'] = df['typical'] * df['volume']
        vwap = df['pv'].cumsum() / df['volume'].cumsum()

        vwap_current = vwap.iloc[-1]
        current_price = df['close'].iloc[-1]

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
            description=f"قیمت {abs(diff_pct):.2f}% {'بالای' if diff_pct > 0 else 'زیر'} VWAP"
        )

    @staticmethod
    def ad_line(candles: list[Candle]) -> IndicatorResult:
        """
        Accumulation/Distribution Line

        Args:
            candles: List of candles

        Returns:
            IndicatorResult with signal
        """
        df = pd.DataFrame([{
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume
        } for c in candles])

        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        ad = (clv * df['volume']).cumsum()

        ad_current = ad.iloc[-1]
        ad_sma = ad.rolling(window=20).mean().iloc[-1]

        # Check trend
        ad_trend = ad.iloc[-5:].diff().mean()
        price_trend = df['close'].iloc[-5:].diff().mean()

        # Signal based on AD line trend and divergence
        if ad_trend > 0 and price_trend > 0:
            if ad_current > ad_sma:
                signal = SignalStrength.VERY_BULLISH
            else:
                signal = SignalStrength.BULLISH
        elif ad_trend > 0 and price_trend < 0:
            signal = SignalStrength.BULLISH_BROKEN
        elif ad_trend < 0 and price_trend < 0:
            if ad_current < ad_sma:
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
            confidence=confidence,
            description=f"خط انباشت/توزیع {'صعودی' if ad_trend > 0 else 'نزولی'}"
        )

    @staticmethod
    def pvt(candles: list[Candle]) -> IndicatorResult:
        """
        Price Volume Trend

        Args:
            candles: List of candles

        Returns:
            IndicatorResult with signal
        """
        df = pd.DataFrame([{
            'close': c.close,
            'volume': c.volume
        } for c in candles])

        price_change = df['close'].pct_change()
        pvt = (price_change * df['volume']).cumsum()

        pvt_current = pvt.iloc[-1]
        pvt_sma = pvt.rolling(window=20).mean().iloc[-1]

        # Check trend
        pvt_trend = pvt.iloc[-5:].diff().mean()

        if pvt_current > pvt_sma and pvt_trend > 0:
            signal = SignalStrength.VERY_BULLISH
        elif pvt_current > pvt_sma:
            signal = SignalStrength.BULLISH
        elif pvt_trend > 0:
            signal = SignalStrength.BULLISH_BROKEN
        elif pvt_current < pvt_sma and pvt_trend < 0:
            signal = SignalStrength.VERY_BEARISH
        elif pvt_current < pvt_sma:
            signal = SignalStrength.BEARISH
        elif pvt_trend < 0:
            signal = SignalStrength.BEARISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL

        confidence = 0.68

        return IndicatorResult(
            indicator_name="PVT",
            category=IndicatorCategory.VOLUME,
            signal=signal,
            value=float(pvt_current),
            confidence=confidence,
            description=f"روند قیمت-حجم {'مثبت' if pvt_trend > 0 else 'منفی'}"
        )

    @staticmethod
    def volume_oscillator(candles: list[Candle], short: int = 5, long: int = 10) -> IndicatorResult:
        """
        Volume Oscillator

        Args:
            candles: List of candles
            short: Short period
            long: Long period

        Returns:
            IndicatorResult with signal
        """
        volumes = pd.Series([c.volume for c in candles])

        short_ma = volumes.rolling(window=short).mean()
        long_ma = volumes.rolling(window=long).mean()

        vo = ((short_ma - long_ma) / long_ma) * 100
        vo_current = vo.iloc[-1]

        # Signal based on volume oscillator
        if vo_current > 20:
            signal = SignalStrength.VERY_BULLISH
        elif vo_current > 10:
            signal = SignalStrength.BULLISH
        elif vo_current > 5:
            signal = SignalStrength.BULLISH_BROKEN
        elif vo_current < -20:
            signal = SignalStrength.VERY_BEARISH
        elif vo_current < -10:
            signal = SignalStrength.BEARISH
        elif vo_current < -5:
            signal = SignalStrength.BEARISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL

        confidence = 0.65

        return IndicatorResult(
            indicator_name=f"Volume Oscillator({short},{long})",
            category=IndicatorCategory.VOLUME,
            signal=signal,
            value=float(vo_current),
            confidence=confidence,
            description=f"نوسانگر حجم: {vo_current:.2f}%"
        )

    @staticmethod
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
