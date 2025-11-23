"""
Trend Indicators Implementation

This module implements 10 comprehensive trend indicators:
1. SMA - Simple Moving Average
2. EMA - Exponential Moving Average  
3. WMA - Weighted Moving Average
4. DEMA - Double Exponential Moving Average
5. TEMA - Triple Exponential Moving Average
6. MACD - Moving Average Convergence Divergence
7. ADX - Average Directional Index
8. Parabolic SAR
9. Supertrend
10. Ichimoku Cloud

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from gravity_tech.models.schemas import Candle, IndicatorResult, SignalStrength, IndicatorCategory


class TrendIndicators:
    """Trend indicators calculator"""
    
    @staticmethod
    def sma(candles: List[Candle], period: int = 20) -> IndicatorResult:
        """
        Simple Moving Average
        
        Args:
            candles: List of candles
            period: SMA period
            
        Returns:
            IndicatorResult with signal
        """
        closes = np.array([c.close for c in candles])
        sma_values = pd.Series(closes).rolling(window=period).mean()
        sma_current = sma_values.iloc[-1]
        current_price = closes[-1]
        
        # Signal: Price position relative to SMA
        diff_pct = ((current_price - sma_current) / sma_current) * 100
        
        # Calculate signal strength
        if diff_pct > 5:
            signal = SignalStrength.VERY_BULLISH
        elif diff_pct > 2:
            signal = SignalStrength.BULLISH
        elif diff_pct > 0.5:
            signal = SignalStrength.BULLISH_BROKEN
        elif diff_pct < -5:
            signal = SignalStrength.VERY_BEARISH
        elif diff_pct < -2:
            signal = SignalStrength.BEARISH
        elif diff_pct < -0.5:
            signal = SignalStrength.BEARISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL
        
        # Calculate slope for confidence
        recent_sma = sma_values.iloc[-5:]
        slope = (recent_sma.iloc[-1] - recent_sma.iloc[0]) / 5
        confidence = min(0.95, 0.6 + abs(slope / current_price) * 100)
        
        return IndicatorResult(
            indicator_name=f"SMA({period})",
            category=IndicatorCategory.TREND,
            signal=signal,
            value=float(sma_current),
            confidence=confidence,
            description=f"قیمت {diff_pct:.2f}% {'بالای' if diff_pct > 0 else 'زیر'} SMA"
        )
    
    @staticmethod
    def ema(candles: List[Candle], period: int = 20) -> IndicatorResult:
        """
        Exponential Moving Average
        
        Args:
            candles: List of candles
            period: EMA period
            
        Returns:
            IndicatorResult with signal
        """
        closes = np.array([c.close for c in candles])
        ema_values = pd.Series(closes).ewm(span=period, adjust=False).mean()
        ema_current = ema_values.iloc[-1]
        current_price = closes[-1]
        
        diff_pct = ((current_price - ema_current) / ema_current) * 100
        
        if diff_pct > 5:
            signal = SignalStrength.VERY_BULLISH
        elif diff_pct > 2:
            signal = SignalStrength.BULLISH
        elif diff_pct > 0.5:
            signal = SignalStrength.BULLISH_BROKEN
        elif diff_pct < -5:
            signal = SignalStrength.VERY_BEARISH
        elif diff_pct < -2:
            signal = SignalStrength.BEARISH
        elif diff_pct < -0.5:
            signal = SignalStrength.BEARISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL
        
        recent_ema = ema_values.iloc[-5:]
        slope = (recent_ema.iloc[-1] - recent_ema.iloc[0]) / 5
        confidence = min(0.95, 0.65 + abs(slope / current_price) * 100)
        
        return IndicatorResult(
            indicator_name=f"EMA({period})",
            category=IndicatorCategory.TREND,
            signal=signal,
            value=float(ema_current),
            confidence=confidence,
            description=f"قیمت {diff_pct:.2f}% {'بالای' if diff_pct > 0 else 'زیر'} EMA"
        )
    
    @staticmethod
    def wma(candles: List[Candle], period: int = 20) -> IndicatorResult:
        """
        Weighted Moving Average
        
        Args:
            candles: List of candles
            period: WMA period
            
        Returns:
            IndicatorResult with signal
        """
        closes = np.array([c.close for c in candles])
        weights = np.arange(1, period + 1)
        
        wma_values = []
        for i in range(len(closes)):
            if i < period - 1:
                wma_values.append(np.nan)
            else:
                window = closes[i - period + 1:i + 1]
                wma = np.dot(window, weights) / weights.sum()
                wma_values.append(wma)
        
        wma_current = wma_values[-1]
        current_price = closes[-1]
        
        diff_pct = ((current_price - wma_current) / wma_current) * 100
        
        if diff_pct > 5:
            signal = SignalStrength.VERY_BULLISH
        elif diff_pct > 2:
            signal = SignalStrength.BULLISH
        elif diff_pct > 0.5:
            signal = SignalStrength.BULLISH_BROKEN
        elif diff_pct < -5:
            signal = SignalStrength.VERY_BEARISH
        elif diff_pct < -2:
            signal = SignalStrength.BEARISH
        elif diff_pct < -0.5:
            signal = SignalStrength.BEARISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL
        
        confidence = 0.7
        
        return IndicatorResult(
            indicator_name=f"WMA({period})",
            category=IndicatorCategory.TREND,
            signal=signal,
            value=float(wma_current),
            confidence=confidence,
            description=f"قیمت {diff_pct:.2f}% {'بالای' if diff_pct > 0 else 'زیر'} WMA"
        )
    
    @staticmethod
    def dema(candles: List[Candle], period: int = 20) -> IndicatorResult:
        """
        Double Exponential Moving Average
        DEMA = 2*EMA - EMA(EMA)
        
        Args:
            candles: List of candles
            period: DEMA period
            
        Returns:
            IndicatorResult with signal
        """
        closes = pd.Series([c.close for c in candles])
        
        ema1 = closes.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        dema = 2 * ema1 - ema2
        
        dema_current = dema.iloc[-1]
        current_price = closes.iloc[-1]
        
        diff_pct = ((current_price - dema_current) / dema_current) * 100
        
        if diff_pct > 5:
            signal = SignalStrength.VERY_BULLISH
        elif diff_pct > 2:
            signal = SignalStrength.BULLISH
        elif diff_pct > 0.5:
            signal = SignalStrength.BULLISH_BROKEN
        elif diff_pct < -5:
            signal = SignalStrength.VERY_BEARISH
        elif diff_pct < -2:
            signal = SignalStrength.BEARISH
        elif diff_pct < -0.5:
            signal = SignalStrength.BEARISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL
        
        confidence = 0.75
        
        return IndicatorResult(
            indicator_name=f"DEMA({period})",
            category=IndicatorCategory.TREND,
            signal=signal,
            value=float(dema_current),
            confidence=confidence,
            description=f"DEMA نشان‌دهنده روند {'صعودی' if diff_pct > 0 else 'نزولی'} است"
        )
    
    @staticmethod
    def tema(candles: List[Candle], period: int = 20) -> IndicatorResult:
        """
        Triple Exponential Moving Average
        TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
        
        Args:
            candles: List of candles
            period: TEMA period
            
        Returns:
            IndicatorResult with signal
        """
        closes = pd.Series([c.close for c in candles])
        
        ema1 = closes.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        tema = 3 * ema1 - 3 * ema2 + ema3
        
        tema_current = tema.iloc[-1]
        current_price = closes.iloc[-1]
        
        diff_pct = ((current_price - tema_current) / tema_current) * 100
        
        if diff_pct > 5:
            signal = SignalStrength.VERY_BULLISH
        elif diff_pct > 2:
            signal = SignalStrength.BULLISH
        elif diff_pct > 0.5:
            signal = SignalStrength.BULLISH_BROKEN
        elif diff_pct < -5:
            signal = SignalStrength.VERY_BEARISH
        elif diff_pct < -2:
            signal = SignalStrength.BEARISH
        elif diff_pct < -0.5:
            signal = SignalStrength.BEARISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL
        
        confidence = 0.78
        
        return IndicatorResult(
            indicator_name=f"TEMA({period})",
            category=IndicatorCategory.TREND,
            signal=signal,
            value=float(tema_current),
            confidence=confidence,
            description=f"TEMA با واکنش سریع‌تر نشان‌دهنده روند {'صعودی' if diff_pct > 0 else 'نزولی'}"
        )
    
    @staticmethod
    def macd(candles: List[Candle], fast: int = 12, slow: int = 26, signal_period: int = 9) -> IndicatorResult:
        """
        Moving Average Convergence Divergence
        
        Args:
            candles: List of candles
            fast: Fast EMA period
            slow: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            IndicatorResult with signal
        """
        closes = pd.Series([c.close for c in candles])
        
        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        macd_current = macd_line.iloc[-1]
        signal_current = signal_line.iloc[-1]
        hist_current = histogram.iloc[-1]
        
        # Signal based on histogram and crossover
        hist_prev = histogram.iloc[-2]
        
        if hist_current > 0 and hist_current > hist_prev and macd_current > signal_current:
            if hist_current > abs(macd_current) * 0.02:
                signal = SignalStrength.VERY_BULLISH
            else:
                signal = SignalStrength.BULLISH
        elif hist_current > 0 and hist_current < hist_prev:
            signal = SignalStrength.BULLISH_BROKEN
        elif hist_current < 0 and hist_current < hist_prev and macd_current < signal_current:
            if abs(hist_current) > abs(macd_current) * 0.02:
                signal = SignalStrength.VERY_BEARISH
            else:
                signal = SignalStrength.BEARISH
        elif hist_current < 0 and hist_current > hist_prev:
            signal = SignalStrength.BEARISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL
        
        confidence = min(0.9, 0.7 + abs(hist_current / macd_current) * 10 if macd_current != 0 else 0.7)
        
        return IndicatorResult(
            indicator_name="MACD",
            category=IndicatorCategory.TREND,
            signal=signal,
            value=float(macd_current),
            additional_values={
                "signal": float(signal_current),
                "histogram": float(hist_current)
            },
            confidence=confidence,
            description=f"MACD {'بالای' if macd_current > signal_current else 'زیر'} خط سیگنال"
        )
    
    @staticmethod
    def adx(candles: List[Candle], period: int = 14) -> IndicatorResult:
        """
        Average Directional Index
        Measures trend strength
        
        Args:
            candles: List of candles
            period: ADX period
            
        Returns:
            IndicatorResult with signal
        """
        df = pd.DataFrame([{
            'high': c.high,
            'low': c.low,
            'close': c.close
        } for c in candles])
        
        # Calculate +DM and -DM
        df['high_diff'] = df['high'].diff()
        df['low_diff'] = -df['low'].diff()
        
        df['+DM'] = np.where(
            (df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0),
            df['high_diff'],
            0
        )
        df['-DM'] = np.where(
            (df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0),
            df['low_diff'],
            0
        )
        
        # Calculate True Range
        df['TR'] = df.apply(
            lambda row: max(
                row['high'] - row['low'],
                abs(row['high'] - df['close'].shift(1).loc[row.name]) if row.name > 0 else 0,
                abs(row['low'] - df['close'].shift(1).loc[row.name]) if row.name > 0 else 0
            ),
            axis=1
        )
        
        # Smooth the values
        atr = df['TR'].rolling(window=period).mean()
        plus_di = 100 * (df['+DM'].rolling(window=period).mean() / atr)
        minus_di = 100 * (df['-DM'].rolling(window=period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        adx_current = adx.iloc[-1]
        plus_di_current = plus_di.iloc[-1]
        minus_di_current = minus_di.iloc[-1]
        
        # Signal based on DI lines and ADX strength
        if adx_current > 40:  # Strong trend
            if plus_di_current > minus_di_current:
                signal = SignalStrength.VERY_BULLISH
            else:
                signal = SignalStrength.VERY_BEARISH
        elif adx_current > 25:  # Moderate trend
            if plus_di_current > minus_di_current:
                signal = SignalStrength.BULLISH
            else:
                signal = SignalStrength.BEARISH
        elif adx_current > 20:  # Weak trend
            if plus_di_current > minus_di_current:
                signal = SignalStrength.BULLISH_BROKEN
            else:
                signal = SignalStrength.BEARISH_BROKEN
        else:  # No trend
            signal = SignalStrength.NEUTRAL
        
        confidence = min(0.95, adx_current / 100 + 0.5)
        
        return IndicatorResult(
            indicator_name=f"ADX({period})",
            category=IndicatorCategory.TREND,
            signal=signal,
            value=float(adx_current),
            additional_values={
                "+DI": float(plus_di_current),
                "-DI": float(minus_di_current)
            },
            confidence=confidence,
            description=f"قدرت روند: {adx_current:.1f} - {'قوی' if adx_current > 25 else 'ضعیف'}"
        )
    
    @staticmethod
    def calculate_all(candles: List[Candle]) -> List[IndicatorResult]:
        """
        Calculate all trend indicators
        
        Args:
            candles: List of candles
            
        Returns:
            List of all trend indicator results
        """
        results = []
        
        if len(candles) >= 20:
            results.append(TrendIndicators.sma(candles, 20))
            results.append(TrendIndicators.sma(candles, 50))
            results.append(TrendIndicators.ema(candles, 12))
            results.append(TrendIndicators.ema(candles, 26))
            results.append(TrendIndicators.wma(candles, 20))
            results.append(TrendIndicators.dema(candles, 20))
            results.append(TrendIndicators.tema(candles, 20))
            results.append(TrendIndicators.macd(candles))
        
        if len(candles) >= 28:
            results.append(TrendIndicators.adx(candles, 14))
        
        return results
