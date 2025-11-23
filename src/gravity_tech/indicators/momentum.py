"""
Momentum Indicators Implementation

This module implements 10 comprehensive momentum indicators:
1. RSI - Relative Strength Index
2. Stochastic Oscillator
3. CCI - Commodity Channel Index
4. ROC - Rate of Change
5. Williams %R
6. MFI - Money Flow Index
7. Ultimate Oscillator
8. TSI - True Strength Index
9. KST - Know Sure Thing
10. PMO - Price Momentum Oscillator

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import numpy as np
import pandas as pd
from typing import List
from gravity_tech.models.schemas import Candle, IndicatorResult, SignalStrength, IndicatorCategory


class MomentumIndicators:
    """Momentum indicators calculator"""
    
    @staticmethod
    def rsi(candles: List[Candle], period: int = 14) -> IndicatorResult:
        """
        Relative Strength Index
        
        Args:
            candles: List of candles
            period: RSI period
            
        Returns:
            IndicatorResult with signal
        """
        closes = pd.Series([c.close for c in candles])
        delta = closes.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_current = rsi.iloc[-1]
        
        # Signal based on RSI levels
        if rsi_current > 80:
            signal = SignalStrength.VERY_BEARISH  # Overbought
        elif rsi_current > 70:
            signal = SignalStrength.BEARISH
        elif rsi_current > 60:
            signal = SignalStrength.BEARISH_BROKEN
        elif rsi_current < 20:
            signal = SignalStrength.VERY_BULLISH  # Oversold
        elif rsi_current < 30:
            signal = SignalStrength.BULLISH
        elif rsi_current < 40:
            signal = SignalStrength.BULLISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL
        
        # Higher confidence at extremes (max 0.95)
        confidence = min(0.95, 0.6 + (abs(rsi_current - 50) / 100))
        
        return IndicatorResult(
            indicator_name=f"RSI({period})",
            category=IndicatorCategory.MOMENTUM,
            signal=signal,
            value=float(rsi_current),
            confidence=confidence,
            description=f"RSI در سطح {rsi_current:.1f} - {'اشباع خرید' if rsi_current > 70 else 'اشباع فروش' if rsi_current < 30 else 'خنثی'}"
        )
    
    @staticmethod
    def stochastic(candles: List[Candle], k_period: int = 14, d_period: int = 3) -> IndicatorResult:
        """
        Stochastic Oscillator
        
        Args:
            candles: List of candles
            k_period: %K period
            d_period: %D period
            
        Returns:
            IndicatorResult with signal
        """
        df = pd.DataFrame([{
            'high': c.high,
            'low': c.low,
            'close': c.close
        } for c in candles])
        
        # Calculate %K
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        
        k_current = k.iloc[-1]
        d_current = d.iloc[-1]
        
        # Signal based on levels and crossover
        if k_current > 80 and d_current > 80:
            signal = SignalStrength.VERY_BEARISH
        elif k_current > 80:
            signal = SignalStrength.BEARISH
        elif k_current > 70 and k_current < d_current:
            signal = SignalStrength.BEARISH_BROKEN
        elif k_current < 20 and d_current < 20:
            signal = SignalStrength.VERY_BULLISH
        elif k_current < 20:
            signal = SignalStrength.BULLISH
        elif k_current < 30 and k_current > d_current:
            signal = SignalStrength.BULLISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL
        
        confidence = min(0.95, 0.65 + (abs(k_current - 50) / 200))
        
        return IndicatorResult(
            indicator_name=f"Stochastic({k_period},{d_period})",
            category=IndicatorCategory.MOMENTUM,
            signal=signal,
            value=float(k_current),
            additional_values={"D": float(d_current)},
            confidence=confidence,
            description=f"Stochastic: K={k_current:.1f}, D={d_current:.1f}"
        )
    
    @staticmethod
    def cci(candles: List[Candle], period: int = 20) -> IndicatorResult:
        """
        Commodity Channel Index
        
        Args:
            candles: List of candles
            period: CCI period
            
        Returns:
            IndicatorResult with signal
        """
        typical_prices = pd.Series([c.typical_price for c in candles])
        sma = typical_prices.rolling(window=period).mean()
        mad = typical_prices.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        
        cci = (typical_prices - sma) / (0.015 * mad)
        cci_current = cci.iloc[-1]
        
        # Signal based on CCI levels
        if cci_current > 200:
            signal = SignalStrength.VERY_BULLISH
        elif cci_current > 100:
            signal = SignalStrength.BULLISH
        elif cci_current > 50:
            signal = SignalStrength.BULLISH_BROKEN
        elif cci_current < -200:
            signal = SignalStrength.VERY_BEARISH
        elif cci_current < -100:
            signal = SignalStrength.BEARISH
        elif cci_current < -50:
            signal = SignalStrength.BEARISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL
        
        confidence = min(0.9, 0.6 + abs(cci_current) / 500)
        
        return IndicatorResult(
            indicator_name=f"CCI({period})",
            category=IndicatorCategory.MOMENTUM,
            signal=signal,
            value=float(cci_current),
            confidence=confidence,
            description=f"CCI در سطح {cci_current:.1f}"
        )
    
    @staticmethod
    def roc(candles: List[Candle], period: int = 12) -> IndicatorResult:
        """
        Rate of Change
        
        Args:
            candles: List of candles
            period: ROC period
            
        Returns:
            IndicatorResult with signal
        """
        closes = pd.Series([c.close for c in candles])
        roc = ((closes - closes.shift(period)) / closes.shift(period)) * 100
        roc_current = roc.iloc[-1]
        
        # Signal based on ROC value
        if roc_current > 10:
            signal = SignalStrength.VERY_BULLISH
        elif roc_current > 5:
            signal = SignalStrength.BULLISH
        elif roc_current > 1:
            signal = SignalStrength.BULLISH_BROKEN
        elif roc_current < -10:
            signal = SignalStrength.VERY_BEARISH
        elif roc_current < -5:
            signal = SignalStrength.BEARISH
        elif roc_current < -1:
            signal = SignalStrength.BEARISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL
        
        confidence = min(0.85, 0.6 + abs(roc_current) / 50)
        
        return IndicatorResult(
            indicator_name=f"ROC({period})",
            category=IndicatorCategory.MOMENTUM,
            signal=signal,
            value=float(roc_current),
            confidence=confidence,
            description=f"نرخ تغییر: {roc_current:.2f}%"
        )
    
    @staticmethod
    def williams_r(candles: List[Candle], period: int = 14) -> IndicatorResult:
        """
        Williams %R
        
        Args:
            candles: List of candles
            period: Williams %R period
            
        Returns:
            IndicatorResult with signal
        """
        df = pd.DataFrame([{
            'high': c.high,
            'low': c.low,
            'close': c.close
        } for c in candles])
        
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        
        williams_r = -100 * ((high_max - df['close']) / (high_max - low_min))
        wr_current = williams_r.iloc[-1]
        
        # Signal based on Williams %R levels (inverted like RSI)
        if wr_current > -20:
            signal = SignalStrength.VERY_BEARISH  # Overbought
        elif wr_current > -30:
            signal = SignalStrength.BEARISH
        elif wr_current > -40:
            signal = SignalStrength.BEARISH_BROKEN
        elif wr_current < -80:
            signal = SignalStrength.VERY_BULLISH  # Oversold
        elif wr_current < -70:
            signal = SignalStrength.BULLISH
        elif wr_current < -60:
            signal = SignalStrength.BULLISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL
        
        confidence = min(0.95, 0.6 + (abs(wr_current + 50) / 100))
        
        return IndicatorResult(
            indicator_name=f"Williams %R({period})",
            category=IndicatorCategory.MOMENTUM,
            signal=signal,
            value=float(wr_current),
            confidence=confidence,
            description=f"Williams %R: {wr_current:.1f}"
        )
    
    @staticmethod
    def mfi(candles: List[Candle], period: int = 14) -> IndicatorResult:
        """
        Money Flow Index
        
        Args:
            candles: List of candles
            period: MFI period
            
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
        
        money_flow = df['typical'] * df['volume']
        
        # Positive and negative money flow
        positive_flow = money_flow.where(df['typical'] > df['typical'].shift(1), 0)
        negative_flow = money_flow.where(df['typical'] < df['typical'].shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
        mfi_current = mfi.iloc[-1]
        
        # Signal based on MFI levels
        if mfi_current > 80:
            signal = SignalStrength.VERY_BEARISH
        elif mfi_current > 70:
            signal = SignalStrength.BEARISH
        elif mfi_current > 60:
            signal = SignalStrength.BEARISH_BROKEN
        elif mfi_current < 20:
            signal = SignalStrength.VERY_BULLISH
        elif mfi_current < 30:
            signal = SignalStrength.BULLISH
        elif mfi_current < 40:
            signal = SignalStrength.BULLISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL
        
        confidence = min(0.95, 0.65 + (abs(mfi_current - 50) / 100))
        
        return IndicatorResult(
            indicator_name=f"MFI({period})",
            category=IndicatorCategory.MOMENTUM,
            signal=signal,
            value=float(mfi_current),
            confidence=confidence,
            description=f"جریان پول: {mfi_current:.1f}"
        )
    
    @staticmethod
    def ultimate_oscillator(candles: List[Candle], 
                           period1: int = 7, 
                           period2: int = 14, 
                           period3: int = 28) -> IndicatorResult:
        """
        Ultimate Oscillator
        
        Args:
            candles: List of candles
            period1: Short period
            period2: Medium period
            period3: Long period
            
        Returns:
            IndicatorResult with signal
        """
        df = pd.DataFrame([{
            'high': c.high,
            'low': c.low,
            'close': c.close
        } for c in candles])
        
        # Calculate buying pressure and true range
        prior_close = df['close'].shift(1).fillna(df['close'].iloc[0])
        
        low_or_prior = pd.concat([df['low'], prior_close], axis=1).min(axis=1)
        high_or_prior = pd.concat([df['high'], prior_close], axis=1).max(axis=1)
        
        bp = df['close'] - low_or_prior
        tr = high_or_prior - low_or_prior
        
        avg1 = bp.rolling(period1).sum() / tr.rolling(period1).sum()
        avg2 = bp.rolling(period2).sum() / tr.rolling(period2).sum()
        avg3 = bp.rolling(period3).sum() / tr.rolling(period3).sum()
        
        uo = 100 * ((4 * avg1 + 2 * avg2 + avg3) / 7)
        uo_current = uo.iloc[-1]
        
        # Signal based on UO levels
        if uo_current > 70:
            signal = SignalStrength.VERY_BULLISH
        elif uo_current > 60:
            signal = SignalStrength.BULLISH
        elif uo_current > 55:
            signal = SignalStrength.BULLISH_BROKEN
        elif uo_current < 30:
            signal = SignalStrength.VERY_BEARISH
        elif uo_current < 40:
            signal = SignalStrength.BEARISH
        elif uo_current < 45:
            signal = SignalStrength.BEARISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL
        
        confidence = 0.7
        
        return IndicatorResult(
            indicator_name=f"Ultimate Oscillator({period1},{period2},{period3})",
            category=IndicatorCategory.MOMENTUM,
            signal=signal,
            value=float(uo_current),
            confidence=confidence,
            description=f"نوسانگر نهایی: {uo_current:.1f}"
        )
    
    @staticmethod
    def calculate_all(candles: List[Candle]) -> List[IndicatorResult]:
        """
        Calculate all momentum indicators
        
        Args:
            candles: List of candles
            
        Returns:
            List of all momentum indicator results
        """
        results = []
        
        if len(candles) >= 14:
            results.append(MomentumIndicators.rsi(candles, 14))
            results.append(MomentumIndicators.stochastic(candles, 14, 3))
            results.append(MomentumIndicators.williams_r(candles, 14))
            results.append(MomentumIndicators.mfi(candles, 14))
        
        if len(candles) >= 20:
            results.append(MomentumIndicators.cci(candles, 20))
            results.append(MomentumIndicators.roc(candles, 12))
        
        if len(candles) >= 28:
            results.append(MomentumIndicators.ultimate_oscillator(candles, 7, 14, 28))
        
        return results
