"""
Market Phase Analysis

Based on Dow Theory and technical analysis principles from Mark Andrew Lim.
This module identifies the current market phase:
1. Accumulation Phase (فاز انباشت)
2. Markup Phase (فاز صعود)
3. Distribution Phase (فاز توزیع)
4. Markdown Phase (فاز نزول)

All analysis adheres to Dow Theory principles:
- The market has three trends (primary, secondary, minor)
- Trends have three phases
- The market discounts all news
- Volume confirms the trend
- Trends persist until definitive reversal signals appear

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from enum import Enum

from gravity_tech.models.schemas import Candle, SignalStrength
from gravity_tech.indicators.trend import TrendIndicators
from gravity_tech.indicators.momentum import MomentumIndicators
from gravity_tech.indicators.volume import VolumeIndicators


class MarketPhase(str, Enum):
    """Market phases based on Dow Theory"""
    ACCUMULATION = "انباشت"  # Accumulation
    MARKUP = "صعود"  # Markup (Bullish trend)
    DISTRIBUTION = "توزیع"  # Distribution
    MARKDOWN = "نزول"  # Markdown (Bearish trend)
    TRANSITION = "انتقال"  # Transition (unclear phase)


class PhaseStrength(str, Enum):
    """Strength of current phase"""
    VERY_STRONG = "بسیار قوی"
    STRONG = "قوی"
    MODERATE = "متوسط"
    WEAK = "ضعیف"
    VERY_WEAK = "بسیار ضعیف"


class MarketPhaseAnalysis:
    """Analyze market phase based on Dow Theory"""
    
    def __init__(self):
        self.trend_analyzer = TrendIndicators()
        self.momentum_analyzer = MomentumIndicators()
        self.volume_analyzer = VolumeIndicators()
    
    @staticmethod
    def identify_trend_structure(candles: List[Candle], period: int = 20) -> Dict:
        """
        Identify higher highs, higher lows, lower highs, lower lows
        Core principle of Dow Theory
        
        Args:
            candles: List of candles
            period: Period for swing analysis
            
        Returns:
            Dictionary with trend structure information
        """
        if len(candles) < period * 2:
            return {"structure": "insufficient_data"}
        
        # Find recent swing highs and lows
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        
        # Find local peaks and troughs
        swing_highs = []
        swing_lows = []
        
        window = 5
        for i in range(window, len(candles) - window):
            # Check for swing high
            if all(highs[i] >= highs[j] for j in range(i-window, i+window+1) if j != i):
                swing_highs.append((i, highs[i]))
            
            # Check for swing low
            if all(lows[i] <= lows[j] for j in range(i-window, i+window+1) if j != i):
                swing_lows.append((i, lows[i]))
        
        # Analyze trend structure
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Check for higher highs and higher lows (uptrend)
            recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
            recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
            
            higher_highs = all(recent_highs[i][1] < recent_highs[i+1][1] 
                              for i in range(len(recent_highs)-1))
            higher_lows = all(recent_lows[i][1] < recent_lows[i+1][1] 
                             for i in range(len(recent_lows)-1))
            
            lower_highs = all(recent_highs[i][1] > recent_highs[i+1][1] 
                             for i in range(len(recent_highs)-1))
            lower_lows = all(recent_lows[i][1] > recent_lows[i+1][1] 
                            for i in range(len(recent_lows)-1))
            
            if higher_highs and higher_lows:
                structure = "uptrend"
            elif lower_highs and lower_lows:
                structure = "downtrend"
            elif higher_highs and lower_lows:
                structure = "expansion"  # Volatility expansion
            elif lower_highs and higher_lows:
                structure = "contraction"  # Range contraction
            else:
                structure = "mixed"
        else:
            structure = "insufficient_swings"
        
        return {
            "structure": structure,
            "swing_highs": swing_highs,
            "swing_lows": swing_lows,
            "last_high": swing_highs[-1][1] if swing_highs else None,
            "last_low": swing_lows[-1][1] if swing_lows else None
        }
    
    @staticmethod
    def analyze_volume_behavior(candles: List[Candle], period: int = 20) -> Dict:
        """
        Analyze volume behavior - key Dow Theory principle
        Volume should confirm the trend
        
        Args:
            candles: List of candles
            period: Analysis period
            
        Returns:
            Dictionary with volume analysis
        """
        if len(candles) < period:
            return {"status": "insufficient_data"}
        
        recent = candles[-period:]
        volumes = [c.volume for c in recent]
        closes = [c.close for c in recent]
        
        avg_volume = np.mean(volumes)
        
        # Volume on up days vs down days
        up_days_volume = []
        down_days_volume = []
        
        for i in range(1, len(recent)):
            if recent[i].close > recent[i-1].close:
                up_days_volume.append(recent[i].volume)
            elif recent[i].close < recent[i-1].close:
                down_days_volume.append(recent[i].volume)
        
        avg_up_volume = np.mean(up_days_volume) if up_days_volume else 0
        avg_down_volume = np.mean(down_days_volume) if down_days_volume else 0
        
        # Volume trend
        first_half_vol = np.mean(volumes[:period//2])
        second_half_vol = np.mean(volumes[period//2:])
        volume_trend = "increasing" if second_half_vol > first_half_vol * 1.1 else \
                      "decreasing" if second_half_vol < first_half_vol * 0.9 else "stable"
        
        return {
            "status": "analyzed",
            "avg_volume": avg_volume,
            "avg_up_volume": avg_up_volume,
            "avg_down_volume": avg_down_volume,
            "volume_trend": volume_trend,
            "up_volume_dominance": avg_up_volume > avg_down_volume * 1.2,
            "down_volume_dominance": avg_down_volume > avg_up_volume * 1.2
        }
    
    @staticmethod
    def calculate_price_momentum(candles: List[Candle], periods: List[int] = [10, 20, 50]) -> Dict:
        """
        Calculate price momentum across multiple timeframes
        
        Args:
            candles: List of candles
            periods: Periods to analyze
            
        Returns:
            Dictionary with momentum information
        """
        if len(candles) < max(periods):
            return {"status": "insufficient_data"}
        
        closes = [c.close for c in candles]
        current_price = closes[-1]
        
        momentum = {}
        for period in periods:
            if len(closes) >= period:
                past_price = closes[-period]
                change = ((current_price - past_price) / past_price) * 100
                momentum[f"period_{period}"] = {
                    "change_pct": change,
                    "direction": "up" if change > 0 else "down" if change < 0 else "flat"
                }
        
        return momentum
    
    def identify_phase(self, candles: List[Candle]) -> Tuple[MarketPhase, PhaseStrength, Dict]:
        """
        Identify current market phase based on Dow Theory
        
        Args:
            candles: List of candles (minimum 100 recommended)
            
        Returns:
            Tuple of (MarketPhase, PhaseStrength, detailed_analysis)
        """
        if len(candles) < 50:
            return MarketPhase.TRANSITION, PhaseStrength.WEAK, {"error": "insufficient_data"}
        
        # Analyze components
        trend_structure = self.identify_trend_structure(candles, period=20)
        volume_behavior = self.analyze_volume_behavior(candles, period=20)
        momentum = self.calculate_price_momentum(candles, periods=[10, 20, 50])
        
        # Get indicator signals
        sma_20 = TrendIndicators.sma(candles, 20)
        sma_50 = TrendIndicators.sma(candles, 50)
        ema_20 = TrendIndicators.ema(candles, 20)
        
        macd = TrendIndicators.macd(candles)
        rsi = MomentumIndicators.rsi(candles, 14)
        
        obv = VolumeIndicators.obv(candles)
        
        closes = [c.close for c in candles]
        current_price = closes[-1]
        
        # Score different aspects (0-100)
        scores = {
            "trend": 50,  # Neutral start
            "volume": 50,
            "momentum": 50,
            "price_position": 50
        }
        
        # 1. Trend Structure Analysis
        if trend_structure["structure"] == "uptrend":
            scores["trend"] = 80
        elif trend_structure["structure"] == "downtrend":
            scores["trend"] = 20
        elif trend_structure["structure"] == "contraction":
            scores["trend"] = 50  # Range-bound
        
        # 2. Volume Analysis (Dow Theory: Volume confirms trend)
        if volume_behavior["status"] == "analyzed":
            if volume_behavior["up_volume_dominance"]:
                scores["volume"] += 20
            elif volume_behavior["down_volume_dominance"]:
                scores["volume"] -= 20
            
            if volume_behavior["volume_trend"] == "increasing":
                # Volume increasing can mean either accumulation or distribution
                # Depends on price action
                pass
        
        # 3. Momentum Analysis
        if momentum.get("period_20"):
            change = momentum["period_20"]["change_pct"]
            if change > 5:
                scores["momentum"] = 80
            elif change > 2:
                scores["momentum"] = 65
            elif change < -5:
                scores["momentum"] = 20
            elif change < -2:
                scores["momentum"] = 35
        
        # 4. Price Position relative to moving averages
        if current_price > sma_50.value:
            scores["price_position"] += 20
        else:
            scores["price_position"] -= 20
        
        if current_price > sma_20.value:
            scores["price_position"] += 10
        else:
            scores["price_position"] -= 10
        
        # Determine Phase based on scores
        overall_score = np.mean(list(scores.values()))
        
        # Phase identification logic
        phase = MarketPhase.TRANSITION
        strength = PhaseStrength.MODERATE
        
        # MARKUP PHASE (Bullish Trend)
        # - Higher highs and higher lows
        # - Price above moving averages
        # - Volume increasing on rallies
        # - Strong momentum
        if (trend_structure["structure"] == "uptrend" and 
            overall_score > 60 and
            scores["momentum"] > 60):
            phase = MarketPhase.MARKUP
            
            if overall_score > 75:
                strength = PhaseStrength.VERY_STRONG
            elif overall_score > 65:
                strength = PhaseStrength.STRONG
            else:
                strength = PhaseStrength.MODERATE
        
        # MARKDOWN PHASE (Bearish Trend)
        # - Lower highs and lower lows
        # - Price below moving averages
        # - Volume increasing on declines
        # - Weak momentum
        elif (trend_structure["structure"] == "downtrend" and 
              overall_score < 40 and
              scores["momentum"] < 40):
            phase = MarketPhase.MARKDOWN
            
            if overall_score < 25:
                strength = PhaseStrength.VERY_STRONG
            elif overall_score < 35:
                strength = PhaseStrength.STRONG
            else:
                strength = PhaseStrength.MODERATE
        
        # ACCUMULATION PHASE
        # - Range-bound price action (contraction)
        # - Price near support
        # - Volume decreasing (quiet period)
        # - Coming after downtrend
        # - Smart money buying
        elif (trend_structure["structure"] in ["contraction", "mixed"] and
              40 <= overall_score <= 60 and
              volume_behavior.get("volume_trend") != "increasing"):
            
            # Check if coming from downtrend (look at longer-term trend)
            if len(candles) >= 100:
                long_term_change = ((closes[-1] - closes[-100]) / closes[-100]) * 100
                if long_term_change < 0:  # Coming from decline
                    phase = MarketPhase.ACCUMULATION
                    
                    # Check for accumulation signs
                    if (scores["volume"] > 50 and 
                        rsi.value < 50 and 
                        volume_behavior.get("volume_trend") == "stable"):
                        strength = PhaseStrength.STRONG
                    else:
                        strength = PhaseStrength.MODERATE
        
        # DISTRIBUTION PHASE
        # - Range-bound after uptrend
        # - Price near resistance
        # - Volume may be high (smart money selling)
        # - Momentum weakening
        elif (trend_structure["structure"] in ["contraction", "mixed", "expansion"] and
              overall_score > 50 and
              scores["momentum"] < 60):
            
            # Check if coming from uptrend
            if len(candles) >= 100:
                long_term_change = ((closes[-1] - closes[-100]) / closes[-100]) * 100
                if long_term_change > 0:  # Coming from rally
                    phase = MarketPhase.DISTRIBUTION
                    
                    # Check for distribution signs
                    if (volume_behavior.get("volume_trend") == "increasing" and
                        rsi.value > 50 and
                        scores["momentum"] < 55):
                        strength = PhaseStrength.STRONG
                    else:
                        strength = PhaseStrength.MODERATE
        
        # Compile detailed analysis
        detailed_analysis = {
            "phase": phase.value,
            "strength": strength.value,
            "scores": scores,
            "overall_score": overall_score,
            "trend_structure": trend_structure["structure"],
            "volume_behavior": volume_behavior,
            "momentum_data": momentum,
            "current_price": current_price,
            "sma_20": sma_20.value,
            "sma_50": sma_50.value,
            "rsi": rsi.value,
            "macd_histogram": macd.additional_values.get("histogram") if macd.additional_values else None,
            "indicators_alignment": {
                "price_above_sma20": current_price > sma_20.value,
                "price_above_sma50": current_price > sma_50.value,
                "sma20_above_sma50": sma_20.value > sma_50.value,
                "rsi_overbought": rsi.value > 70,
                "rsi_oversold": rsi.value < 30
            }
        }
        
        return phase, strength, detailed_analysis
    
    def generate_analysis_report(self, candles: List[Candle]) -> Dict:
        """
        Generate comprehensive market phase analysis report
        
        Args:
            candles: List of candles
            
        Returns:
            Complete analysis report
        """
        phase, strength, detailed = self.identify_phase(candles)
        
        # Generate Persian description
        descriptions = {
            MarketPhase.ACCUMULATION: "بازار در فاز انباشت است. سرمایه‌گذاران باتجربه در حال خرید تدریجی هستند. این فاز معمولاً پس از یک روند نزولی اتفاق می‌افتد.",
            MarketPhase.MARKUP: "بازار در فاز صعودی است. روند صعودی قوی و عموم سرمایه‌گذاران وارد بازار می‌شوند. قیمت‌ها در حال افزایش هستند.",
            MarketPhase.DISTRIBUTION: "بازار در فاز توزیع است. سرمایه‌گذاران باتجربه در حال فروش تدریجی هستند. این فاز معمولاً پس از یک روند صعودی اتفاق می‌افتد.",
            MarketPhase.MARKDOWN: "بازار در فاز نزولی است. روند نزولی قوی و فشار فروش بالا است. قیمت‌ها در حال کاهش هستند.",
            MarketPhase.TRANSITION: "بازار در فاز انتقالی است. جهت واضحی وجود ندارد و بازار در حال تصمیم‌گیری برای حرکت بعدی است."
        }
        
        # Trading recommendations based on Dow Theory
        recommendations = self._generate_recommendations(phase, strength, detailed)
        
        return {
            "timestamp": datetime.utcnow(),
            "market_phase": phase.value,
            "phase_strength": strength.value,
            "description": descriptions[phase],
            "detailed_analysis": detailed,
            "recommendations": recommendations,
            "dow_theory_compliance": True  # Always compliant with Dow Theory
        }
    
    def _generate_recommendations(self, phase: MarketPhase, 
                                  strength: PhaseStrength, 
                                  detailed: Dict) -> List[str]:
        """Generate trading recommendations based on phase"""
        recommendations = []
        
        if phase == MarketPhase.ACCUMULATION:
            recommendations.append("فرصت مناسب برای خرید تدریجی و ایجاد موقعیت")
            recommendations.append("حجم معاملات پایین است - نشانه انباشت")
            recommendations.append("صبر برای تایید شروع روند صعودی")
            if strength in [PhaseStrength.STRONG, PhaseStrength.VERY_STRONG]:
                recommendations.append("سیگنال‌های انباشت قوی - افزایش سایز موقعیت")
        
        elif phase == MarketPhase.MARKUP:
            recommendations.append("ادامه نگهداری موقعیت‌های خرید")
            recommendations.append("استفاده از استراتژی trend-following")
            recommendations.append("جابجایی حد ضرر به بالای نقاط پایین قبلی")
            if strength == PhaseStrength.VERY_STRONG:
                recommendations.append("روند بسیار قوی - نگهداری موقعیت‌ها")
            elif strength in [PhaseStrength.WEAK, PhaseStrength.VERY_WEAK]:
                recommendations.append("روند در حال ضعیف شدن - آماده خروج باشید")
        
        elif phase == MarketPhase.DISTRIBUTION:
            recommendations.append("زمان کاهش تدریجی موقعیت‌های خرید")
            recommendations.append("حجم بالا با حرکت جانبی - نشانه توزیع")
            recommendations.append("از ورود به موقعیت‌های جدید خودداری کنید")
            if strength in [PhaseStrength.STRONG, PhaseStrength.VERY_STRONG]:
                recommendations.append("سیگنال‌های توزیع قوی - خروج از موقعیت‌ها")
        
        elif phase == MarketPhase.MARKDOWN:
            recommendations.append("از ورود به موقعیت‌های خرید خودداری کنید")
            recommendations.append("در صورت مهارت، موقعیت‌های فروش در نظر بگیرید")
            recommendations.append("منتظر علائم پایان روند نزولی باشید")
            if strength == PhaseStrength.VERY_STRONG:
                recommendations.append("روند نزولی بسیار قوی - دوری از بازار")
        
        else:  # TRANSITION
            recommendations.append("بازار در وضعیت نامشخص - انتظار برای شفافیت")
            recommendations.append("کاهش سایز معاملات")
            recommendations.append("منتظر تایید جهت جدید باشید")
        
        # Add indicator-based recommendations
        if detailed.get("indicators_alignment"):
            align = detailed["indicators_alignment"]
            if align.get("rsi_oversold"):
                recommendations.append("RSI در ناحیه اشباع فروش - احتمال بازگشت")
            elif align.get("rsi_overbought"):
                recommendations.append("RSI در ناحیه اشباع خرید - احتمال اصلاح")
        
        return recommendations


def analyze_market_phase(candles: List[Candle]) -> Dict:
    """
    Convenience function for market phase analysis
    
    Args:
        candles: List of candles (minimum 50, recommended 100+)
        
    Returns:
        Complete market phase analysis report
    """
    analyzer = MarketPhaseAnalysis()
    return analyzer.generate_analysis_report(candles)
