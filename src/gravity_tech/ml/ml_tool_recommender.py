"""
ML-Based Dynamic Tool Recommender

این ماژول پیشنهادات پویا برای انتخاب بهترین ابزارهای تحلیل تکنیکال ارائه می‌دهد
بر اساس:
- وزن‌های یادگرفته شده ML
- رژیم بازار (trending/ranging/volatile)
- عملکرد تاریخی ابزارها
- مشخصات دارایی (volatility، timeframe، نوع)

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


import numpy as np
import pandas as pd
from datetime import timezone

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False



@dataclass
class ToolRecommendation:
    """پیشنهاد یک ابزار"""
    tool_name: str
    category: str
    ml_weight: float
    confidence: float
    historical_accuracy: float
    reason: str
    priority: str  # "must_use", "recommended", "optional", "avoid"
    best_for: list[str]


@dataclass
class MarketContext:
    """کانتکست بازار برای پیشنهاد ابزار"""
    symbol: str
    timeframe: str
    market_regime: str  # trending_bullish, trending_bearish, ranging, volatile
    volatility_level: float  # 0-100
    trend_strength: float  # 0-100
    volume_profile: str  # high, medium, low
    trading_style: str | None = "swing"  # scalp, day, swing, position


class DynamicToolRecommender:
    """
    سیستم پیشنهاد پویای ابزارها بر اساس ML

    ویژگی‌ها:
    - پیشنهاد بر اساس وزن‌های یادگرفته شده ML
    - تطبیق با رژیم بازار فعلی
    - یادگیری از عملکرد تاریخی
    - شخصی‌سازی بر اساس سبک معامله‌گری
    """

    # دسته‌بندی 95+ ابزار
    TOOL_CATEGORIES = {
        "trend_indicators": [
            "SMA", "EMA", "WMA", "DEMA", "TEMA", "HMA",
            "MACD", "ADX", "Parabolic_SAR", "Supertrend"
        ],
        "momentum_indicators": [
            "RSI", "Stochastic", "CCI", "Williams_R",
            "ROC", "MFI", "Ultimate_Oscillator", "TSI"
        ],
        "volatility_indicators": [
            "Bollinger_Bands", "ATR", "Keltner_Channels",
            "Standard_Deviation", "Historical_Volatility",
            "Chaikin_Volatility", "Donchian_Channels",
            "Mass_Index", "Volatility_Ratio", "True_Range"
        ],
        "volume_indicators": [
            "OBV", "VWAP", "Volume_Profile", "Accumulation_Distribution",
            "Chaikin_Money_Flow", "Money_Flow_Index", "Force_Index",
            "Ease_of_Movement", "Volume_Oscillator", "Klinger_Oscillator"
        ],
        "cycle_indicators": [
            "Detrended_Price", "Schaff_Trend_Cycle", "Ehlers_Fisher",
            "Hilbert_Transform", "Mesa_Adaptive", "Cycle_Period",
            "Dominant_Cycle", "Phase_Accumulation", "Sine_Wave", "Lead_Sine"
        ],
        "support_resistance": [
            "Pivot_Points", "Fibonacci_Retracement", "Fibonacci_Extension",
            "Fibonacci_Fan", "Fibonacci_Arc", "Gann_Levels",
            "Camarilla_Pivots", "Woodie_Pivots", "Floor_Pivots",
            "CPR", "Support_Zones", "Resistance_Zones"
        ],
        "candlestick_patterns": [
            "Doji", "Hammer", "Hanging_Man", "Shooting_Star",
            "Engulfing_Bullish", "Engulfing_Bearish", "Morning_Star", "Evening_Star",
            "Three_White_Soldiers", "Three_Black_Crows", "Harami", "Piercing",
            "Dark_Cloud", "Tweezer_Top", "Tweezer_Bottom", "Marubozu",
            # ... 40 الگو
        ],
        "classical_patterns": [
            "Head_Shoulders", "Inverse_Head_Shoulders", "Double_Top", "Double_Bottom",
            "Triple_Top", "Triple_Bottom", "Ascending_Triangle", "Descending_Triangle",
            "Symmetrical_Triangle", "Wedge_Rising", "Wedge_Falling",
            "Flag_Bullish", "Flag_Bearish", "Pennant", "Cup_Handle"
        ],
        "elliott_wave": ["Elliott_Wave_Analysis"],
        "divergence": ["RSI_Divergence", "MACD_Divergence", "Volume_Divergence"]
    }

    # وزن پایه هر دسته (قابل یادگیری)
    BASE_CATEGORY_WEIGHTS = {
        "trend_indicators": 0.25,
        "momentum_indicators": 0.20,
        "volatility_indicators": 0.15,
        "volume_indicators": 0.15,
        "cycle_indicators": 0.10,
        "support_resistance": 0.10,
        "candlestick_patterns": 0.03,
        "classical_patterns": 0.01,
        "elliott_wave": 0.005,
        "divergence": 0.005
    }

    def __init__(self, model_type: str = "lightgbm"):
        """
        Initialize Dynamic Tool Recommender

        Args:
            model_type: "lightgbm", "xgboost", or "sklearn"
        """
        self.model_type = model_type
        self.classifier = None
        self.tool_weights_history = {}
        self.performance_tracker = {}

        self.model_path = Path("ml_models/tool_recommender")
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Validate model availability
        if model_type == "lightgbm" and not LIGHTGBM_AVAILABLE:
            print("⚠️ LightGBM not available, falling back to sklearn")
            self.model_type = "sklearn"
        elif model_type == "xgboost" and not XGBOOST_AVAILABLE:
            print("⚠️ XGBoost not available, falling back to sklearn")
            self.model_type = "sklearn"

    def recommend_tools(
        self,
        context: MarketContext,
        ml_weights: dict[str, float | None] = None,
        top_n: int = 15
    ) -> list[ToolRecommendation]:
        """
        پیشنهاد ابزارها بر اساس کانتکست بازار و وزن‌های ML

        Args:
            context: اطلاعات بازار و دارایی
            ml_weights: وزن‌های یادگرفته شده از ML (اگر موجود باشد)
            top_n: تعداد ابزارهای پیشنهادی

        Returns:
            لیست ابزارهای پیشنهادی به ترتیب اولویت
        """
        recommendations = []

        # 1. دریافت وزن‌ها برای این رژیم بازار
        if ml_weights is None:
            ml_weights = self._get_regime_based_weights(context.market_regime)

        # 2. رتبه‌بندی ابزارها در هر دسته
        for category, tools in self.TOOL_CATEGORIES.items():
            category_weight = ml_weights.get(category, self.BASE_CATEGORY_WEIGHTS[category])

            for tool in tools:
                # محاسبه امتیاز ابزار
                tool_score = self._calculate_tool_score(
                    tool=tool,
                    category=category,
                    category_weight=category_weight,
                    context=context
                )

                # دریافت عملکرد تاریخی
                historical_accuracy = self._get_historical_accuracy(
                    tool=tool,
                    market_regime=context.market_regime
                )

                # تعیین اولویت
                priority = self._determine_priority(tool_score, historical_accuracy)

                # تولید دلیل
                reason = self._generate_reason(
                    tool=tool,
                    category=category,
                    context=context,
                    score=tool_score
                )

                # بهترین کاربردها
                best_for = self._get_best_use_cases(tool, context)

                rec = ToolRecommendation(
                    tool_name=tool,
                    category=category,
                    ml_weight=category_weight * tool_score,
                    confidence=min(tool_score * historical_accuracy, 1.0),
                    historical_accuracy=historical_accuracy,
                    reason=reason,
                    priority=priority,
                    best_for=best_for
                )

                recommendations.append(rec)

        # 3. مرتب‌سازی بر اساس confidence
        recommendations.sort(key=lambda x: x.confidence, reverse=True)

        return recommendations[:top_n]

    def _get_regime_based_weights(self, market_regime: str) -> dict[str, float]:
        """
        دریافت وزن‌های بهینه برای رژیم خاص بازار

        رژیم‌های مختلف نیاز به ابزارهای متفاوت دارند:
        - Trending: ADX, MACD, Moving Averages
        - Ranging: RSI, Stochastic, Bollinger Bands
        - Volatile: ATR, Bollinger Bands, Volatility Indicators
        """
        regime_weights = {
            "trending_bullish": {
                "trend_indicators": 0.35,
                "momentum_indicators": 0.25,
                "volume_indicators": 0.15,
                "volatility_indicators": 0.10,
                "cycle_indicators": 0.08,
                "support_resistance": 0.05,
                "classical_patterns": 0.01,
                "candlestick_patterns": 0.005,
                "elliott_wave": 0.003,
                "divergence": 0.002
            },
            "trending_bearish": {
                "trend_indicators": 0.35,
                "momentum_indicators": 0.25,
                "volume_indicators": 0.15,
                "volatility_indicators": 0.10,
                "cycle_indicators": 0.08,
                "support_resistance": 0.05,
                "classical_patterns": 0.01,
                "candlestick_patterns": 0.005,
                "elliott_wave": 0.003,
                "divergence": 0.002
            },
            "ranging": {
                "momentum_indicators": 0.30,
                "volatility_indicators": 0.25,
                "support_resistance": 0.20,
                "trend_indicators": 0.10,
                "volume_indicators": 0.08,
                "cycle_indicators": 0.05,
                "candlestick_patterns": 0.01,
                "classical_patterns": 0.005,
                "divergence": 0.003,
                "elliott_wave": 0.002
            },
            "volatile": {
                "volatility_indicators": 0.35,
                "momentum_indicators": 0.25,
                "support_resistance": 0.15,
                "trend_indicators": 0.10,
                "volume_indicators": 0.08,
                "cycle_indicators": 0.05,
                "candlestick_patterns": 0.01,
                "classical_patterns": 0.005,
                "divergence": 0.003,
                "elliott_wave": 0.002
            }
        }

        return regime_weights.get(market_regime, self.BASE_CATEGORY_WEIGHTS)

    def _calculate_tool_score(
        self,
        tool: str,
        category: str,
        category_weight: float,
        context: MarketContext
    ) -> float:
        """
        محاسبه امتیاز یک ابزار در کانتکست فعلی

        امتیاز بر اساس:
        - تطبیق با رژیم بازار
        - سازگاری با timeframe
        - مناسب بودن برای سبک معامله‌گری
        """
        score = 0.5  # امتیاز پایه

        # تطبیق با رژیم بازار
        regime_match = self._check_regime_compatibility(tool, context.market_regime)
        score += regime_match * 0.3

        # تطبیق با timeframe
        timeframe_match = self._check_timeframe_compatibility(tool, context.timeframe)
        score += timeframe_match * 0.2

        # تطبیق با volatility
        volatility_match = self._check_volatility_compatibility(tool, context.volatility_level)
        score += volatility_match * 0.2

        # تطبیق با سبک معامله‌گری
        if context.trading_style:
            style_match = self._check_trading_style_compatibility(tool, context.trading_style)
            score += style_match * 0.1

        return min(score, 1.0)

    def _check_regime_compatibility(self, tool: str, regime: str) -> float:
        """بررسی سازگاری ابزار با رژیم بازار"""
        # نقشه سازگاری ابزارها با رژیم‌های مختلف
        regime_compatibility = {
            "trending_bullish": {
                "ADX": 1.0, "MACD": 0.9, "EMA": 0.9, "Parabolic_SAR": 0.8,
                "RSI": 0.6, "Bollinger_Bands": 0.5
            },
            "trending_bearish": {
                "ADX": 1.0, "MACD": 0.9, "EMA": 0.9, "Parabolic_SAR": 0.8,
                "RSI": 0.6, "Bollinger_Bands": 0.5
            },
            "ranging": {
                "RSI": 1.0, "Stochastic": 0.9, "Bollinger_Bands": 0.9,
                "Support_Zones": 0.8, "Resistance_Zones": 0.8,
                "ADX": 0.3, "MACD": 0.4
            },
            "volatile": {
                "ATR": 1.0, "Bollinger_Bands": 0.9, "Keltner_Channels": 0.8,
                "Standard_Deviation": 0.8, "Historical_Volatility": 0.9
            }
        }

        return regime_compatibility.get(regime, {}).get(tool, 0.5)

    def _check_timeframe_compatibility(self, tool: str, timeframe: str) -> float:
        """بررسی سازگاری ابزار با بازه زمانی"""
        # ابزارهای مناسب برای timeframe‌های مختلف
        timeframe_scores = {
            "1m": {"RSI": 0.8, "MACD": 0.6, "Stochastic": 0.8},
            "5m": {"RSI": 0.9, "MACD": 0.7, "Stochastic": 0.9},
            "15m": {"RSI": 0.9, "MACD": 0.8, "EMA": 0.8},
            "1h": {"MACD": 0.9, "RSI": 0.9, "EMA": 0.9, "ADX": 0.8},
            "4h": {"MACD": 0.9, "ADX": 0.9, "EMA": 0.9},
            "1d": {"MACD": 1.0, "ADX": 1.0, "EMA": 1.0, "Elliott_Wave_Analysis": 0.9}
        }

        return timeframe_scores.get(timeframe, {}).get(tool, 0.7)

    def _check_volatility_compatibility(self, tool: str, volatility: float) -> float:
        """بررسی سازگاری ابزار با سطح نوسان"""
        # ابزارهای مناسب برای volatility بالا/پایین
        if volatility > 70:  # High volatility
            high_vol_tools = ["ATR", "Bollinger_Bands", "Keltner_Channels", "Standard_Deviation"]
            return 1.0 if tool in high_vol_tools else 0.5
        elif volatility < 30:  # Low volatility
            low_vol_tools = ["RSI", "Stochastic", "Support_Zones", "Resistance_Zones"]
            return 1.0 if tool in low_vol_tools else 0.6
        else:  # Medium volatility
            return 0.8

    def _check_trading_style_compatibility(self, tool: str, style: str) -> float:
        """بررسی سازگاری ابزار با سبک معامله‌گری"""
        style_tools = {
            "scalp": {"RSI": 1.0, "Stochastic": 0.9, "MACD": 0.7},
            "day": {"MACD": 1.0, "RSI": 0.9, "ADX": 0.8, "VWAP": 0.9},
            "swing": {"MACD": 1.0, "ADX": 0.9, "EMA": 0.9, "Fibonacci_Retracement": 0.8},
            "position": {"ADX": 1.0, "MACD": 0.9, "EMA": 1.0, "Elliott_Wave_Analysis": 0.9}
        }

        return style_tools.get(style, {}).get(tool, 0.7)

    def _get_historical_accuracy(self, tool: str, market_regime: str) -> float:
        """
        دریافت دقت تاریخی ابزار در رژیم خاص

        در واقعیت، این از دیتابیس خوانده می‌شود
        فعلاً مقادیر تقریبی برمی‌گردانیم
        """
        # TODO: Load from database
        # این باید از جدول tool_performance_history خوانده شود

        # فعلاً مقادیر شبیه‌سازی شده
        base_accuracy = {
            "ADX": 0.82, "MACD": 0.79, "RSI": 0.76, "EMA": 0.78,
            "Bollinger_Bands": 0.74, "ATR": 0.71, "Stochastic": 0.75,
            "VWAP": 0.77, "Fibonacci_Retracement": 0.68
        }

        return base_accuracy.get(tool, 0.70)

    def _determine_priority(self, score: float, accuracy: float) -> str:
        """تعیین اولویت استفاده از ابزار"""
        combined = score * accuracy

        if combined > 0.75:
            return "must_use"
        elif combined > 0.60:
            return "recommended"
        elif combined > 0.40:
            return "optional"
        else:
            return "avoid"

    def _generate_reason(
        self,
        tool: str,
        category: str,
        context: MarketContext,
        score: float
    ) -> str:
        """تولید دلیل پیشنهاد ابزار"""
        reasons = []

        if context.market_regime.startswith("trending"):
            if tool in ["ADX", "MACD", "EMA"]:
                reasons.append("در بازار روندی بسیار موثر است")

        if context.market_regime == "ranging":
            if tool in ["RSI", "Stochastic", "Bollinger_Bands"]:
                reasons.append("برای بازار رنج بهترین انتخاب است")

        if context.volatility_level > 70:
            if tool in ["ATR", "Bollinger_Bands"]:
                reasons.append("نوسانات بالا را به خوبی شناسایی می‌کند")

        if score > 0.8:
            reasons.append(f"وزن ML بالا ({score:.1%})")

        if not reasons:
            reasons.append("ابزار استандارد برای این شرایط")

        return " | ".join(reasons)

    def _get_best_use_cases(self, tool: str, context: MarketContext) -> list[str]:
        """دریافت بهترین موارد استفاده ابزار"""
        use_cases = {
            "MACD": ["تشخیص ترند", "سیگنال‌های خرید/فروش", "واگرایی"],
            "RSI": ["شناسایی اشباع خرید/فروش", "واگرایی", "سیگنال برگشت"],
            "ADX": ["قدرت ترند", "تایید جهت حرکت"],
            "Bollinger_Bands": ["محدوده قیمت", "شناسایی نوسانات", "شکست"],
            "ATR": ["محاسبه حد ضرر", "اندازه پوزیشن", "نوسانات"],
            "EMA": ["تشخیص ترند", "سطوح حمایت/مقاومت پویا"],
            "VWAP": ["قیمت میانگین", "ورود نهادی"]
        }

        return use_cases.get(tool, ["تحلیل تکنیکال عمومی"])

    def get_contextual_recommendations(
        self,
        symbol: str,
        candles: pd.DataFrame,
        analysis_goal: str = "entry_signal"
    ) -> dict:
        """
        پیشنهاد ابزارها با تحلیل کامل کانتکست بازار

        Args:
            symbol: نماد دارایی
            candles: داده‌های قیمتی
            analysis_goal: هدف تحلیل (entry_signal, exit_signal, risk_management)

        Returns:
            دیکشنری کامل با پیشنهادات و استراتژی
        """
        # 1. شناسایی کانتکست بازار
        context = self._analyze_market_context(symbol, candles)

        # 2. دریافت وزن‌های ML
        ml_weights = self._get_regime_based_weights(context.market_regime)

        # 3. پیشنهاد ابزارها
        recommendations = self.recommend_tools(context, ml_weights, top_n=15)

        # 4. تفکیک بر اساس اولویت
        must_use = [r for r in recommendations if r.priority == "must_use"]
        recommended = [r for r in recommendations if r.priority == "recommended"]
        optional = [r for r in recommendations if r.priority == "optional"]
        avoid = [r for r in recommendations if r.priority == "avoid"]

        # 5. ساخت استراتژی
        strategy = self._build_strategy(must_use, recommended, context)

        return {
            "symbol": symbol,
            "market_context": {
                "regime": context.market_regime,
                "volatility": context.volatility_level,
                "trend_strength": context.trend_strength,
                "volume_profile": context.volume_profile
            },
            "analysis_goal": analysis_goal,
            "recommendations": {
                "must_use": [self._rec_to_dict(r) for r in must_use],
                "recommended": [self._rec_to_dict(r) for r in recommended],
                "optional": [self._rec_to_dict(r) for r in optional],
                "avoid": [self._rec_to_dict(r) for r in avoid]
            },
            "dynamic_strategy": strategy,
            "ml_metadata": {
                "model_type": self.model_type,
                "regime_weights": ml_weights,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    def _analyze_market_context(self, symbol: str, candles: pd.DataFrame) -> MarketContext:
        """تحلیل کانتکست بازار از داده‌های قیمتی"""
        # محاسبات ساده برای شناسایی رژیم
        # در production باید از indicator calculators استفاده شود

        # محاسبه volatility
        returns = candles['close'].pct_change()
        volatility = returns.std() * 100

        # محاسبه trend strength (ساده شده)
        sma_20 = candles['close'].rolling(20).mean()
        current_price = candles['close'].iloc[-1]
        trend_strength = abs((current_price - sma_20.iloc[-1]) / sma_20.iloc[-1]) * 100

        # تشخیص رژیم
        if trend_strength > 5:
            if current_price > sma_20.iloc[-1]:
                regime = "trending_bullish"
            else:
                regime = "trending_bearish"
        elif volatility > 3:
            regime = "volatile"
        else:
            regime = "ranging"

        # Volume profile
        avg_volume = candles['volume'].mean()
        recent_volume = candles['volume'].iloc[-10:].mean()
        volume_ratio = recent_volume / avg_volume

        if volume_ratio > 1.5:
            volume_profile = "high"
        elif volume_ratio < 0.7:
            volume_profile = "low"
        else:
            volume_profile = "medium"

        return MarketContext(
            symbol=symbol,
            timeframe="1d",  # باید از ورودی دریافت شود
            market_regime=regime,
            volatility_level=min(volatility * 10, 100),
            trend_strength=min(trend_strength * 10, 100),
            volume_profile=volume_profile,
            trading_style="swing"
        )

    def _build_strategy(
        self,
        must_use: list[ToolRecommendation],
        recommended: list[ToolRecommendation],
        context: MarketContext
    ) -> dict:
        """ساخت استراتژی پیشنهادی"""
        primary_tools = [r.tool_name for r in must_use[:5]]
        supporting_tools = [r.tool_name for r in recommended[:5]]

        avg_confidence = np.mean([r.confidence for r in must_use + recommended])

        return {
            "primary_tools": primary_tools,
            "supporting_tools": supporting_tools,
            "confidence": float(avg_confidence),
            "based_on": f"تحلیل {len(must_use) + len(recommended)} ابزار برتر",
            "regime": context.market_regime,
            "expected_accuracy": f"{avg_confidence * 100:.1f}%"
        }

    def _rec_to_dict(self, rec: ToolRecommendation) -> dict:
        """تبدیل ToolRecommendation به dictionary"""
        return {
            "name": rec.tool_name,
            "category": rec.category,
            "ml_weight": float(rec.ml_weight),
            "confidence": float(rec.confidence),
            "historical_accuracy": f"{rec.historical_accuracy * 100:.1f}%",
            "reason": rec.reason,
            "best_for": rec.best_for
        }

    def train_recommender(
        self,
        training_data: pd.DataFrame,
        test_size: float = 0.2
    ) -> dict:
        """
        آموزش مدل ML برای پیشنهاد ابزارها

        Args:
            training_data: DataFrame شامل:
                - features: market regime, volatility, trend_strength, etc.
                - target: best_tool_category or best_tool
            test_size: نسبت داده تست

        Returns:
            متریک‌های عملکرد
        """
        print("\n🎓 Training Tool Recommender Model...")

        # TODO: Implement full training pipeline
        # این نیاز به داده واقعی تریدها دارد

        print("⚠️ Training pipeline not implemented yet")
        print("   Needs historical trade data with tool performance")

        return {
            "status": "not_implemented",
            "message": "Training requires historical performance data"
        }

    def save_model(self, filename: str = "tool_recommender.pkl"):
        """ذخیره مدل"""
        model_file = self.model_path / filename
        # TODO: Implement model saving
        print("💾 Model saving not implemented yet")

    def load_model(self, filename: str = "tool_recommender.pkl"):
        """بارگذاری مدل"""
        model_file = self.model_path / filename
        # TODO: Implement model loading
        print("📂 Model loading not implemented yet")


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("🔧 Dynamic Tool Recommender - Example Usage")
    print("=" * 70)

    # مثال 1: پیشنهاد بر اساس کانتکست دستی
    print("\n📋 Example 1: Manual Context")

    context = MarketContext(
        symbol="BTCUSDT",
        timeframe="1d",
        market_regime="trending_bullish",
        volatility_level=45.0,
        trend_strength=75.0,
        volume_profile="high",
        trading_style="swing"
    )

    recommender = DynamicToolRecommender(model_type="lightgbm")
    recommendations = recommender.recommend_tools(context, top_n=10)

    print(f"\n🎯 Top 10 Recommended Tools for {context.symbol}:")
    print(f"   Market Regime: {context.market_regime}")
    print(f"   Volatility: {context.volatility_level:.1f}")
    print(f"   Trend Strength: {context.trend_strength:.1f}")
    print()

    for i, rec in enumerate(recommendations, 1):
        print(f"{i:2d}. {rec.tool_name:25s} ({rec.category})")
        print(f"    Priority: {rec.priority:12s} | Confidence: {rec.confidence:.1%}")
        print(f"    Reason: {rec.reason}")
        print()

    print("\n" + "=" * 70)
    print("✅ Dynamic Tool Recommender Ready!")
    print("=" * 70)
