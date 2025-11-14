"""
Tool Recommendation Service

این سرویس business logic برای پیشنهاد ابزارها را فراهم می‌کند
و بین API، ML models، و database ارتباط برقرار می‌کند.
"""

import asyncio
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path

# TODO: Import actual modules when ready
# from ml.ml_tool_recommender import DynamicToolRecommender, MarketContext, ToolRecommendation
# from src.gravity_tech.clients.data_service_client import DataServiceClient
# from database.tool_performance_manager import ToolPerformanceManager


class ToolRecommendationService:
    """
    سرویس پیشنهاد ابزارها
    
    مسئولیت‌ها:
    - دریافت داده از Data Service
    - تحلیل کانتکست بازار
    - فراخوانی ML models برای پیشنهاد
    - ذخیره و بازیابی عملکرد تاریخی
    - کش کردن نتایج
    """
    
    def __init__(
        self,
        data_service_url: Optional[str] = None,
        redis_url: Optional[str] = None,
        db_connection_string: Optional[str] = None
    ):
        """
        Initialize Tool Recommendation Service
        
        Args:
            data_service_url: URL سرویس داده
            redis_url: URL Redis برای کش
            db_connection_string: رشته اتصال دیتابیس
        """
        self.data_service_url = data_service_url or "http://localhost:8001"
        self.redis_url = redis_url
        self.db_connection = db_connection_string
        
        # TODO: Initialize actual components
        # self.data_client = DataServiceClient(self.data_service_url)
        # self.recommender = DynamicToolRecommender(model_type="lightgbm")
        # self.performance_manager = ToolPerformanceManager(self.db_connection)
        # self.cache = CacheService(self.redis_url)
        
        self._cache = {}  # Simple in-memory cache for now
    
    async def get_tool_recommendations(
        self,
        symbol: str,
        timeframe: str = "1d",
        analysis_goal: str = "entry_signal",
        trading_style: str = "swing",
        limit_candles: int = 200,
        top_n: int = 15
    ) -> Dict[str, Any]:
        """
        دریافت پیشنهادات ابزارها برای یک نماد
        
        Args:
            symbol: نماد دارایی
            timeframe: بازه زمانی
            analysis_goal: هدف تحلیل
            trading_style: سبک معامله‌گری
            limit_candles: تعداد کندل
            top_n: تعداد ابزارهای پیشنهادی
        
        Returns:
            دیکشنری کامل با پیشنهادات
        """
        
        # 1. Check cache
        cache_key = f"tool_rec:{symbol}:{timeframe}:{analysis_goal}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        # 2. Fetch market data from Data Service
        candles_df = await self._fetch_market_data(symbol, timeframe, limit_candles)
        
        # 3. Analyze market context
        market_context = self._analyze_market_context(
            symbol=symbol,
            candles=candles_df,
            timeframe=timeframe,
            trading_style=trading_style
        )
        
        # 4. Get ML weights for current regime
        ml_weights = self._get_ml_weights(market_context["regime"])
        
        # 5. Get tool recommendations
        recommendations = self._get_recommendations(
            market_context=market_context,
            ml_weights=ml_weights,
            top_n=top_n
        )
        
        # 6. Build dynamic strategy
        strategy = self._build_strategy(
            recommendations=recommendations,
            market_context=market_context,
            analysis_goal=analysis_goal
        )
        
        # 7. Prepare response
        response = {
            "symbol": symbol,
            "market_context": market_context,
            "analysis_goal": analysis_goal,
            "recommendations": recommendations,
            "dynamic_strategy": strategy,
            "ml_metadata": {
                "model_type": "lightgbm",
                "regime_weights": ml_weights,
                "total_tools_analyzed": 95,
                "timestamp": datetime.utcnow().isoformat()
            },
            "timestamp": datetime.utcnow()
        }
        
        # 8. Cache result
        self._save_to_cache(cache_key, response, ttl=300)  # 5 minutes
        
        return response
    
    async def analyze_with_custom_tools(
        self,
        symbol: str,
        timeframe: str,
        selected_tools: List[str],
        include_ml_scoring: bool = True,
        include_patterns: bool = True,
        limit_candles: int = 200
    ) -> Dict[str, Any]:
        """
        تحلیل با ابزارهای انتخابی کاربر
        
        Args:
            symbol: نماد دارایی
            timeframe: بازه زمانی
            selected_tools: لیست ابزارهای انتخاب شده
            include_ml_scoring: شامل امتیازدهی ML
            include_patterns: شامل تشخیص الگو
            limit_candles: تعداد کندل
        
        Returns:
            نتایج تحلیل
        """
        
        # 1. Fetch market data
        candles_df = await self._fetch_market_data(symbol, timeframe, limit_candles)
        
        # 2. Calculate selected indicators
        tool_results = {}
        for tool in selected_tools:
            result = await self._calculate_indicator(tool, candles_df)
            tool_results[tool] = result
        
        # 3. ML Scoring (if requested)
        ml_scoring = None
        if include_ml_scoring:
            ml_scoring = self._calculate_ml_scoring(tool_results, candles_df)
        
        # 4. Pattern Detection (if requested)
        patterns = None
        if include_patterns:
            patterns = await self._detect_patterns(candles_df)
        
        # 5. Summary
        summary = self._create_summary(tool_results, ml_scoring, patterns)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "selected_tools": selected_tools,
            "tool_results": tool_results,
            "ml_scoring": ml_scoring,
            "patterns_detected": patterns,
            "summary": summary,
            "timestamp": datetime.utcnow()
        }
    
    async def _fetch_market_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> pd.DataFrame:
        """
        دریافت داده بازار از Data Service
        
        TODO: Integrate with actual DataServiceClient
        """
        
        # فعلاً داده شبیه‌سازی شده برمی‌گرداند
        # در production باید از DataServiceClient استفاده شود
        
        dates = pd.date_range(end=datetime.utcnow(), periods=limit, freq='D')
        
        # Generate sample OHLCV data
        np.random.seed(42)
        base_price = 50000
        
        data = {
            'timestamp': dates,
            'open': base_price + np.random.randn(limit).cumsum() * 100,
            'high': base_price + np.random.randn(limit).cumsum() * 100 + 200,
            'low': base_price + np.random.randn(limit).cumsum() * 100 - 200,
            'close': base_price + np.random.randn(limit).cumsum() * 100,
            'volume': np.random.randint(1000000, 5000000, limit)
        }
        
        df = pd.DataFrame(data)
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    def _analyze_market_context(
        self,
        symbol: str,
        candles: pd.DataFrame,
        timeframe: str,
        trading_style: str
    ) -> Dict[str, Any]:
        """
        تحلیل کانتکست بازار
        
        شامل:
        - رژیم بازار (trending/ranging/volatile)
        - سطح نوسان
        - قدرت ترند
        - پروفایل حجم
        """
        
        # محاسبه volatility
        returns = candles['close'].pct_change()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized
        
        # محاسبه trend strength
        sma_20 = candles['close'].rolling(20).mean()
        sma_50 = candles['close'].rolling(50).mean()
        current_price = candles['close'].iloc[-1]
        
        trend_strength = abs((current_price - sma_20.iloc[-1]) / sma_20.iloc[-1]) * 100
        
        # تشخیص رژیم
        if trend_strength > 5:
            if current_price > sma_20.iloc[-1] and sma_20.iloc[-1] > sma_50.iloc[-1]:
                regime = "trending_bullish"
            elif current_price < sma_20.iloc[-1] and sma_20.iloc[-1] < sma_50.iloc[-1]:
                regime = "trending_bearish"
            else:
                regime = "ranging"
        elif volatility > 30:
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
        
        return {
            "regime": regime,
            "volatility": min(volatility, 100.0),
            "trend_strength": min(trend_strength * 10, 100.0),
            "volume_profile": volume_profile,
            "current_price": float(current_price),
            "sma_20": float(sma_20.iloc[-1]),
            "sma_50": float(sma_50.iloc[-1])
        }
    
    def _get_ml_weights(self, market_regime: str) -> Dict[str, float]:
        """
        دریافت وزن‌های ML برای رژیم بازار
        
        TODO: Load from trained ML model
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
        
        return regime_weights.get(market_regime, regime_weights["ranging"])
    
    def _get_recommendations(
        self,
        market_context: Dict[str, Any],
        ml_weights: Dict[str, float],
        top_n: int
    ) -> Dict[str, List[Dict]]:
        """
        دریافت پیشنهادات ابزارها
        
        TODO: Use actual DynamicToolRecommender
        """
        
        # فعلاً پیشنهادات شبیه‌سازی شده
        must_use = [
            {
                "name": "ADX",
                "category": "trend_indicators",
                "ml_weight": 0.28,
                "confidence": 0.87,
                "historical_accuracy": "82.0%",
                "reason": "در بازار روندی بسیار موثر است | وزن ML بالا",
                "priority": "must_use",
                "best_for": ["قدرت ترند", "تایید جهت حرکت"]
            },
            {
                "name": "MACD",
                "category": "trend_indicators",
                "ml_weight": 0.24,
                "confidence": 0.83,
                "historical_accuracy": "79.0%",
                "reason": "در بازار روندی بسیار موثر است",
                "priority": "must_use",
                "best_for": ["تشخیص ترند", "سیگنال خرید/فروش"]
            }
        ]
        
        recommended = [
            {
                "name": "RSI",
                "category": "momentum_indicators",
                "ml_weight": 0.18,
                "confidence": 0.76,
                "historical_accuracy": "76.0%",
                "reason": "برای تشخیص نقاط اصلاح در روند",
                "priority": "recommended",
                "best_for": ["اشباع خرید/فروش", "واگرایی"]
            }
        ]
        
        optional = []
        avoid = []
        
        return {
            "must_use": must_use[:top_n],
            "recommended": recommended[:top_n],
            "optional": optional,
            "avoid": avoid
        }
    
    def _build_strategy(
        self,
        recommendations: Dict[str, List[Dict]],
        market_context: Dict[str, Any],
        analysis_goal: str
    ) -> Dict[str, Any]:
        """ساخت استراتژی معاملاتی"""
        
        must_use = recommendations.get("must_use", [])
        recommended = recommendations.get("recommended", [])
        
        primary_tools = [tool["name"] for tool in must_use[:5]]
        supporting_tools = [tool["name"] for tool in recommended[:5]]
        
        # محاسبه confidence میانگین
        all_tools = must_use + recommended
        if all_tools:
            avg_confidence = np.mean([tool["confidence"] for tool in all_tools])
        else:
            avg_confidence = 0.0
        
        return {
            "primary_tools": primary_tools,
            "supporting_tools": supporting_tools,
            "confidence": float(avg_confidence),
            "based_on": f"تحلیل {len(all_tools)} ابزار برتر",
            "regime": market_context["regime"],
            "expected_accuracy": f"{avg_confidence * 100:.1f}%",
            "analysis_goal": analysis_goal
        }
    
    async def _calculate_indicator(
        self,
        tool_name: str,
        candles: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        محاسبه یک اندیکاتور
        
        TODO: Use actual indicator calculators
        """
        
        # فعلاً نتایج شبیه‌سازی شده
        if tool_name.upper() == "MACD":
            return {
                "macd": 125.3,
                "signal": 118.7,
                "histogram": 6.6,
                "signal_type": "bullish",
                "strength": 0.72
            }
        elif tool_name.upper() == "RSI":
            return {
                "value": 58.3,
                "signal": "neutral",
                "overbought": False,
                "oversold": False
            }
        elif tool_name.upper() == "ADX":
            return {
                "adx": 32.5,
                "plus_di": 28.3,
                "minus_di": 18.7,
                "trend_strength": "strong",
                "direction": "bullish"
            }
        else:
            return {
                "value": 0.0,
                "signal": "neutral"
            }
    
    def _calculate_ml_scoring(
        self,
        tool_results: Dict[str, Any],
        candles: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        امتیازدهی ML بر اساس نتایج ابزارها
        
        TODO: Use actual ML scoring models
        """
        
        return {
            "trend_score": 72.5,
            "momentum_score": 68.3,
            "volatility_score": 55.2,
            "volume_score": 65.0,
            "combined_score": 67.8,
            "signal": "buy",
            "confidence": 0.78,
            "ml_model": "lightgbm",
            "version": "1.0.0"
        }
    
    async def _detect_patterns(
        self,
        candles: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        تشخیص الگوهای قیمتی
        
        TODO: Use actual pattern detection
        """
        
        return [
            {
                "type": "Bullish_Engulfing",
                "confidence": 0.85,
                "location": "recent",
                "candle_index": len(candles) - 2,
                "significance": "high",
                "expected_move": "bullish"
            }
        ]
    
    def _create_summary(
        self,
        tool_results: Dict[str, Any],
        ml_scoring: Optional[Dict[str, Any]],
        patterns: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """ساخت خلاصه نتایج"""
        
        # شمارش سیگنال‌ها
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for tool, result in tool_results.items():
            signal = result.get("signal_type") or result.get("signal", "neutral")
            if "bull" in signal.lower():
                bullish_count += 1
            elif "bear" in signal.lower():
                bearish_count += 1
            else:
                neutral_count += 1
        
        # تعیین سیگنال کلی
        if bullish_count > bearish_count:
            overall_signal = "bullish"
            if bullish_count >= len(tool_results) * 0.7:
                consensus = "strong_buy"
            else:
                consensus = "buy"
        elif bearish_count > bullish_count:
            overall_signal = "bearish"
            if bearish_count >= len(tool_results) * 0.7:
                consensus = "strong_sell"
            else:
                consensus = "sell"
        else:
            overall_signal = "neutral"
            consensus = "hold"
        
        return {
            "overall_signal": overall_signal,
            "consensus": consensus,
            "tools_analyzed": len(tool_results),
            "bullish_tools": bullish_count,
            "bearish_tools": bearish_count,
            "neutral_tools": neutral_count,
            "ml_score": ml_scoring.get("combined_score") if ml_scoring else None,
            "patterns_found": len(patterns) if patterns else 0
        }
    
    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """دریافت از کش"""
        cached_item = self._cache.get(key)
        if cached_item:
            data, expiry = cached_item
            if datetime.utcnow() < expiry:
                return data
            else:
                del self._cache[key]
        return None
    
    def _save_to_cache(self, key: str, data: Dict, ttl: int = 300):
        """ذخیره در کش"""
        expiry = datetime.utcnow() + timedelta(seconds=ttl)
        self._cache[key] = (data, expiry)
    
    async def record_tool_performance(
        self,
        tool_name: str,
        market_regime: str,
        prediction: str,
        actual_result: str,
        accuracy: float,
        metadata: Optional[Dict] = None
    ):
        """
        ثبت عملکرد ابزار برای یادگیری آینده
        
        Args:
            tool_name: نام ابزار
            market_regime: رژیم بازار
            prediction: پیش‌بینی ابزار
            actual_result: نتیجه واقعی
            accuracy: دقت
            metadata: اطلاعات اضافی
        """
        
        # TODO: Save to database
        performance_record = {
            "tool_name": tool_name,
            "market_regime": market_regime,
            "prediction": prediction,
            "actual_result": actual_result,
            "accuracy": accuracy,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow()
        }
        
        # فعلاً فقط log می‌کنیم
        print(f"📊 Performance recorded: {tool_name} - {accuracy:.1%} accuracy")
    
    def get_tool_statistics(
        self,
        tool_name: str,
        market_regime: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        دریافت آمار عملکرد یک ابزار
        
        TODO: Load from database
        """
        
        return {
            "tool_name": tool_name,
            "market_regime": market_regime or "all",
            "period_days": days,
            "total_predictions": 150,
            "correct_predictions": 123,
            "accuracy": 0.82,
            "avg_confidence": 0.75,
            "best_regime": "trending_bullish",
            "worst_regime": "volatile"
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    print("=" * 70)
    print("🔧 Tool Recommendation Service - Example Usage")
    print("=" * 70)
    
    async def test_service():
        service = ToolRecommendationService()
        
        # Test 1: Get recommendations
        print("\n📋 Test 1: Get Tool Recommendations")
        result = await service.get_tool_recommendations(
            symbol="BTCUSDT",
            timeframe="1d",
            analysis_goal="entry_signal",
            top_n=10
        )
        
        print(f"\n✅ Recommendations for {result['symbol']}:")
        print(f"   Market Regime: {result['market_context']['regime']}")
        print(f"   Must Use Tools: {len(result['recommendations']['must_use'])}")
        print(f"   Recommended Tools: {len(result['recommendations']['recommended'])}")
        
        # Test 2: Custom analysis
        print("\n📋 Test 2: Custom Tool Analysis")
        custom_result = await service.analyze_with_custom_tools(
            symbol="BTCUSDT",
            timeframe="1h",
            selected_tools=["MACD", "RSI", "ADX"],
            include_ml_scoring=True
        )
        
        print(f"\n✅ Custom Analysis Results:")
        print(f"   Tools Used: {', '.join(custom_result['selected_tools'])}")
        print(f"   Overall Signal: {custom_result['summary']['overall_signal']}")
        print(f"   Consensus: {custom_result['summary']['consensus']}")
    
    asyncio.run(test_service())
    
    print("\n" + "=" * 70)
    print("✅ Tool Recommendation Service Ready!")
    print("=" * 70)
