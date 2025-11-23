"""
Multi-Horizon Volatility Analysis System

سیستم تحلیل نوسان با سه امتیاز مستقل:
- 3-Day Volatility Score
- 7-Day Volatility Score
- 30-Day Volatility Score

با پیش‌بینی تغییرات نوسان
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from gravity_tech.models.schemas import SignalStrength
from gravity_tech.ml.multi_horizon_weights import MultiHorizonWeightLearner, HorizonWeights


@dataclass
class VolatilityScore:
    """امتیاز نوسان یک افق"""
    horizon: str
    score: float  # [-1, 1] (negative=کاهش نوسان, positive=افزایش نوسان)
    confidence: float  # [0, 1]
    signal: SignalStrength
    
    def get_strength(self) -> str:
        """
        قدرت نوسان
        
        Returns:
            - EXPLOSIVE: نوسان در حال انفجار (بسیار بالا)
            - HIGH: نوسان بالا
            - MODERATE: نوسان متوسط
            - LOW: نوسان پایین
            - COMPRESSED: نوسان فشرده (احتمال شکست قریب‌الوقوع)
        """
        if self.score > 0.7:
            return "EXPLOSIVE"  # نوسان در حال انفجار
        elif self.score > 0.3:
            return "HIGH"  # نوسان بالا
        elif self.score > -0.3:
            return "MODERATE"  # نوسان متوسط
        elif self.score > -0.7:
            return "LOW"  # نوسان پایین
        else:
            return "COMPRESSED"  # نوسان فشرده
    
    def get_direction(self) -> str:
        """جهت تغییر نوسان"""
        if self.score > 0.2:
            return "EXPANDING"  # در حال افزایش
        elif self.score < -0.2:
            return "CONTRACTING"  # در حال کاهش
        else:
            return "STABLE"  # پایدار


@dataclass
class MultiHorizonVolatilityAnalysis:
    """نتیجه تحلیل نوسان چند افقی"""
    timestamp: str
    
    # امتیازهای نوسان
    volatility_3d: VolatilityScore
    volatility_7d: VolatilityScore
    volatility_30d: VolatilityScore
    
    # امتیاز ترکیبی
    combined_volatility: float
    combined_confidence: float
    
    # توصیه‌ها
    recommendation_3d: str
    recommendation_7d: str
    recommendation_30d: str
    
    # تشخیص فاز نوسان
    volatility_phase: str  # EXPANSION, CONTRACTION, SQUEEZE, BREAKOUT
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'volatility_scores': {
                '3d': {
                    'score': self.volatility_3d.score,
                    'confidence': self.volatility_3d.confidence,
                    'signal': self.volatility_3d.signal.name,
                    'strength': self.volatility_3d.get_strength(),
                    'direction': self.volatility_3d.get_direction()
                },
                '7d': {
                    'score': self.volatility_7d.score,
                    'confidence': self.volatility_7d.confidence,
                    'signal': self.volatility_7d.signal.name,
                    'strength': self.volatility_7d.get_strength(),
                    'direction': self.volatility_7d.get_direction()
                },
                '30d': {
                    'score': self.volatility_30d.score,
                    'confidence': self.volatility_30d.confidence,
                    'signal': self.volatility_30d.signal.name,
                    'strength': self.volatility_30d.get_strength(),
                    'direction': self.volatility_30d.get_direction()
                }
            },
            'combined': {
                'volatility': self.combined_volatility,
                'confidence': self.combined_confidence
            },
            'recommendations': {
                '3d': self.recommendation_3d,
                '7d': self.recommendation_7d,
                '30d': self.recommendation_30d
            },
            'volatility_phase': self.volatility_phase
        }


class MultiHorizonVolatilityAnalyzer:
    """تحلیلگر نوسان چند افقی"""
    
    def __init__(
        self,
        weight_learner: MultiHorizonWeightLearner
    ):
        """
        Initialize analyzer
        
        Args:
            weight_learner: مدل یادگیری وزن‌ها
        """
        self.weight_learner = weight_learner
        self.horizons = ['3d', '7d', '30d']
    
    def analyze(
        self,
        features: Dict[str, float]
    ) -> MultiHorizonVolatilityAnalysis:
        """
        تحلیل چند افقی نوسان
        
        Args:
            features: ویژگی‌های نوسان استخراج شده
            
        Returns:
            نتیجه تحلیل نوسان
        """
        # ایجاد DataFrame برای پیش‌بینی
        X = pd.DataFrame([features])
        
        # پیش‌بینی امتیازها
        predictions = self.weight_learner.predict_multi_horizon(X)
        
        # ایجاد VolatilityScore برای هر افق
        volatility_scores = {}
        for horizon in self.horizons:
            pred_col = f'pred_{horizon}'
            raw_score = predictions[pred_col].iloc[0]
            
            # دریافت وزن‌ها و confidence
            horizon_weights = self.weight_learner.get_horizon_weights(horizon)
            confidence = horizon_weights.confidence
            
            # نرمال‌سازی score به [-1, +1]
            normalized_score = np.clip(raw_score, -1, 1)
            
            # تعیین SignalStrength
            signal = self._score_to_signal(normalized_score)
            
            volatility_scores[horizon] = VolatilityScore(
                horizon=horizon,
                score=normalized_score,
                confidence=confidence,
                signal=signal
            )
        
        # محاسبه امتیاز ترکیبی
        combined_volatility, combined_confidence = self._smart_combination(volatility_scores)
        
        # تشخیص فاز نوسان
        volatility_phase = self._detect_volatility_phase(volatility_scores, features)
        
        # ایجاد توصیه‌ها
        rec_3d = self._generate_recommendation(volatility_scores['3d'], '3d')
        rec_7d = self._generate_recommendation(volatility_scores['7d'], '7d')
        rec_30d = self._generate_recommendation(volatility_scores['30d'], '30d')
        
        return MultiHorizonVolatilityAnalysis(
            timestamp=datetime.now().isoformat(),
            volatility_3d=volatility_scores['3d'],
            volatility_7d=volatility_scores['7d'],
            volatility_30d=volatility_scores['30d'],
            combined_volatility=combined_volatility,
            combined_confidence=combined_confidence,
            recommendation_3d=rec_3d,
            recommendation_7d=rec_7d,
            recommendation_30d=rec_30d,
            volatility_phase=volatility_phase
        )
    
    def _score_to_signal(self, score: float) -> SignalStrength:
        """
        تبدیل امتیاز به سیگنال
        
        برای نوسان:
        - مثبت = افزایش نوسان = HIGH VOLATILITY
        - منفی = کاهش نوسان = LOW VOLATILITY
        """
        if score > 0.7:
            return SignalStrength.VERY_BULLISH  # نوسان بسیار بالا
        elif score > 0.3:
            return SignalStrength.BULLISH  # نوسان بالا
        elif score > -0.3:
            return SignalStrength.NEUTRAL  # نوسان متوسط
        elif score > -0.7:
            return SignalStrength.BEARISH  # نوسان پایین
        else:
            return SignalStrength.VERY_BEARISH  # نوسان بسیار پایین (فشردگی)
    
    def _smart_combination(
        self,
        volatility_scores: Dict[str, VolatilityScore]
    ) -> tuple[float, float]:
        """
        ترکیب هوشمند امتیازها
        
        وزن‌های پیشنهادی:
        - 3d: 50% (کوتاه‌مدت مهم‌تر)
        - 7d: 30% (میان‌مدت)
        - 30d: 20% (بلندمدت)
        """
        # وزن‌های adaptive بر اساس confidence
        scores = []
        confidences = []
        
        for horizon in self.horizons:
            vs = volatility_scores[horizon]
            scores.append(vs.score)
            confidences.append(vs.confidence)
        
        total_confidence = sum(confidences)
        
        if total_confidence > 0:
            # میانگین وزن‌دار بر اساس confidence
            weighted_score = sum(
                s * c for s, c in zip(scores, confidences)
            ) / total_confidence
            combined_confidence = total_confidence / len(confidences)
        else:
            weighted_score = 0.0
            combined_confidence = 0.0
        
        return weighted_score, combined_confidence
    
    def _detect_volatility_phase(
        self,
        volatility_scores: Dict[str, VolatilityScore],
        features: Dict[str, float]
    ) -> str:
        """
        تشخیص فاز نوسان بازار
        
        Phases:
        - EXPANSION: نوسان در حال افزایش (انتظار حرکت بزرگ)
        - CONTRACTION: نوسان در حال کاهش (آرامش بازار)
        - SQUEEZE: نوسان بسیار پایین (قبل از شکست)
        - BREAKOUT: شکست از فشردگی (نوسان ناگهانی افزایش)
        - STABLE: نوسان پایدار
        """
        score_3d = volatility_scores['3d'].score
        score_7d = volatility_scores['7d'].score
        score_30d = volatility_scores['30d'].score
        
        # بررسی ATR برای تشخیص squeeze
        atr_percentile = features.get('atr_percentile', 50)
        bb_percentile = features.get('bollinger_bands_percentile', 50)
        
        # SQUEEZE: همه نوسان‌ها پایین + اندیکاتورها در پایین‌ترین سطح
        if (score_3d < -0.5 and score_7d < -0.5 and score_30d < -0.5 and
            atr_percentile < 25 and bb_percentile < 25):
            return "SQUEEZE"
        
        # BREAKOUT: نوسان کوتاه‌مدت ناگهان بالا رفته ولی میان‌مدت هنوز پایین
        if score_3d > 0.5 and score_7d < 0 and score_30d < 0:
            return "BREAKOUT"
        
        # EXPANSION: همه افق‌ها نشان‌دهنده افزایش نوسان
        if score_3d > 0.3 and score_7d > 0.2 and score_30d > 0:
            return "EXPANSION"
        
        # CONTRACTION: همه افق‌ها نشان‌دهنده کاهش نوسان
        if score_3d < -0.2 and score_7d < -0.2 and score_30d < -0.1:
            return "CONTRACTION"
        
        # STABLE: نوسان پایدار
        return "STABLE"
    
    def _generate_recommendation(
        self,
        volatility_score: VolatilityScore,
        horizon: str
    ) -> str:
        """
        ایجاد توصیه بر اساس امتیاز نوسان
        
        Args:
            volatility_score: امتیاز نوسان
            horizon: افق زمانی
            
        Returns:
            توصیه فارسی
        """
        score = volatility_score.score
        confidence = volatility_score.confidence
        strength = volatility_score.get_strength()
        direction = volatility_score.get_direction()
        
        # فرمت افق
        horizon_fa = {
            '3d': 'کوتاه‌مدت (3 روز)',
            '7d': 'میان‌مدت (هفته)',
            '30d': 'بلندمدت (ماه)'
        }.get(horizon, horizon)
        
        if strength == "EXPLOSIVE":
            if confidence > 0.7:
                return f"⚠️ {horizon_fa}: نوسان در حال انفجار - خطر بسیار بالا - از پوزیشن‌های بزرگ پرهیز کنید"
            else:
                return f"⚠️ {horizon_fa}: نوسان بالا محتمل - احتیاط در معاملات"
        
        elif strength == "HIGH":
            return f"📊 {horizon_fa}: نوسان بالا - بازار فعال - فرصت برای معامله‌گران روزانه"
        
        elif strength == "MODERATE":
            if direction == "EXPANDING":
                return f"📈 {horizon_fa}: نوسان در حال افزایش - آماده باشید برای حرکت قیمتی"
            elif direction == "CONTRACTING":
                return f"📉 {horizon_fa}: نوسان در حال کاهش - بازار آرام می‌شود"
            else:
                return f"➡️ {horizon_fa}: نوسان متوسط - شرایط عادی بازار"
        
        elif strength == "LOW":
            return f"🔻 {horizon_fa}: نوسان پایین - بازار آرام - معامله‌گران صبور باشند"
        
        else:  # COMPRESSED
            if confidence > 0.7:
                return f"🎯 {horizon_fa}: فشردگی نوسان - انتظار شکست و حرکت بزرگ قریب‌الوقوع"
            else:
                return f"⏳ {horizon_fa}: نوسان بسیار پایین - صبر برای فرصت مناسب"
    
    def get_trading_advice(
        self,
        analysis: MultiHorizonVolatilityAnalysis
    ) -> Dict[str, str]:
        """
        مشاوره معاملاتی بر اساس تحلیل نوسان
        
        Args:
            analysis: نتیجه تحلیل
            
        Returns:
            مشاوره برای انواع معامله‌گران
        """
        phase = analysis.volatility_phase
        combined_score = analysis.combined_volatility
        
        advice = {}
        
        # مشاوره برای Day Traders
        if phase == "SQUEEZE":
            advice['day_trader'] = "⏳ صبر کنید - بازار در حال فشردگی. بعد از شکست وارد شوید."
        elif phase == "BREAKOUT":
            advice['day_trader'] = "🚀 فرصت عالی - شکست اتفاق افتاده. با حد ضرر مناسب وارد شوید."
        elif phase == "EXPANSION":
            advice['day_trader'] = "💰 شرایط ایده‌آل - نوسان بالا = فرصت سود بیشتر"
        else:
            advice['day_trader'] = "😴 شرایط معمولی - فرصت‌های محدود"
        
        # مشاوره برای Swing Traders
        if combined_score > 0.5:
            advice['swing_trader'] = "⚠️ احتیاط - نوسان بالا می‌تواند استاپ‌ها را فعال کند"
        elif combined_score < -0.5:
            advice['swing_trader'] = "✅ مناسب - نوسان پایین برای نگهداری میان‌مدت"
        else:
            advice['swing_trader'] = "➡️ شرایط متوسط - با استاپ‌های محافظانه"
        
        # مشاوره برای Long-term Investors
        if phase in ["SQUEEZE", "CONTRACTION"]:
            advice['long_term'] = "🎯 فرصت خوب - نوسان پایین برای ورود بلندمدت"
        elif phase == "EXPANSION":
            advice['long_term'] = "⏸️ صبر کنید - بگذارید بازار آرام شود"
        else:
            advice['long_term'] = "➡️ شرایط عادی - سرمایه‌گذاری تدریجی"
        
        # مشاوره Position Sizing
        if combined_score > 0.7:
            advice['position_size'] = "🔻 کوچک - نوسان بالا = ریسک بالا - حجم کم معامله کنید"
        elif combined_score < -0.5:
            advice['position_size'] = "🔺 بزرگ‌تر - نوسان پایین = ریسک کم - می‌توانید حجم بیشتر بگیرید"
        else:
            advice['position_size'] = "➡️ متوسط - مدیریت ریسک استاندارد"
        
        return advice
