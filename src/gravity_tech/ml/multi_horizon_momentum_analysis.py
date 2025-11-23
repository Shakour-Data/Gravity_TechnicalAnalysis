"""
Multi-Horizon Momentum Analysis System

سیستم تحلیل مومنتوم با سه امتیاز مستقل:
- 3-Day Momentum Score
- 7-Day Momentum Score
- 30-Day Momentum Score

با تشخیص الگوهای مومنتوم
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass

from gravity_tech.models.schemas import SignalStrength
from gravity_tech.ml.multi_horizon_weights import MultiHorizonWeightLearner, HorizonWeights


@dataclass
class MomentumScore:
    """امتیاز مومنتوم یک افق"""
    horizon: str
    score: float  # [-1, 1]
    confidence: float  # [0, 1]
    signal: SignalStrength
    
    def get_strength(self) -> str:
        abs_score = abs(self.score)
        if abs_score > 0.7:
            return "STRONG"
        elif abs_score > 0.4:
            return "MODERATE"
        elif abs_score > 0.2:
            return "WEAK"
        else:
            return "NEUTRAL"


@dataclass
class MultiHorizonMomentumAnalysis:
    """نتیجه تحلیل مومنتوم چند افقی"""
    timestamp: str
    
    # امتیازهای مومنتوم
    momentum_3d: MomentumScore
    momentum_7d: MomentumScore
    momentum_30d: MomentumScore
    
    # امتیاز ترکیبی
    combined_momentum: float
    combined_confidence: float
    
    # توصیه‌ها
    recommendation_3d: str
    recommendation_7d: str
    recommendation_30d: str
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'momentum_scores': {
                '3d': {
                    'score': self.momentum_3d.score,
                    'confidence': self.momentum_3d.confidence,
                    'trend': self.momentum_3d.trend_signal.name,
                    'strength': self.momentum_3d.get_strength()
                },
                '7d': {
                    'score': self.momentum_7d.score,
                    'confidence': self.momentum_7d.confidence,
                    'trend': self.momentum_7d.trend_signal.name,
                    'strength': self.momentum_7d.get_strength()
                },
                '30d': {
                    'score': self.momentum_30d.score,
                    'confidence': self.momentum_30d.confidence,
                    'trend': self.momentum_30d.trend_signal.name,
                    'strength': self.momentum_30d.get_strength()
                }
            },
            'combined': {
                'momentum': self.combined_momentum,
                'confidence': self.combined_confidence
            },
            'recommendations': {
                '3d': self.recommendation_3d,
                '7d': self.recommendation_7d,
                '30d': self.recommendation_30d
            }
        }


class MultiHorizonMomentumAnalyzer:
    """تحلیلگر مومنتوم چند افقی"""
    
    def __init__(
        self,
        weight_learner: MultiHorizonWeightLearner
    ):
        self.weight_learner = weight_learner
        self.horizons = ['3d', '7d', '30d']
    
    def analyze(
        self,
        features: Dict[str, float]
    ) -> MultiHorizonMomentumAnalysis:
        """
        تحلیل چند افقی مومنتوم
        
        Args:
            features: ویژگی‌های مومنتوم استخراج شده
        """
        # ایجاد DataFrame برای پیش‌بینی
        X = pd.DataFrame([features])
        
        # پیش‌بینی امتیازها
        predictions = self.weight_learner.predict_multi_horizon(X)
        
        # ایجاد MomentumScore برای هر افق
        momentum_scores = {}
        for horizon in self.horizons:
            pred_col = f'pred_{horizon}'
            raw_score = predictions[pred_col].iloc[0]
            
            # دریافت وزن‌ها و confidence
            horizon_weights = self.weight_learner.get_horizon_weights(horizon)
            confidence = horizon_weights.confidence
            
            # نرمال‌سازی score
            normalized_score = np.clip(raw_score * 10, -1, 1)
            
            # تعیین SignalStrength
            signal = self._score_to_signal(normalized_score)
            
            momentum_scores[horizon] = MomentumScore(
                horizon=horizon,
                score=normalized_score,
                confidence=confidence,
                signal=signal
            )
        
        # محاسبه امتیاز ترکیبی
        combined_momentum, combined_confidence = self._smart_combination(momentum_scores)
        
        # ایجاد توصیه‌ها
        rec_3d = self._generate_recommendation(momentum_scores['3d'])
        rec_7d = self._generate_recommendation(momentum_scores['7d'])
        rec_30d = self._generate_recommendation(momentum_scores['30d'])
        
        return MultiHorizonMomentumAnalysis(
            timestamp=pd.Timestamp.now().isoformat(),
            momentum_3d=momentum_scores['3d'],
            momentum_7d=momentum_scores['7d'],
            momentum_30d=momentum_scores['30d'],
            combined_momentum=combined_momentum,
            combined_confidence=combined_confidence,
            recommendation_3d=rec_3d,
            recommendation_7d=rec_7d,
            recommendation_30d=rec_30d
        )
    
    def _score_to_signal(self, score: float) -> SignalStrength:
        if score > 0.7:
            return SignalStrength.VERY_BULLISH
        elif score > 0.3:
            return SignalStrength.BULLISH
        elif score > -0.3:
            return SignalStrength.NEUTRAL
        elif score > -0.7:
            return SignalStrength.BEARISH
        else:
            return SignalStrength.VERY_BEARISH
    
    def _smart_combination(
        self,
        momentum_scores: Dict[str, MomentumScore]
    ) -> tuple[float, float]:
        """ترکیب هوشمند امتیازها"""
        scores = []
        confidences = []
        
        for horizon in self.horizons:
            ms = momentum_scores[horizon]
            scores.append(ms.score)
            confidences.append(ms.confidence)
        
        total_confidence = sum(confidences)
        
        if total_confidence > 0:
            weighted_score = sum(
                s * c for s, c in zip(scores, confidences)
            ) / total_confidence
            combined_confidence = total_confidence / len(confidences)
        else:
            weighted_score = 0.0
            combined_confidence = 0.0
        
        return weighted_score, combined_confidence
    
    def _generate_recommendation(
        self,
        momentum_score: MomentumScore
    ) -> str:
        """ایجاد توصیه"""
        score = momentum_score.score
        confidence = momentum_score.confidence
        horizon = momentum_score.horizon
        
        if confidence < 0.3:
            return f"⚠️ UNCERTAIN - Low confidence ({confidence:.0%})"
        
        if score > 0.7:
            return f"🚀 STRONG MOMENTUM UP - {horizon} ({confidence:.0%})"
        elif score > 0.4:
            return f"📈 MOMENTUM UP - {horizon} ({confidence:.0%})"
        elif score > 0.2:
            return f"↗️ WEAK MOMENTUM UP - {horizon}"
        elif score > -0.2:
            return f"➡️ NEUTRAL - {horizon}"
        elif score > -0.4:
            return f"↘️ WEAK MOMENTUM DOWN - {horizon}"
        elif score > -0.7:
            return f"📉 MOMENTUM DOWN - {horizon} ({confidence:.0%})"
        else:
            return f"⛔ STRONG MOMENTUM DOWN - {horizon} ({confidence:.0%})"
    
    def print_analysis(
        self,
        analysis: MultiHorizonMomentumAnalysis
    ):
        """نمایش زیبای نتیجه"""
        print("\n" + "="*70)
        print("🔮 MULTI-HORIZON MOMENTUM ANALYSIS")
        print("="*70)
        
        print(f"\n📅 Timestamp: {analysis.timestamp}")
        
        print("\n" + "-"*70)
        print("📊 MOMENTUM SCORES")
        print("-"*70)
        
        for horizon in ['3d', '7d', '30d']:
            if horizon == '3d':
                ms = analysis.momentum_3d
                rec = analysis.recommendation_3d
            elif horizon == '7d':
                ms = analysis.momentum_7d
                rec = analysis.recommendation_7d
            else:
                ms = analysis.momentum_30d
                rec = analysis.recommendation_30d
            
            print(f"\n{horizon.upper()}:")
            print(f"  Score:      {ms.score:+.3f} ({ms.get_strength()})")
            print(f"  Confidence: {ms.confidence:.0%}")
            print(f"  Signal:     {ms.signal.name}")
            print(f"  💡 {rec}")
        
        print("\n" + "-"*70)
        print("🧠 COMBINED MOMENTUM")
        print("-"*70)
        print(f"  Combined Score:      {analysis.combined_momentum:+.3f}")
        print(f"  Combined Confidence: {analysis.combined_confidence:.0%}")
        
        print("\n" + "="*70)
