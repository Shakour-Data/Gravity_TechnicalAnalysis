"""
سیستم تحلیل ترکیبی Trend + Momentum

ترکیب هوشمند دو تحلیل جداگانه:
1. تحلیل روند (10 اندیکاتور روند)
2. تحلیل مومنتوم (8 اندیکاتور مومنتوم)

هر کدام امتیاز مستقل دارند و سپس ترکیب می‌شوند
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum

from gravity_tech.models.schemas import SignalStrength
from gravity_tech.ml.multi_horizon_analysis import MultiHorizonAnalysis, MultiHorizonAnalyzer
from gravity_tech.ml.multi_horizon_momentum_analysis import MultiHorizonMomentumAnalysis, MultiHorizonMomentumAnalyzer


class ActionRecommendation(Enum):
    """توصیه نهایی"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    ACCUMULATE = "ACCUMULATE"
    HOLD = "HOLD"
    TAKE_PROFIT = "TAKE_PROFIT"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class CombinedAnalysis:
    """نتیجه تحلیل ترکیبی"""
    timestamp: str
    
    # تحلیل‌های جداگانه
    trend_analysis: MultiHorizonAnalysis
    momentum_analysis: MultiHorizonMomentumAnalysis
    
    # امتیازهای ترکیبی
    combined_score_3d: float  # [-1, 1]
    combined_score_7d: float
    combined_score_30d: float
    
    # اعتماد
    confidence_3d: float  # [0, 1]
    confidence_7d: float
    confidence_30d: float
    
    # توصیه‌ها
    action_3d: ActionRecommendation
    action_7d: ActionRecommendation
    action_30d: ActionRecommendation
    
    # توصیه نهایی (میانگین وزنی)
    final_action: ActionRecommendation
    final_confidence: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'combined_scores': {
                '3d': self.combined_score_3d,
                '7d': self.combined_score_7d,
                '30d': self.combined_score_30d
            },
            'confidence': {
                '3d': self.confidence_3d,
                '7d': self.confidence_7d,
                '30d': self.confidence_30d
            },
            'actions': {
                '3d': self.action_3d.value,
                '7d': self.action_7d.value,
                '30d': self.action_30d.value
            },
            'final': {
                'action': self.final_action.value,
                'confidence': self.final_confidence
            }
        }


class CombinedTrendMomentumAnalyzer:
    """تحلیلگر ترکیبی Trend + Momentum"""
    
    def __init__(
        self,
        trend_analyzer: MultiHorizonAnalyzer,
        momentum_analyzer: MultiHorizonMomentumAnalyzer,
        trend_weight: float = 0.5,
        momentum_weight: float = 0.5
    ):
        """
        Args:
            trend_analyzer: تحلیلگر روند
            momentum_analyzer: تحلیلگر مومنتوم
            trend_weight: وزن روند (0-1)
            momentum_weight: وزن مومنتوم (0-1)
        """
        self.trend_analyzer = trend_analyzer
        self.momentum_analyzer = momentum_analyzer
        
        # نرمال‌سازی وزن‌ها
        total = trend_weight + momentum_weight
        self.trend_weight = trend_weight / total
        self.momentum_weight = momentum_weight / total
    
    def analyze(
        self,
        trend_features: Dict[str, float],
        momentum_features: Dict[str, float]
    ) -> CombinedAnalysis:
        """
        تحلیل ترکیبی
        
        Args:
            trend_features: ویژگی‌های روند
            momentum_features: ویژگی‌های مومنتوم
        """
        # تحلیل‌های جداگانه
        trend_analysis = self.trend_analyzer.analyze(trend_features)
        momentum_analysis = self.momentum_analyzer.analyze(momentum_features)
        
        # ترکیب امتیازها
        combined_3d, conf_3d = self._combine_scores(
            trend_analysis.score_3d.score,
            trend_analysis.score_3d.confidence,
            momentum_analysis.momentum_3d.score,
            momentum_analysis.momentum_3d.confidence
        )
        
        combined_7d, conf_7d = self._combine_scores(
            trend_analysis.score_7d.score,
            trend_analysis.score_7d.confidence,
            momentum_analysis.momentum_7d.score,
            momentum_analysis.momentum_7d.confidence
        )
        
        combined_30d, conf_30d = self._combine_scores(
            trend_analysis.score_30d.score,
            trend_analysis.score_30d.confidence,
            momentum_analysis.momentum_30d.score,
            momentum_analysis.momentum_30d.confidence
        )
        
        # تعیین توصیه‌ها
        action_3d = self._score_to_action(combined_3d, conf_3d)
        action_7d = self._score_to_action(combined_7d, conf_7d)
        action_30d = self._score_to_action(combined_30d, conf_30d)
        
        # توصیه نهایی (میانگین وزنی بر اساس confidence)
        final_action, final_conf = self._final_recommendation(
            [(combined_3d, conf_3d), (combined_7d, conf_7d), (combined_30d, conf_30d)]
        )
        
        return CombinedAnalysis(
            timestamp=pd.Timestamp.now().isoformat(),
            trend_analysis=trend_analysis,
            momentum_analysis=momentum_analysis,
            combined_score_3d=combined_3d,
            combined_score_7d=combined_7d,
            combined_score_30d=combined_30d,
            confidence_3d=conf_3d,
            confidence_7d=conf_7d,
            confidence_30d=conf_30d,
            action_3d=action_3d,
            action_7d=action_7d,
            action_30d=action_30d,
            final_action=final_action,
            final_confidence=final_conf
        )
    
    def _combine_scores(
        self,
        trend_score: float,
        trend_conf: float,
        momentum_score: float,
        momentum_conf: float
    ) -> tuple[float, float]:
        """ترکیب امتیاز روند و مومنتوم"""
        # اگر یکی از اعتمادها خیلی پایین است، وزن بیشتری به دیگری
        if trend_conf < 0.2 and momentum_conf > 0.5:
            combined = momentum_score
            conf = momentum_conf
        elif momentum_conf < 0.2 and trend_conf > 0.5:
            combined = trend_score
            conf = trend_conf
        else:
            # ترکیب عادی
            combined = (
                self.trend_weight * trend_score +
                self.momentum_weight * momentum_score
            )
            conf = (trend_conf + momentum_conf) / 2
        
        return combined, conf
    
    def _score_to_action(
        self,
        score: float,
        confidence: float
    ) -> ActionRecommendation:
        """تبدیل امتیاز به توصیه"""
        if confidence < 0.3:
            return ActionRecommendation.HOLD
        
        if score > 0.7:
            return ActionRecommendation.STRONG_BUY
        elif score > 0.4:
            return ActionRecommendation.BUY
        elif score > 0.2:
            return ActionRecommendation.ACCUMULATE
        elif score > -0.2:
            return ActionRecommendation.HOLD
        elif score > -0.4:
            return ActionRecommendation.TAKE_PROFIT
        elif score > -0.7:
            return ActionRecommendation.SELL
        else:
            return ActionRecommendation.STRONG_SELL
    
    def _final_recommendation(
        self,
        scores_confs: list[tuple[float, float]]
    ) -> tuple[ActionRecommendation, float]:
        """توصیه نهایی با میانگین وزنی"""
        scores = [s for s, c in scores_confs]
        confs = [c for s, c in scores_confs]
        
        total_conf = sum(confs)
        if total_conf > 0:
            weighted_score = sum(s * c for s, c in zip(scores, confs)) / total_conf
            avg_conf = total_conf / len(confs)
        else:
            weighted_score = 0.0
            avg_conf = 0.0
        
        action = self._score_to_action(weighted_score, avg_conf)
        return action, avg_conf
    
    def print_analysis(
        self,
        analysis: CombinedAnalysis
    ):
        """نمایش زیبای تحلیل ترکیبی"""
        print("\n" + "="*80)
        print("🎯 COMBINED TREND + MOMENTUM ANALYSIS")
        print("="*80)
        
        print(f"\n📅 Timestamp: {analysis.timestamp}")
        
        # امتیازهای ترکیبی
        print("\n" + "-"*80)
        print("📊 COMBINED SCORES (Trend + Momentum)")
        print("-"*80)
        
        for horizon in ['3d', '7d', '30d']:
            if horizon == '3d':
                score = analysis.combined_score_3d
                conf = analysis.confidence_3d
                action = analysis.action_3d
                t_score = analysis.trend_analysis.trend_3d.score
                m_score = analysis.momentum_analysis.momentum_3d.score
            elif horizon == '7d':
                score = analysis.combined_score_7d
                conf = analysis.confidence_7d
                action = analysis.action_7d
                t_score = analysis.trend_analysis.trend_7d.score
                m_score = analysis.momentum_analysis.momentum_7d.score
            else:
                score = analysis.combined_score_30d
                conf = analysis.confidence_30d
                action = analysis.action_30d
                t_score = analysis.trend_analysis.trend_30d.score
                m_score = analysis.momentum_analysis.momentum_30d.score
            
            print(f"\n{horizon.upper()}:")
            print(f"  Trend Score:    {t_score:+.3f}")
            print(f"  Momentum Score: {m_score:+.3f}")
            print(f"  Combined:       {score:+.3f}")
            print(f"  Confidence:     {conf:.0%}")
            print(f"  💡 Action:      {action.value}")
        
        # توصیه نهایی
        print("\n" + "-"*80)
        print("🎓 FINAL RECOMMENDATION")
        print("-"*80)
        print(f"  Action:     {analysis.final_action.value}")
        print(f"  Confidence: {analysis.final_confidence:.0%}")
        
        # توضیح
        self._print_explanation(analysis)
        
        print("\n" + "="*80)
    
    def _print_explanation(self, analysis: CombinedAnalysis):
        """توضیح توصیه"""
        print("\n💬 Explanation:")
        
        if analysis.final_action == ActionRecommendation.STRONG_BUY:
            print("   Strong alignment between trend and momentum.")
            print("   Both indicators suggest strong upward movement.")
        elif analysis.final_action == ActionRecommendation.BUY:
            print("   Positive trend with supportive momentum.")
            print("   Good entry point for position building.")
        elif analysis.final_action == ActionRecommendation.ACCUMULATE:
            print("   Moderate positive signals.")
            print("   Consider gradual accumulation.")
        elif analysis.final_action == ActionRecommendation.HOLD:
            print("   Mixed or neutral signals.")
            print("   Wait for clearer direction.")
        elif analysis.final_action == ActionRecommendation.TAKE_PROFIT:
            print("   Weakening momentum or trend reversal signs.")
            print("   Consider securing profits.")
        elif analysis.final_action == ActionRecommendation.SELL:
            print("   Negative alignment between trend and momentum.")
            print("   Risk management suggests reducing exposure.")
        else:  # STRONG_SELL
            print("   Strong bearish signals from both systems.")
            print("   Exit positions to preserve capital.")
