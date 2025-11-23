"""
Volume-Dimension Matrix Analysis
================================

این ماژول ماتریس دوبُعدی Volume × Dimensions را محاسبه می‌کند.
برخلاف confirmation layer ساده، هر dimension با حجم به صورت منحصربفرد تعامل دارد.

5 Interaction:
1. Volume × Trend: تایید/رد روند بر اساس حجم در کندل‌های صعودی/نزولی
2. Volume × Momentum: تشخیص واگرایی RSI/MFI و شرایط اشباع
3. Volume × Volatility: تحلیل BB Squeeze + Volume Spike
4. Volume × Cycle: حجم در فازهای مختلف بازار (Accumulation, Markup, etc.)
5. Volume × S/R: تایید Bounce/Breakout با حجم در سطوح کلیدی
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum
import numpy as np
from gravity_tech.models.schemas import (
    Candle,
    TrendScore,
    MomentumScore,
    VolatilityScore,
    CycleScore,
    SupportResistanceScore
)


class InteractionType(Enum):
    """نوع تعامل Volume با Dimension"""
    STRONG_CONFIRM = "STRONG_CONFIRM"      # تایید قوی
    CONFIRM = "CONFIRM"                     # تایید معمولی
    NEUTRAL = "NEUTRAL"                     # خنثی
    WARN = "WARN"                           # هشدار
    DIVERGENCE = "DIVERGENCE"               # واگرایی
    FAKE = "FAKE"                           # سیگنال جعلی


@dataclass
class VolumeMetrics:
    """معیارهای پایه حجم"""
    volume_ratio: float          # نسبت به میانگین 20 دوره
    volume_trend: float          # روند حجم (شیب)
    volume_spike: bool           # آیا spike وجود دارد (>2×)
    volume_in_bullish: float     # درصد حجم در کندل‌های صعودی
    volume_in_bearish: float     # درصد حجم در کندل‌های نزولی
    obv_trend: float             # روند On-Balance Volume
    avg_volume: float            # میانگین حجم


@dataclass
class VolumeDimensionInteraction:
    """نتیجه تعامل Volume با یک Dimension"""
    dimension: str                        # نام dimension
    
    # معیارهای حجم
    volume_metrics: VolumeMetrics
    
    # معیارهای dimension
    dimension_score: float                # امتیاز dimension [-1, +1]
    dimension_strength: float             # قدرت dimension [0, 1]
    dimension_state: str                  # وضعیت (مثلاً "BULLISH", "BEARISH")
    
    # نتیجه interaction
    interaction_score: float              # امتیاز interaction [-0.35, +0.35]
    interaction_type: InteractionType     # نوع interaction
    confidence: float                     # اطمینان [0, 1]
    
    # توضیحات
    explanation: str                      # توضیح interaction
    signals: List[str]                    # سیگنال‌های شناسایی شده


class VolumeDimensionMatrix:
    """
    محاسبه ماتریس دوبُعدی Volume × Dimensions
    
    هر dimension به صورت مستقل با حجم تعامل دارد:
    - Trend: حجم در جهت روند؟
    - Momentum: واگرایی RSI/MFI؟
    - Volatility: BB Squeeze + Volume؟
    - Cycle: حجم در فاز فعلی منطقی است؟
    - S/R: حجم در سطوح کلیدی؟
    """
    
    def __init__(self, candles: List[Candle]):
        """
        Args:
            candles: لیست کندل‌ها (حداقل 50 کندل برای محاسبات دقیق)
        """
        self.candles = candles
        self.volume_metrics = self._calculate_volume_metrics()
    
    def _calculate_volume_metrics(self) -> VolumeMetrics:
        """محاسبه معیارهای پایه حجم"""
        volumes = [c.volume for c in self.candles]
        avg_volume = np.mean(volumes[-20:])  # میانگین 20 دوره
        current_volume = volumes[-1]
        
        # نسبت حجم
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # روند حجم (شیب خط رگرسیون)
        x = np.arange(len(volumes[-20:]))
        y = np.array(volumes[-20:])
        if len(x) > 1 and np.std(y) > 0:
            slope = np.polyfit(x, y, 1)[0]
            volume_trend = slope / avg_volume  # نرمال‌سازی
        else:
            volume_trend = 0.0
        
        # Volume Spike
        volume_spike = volume_ratio > 2.0
        
        # حجم در کندل‌های صعودی/نزولی (20 کندل اخیر)
        bullish_volume = sum(c.volume for c in self.candles[-20:] if c.close > c.open)
        bearish_volume = sum(c.volume for c in self.candles[-20:] if c.close <= c.open)
        total_volume = bullish_volume + bearish_volume
        
        volume_in_bullish = bullish_volume / total_volume if total_volume > 0 else 0.5
        volume_in_bearish = bearish_volume / total_volume if total_volume > 0 else 0.5
        
        # On-Balance Volume trend
        obv = self._calculate_obv()
        obv_trend = self._calculate_obv_trend(obv)
        
        return VolumeMetrics(
            volume_ratio=volume_ratio,
            volume_trend=volume_trend,
            volume_spike=volume_spike,
            volume_in_bullish=volume_in_bullish,
            volume_in_bearish=volume_in_bearish,
            obv_trend=obv_trend,
            avg_volume=avg_volume
        )
    
    def _calculate_obv(self) -> List[float]:
        """محاسبه On-Balance Volume"""
        obv = [0.0]
        for i in range(1, len(self.candles)):
            if self.candles[i].close > self.candles[i-1].close:
                obv.append(obv[-1] + self.candles[i].volume)
            elif self.candles[i].close < self.candles[i-1].close:
                obv.append(obv[-1] - self.candles[i].volume)
            else:
                obv.append(obv[-1])
        return obv
    
    def _calculate_obv_trend(self, obv: List[float]) -> float:
        """محاسبه روند OBV"""
        if len(obv) < 20:
            return 0.0
        
        recent_obv = obv[-20:]
        x = np.arange(len(recent_obv))
        y = np.array(recent_obv)
        
        if np.std(y) > 0:
            slope = np.polyfit(x, y, 1)[0]
            # نرمال‌سازی بر اساس محدوده OBV
            obv_range = max(y) - min(y)
            if obv_range > 0:
                normalized_slope = slope / obv_range
                return np.clip(normalized_slope * 10, -1.0, 1.0)
        
        return 0.0
    
    # ═══════════════════════════════════════════════════════════════════
    # 1️⃣ Volume × Trend Interaction
    # ═══════════════════════════════════════════════════════════════════
    
    def calculate_volume_trend_interaction(
        self,
        trend_score: TrendScore
    ) -> VolumeDimensionInteraction:
        """
        تعامل حجم با روند
        
        منطق:
        - روند صعودی + حجم بالا در کندل‌های سبز → تایید قوی (+0.20)
        - روند صعودی + حجم پایین → هشدار ضعف (-0.15)
        - روند صعودی + OBV نزولی → واگرایی (-0.20)
        - روند نزولی + حجم بالا در کندل‌های قرمز → تایید قوی (-0.20)
        
        وزن‌ها:
        - volume_in_direction: 40%
        - obv_alignment: 30%
        - volume_ratio: 20%
        - trend_strength: 10%
        """
        vm = self.volume_metrics
        signals = []
        
        # تعیین جهت روند
        is_bullish = trend_score.score > 0.2
        is_bearish = trend_score.score < -0.2
        is_neutral = not is_bullish and not is_bearish
        
        if is_neutral:
            return VolumeDimensionInteraction(
                dimension="Trend",
                volume_metrics=vm,
                dimension_score=trend_score.score,
                dimension_strength=abs(trend_score.score),
                dimension_state="NEUTRAL",
                interaction_score=0.0,
                interaction_type=InteractionType.NEUTRAL,
                confidence=0.5,
                explanation="روند خنثی - حجم تاثیر کمی دارد",
                signals=["NEUTRAL_TREND"]
            )
        
        # ═══ بررسی هم‌جهتی حجم با روند ═══
        
        # 1. حجم در جهت روند
        if is_bullish:
            volume_in_direction = vm.volume_in_bullish
            direction_name = "صعودی"
        else:
            volume_in_direction = vm.volume_in_bearish
            direction_name = "نزولی"
        
        # 2. هم‌ترازی OBV با روند
        obv_aligned = (is_bullish and vm.obv_trend > 0.1) or \
                     (is_bearish and vm.obv_trend < -0.1)
        obv_divergent = (is_bullish and vm.obv_trend < -0.1) or \
                       (is_bearish and vm.obv_trend > 0.1)
        
        # 3. نسبت حجم
        high_volume = vm.volume_ratio > 1.3
        low_volume = vm.volume_ratio < 0.8
        
        # ═══ محاسبه امتیاز interaction ═══
        
        interaction_score = 0.0
        
        # وزن 1: حجم در جهت روند (40%)
        if volume_in_direction > 0.65:  # بیش از 65% در جهت روند
            interaction_score += 0.14  # 0.35 × 0.4
            signals.append(f"VOLUME_IN_{direction_name.upper()}")
        elif volume_in_direction < 0.35:  # کمتر از 35% در جهت روند
            interaction_score -= 0.14
            signals.append(f"VOLUME_AGAINST_{direction_name.upper()}")
        
        # وزن 2: هم‌ترازی OBV (30%)
        if obv_aligned:
            interaction_score += 0.105  # 0.35 × 0.3
            signals.append("OBV_ALIGNED")
        elif obv_divergent:
            interaction_score -= 0.105
            signals.append("OBV_DIVERGENCE")
        
        # وزن 3: نسبت حجم (20%)
        if high_volume:
            interaction_score += 0.07 * (1 if obv_aligned else -0.5)
            signals.append("HIGH_VOLUME")
        elif low_volume:
            interaction_score -= 0.07
            signals.append("LOW_VOLUME")
        
        # وزن 4: قدرت روند (10%)
        trend_strength = abs(trend_score.score)
        if trend_strength > 0.7:
            interaction_score += 0.035  # 0.35 × 0.1
            signals.append("STRONG_TREND")
        
        # معکوس کردن برای روند نزولی
        if is_bearish:
            interaction_score = -interaction_score
        
        # محدود کردن به [-0.35, +0.35]
        interaction_score = np.clip(interaction_score, -0.35, 0.35)
        
        # ═══ تعیین نوع interaction ═══
        
        if abs(interaction_score) > 0.20:
            if interaction_score > 0:
                interaction_type = InteractionType.STRONG_CONFIRM
            else:
                interaction_type = InteractionType.DIVERGENCE
        elif abs(interaction_score) > 0.10:
            if interaction_score > 0:
                interaction_type = InteractionType.CONFIRM
            else:
                interaction_type = InteractionType.WARN
        else:
            interaction_type = InteractionType.NEUTRAL
        
        # محاسبه اطمینان
        confidence = min(0.95, 0.6 + abs(interaction_score))
        
        # توضیحات
        if interaction_type == InteractionType.STRONG_CONFIRM:
            explanation = f"حجم به شدت روند {direction_name} را تایید می‌کند"
        elif interaction_type == InteractionType.CONFIRM:
            explanation = f"حجم روند {direction_name} را تایید می‌کند"
        elif interaction_type == InteractionType.DIVERGENCE:
            explanation = f"واگرایی قوی: روند {direction_name} اما حجم مخالف"
        elif interaction_type == InteractionType.WARN:
            explanation = f"هشدار: روند {direction_name} اما حجم ضعیف"
        else:
            explanation = "حجم در تایید یا رد روند نقش کمی دارد"
        
        return VolumeDimensionInteraction(
            dimension="Trend",
            volume_metrics=vm,
            dimension_score=trend_score.score,
            dimension_strength=trend_strength,
            dimension_state=direction_name.upper(),
            interaction_score=interaction_score,
            interaction_type=interaction_type,
            confidence=confidence,
            explanation=explanation,
            signals=signals
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # 2️⃣ Volume × Momentum Interaction
    # ═══════════════════════════════════════════════════════════════════
    
    def calculate_volume_momentum_interaction(
        self,
        momentum_score: MomentumScore
    ) -> VolumeDimensionInteraction:
        """
        تعامل حجم با مومنتوم
        
        منطق:
        - RSI > 70 + حجم بالا → هشدار اشباع خرید (-0.15)
        - RSI < 30 + حجم بالا → فرصت خرید (+0.15)
        - MFI واگرایی + حجم بالا → هشدار قوی (-0.20)
        - مومنتوم قوی + حجم پایین → مشکوک (-0.10)
        
        وزن‌ها:
        - divergence: 35%
        - mfi_direction: 30%
        - momentum_level: 20% (overbought/oversold)
        - volume_trend: 15%
        """
        vm = self.volume_metrics
        signals = []
        
        # فرض: momentum_score دارای فیلدهای rsi و mfi است
        # اگر ندارد، از score کلی استفاده می‌کنیم
        
        # شبیه‌سازی RSI و MFI (در واقعیت باید از momentum_score بیاید)
        rsi = self._estimate_rsi()
        mfi = self._estimate_mfi()
        
        # تعیین وضعیت مومنتوم
        is_overbought = rsi > 70 or mfi > 80
        is_oversold = rsi < 30 or mfi < 20
        is_neutral = not is_overbought and not is_oversold
        
        momentum_strength = abs(momentum_score.score)
        
        # ═══ بررسی واگرایی ═══
        
        # واگرایی: قیمت بالا می‌رود اما MFI پایین می‌آید (bearish divergence)
        # یا قیمت پایین می‌رود اما MFI بالا می‌آید (bullish divergence)
        
        price_trend = self._calculate_price_trend()
        mfi_trend = vm.volume_trend  # تقریباً معادل MFI trend
        
        bearish_divergence = price_trend > 0.1 and mfi_trend < -0.1
        bullish_divergence = price_trend < -0.1 and mfi_trend > 0.1
        
        # ═══ محاسبه امتیاز interaction ═══
        
        interaction_score = 0.0
        
        # وزن 1: واگرایی (35%)
        if bearish_divergence:
            interaction_score -= 0.1225  # 0.35 × 0.35
            signals.append("BEARISH_DIVERGENCE")
        elif bullish_divergence:
            interaction_score += 0.1225
            signals.append("BULLISH_DIVERGENCE")
        
        # وزن 2: جهت MFI با حجم (30%)
        if mfi_trend > 0.1 and vm.volume_ratio > 1.2:
            interaction_score += 0.105  # 0.35 × 0.3
            signals.append("MFI_BULLISH")
        elif mfi_trend < -0.1 and vm.volume_ratio > 1.2:
            interaction_score -= 0.105
            signals.append("MFI_BEARISH")
        
        # وزن 3: سطح مومنتوم (20%)
        if is_overbought and vm.volume_ratio > 1.3:
            interaction_score -= 0.07  # 0.35 × 0.2
            signals.append("OVERBOUGHT_HIGH_VOL")
        elif is_oversold and vm.volume_ratio > 1.3:
            interaction_score += 0.07
            signals.append("OVERSOLD_HIGH_VOL")
        elif is_overbought:
            signals.append("OVERBOUGHT")
        elif is_oversold:
            signals.append("OVERSOLD")
        
        # وزن 4: روند حجم (15%)
        if vm.volume_spike:
            if momentum_score.score > 0.5:
                interaction_score += 0.0525  # 0.35 × 0.15
                signals.append("VOLUME_SPIKE_BULLISH")
            elif momentum_score.score < -0.5:
                interaction_score -= 0.0525
                signals.append("VOLUME_SPIKE_BEARISH")
        
        # محدود کردن
        interaction_score = np.clip(interaction_score, -0.35, 0.35)
        
        # ═══ تعیین نوع interaction ═══
        
        if bearish_divergence or (is_overbought and vm.volume_ratio > 1.5):
            interaction_type = InteractionType.DIVERGENCE
        elif bullish_divergence or (is_oversold and vm.volume_ratio > 1.5):
            interaction_type = InteractionType.STRONG_CONFIRM
        elif abs(interaction_score) > 0.15:
            interaction_type = InteractionType.WARN if interaction_score < 0 else InteractionType.CONFIRM
        else:
            interaction_type = InteractionType.NEUTRAL
        
        # محاسبه اطمینان
        confidence = min(0.95, 0.65 + abs(interaction_score))
        
        # توضیحات
        if bearish_divergence:
            explanation = "واگرایی نزولی: قیمت بالا اما MFI پایین (هشدار!)"
        elif bullish_divergence:
            explanation = "واگرایی صعودی: قیمت پایین اما MFI بالا (فرصت!)"
        elif is_overbought and vm.volume_ratio > 1.3:
            explanation = f"اشباع خرید با حجم بالا (RSI={rsi:.0f}) - احتمال اصلاح"
        elif is_oversold and vm.volume_ratio > 1.3:
            explanation = f"اشباع فروش با حجم بالا (RSI={rsi:.0f}) - فرصت خرید"
        else:
            explanation = "تعامل معمولی حجم با مومنتوم"
        
        state = "OVERBOUGHT" if is_overbought else "OVERSOLD" if is_oversold else "NEUTRAL"
        
        return VolumeDimensionInteraction(
            dimension="Momentum",
            volume_metrics=vm,
            dimension_score=momentum_score.score,
            dimension_strength=momentum_strength,
            dimension_state=state,
            interaction_score=interaction_score,
            interaction_type=interaction_type,
            confidence=confidence,
            explanation=explanation,
            signals=signals
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # 3️⃣ Volume × Volatility Interaction
    # ═══════════════════════════════════════════════════════════════════
    
    def calculate_volume_volatility_interaction(
        self,
        volatility_score: VolatilityScore
    ) -> VolumeDimensionInteraction:
        """
        تعامل حجم با نوسان
        
        منطق:
        - BB Squeeze + Volume Spike → آماده شکست (+0.20)
        - نوسان بالا + حجم پایین → حرکت غیرواقعی (-0.15)
        - نوسان در حال افزایش + حجم بالا → تایید حرکت (+0.15)
        - نوسان پایین + حجم پایین → بازار خواب آلود (0.0)
        
        وزن‌ها:
        - bb_squeeze: 35%
        - volatility_expansion: 30%
        - volume_confirmation: 25%
        - atr_trend: 10%
        """
        vm = self.volume_metrics
        signals = []
        
        # شبیه‌سازی BB Squeeze و ATR
        bb_squeeze = self._detect_bb_squeeze()
        volatility_expanding = volatility_score.score > 0.3
        volatility_contracting = volatility_score.score < -0.3
        
        volatility_strength = abs(volatility_score.score)
        
        # ═══ محاسبه امتیاز interaction ═══
        
        interaction_score = 0.0
        
        # وزن 1: BB Squeeze (35%)
        if bb_squeeze:
            if vm.volume_spike:
                interaction_score += 0.1225  # 0.35 × 0.35 - آماده شکست!
                signals.append("BB_SQUEEZE_VOLUME_SPIKE")
            else:
                interaction_score += 0.0875  # 0.25 × 0.35 - squeeze اما هنوز بدون حجم
                signals.append("BB_SQUEEZE")
        
        # وزن 2: انبساط/انقباض نوسان (30%)
        if volatility_expanding:
            if vm.volume_ratio > 1.3:
                interaction_score += 0.105  # 0.35 × 0.3 - حرکت معتبر
                signals.append("VOLATILITY_EXPANSION_CONFIRMED")
            else:
                interaction_score -= 0.0525  # حرکت مشکوک
                signals.append("VOLATILITY_EXPANSION_LOW_VOLUME")
        elif volatility_contracting:
            signals.append("VOLATILITY_CONTRACTION")
        
        # وزن 3: تایید حجم (25%)
        if volatility_strength > 0.5:  # نوسان قابل توجه
            if vm.volume_ratio > 1.5:
                interaction_score += 0.0875  # 0.35 × 0.25
                signals.append("HIGH_VOLATILITY_HIGH_VOLUME")
            elif vm.volume_ratio < 0.8:
                interaction_score -= 0.0875
                signals.append("HIGH_VOLATILITY_LOW_VOLUME")
        
        # وزن 4: روند ATR (10%)
        atr_increasing = volatility_score.score > 0
        if atr_increasing and vm.volume_trend > 0:
            interaction_score += 0.035  # 0.35 × 0.1
            signals.append("ATR_VOLUME_ALIGNED")
        
        # محدود کردن
        interaction_score = np.clip(interaction_score, -0.35, 0.35)
        
        # ═══ تعیین نوع interaction ═══
        
        if bb_squeeze and vm.volume_spike:
            interaction_type = InteractionType.STRONG_CONFIRM
        elif volatility_expanding and vm.volume_ratio < 0.8:
            interaction_type = InteractionType.FAKE
        elif abs(interaction_score) > 0.15:
            interaction_type = InteractionType.CONFIRM if interaction_score > 0 else InteractionType.WARN
        else:
            interaction_type = InteractionType.NEUTRAL
        
        # محاسبه اطمینان
        confidence = min(0.95, 0.6 + abs(interaction_score))
        
        # توضیحات
        if bb_squeeze and vm.volume_spike:
            explanation = "BB Squeeze + Volume Spike: آماده شکست قیمتی!"
        elif bb_squeeze:
            explanation = "BB Squeeze: نوسان فشرده، منتظر volume spike"
        elif volatility_expanding and vm.volume_ratio > 1.3:
            explanation = "انبساط نوسان با حجم بالا: حرکت معتبر"
        elif volatility_expanding and vm.volume_ratio < 0.8:
            explanation = "انبساط نوسان با حجم پایین: حرکت مشکوک"
        else:
            explanation = "تعامل معمولی حجم با نوسان"
        
        state = "SQUEEZE" if bb_squeeze else "EXPANDING" if volatility_expanding else "CONTRACTING" if volatility_contracting else "NORMAL"
        
        return VolumeDimensionInteraction(
            dimension="Volatility",
            volume_metrics=vm,
            dimension_score=volatility_score.score,
            dimension_strength=volatility_strength,
            dimension_state=state,
            interaction_score=interaction_score,
            interaction_type=interaction_type,
            confidence=confidence,
            explanation=explanation,
            signals=signals
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # 4️⃣ Volume × Cycle Interaction
    # ═══════════════════════════════════════════════════════════════════
    
    def calculate_volume_cycle_interaction(
        self,
        cycle_score: CycleScore
    ) -> VolumeDimensionInteraction:
        """
        تعامل حجم با سیکل بازار
        
        منطق:
        - Accumulation + حجم پایین → طبیعی (+0.15)
        - Accumulation + volume spike → شروع Markup (+0.25)
        - Markup + حجم بالا در کندل‌های صعودی → تایید (+0.20)
        - Distribution + حجم بالا در کندل‌های نزولی → تایید (+0.20)
        - Markdown + حجم بالا → تایید (-0.20)
        
        وزن‌ها:
        - phase_volume_pattern: 40%
        - volume_in_phase_direction: 30%
        - phase_transition: 20%
        - cycle_strength: 10%
        """
        vm = self.volume_metrics
        signals = []
        
        # فرض: cycle_score دارای فیلد phase است
        # اگر نباشد، از score برای حدس phase استفاده می‌کنیم
        phase = self._estimate_phase(cycle_score)
        
        cycle_strength = abs(cycle_score.score)
        
        # ═══ الگوهای حجم مورد انتظار در هر فاز ═══
        
        expected_volume = {
            "ACCUMULATION": "LOW",       # حجم پایین
            "MARKUP": "HIGH_BULLISH",    # حجم بالا در کندل‌های صعودی
            "DISTRIBUTION": "MEDIUM",    # حجم متوسط
            "MARKDOWN": "HIGH_BEARISH"   # حجم بالا در کندل‌های نزولی
        }
        
        # ═══ محاسبه امتیاز interaction ═══
        
        interaction_score = 0.0
        
        # وزن 1: الگوی حجم در فاز (40%)
        if phase == "ACCUMULATION":
            if vm.volume_ratio < 0.9:  # حجم پایین - طبیعی
                interaction_score += 0.084  # 0.35 × 0.4 × 0.6
                signals.append("ACCUMULATION_LOW_VOLUME")
            elif vm.volume_spike:  # volume spike - احتمال شروع markup
                interaction_score += 0.14  # 0.35 × 0.4
                signals.append("ACCUMULATION_VOLUME_SPIKE")
        
        elif phase == "MARKUP":
            if vm.volume_in_bullish > 0.65 and vm.volume_ratio > 1.2:
                interaction_score += 0.14  # 0.35 × 0.4
                signals.append("MARKUP_STRONG_VOLUME")
            elif vm.volume_ratio < 0.8:
                interaction_score -= 0.07  # حجم پایین در markup - مشکوک
                signals.append("MARKUP_WEAK_VOLUME")
        
        elif phase == "DISTRIBUTION":
            if vm.volume_in_bearish > 0.60 and vm.volume_ratio > 1.1:
                interaction_score -= 0.14  # تایید distribution
                signals.append("DISTRIBUTION_SELLING_VOLUME")
            elif vm.volume_ratio < 0.9:
                signals.append("DISTRIBUTION_LOW_VOLUME")
        
        elif phase == "MARKDOWN":
            if vm.volume_in_bearish > 0.65 and vm.volume_ratio > 1.2:
                interaction_score -= 0.14  # 0.35 × 0.4
                signals.append("MARKDOWN_STRONG_VOLUME")
            elif vm.volume_ratio < 0.8:
                interaction_score += 0.07  # حجم پایین - احتمال پایان markdown
                signals.append("MARKDOWN_WEAK_VOLUME")
        
        # وزن 2: حجم در جهت فاز (30%)
        if phase in ["MARKUP", "ACCUMULATION"]:
            if vm.volume_in_bullish > 0.6:
                interaction_score += 0.105  # 0.35 × 0.3
                signals.append("VOLUME_IN_PHASE_DIRECTION")
        elif phase in ["MARKDOWN", "DISTRIBUTION"]:
            if vm.volume_in_bearish > 0.6:
                interaction_score -= 0.105
                signals.append("VOLUME_IN_PHASE_DIRECTION")
        
        # وزن 3: انتقال فاز (20%)
        # تشخیص انتقال از روی تغییرات ناگهانی حجم
        if phase == "ACCUMULATION" and vm.volume_spike:
            interaction_score += 0.07  # 0.35 × 0.2
            signals.append("PHASE_TRANSITION_TO_MARKUP")
        elif phase == "MARKUP" and vm.volume_in_bearish > 0.65:
            interaction_score -= 0.07  # احتمال شروع distribution
            signals.append("PHASE_TRANSITION_TO_DISTRIBUTION")
        
        # وزن 4: قدرت سیکل (10%)
        if cycle_strength > 0.7:
            interaction_score += 0.035 * np.sign(interaction_score or 1)
            signals.append("STRONG_CYCLE")
        
        # محدود کردن
        interaction_score = np.clip(interaction_score, -0.35, 0.35)
        
        # ═══ تعیین نوع interaction ═══
        
        if abs(interaction_score) > 0.20:
            interaction_type = InteractionType.STRONG_CONFIRM
        elif abs(interaction_score) > 0.10:
            interaction_type = InteractionType.CONFIRM
        else:
            interaction_type = InteractionType.NEUTRAL
        
        # محاسبه اطمینان
        confidence = min(0.95, 0.65 + abs(interaction_score))
        
        # توضیحات
        phase_persian = {
            "ACCUMULATION": "انباشت",
            "MARKUP": "صعود",
            "DISTRIBUTION": "توزیع",
            "MARKDOWN": "نزول"
        }
        
        if phase == "ACCUMULATION" and vm.volume_spike:
            explanation = f"فاز {phase_persian[phase]}: volume spike احتمالاً شروع فاز صعود"
        elif phase == "MARKUP" and vm.volume_in_bullish > 0.65:
            explanation = f"فاز {phase_persian[phase]}: حجم قوی در کندل‌های صعودی - تایید"
        elif phase == "DISTRIBUTION" and vm.volume_in_bearish > 0.60:
            explanation = f"فاز {phase_persian[phase]}: حجم فروش بالا - هشدار!"
        elif phase == "MARKDOWN" and vm.volume_in_bearish > 0.65:
            explanation = f"فاز {phase_persian[phase]}: حجم نزولی قوی - تایید روند"
        else:
            explanation = f"فاز {phase_persian.get(phase, phase)}: الگوی حجم معمولی"
        
        return VolumeDimensionInteraction(
            dimension="Cycle",
            volume_metrics=vm,
            dimension_score=cycle_score.score,
            dimension_strength=cycle_strength,
            dimension_state=phase,
            interaction_score=interaction_score,
            interaction_type=interaction_type,
            confidence=confidence,
            explanation=explanation,
            signals=signals
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # 5️⃣ Volume × S/R Interaction
    # ═══════════════════════════════════════════════════════════════════
    
    def calculate_volume_sr_interaction(
        self,
        sr_score: SupportResistanceScore
    ) -> VolumeDimensionInteraction:
        """
        تعامل حجم با حمایت/مقاومت
        
        منطق:
        - نزدیک Support + rejection candle + حجم بالا → bounce قوی (+0.25)
        - نزدیک Resistance + rejection candle + حجم بالا → bounce قوی (+0.25)
        - شکست مقاومت + حجم > 3× → breakout معتبر (+0.35)
        - شکست + حجم پایین → fake breakout (-0.30)
        
        وزن‌ها:
        - breakout_volume: 40%
        - rejection_volume: 30%
        - distance_to_level: 20%
        - level_strength: 10%
        """
        vm = self.volume_metrics
        signals = []
        
        # فرض: sr_score دارای فیلدهای distance و level_type است
        sr_strength = abs(sr_score.score)
        
        # شبیه‌سازی موقعیت نسبت به سطوح
        near_support = sr_score.score > 0.5  # نزدیک support
        near_resistance = sr_score.score < -0.5  # نزدیک resistance
        at_level = abs(sr_score.score) > 0.7
        
        # تشخیص breakout (قیمت از سطح عبور کرده)
        breakout_detected = self._detect_breakout()
        
        # ═══ محاسبه امتیاز interaction ═══
        
        interaction_score = 0.0
        
        # وزن 1: حجم در breakout (40%)
        if breakout_detected:
            if vm.volume_ratio > 3.0:  # حجم بسیار بالا
                interaction_score += 0.14  # 0.35 × 0.4 - breakout معتبر
                signals.append("BREAKOUT_VERY_HIGH_VOLUME")
            elif vm.volume_ratio > 2.0:
                interaction_score += 0.105  # 0.35 × 0.4 × 0.75
                signals.append("BREAKOUT_HIGH_VOLUME")
            elif vm.volume_ratio < 1.2:  # حجم پایین
                interaction_score -= 0.12  # 0.35 × 0.4 × 0.85 - fake breakout
                signals.append("BREAKOUT_LOW_VOLUME_FAKE")
        
        # وزن 2: حجم در rejection (30%)
        rejection_candle = self._detect_rejection_candle()
        if rejection_candle and at_level:
            if vm.volume_ratio > 1.5:
                if near_support:
                    interaction_score += 0.105  # 0.35 × 0.3 - bounce from support
                    signals.append("SUPPORT_BOUNCE_HIGH_VOLUME")
                elif near_resistance:
                    interaction_score += 0.105  # bounce from resistance (برگشت نزولی)
                    signals.append("RESISTANCE_REJECTION_HIGH_VOLUME")
        
        # وزن 3: فاصله تا سطح (20%)
        if at_level:  # نزدیک سطح
            if vm.volume_ratio > 1.3:
                interaction_score += 0.07 * np.sign(sr_score.score)
                signals.append("AT_LEVEL_HIGH_VOLUME")
        
        # وزن 4: قدرت سطح (10%)
        # اگر سطح قوی است (sr_strength بالا) و حجم تایید می‌کند
        if sr_strength > 0.7 and vm.volume_ratio > 1.2:
            interaction_score += 0.035
            signals.append("STRONG_LEVEL")
        
        # محدود کردن
        interaction_score = np.clip(interaction_score, -0.35, 0.35)
        
        # ═══ تعیین نوع interaction ═══
        
        if breakout_detected and vm.volume_ratio > 2.5:
            interaction_type = InteractionType.STRONG_CONFIRM
        elif breakout_detected and vm.volume_ratio < 1.2:
            interaction_type = InteractionType.FAKE
        elif rejection_candle and vm.volume_ratio > 1.5:
            interaction_type = InteractionType.CONFIRM
        elif abs(interaction_score) > 0.15:
            interaction_type = InteractionType.CONFIRM if interaction_score > 0 else InteractionType.WARN
        else:
            interaction_type = InteractionType.NEUTRAL
        
        # محاسبه اطمینان
        confidence = min(0.95, 0.65 + abs(interaction_score))
        
        # توضیحات
        if breakout_detected and vm.volume_ratio > 2.5:
            explanation = f"شکست با حجم {vm.volume_ratio:.1f}× - breakout معتبر!"
        elif breakout_detected and vm.volume_ratio < 1.2:
            explanation = f"شکست با حجم پایین ({vm.volume_ratio:.1f}×) - احتمال fake!"
        elif near_support and rejection_candle and vm.volume_ratio > 1.5:
            explanation = "Bounce قوی از سطح حمایت با حجم بالا"
        elif near_resistance and rejection_candle and vm.volume_ratio > 1.5:
            explanation = "Rejection قوی از سطح مقاومت با حجم بالا"
        else:
            explanation = "تعامل معمولی حجم با سطوح S/R"
        
        state = "AT_SUPPORT" if near_support else "AT_RESISTANCE" if near_resistance else "BETWEEN_LEVELS"
        
        return VolumeDimensionInteraction(
            dimension="SupportResistance",
            volume_metrics=vm,
            dimension_score=sr_score.score,
            dimension_strength=sr_strength,
            dimension_state=state,
            interaction_score=interaction_score,
            interaction_type=interaction_type,
            confidence=confidence,
            explanation=explanation,
            signals=signals
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # Helper Methods
    # ═══════════════════════════════════════════════════════════════════
    
    def _estimate_rsi(self, period: int = 14) -> float:
        """تخمین RSI از کندل‌ها"""
        if len(self.candles) < period + 1:
            return 50.0
        
        gains = []
        losses = []
        
        for i in range(len(self.candles) - period, len(self.candles)):
            change = self.candles[i].close - self.candles[i-1].close
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _estimate_mfi(self, period: int = 14) -> float:
        """تخمین Money Flow Index"""
        # ساده‌سازی: از نسبت حجم در کندل‌های صعودی/نزولی
        vm = self.volume_metrics
        
        # MFI تقریباً معادل RSI اما با حجم
        # نسبت حجم صعودی به کل = تقریب MFI
        mfi = vm.volume_in_bullish * 100
        
        return mfi
    
    def _calculate_price_trend(self, period: int = 20) -> float:
        """محاسبه روند قیمت"""
        if len(self.candles) < period:
            return 0.0
        
        closes = [c.close for c in self.candles[-period:]]
        x = np.arange(len(closes))
        y = np.array(closes)
        
        if np.std(y) > 0:
            slope = np.polyfit(x, y, 1)[0]
            return slope / np.mean(y)  # نرمال‌سازی
        
        return 0.0
    
    def _detect_bb_squeeze(self) -> bool:
        """تشخیص Bollinger Bands Squeeze"""
        if len(self.candles) < 20:
            return False
        
        closes = [c.close for c in self.candles[-20:]]
        sma = np.mean(closes)
        std = np.std(closes)
        
        # squeeze زمانی است که std بسیار کم باشد
        bb_width = (2 * std) / sma
        
        # squeeze: bb_width < 0.04 (4%)
        return bb_width < 0.04
    
    def _estimate_phase(self, cycle_score: CycleScore) -> str:
        """تخمین فاز بازار از cycle_score"""
        score = cycle_score.score
        
        if score > 0.5:
            return "MARKUP"
        elif score > 0:
            return "ACCUMULATION"
        elif score > -0.5:
            return "DISTRIBUTION"
        else:
            return "MARKDOWN"
    
    def _detect_breakout(self) -> bool:
        """تشخیص breakout از سطح"""
        # ساده‌سازی: بررسی آیا کندل اخیر از محدوده قبلی خارج شده
        if len(self.candles) < 21:
            return False
        
        recent_highs = [c.high for c in self.candles[-20:-1]]
        recent_lows = [c.low for c in self.candles[-20:-1]]
        
        resistance = max(recent_highs)
        support = min(recent_lows)
        
        current = self.candles[-1]
        
        # breakout: قیمت از resistance یا support عبور کرده
        breakout_up = current.close > resistance * 1.005  # 0.5% بالاتر
        breakout_down = current.close < support * 0.995   # 0.5% پایین‌تر
        
        return breakout_up or breakout_down
    
    def _detect_rejection_candle(self) -> bool:
        """تشخیص کندل rejection (سایه بلند)"""
        if not self.candles:
            return False
        
        candle = self.candles[-1]
        
        body = abs(candle.close - candle.open)
        upper_shadow = candle.high - max(candle.open, candle.close)
        lower_shadow = min(candle.open, candle.close) - candle.low
        
        # rejection: سایه بلند (>2× body)
        has_upper_rejection = upper_shadow > 2 * body
        has_lower_rejection = lower_shadow > 2 * body
        
        return has_upper_rejection or has_lower_rejection
    
    # ═══════════════════════════════════════════════════════════════════
    # Main Method: Calculate All Interactions
    # ═══════════════════════════════════════════════════════════════════
    
    def calculate_all_interactions(
        self,
        trend_score: TrendScore,
        momentum_score: MomentumScore,
        volatility_score: VolatilityScore,
        cycle_score: CycleScore,
        sr_score: SupportResistanceScore
    ) -> Dict[str, VolumeDimensionInteraction]:
        """
        محاسبه همه 5 interaction به صورت همزمان
        
        Returns:
            دیکشنری با کلیدهای: "Trend", "Momentum", "Volatility", "Cycle", "SupportResistance"
        """
        return {
            "Trend": self.calculate_volume_trend_interaction(trend_score),
            "Momentum": self.calculate_volume_momentum_interaction(momentum_score),
            "Volatility": self.calculate_volume_volatility_interaction(volatility_score),
            "Cycle": self.calculate_volume_cycle_interaction(cycle_score),
            "SupportResistance": self.calculate_volume_sr_interaction(sr_score)
        }
