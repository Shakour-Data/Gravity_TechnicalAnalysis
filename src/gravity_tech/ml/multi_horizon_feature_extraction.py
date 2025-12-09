"""
Multi-Horizon Feature Extraction for ML Weight Learning

استخراج ویژگی‌ها برای پیش‌بینی در چندین افق زمانی:
- 3 روز (کوتاه‌مدت - Day Trading)
- 7 روز (میان‌مدت - Swing Trading)
- 30 روز (بلندمدت - Position Trading)
"""


import logging
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from gravity_tech.core.domain.entities import Candle
from gravity_tech.core.indicators.trend import TrendIndicators
from gravity_tech.patterns.candlestick import CandlestickPatterns
from gravity_tech.patterns.classical import ClassicalPatterns
from gravity_tech.patterns.elliott_wave import ElliottWaveAnalyzer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrendSeriesCache:
    """Cached arrays to avoid recomputing series per sliding window."""

    closes: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    volumes: np.ndarray

logger = logging.getLogger(__name__)

class MultiHorizonFeatureExtractor:
    """
    استخراج ویژگی‌ها با چندین افق زمانی برای پیش‌بینی
    """

    # افق‌های زمانی (روز)
    HORIZONS = [3, 7, 30]

    def __init__(
        self,
        lookback_period: int = 100,
        horizons: list[int | str] | None = None
    ):
        """
        Initialize multi-horizon feature extractor

        Args:
            lookback_period: تعداد کندل‌های گذشته برای محاسبه اندیکاتورها
            horizons: لیست افق‌های زمانی (پیش‌فرض: [3, 7, 30])
                     می‌تواند int باشد [3, 7, 30] یا string ['3d', '7d', '30d']
            (پارامترها قابل‌تنظیم برای اسکریپت‌های آموزش)
        """
        self.lookback_period = lookback_period
        # تبدیل robust horizons به int (حتی اگر ورودی ['3d', ...] یا ترکیبی باشد)
        def _to_int_horizon(h):
            if isinstance(h, str):
                return int(h.lower().replace('d','').strip())
            return int(h)
        if horizons is None:
            self.horizons = self.HORIZONS
        else:
            self.horizons = [_to_int_horizon(h) for h in horizons]
        # حداکثر افق برای محاسبات
        self.max_horizon = max(self.horizons)

    def extract_indicator_features(
        self,
        candles: list[Candle]
    ) -> dict[str, float]:
        """
        استخراج ویژگی‌های اندیکاتورها (سطح 1)

        Returns:
            21 ویژگی از 7 اندیکاتور (signal, confidence, weighted)
        """
        if len(candles) < self.lookback_period:
            raise ValueError(f"Need at least {self.lookback_period} candles")

        # 7 اندیکاتور موجود
        indicators = {
            'sma': TrendIndicators.sma(candles, period=20),
            'ema': TrendIndicators.ema(candles, period=20),
            'wma': TrendIndicators.wma(candles, period=20),
            'dema': TrendIndicators.dema(candles, period=20),
            'tema': TrendIndicators.tema(candles, period=20),
            'macd': TrendIndicators.macd(candles),
            'adx': TrendIndicators.adx(candles)
        }

        features = {}

        for name, result in indicators.items():
            # سیگنال نرمال شده [-1, 1]
            signal_score = result.signal.get_score() / 2.0

            features[f'{name}_signal'] = signal_score
            features[f'{name}_confidence'] = result.confidence
            features[f'{name}_weighted'] = signal_score * result.confidence

        return features

    def extract_dimension_features(
        self,
        candles: list[Candle]
    ) -> dict[str, float]:
        """
        استخراج ویژگی‌های 4 بُعد (سطح 2)

        Returns:
            12 ویژگی از 4 بُعد (score, confidence, weighted)
        """
        if len(candles) < self.lookback_period:
            raise ValueError(f"Need at least {self.lookback_period} candles")

        features = {}

        # ═══════════════════════════════════════════════════════
        # بُعد 1: اندیکاتورهای تکنیکال
        # ═══════════════════════════════════════════════════════
        try:
            indicator_results = [
                TrendIndicators.sma(candles, 20),
                TrendIndicators.ema(candles, 20),
                TrendIndicators.wma(candles, 20),
                TrendIndicators.dema(candles, 20),
                TrendIndicators.tema(candles, 20),
                TrendIndicators.macd(candles),
                TrendIndicators.adx(candles)
            ]

            weighted_sum = sum(ind.signal.get_score() * ind.confidence for ind in indicator_results)
            total_weight = sum(ind.confidence for ind in indicator_results)

            dim1_score = weighted_sum / total_weight if total_weight > 0 else 0.0
            dim1_confidence = total_weight / len(indicator_results)

        except Exception as e:
            logger.warning("Indicator calculation error", exc_info=e)
            dim1_score = 0.0
            dim1_confidence = 0.5

        features['dim1_indicators_score'] = dim1_score / 2.0  # نرمال به [-1, 1]
        features['dim1_indicators_confidence'] = dim1_confidence
        features['dim1_indicators_weighted'] = (dim1_score / 2.0) * dim1_confidence

        # ═══════════════════════════════════════════════════════
        # بُعد 2: الگوهای شمعی
        # ═══════════════════════════════════════════════════════
        try:
            patterns = CandlestickPatterns.detect_patterns(candles[-10:])

            if patterns:
                # میانگین سیگنال و اعتماد
                avg_signal = sum(p.signal.get_score() for p in patterns) / len(patterns)
                avg_confidence = sum(p.confidence for p in patterns) / len(patterns)

                dim2_score = avg_signal / 2.0
                dim2_confidence = avg_confidence
            else:
                dim2_score = 0.0
                dim2_confidence = 0.5

        except Exception as e:
            logger.warning("Candlestick pattern error", exc_info=e)
            dim2_score = 0.0
            dim2_confidence = 0.5

        features['dim2_candlestick_score'] = dim2_score
        features['dim2_candlestick_confidence'] = dim2_confidence
        features['dim2_candlestick_weighted'] = dim2_score * dim2_confidence

        # ═══════════════════════════════════════════════════════
        # بُعد 3: امواج الیوت
        # ═══════════════════════════════════════════════════════
        try:
            elliott = ElliottWaveAnalyzer.analyze(candles)

            dim3_score = elliott.signal.get_score() / 2.0
            dim3_confidence = elliott.confidence

        except Exception as e:
            logger.warning("Elliott Wave error", exc_info=e)
            dim3_score = 0.0
            dim3_confidence = 0.5

        features['dim3_elliott_score'] = dim3_score
        features['dim3_elliott_confidence'] = dim3_confidence
        features['dim3_elliott_weighted'] = dim3_score * dim3_confidence

        # ═══════════════════════════════════════════════════════
        # بُعد 4: الگوهای کلاسیک
        # ═══════════════════════════════════════════════════════
        try:
            classical = ClassicalPatterns.detect_all(candles)

            if classical:
                avg_signal = sum(p.signal.get_score() for p in classical) / len(classical)
                avg_confidence = sum(p.confidence for p in classical) / len(classical)

                dim4_score = avg_signal / 2.0
                dim4_confidence = avg_confidence
            else:
                dim4_score = 0.0
                dim4_confidence = 0.5

        except Exception as e:
            logger.warning("Classical pattern error", exc_info=e)
            dim4_score = 0.0
            dim4_confidence = 0.5

        features['dim4_classical_score'] = dim4_score
        features['dim4_classical_confidence'] = dim4_confidence
        features['dim4_classical_weighted'] = dim4_score * dim4_confidence

        return features

    def calculate_multi_horizon_returns(
        self,
        candles: list[Candle],
        current_idx: int
    ) -> dict[str, float]:
        """
        محاسبه بازدهی برای چندین افق زمانی

        Returns:
            {
                'return_3d': 0.025,   # 2.5%
                'return_7d': 0.058,   # 5.8%
                'return_30d': 0.152   # 15.2%
            }
        """
        # Validate current candle
        current_candle = candles[current_idx]
        if not isinstance(current_candle, Candle):
            raise TypeError(f"Expected Candle at index {current_idx}, got {type(current_candle)}")

        current_price = current_candle.close
        returns = {}

        for horizon in self.horizons:
            future_idx = current_idx + horizon

            if future_idx < len(candles):
                future_candle = candles[future_idx]
                # Validate future candle
                if not isinstance(future_candle, Candle):
                    raise TypeError(f"Expected Candle at index {future_idx}, got {type(future_candle)}")
                future_price = future_candle.close
                return_pct = (future_price - current_price) / current_price
                returns[f'return_{horizon}d'] = return_pct
            else:
                # داده کافی نیست
                returns[f'return_{horizon}d'] = None

        return returns

    def extract_training_dataset(
        self,
        candles: list[Candle],
        level: str = "indicators"  # "indicators" or "dimensions"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        استخراج دیتاست آموزشی با چندین هدف

        Args:
            candles: کندل‌های تاریخی
            level: سطح ویژگی‌ها (indicators یا dimensions)

        Returns:
            X: DataFrame ویژگی‌ها
            Y: DataFrame اهداف (ستون‌های return_3d, return_7d, return_30d)
        """
        total_needed = self.lookback_period + self.max_horizon
        if len(candles) < total_needed:
            raise ValueError(f"Need at least {total_needed} candles for multi-horizon dataset")

        cache = self._build_series_cache(candles)
        return_matrix = self._compute_return_matrix(cache.closes)

        logger.info(
            "Extracting multi-horizon features",
            extra={"level": level, "horizons": self.horizons},
        )

        features_rows: list[dict[str, float]] = []
        targets_rows: list[dict[str, float]] = []

        usable_length = len(candles) - self.max_horizon - self.lookback_period + 1
        window_deque: deque[Candle] = deque(
            candles[: self.lookback_period],
            maxlen=self.lookback_period,
        )

        for offset in range(usable_length):
            start = offset
            end = offset + self.lookback_period
            current_idx = end - 1

            targets = self._collect_targets(return_matrix, current_idx)
            if targets is None:
                if end < len(candles):
                    window_deque.append(candles[end])
                continue

            try:
                if level == "indicators":
                    features = self.extract_indicator_features(list(window_deque))
                elif level == "dimensions":
                    features = self.extract_dimension_features(list(window_deque))
                else:
                    raise ValueError(f"Unknown level: {level}")

                features_rows.append(features)
                targets_rows.append(targets)
            except Exception as e:
                logger.warning("Error processing sample", extra={"index": offset}, exc_info=e)

            next_idx = end
            if next_idx < len(candles):
                window_deque.append(candles[next_idx])

        X = pd.DataFrame(features_rows)
        Y = pd.DataFrame(targets_rows)

        logger.info(
            "Extracted multi-horizon dataset",
            extra={"samples": len(X), "features": X.shape[1], "targets": Y.shape[1]},
        )

        for horizon in self.horizons:
            col = f'return_{horizon}d'
            mean_return = Y[col].mean()
            std_return = Y[col].std()
            logger.debug(
                "Target stats",
                extra={
                    "column": col,
                    "mean": float(mean_return),
                    "std": float(std_return),
                    "mean_pct": float(mean_return * 100),
                },
            )

        return X, Y

    def _build_series_cache(self, candles: list[Candle]) -> TrendSeriesCache:
        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)
        return TrendSeriesCache(closes=closes, highs=highs, lows=lows, volumes=volumes)

    def _compute_return_matrix(self, closes: np.ndarray) -> dict[int, np.ndarray]:
        return_matrix: dict[int, np.ndarray] = {}
        for horizon in self.horizons:
            if horizon <= 0 or horizon >= len(closes):
                continue
            fwd = closes[horizon:]
            curr = closes[:-horizon]
            returns = (fwd - curr) / curr
            padded = np.concatenate([np.full(horizon, np.nan), returns])
            return_matrix[horizon] = padded
        return return_matrix

    def _collect_targets(
        self,
        return_matrix: dict[int, np.ndarray],
        idx: int,
    ) -> Optional[dict[str, float]]:
        targets: dict[str, float] = {}
        for horizon in self.horizons:
            series = return_matrix.get(horizon)
            if series is None or idx >= len(series):
                return None
            val = series[idx]
            if np.isnan(val):
                return None
            targets[f'return_{horizon}d'] = float(val)
        return targets

    def create_summary_statistics(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame
    ) -> dict:
        """
        آمار خلاصه از دیتاست
        """
        return {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_horizons': Y.shape[1],
            'horizons': self.horizons,
            'feature_names': list(X.columns),
            'target_statistics': {
                f'return_{h}d': {
                    'mean': Y[f'return_{h}d'].mean(),
                    'std': Y[f'return_{h}d'].std(),
                    'min': Y[f'return_{h}d'].min(),
                    'max': Y[f'return_{h}d'].max(),
                    'positive_ratio': (Y[f'return_{h}d'] > 0).mean()
                }
                for h in self.horizons
            }
        }
