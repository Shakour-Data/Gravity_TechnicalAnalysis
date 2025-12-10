"""
Multi-Horizon Feature Extraction for Volatility
"""

import logging
from collections import deque
from dataclasses import dataclass


import numpy as np
import pandas as pd
from gravity_tech.core.domain.entities import Candle
from gravity_tech.core.indicators.volatility import VolatilityIndicators

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VolatilitySeriesCache:
    """Cached series to avoid recomputing true range/ATR and returns."""

    closes: np.ndarray
    true_range: np.ndarray
    atr: np.ndarray


class MultiHorizonVolatilityFeatureExtractor:
    """
    استخراج ویژگی‌های نوسان برای چندین افق زمانی
    """

    HORIZONS = [3, 7, 30]

    # اندیکاتورهای نوسان
    VOLATILITY_INDICATORS = [
        'atr',
        'bollinger_bands',
        'keltner_channel',
        'donchian_channel',
        'standard_deviation',
        'historical_volatility',
        'atr_percentage',
        'chaikin_volatility'
    ]

    def __init__(
        self,
        lookback_period: int = 100,
        horizons: list = None,
        atr_span: int = 14,
    ):
        """
        Initialize feature extractor

        Args:
            lookback_period: تعداد کندل‌های گذشته
            horizons: لیست افق‌های زمانی (پیش‌فرض: [3, 7, 30])
                     می‌تواند int باشد [3, 7, 30] یا string ['3d', '7d', '30d']
            atr_span: طول EMA برای ATR (پیش‌فرض 14)
        """
        self.lookback_period = lookback_period
        self.atr_span = atr_span

        # تبدیل horizons به int اگر string است
        if horizons is None:
            self.horizons = self.HORIZONS
        else:
            self.horizons = []
            for h in horizons:
                if isinstance(h, str):
                    self.horizons.append(int(h.replace('d', '')))
                else:
                    self.horizons.append(int(h))

        self.max_horizon = max(self.horizons)

    def extract_volatility_features(
        self,
        candles: list[Candle]
    ) -> dict[str, float]:
        """
        استخراج ویژگی‌های نوسان

        Returns:
            ویژگی‌ها شامل:
            - {indicator}_signal: سیگنال نرمال شده [-1, 1]
            - {indicator}_confidence: دقت [0, 1]
            - {indicator}_weighted: signal × confidence
            - {indicator}_normalized: مقدار نرمال شده [-1, 1]
            - {indicator}_percentile: صدک تاریخی [0, 100]
        """
        if len(candles) < self.lookback_period:
            raise ValueError(f"Need at least {self.lookback_period} candles")

        features = {}

        # ═══════════════════════════════════════════════════════
        # 1. ATR (Average True Range)
        # ═══════════════════════════════════════════════════════
        try:
            atr_result = VolatilityIndicators.atr(candles, period=14)

            features['atr_signal'] = atr_result.signal.get_score() / 2.0  # نرمال به [-1, 1]
            features['atr_confidence'] = atr_result.confidence
            features['atr_weighted'] = features['atr_signal'] * features['atr_confidence']
            features['atr_normalized'] = atr_result.normalized
            features['atr_percentile'] = atr_result.percentile
        except Exception as e:
            logger.warning("ATR calculation error", exc_info=e)
            features.update({
                'atr_signal': 0.0, 'atr_confidence': 0.5,
                'atr_weighted': 0.0, 'atr_normalized': 0.0, 'atr_percentile': 50.0
            })

        # ═══════════════════════════════════════════════════════
        # 2. Bollinger Bands
        # ═══════════════════════════════════════════════════════
        try:
            bb_result = VolatilityIndicators.bollinger_bands(candles, period=20)

            features['bollinger_bands_signal'] = bb_result.signal.get_score() / 2.0
            features['bollinger_bands_confidence'] = bb_result.confidence
            features['bollinger_bands_weighted'] = features['bollinger_bands_signal'] * features['bollinger_bands_confidence']
            features['bollinger_bands_normalized'] = bb_result.normalized
            features['bollinger_bands_percentile'] = bb_result.percentile
        except Exception as e:
            logger.warning("Bollinger Bands calculation error", exc_info=e)
            features.update({
                'bollinger_bands_signal': 0.0, 'bollinger_bands_confidence': 0.5,
                'bollinger_bands_weighted': 0.0, 'bollinger_bands_normalized': 0.0,
                'bollinger_bands_percentile': 50.0
            })

        # ═══════════════════════════════════════════════════════
        # 3. Keltner Channel
        # ═══════════════════════════════════════════════════════
        try:
            keltner_result = VolatilityIndicators.keltner_channel(candles, period=20)

            features['keltner_channel_signal'] = keltner_result.signal.get_score() / 2.0
            features['keltner_channel_confidence'] = keltner_result.confidence
            features['keltner_channel_weighted'] = features['keltner_channel_signal'] * features['keltner_channel_confidence']
            features['keltner_channel_normalized'] = keltner_result.normalized
            features['keltner_channel_percentile'] = keltner_result.percentile
        except Exception as e:
            logger.warning("Keltner Channel calculation error", exc_info=e)
            features.update({
                'keltner_channel_signal': 0.0, 'keltner_channel_confidence': 0.5,
                'keltner_channel_weighted': 0.0, 'keltner_channel_normalized': 0.0,
                'keltner_channel_percentile': 50.0
            })

        # ═══════════════════════════════════════════════════════
        # 4. Donchian Channel
        # ═══════════════════════════════════════════════════════
        try:
            donchian_result = VolatilityIndicators.donchian_channel(candles, period=20)

            features['donchian_channel_signal'] = donchian_result.signal.get_score() / 2.0
            features['donchian_channel_confidence'] = donchian_result.confidence
            features['donchian_channel_weighted'] = features['donchian_channel_signal'] * features['donchian_channel_confidence']
            features['donchian_channel_normalized'] = donchian_result.normalized
            features['donchian_channel_percentile'] = donchian_result.percentile
        except Exception as e:
            logger.warning("Donchian Channel calculation error", exc_info=e)
            features.update({
                'donchian_channel_signal': 0.0, 'donchian_channel_confidence': 0.5,
                'donchian_channel_weighted': 0.0, 'donchian_channel_normalized': 0.0,
                'donchian_channel_percentile': 50.0
            })

        # ═══════════════════════════════════════════════════════
        # 5. Standard Deviation
        # ═══════════════════════════════════════════════════════
        try:
            std_result = VolatilityIndicators.standard_deviation(candles, period=20)

            features['standard_deviation_signal'] = std_result.signal.get_score() / 2.0
            features['standard_deviation_confidence'] = std_result.confidence
            features['standard_deviation_weighted'] = features['standard_deviation_signal'] * features['standard_deviation_confidence']
            features['standard_deviation_normalized'] = std_result.normalized
            features['standard_deviation_percentile'] = std_result.percentile
        except Exception as e:
            logger.warning("Standard Deviation calculation error", exc_info=e)
            features.update({
                'standard_deviation_signal': 0.0, 'standard_deviation_confidence': 0.5,
                'standard_deviation_weighted': 0.0, 'standard_deviation_normalized': 0.0,
                'standard_deviation_percentile': 50.0
            })

        # ═══════════════════════════════════════════════════════
        # 6. Historical Volatility
        # ═══════════════════════════════════════════════════════
        try:
            hv_result = VolatilityIndicators.historical_volatility(candles, period=20)

            features['historical_volatility_signal'] = hv_result.signal.get_score() / 2.0
            features['historical_volatility_confidence'] = hv_result.confidence
            features['historical_volatility_weighted'] = features['historical_volatility_signal'] * features['historical_volatility_confidence']
            features['historical_volatility_normalized'] = hv_result.normalized
            features['historical_volatility_percentile'] = hv_result.percentile
        except Exception as e:
            logger.warning("Historical Volatility calculation error", exc_info=e)
            features.update({
                'historical_volatility_signal': 0.0, 'historical_volatility_confidence': 0.5,
                'historical_volatility_weighted': 0.0, 'historical_volatility_normalized': 0.0,
                'historical_volatility_percentile': 50.0
            })

        # ═══════════════════════════════════════════════════════
        # 7. ATR Percentage
        # ═══════════════════════════════════════════════════════
        try:
            atr_pct_result = VolatilityIndicators.atr_percentage(candles, period=14)

            features['atr_percentage_signal'] = atr_pct_result.signal.get_score() / 2.0
            features['atr_percentage_confidence'] = atr_pct_result.confidence
            features['atr_percentage_weighted'] = features['atr_percentage_signal'] * features['atr_percentage_confidence']
            features['atr_percentage_normalized'] = atr_pct_result.normalized
            features['atr_percentage_percentile'] = atr_pct_result.percentile
        except Exception as e:
            logger.warning("ATR Percentage calculation error", exc_info=e)
            features.update({
                'atr_percentage_signal': 0.0, 'atr_percentage_confidence': 0.5,
                'atr_percentage_weighted': 0.0, 'atr_percentage_normalized': 0.0,
                'atr_percentage_percentile': 50.0
            })

        # ═══════════════════════════════════════════════════════
        # 8. Chaikin Volatility
        # ═══════════════════════════════════════════════════════
        try:
            chaikin_result = VolatilityIndicators.chaikin_volatility(candles, period=10)

            features['chaikin_volatility_signal'] = chaikin_result.signal.get_score() / 2.0
            features['chaikin_volatility_confidence'] = chaikin_result.confidence
            features['chaikin_volatility_weighted'] = features['chaikin_volatility_signal'] * features['chaikin_volatility_confidence']
            features['chaikin_volatility_normalized'] = chaikin_result.normalized
            features['chaikin_volatility_percentile'] = chaikin_result.percentile
        except Exception as e:
            logger.warning("Chaikin Volatility calculation error", exc_info=e)
            features.update({
                'chaikin_volatility_signal': 0.0, 'chaikin_volatility_confidence': 0.5,
                'chaikin_volatility_weighted': 0.0, 'chaikin_volatility_normalized': 0.0,
                'chaikin_volatility_percentile': 50.0
            })

        return features

    def calculate_future_volatility_change(
        self,
        candles: list[Candle],
        horizon: int
    ) -> float:
        """
        محاسبه تغییر نوسان در آینده

        برای آموزش ML، باید بدانیم نوسان در آینده افزایش یا کاهش می‌یابد.

        از ATR استفاده می‌کنیم به عنوان معیار نوسان:
        - اگر ATR افزایش یابد → نوسان در حال افزایش (مثبت)
        - اگر ATR کاهش یابد → نوسان در حال کاهش (منفی)

        Args:
            candles: لیست کندل‌ها
            horizon: افق زمانی (چند روز آینده)

        Returns:
            تغییر نوسان نرمال شده [-1, +1]
        """
        if len(candles) < horizon + 20:
            return 0.0

        try:
            # محاسبه ATR فعلی
            current_candles = candles[:-horizon]
            tr_current = VolatilityIndicators.true_range(current_candles)
            atr_current = pd.Series(tr_current).ewm(span=14, adjust=False).mean().iloc[-1]

            # محاسبه ATR آینده
            future_candles = candles
            tr_future = VolatilityIndicators.true_range(future_candles)
            atr_future = pd.Series(tr_future).ewm(span=14, adjust=False).mean().iloc[-1]

            # درصد تغییر
            pct_change = ((atr_future - atr_current) / atr_current) * 100

            # نرمال سازی به [-1, +1]
            # +50% = +1, -50% = -1
            normalized_change = np.clip(pct_change / 50.0, -1.0, 1.0)

            return normalized_change

        except Exception as e:
            logger.warning("Future volatility calculation error", exc_info=e)
            return 0.0

    def extract_features_with_target(
        self,
        candles: list[Candle],
        horizon: int
    ) -> tuple[dict[str, float], float]:
        """
        استخراج ویژگی‌ها به همراه target برای آموزش

        Args:
            candles: لیست کندل‌ها
            horizon: افق زمانی

        Returns:
            (features, target)
            - features: ویژگی‌های نوسان
            - target: تغییر نوسان در آینده [-1, +1]
        """
        # استخراج ویژگی‌ها از کندل‌های قبل از horizon
        training_candles = candles[:-horizon] if horizon > 0 else candles
        features = self.extract_volatility_features(training_candles)

        # محاسبه target
        target = self.calculate_future_volatility_change(candles, horizon)

        return features, target

    def create_training_dataset(
        self,
        candles: list[Candle],
        horizons: list[int] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        ایجاد دیتاست آموزشی برای چند افق زمانی

        Args:
            candles: لیست کندل‌ها
            horizons: لیست افق‌های زمانی (پیش‌فرض: [3, 7, 30])

        Returns:
            (X, y)
            - X: DataFrame ویژگی‌ها
            - y: DataFrame اهداف (یک ستون برای هر horizon)
        """
        if horizons is None:
            horizons = self.horizons

        total_needed = self.lookback_period + max(horizons)
        if len(candles) < total_needed:
            raise ValueError(f"Need at least {total_needed} candles for volatility dataset")

        cache = self._build_series_cache(candles)
        change_matrix = self._compute_vol_change_matrix(cache.atr, horizons)

        X_rows: list[dict[str, float]] = []
        y_rows: list[dict[str, float]] = []

        usable_length = len(candles) - max(horizons) - self.lookback_period + 1
        window_deque: deque[Candle] = deque(
            candles[: self.lookback_period],
            maxlen=self.lookback_period,
        )

        for offset in range(usable_length):
            end = offset + self.lookback_period
            current_idx = end - 1

            targets = self._collect_vol_targets(change_matrix, horizons, current_idx)
            if targets is None:
                if end < len(candles):
                    window_deque.append(candles[end])
                continue

            try:
                features = self.extract_volatility_features(list(window_deque))
                X_rows.append(features)
                y_rows.append(targets)
            except Exception as e:
                logger.warning("Volatility dataset error", extra={"index": offset}, exc_info=e)

            if end < len(candles):
                window_deque.append(candles[end])

        X = pd.DataFrame(X_rows)
        y = pd.DataFrame(y_rows)

        logger.info(
            "Extracted volatility dataset",
            extra={"samples": len(X), "features": X.shape[1], "targets": y.shape[1]},
        )
        return X, y

    def _build_series_cache(self, candles: list[Candle]) -> VolatilitySeriesCache:
        closes = np.array([c.close for c in candles], dtype=float)
        true_range = np.array(VolatilityIndicators.true_range(candles), dtype=float)
        atr_series = pd.Series(true_range).ewm(span=self.atr_span, adjust=False).mean().to_numpy()
        return VolatilitySeriesCache(closes=closes, true_range=true_range, atr=atr_series)

    def _compute_vol_change_matrix(
        self,
        atr: np.ndarray,
        horizons: list[int],
    ) -> dict[int, np.ndarray]:
        change_matrix: dict[int, np.ndarray] = {}
        for h in horizons:
            if h <= 0 or h >= len(atr):
                continue
            future = atr[h:]
            current = atr[:-h]
            pct_change = ((future - current) / current) * 100
            padded = np.concatenate([np.full(h, np.nan), pct_change])
            change_matrix[h] = padded
        return change_matrix

    def _collect_vol_targets(
        self,
        change_matrix: dict[int, np.ndarray],
        horizons: list[int],
        idx: int,
    ) -> dict[str, float | None]:
        targets: dict[str, float] = {}
        for h in horizons:
            series = change_matrix.get(h)
            if series is None or idx >= len(series):
                return None
            val = series[idx]
            if np.isnan(val):
                return None
            targets[f'target_{h}d'] = float(np.clip(val / 50.0, -1.0, 1.0))
        return targets

    def get_feature_names(self) -> list[str]:
        """
        لیست نام ویژگی‌ها
        """
        feature_names = []
        for indicator in self.VOLATILITY_INDICATORS:
            feature_names.extend([
                f'{indicator}_signal',
                f'{indicator}_confidence',
                f'{indicator}_weighted',
                f'{indicator}_normalized',
                f'{indicator}_percentile'
            ])
        return feature_names
