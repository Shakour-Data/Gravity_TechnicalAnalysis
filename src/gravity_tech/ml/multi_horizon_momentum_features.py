"""
Multi-Horizon Feature Extraction for Momentum
============================================

- 8 اندیکاتور مومنتوم/حجم: RSI، Stochastic، CCI، Williams %R، ROC، Momentum،
  OBV، CMF
- استخراج سیگنال، اطمینان، مقدار وزن‌دهی و واگرایی برای هر اندیکاتور
- پشتیبانی از یادگیری چند افقی (3، 7 و 30 روزه)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from gravity_tech.core.domain.entities import Candle
from gravity_tech.core.domain.entities import CoreSignalStrength as SignalStrength
from gravity_tech.patterns.divergence import DivergenceDetector


@dataclass(slots=True)
class MomentumSeriesCache:
    """نگهدارنده سری‌های محاسبه‌شده برای تمام اندیکاتورها."""

    closes: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    volumes: np.ndarray
    rsi: np.ndarray
    stochastic_k: np.ndarray
    stochastic_d: np.ndarray
    cci: np.ndarray
    williams: np.ndarray
    roc: np.ndarray
    momentum: np.ndarray
    obv: np.ndarray
    obv_sma: np.ndarray
    obv_trend: np.ndarray
    price_trend: np.ndarray
    cmf: np.ndarray


class MultiHorizonMomentumFeatureExtractor:
    """
    استخراج ویژگی‌های مومنتوم برای مدل‌های چند افقی.
    """

    HORIZONS = [3, 7, 30]
    MOMENTUM_INDICATORS = [
        "rsi",
        "stochastic",
        "cci",
        "williams_r",
        "roc",
        "momentum",
        "obv",
        "cmf",
    ]

    def __init__(
        self,
        lookback_period: int = 100,
        horizons: list[int | str] | None = None,
    ):
        self.lookback_period = lookback_period
        if horizons is None:
            self.horizons = self.HORIZONS
        else:
            self.horizons = []
            for horizon in horizons:
                if isinstance(horizon, str):
                    self.horizons.append(int(horizon.replace("d", "")))
                else:
                    self.horizons.append(int(horizon))
        self.max_horizon = max(self.horizons)
        self.divergence_detector = DivergenceDetector(lookback=20)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def extract_momentum_features(
        self,
        candles: list[Candle],
    ) -> dict[str, float]:
        """
        استخراج ویژگی‌های مومنتوم برای یک پنجره مشخص.
        """
        if len(candles) < self.lookback_period:
            raise ValueError(f"Need at least {self.lookback_period} candles")

        cache = self._build_series_cache(candles)
        return self._build_feature_row(
            window_candles=candles,
            cache=cache,
            start_index=0,
            end_index=len(candles),
            current_index=len(candles) - 1,
        )

    def extract_training_dataset(
        self,
        candles: list[Candle],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        تولید دیتاست کامل آموزش (ویژگی‌ها و اهداف) برای یادگیری وزن‌ها.
        """
        total_needed = self.lookback_period + self.max_horizon
        if len(candles) < total_needed:
            raise ValueError(
                f"Need at least {total_needed} candles for multi-horizon dataset"
            )

        print("\n?? Extracting multi-horizon momentum features...")
        print(f"   Horizons: {self.horizons} days")
        print(f"   Indicators: {len(self.MOMENTUM_INDICATORS)}")

        cache = self._build_series_cache(candles)
        return_matrix = self._compute_return_matrix(cache.closes)

        features_rows: list[dict[str, float]] = []
        targets_rows: list[dict[str, float]] = []

        usable_length = len(candles) - self.max_horizon - self.lookback_period + 1
        candle_deque: deque[Candle] = deque(
            candles[: self.lookback_period], maxlen=self.lookback_period
        )

        for offset in range(usable_length):
            start = offset
            end = offset + self.lookback_period
            idx = end - 1

            # هدف‌ها
            targets = self._collect_targets(return_matrix, idx)
            if targets is None:
                # skip incomplete target rows (قیمت آینده در دسترس نیست)
                next_candle = candles[end]
                candle_deque.append(next_candle)
                continue

            window_candles = list(candle_deque)
            feature_row = self._build_feature_row(
                window_candles=window_candles,
                cache=cache,
                start_index=start,
                end_index=end,
                current_index=idx,
            )

            features_rows.append(feature_row)
            targets_rows.append(targets)

            # Slide window
            next_idx = end
            if next_idx < len(candles):
                candle_deque.append(candles[next_idx])

        X = pd.DataFrame(features_rows)
        Y = pd.DataFrame(targets_rows)

        print(f"? Extracted {len(X)} complete training samples")
        print(f"   Features: {X.shape[1]} columns")
        print(f"   Targets: {Y.shape[1]} horizons")
        for horizon in self.horizons:
            col = f"return_{horizon}d"
            mean_return = Y[col].mean()
            std_return = Y[col].std()
            print(
                f"   {col}: mean={mean_return:.4f} ({mean_return*100:.2f}%), "
                f"std={std_return:.4f}"
            )

        return X, Y

    def create_summary_statistics(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
    ) -> dict[str, Any]:
        """آمار خلاصه دیتاست تولید شده."""
        return {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_horizons": Y.shape[1],
            "horizons": self.horizons,
            "feature_names": list(X.columns),
            "target_statistics": {
                f"return_{h}d": {
                    "mean": Y[f"return_{h}d"].mean(),
                    "std": Y[f"return_{h}d"].std(),
                    "min": Y[f"return_{h}d"].min(),
                    "max": Y[f"return_{h}d"].max(),
                    "positive_ratio": (Y[f"return_{h}d"] > 0).mean(),
                }
                for h in self.horizons
            },
        }

    # ------------------------------------------------------------------ #
    # Feature construction helpers
    # ------------------------------------------------------------------ #
    def _build_feature_row(
        self,
        window_candles: list[Candle],
        cache: MomentumSeriesCache,
        start_index: int,
        end_index: int,
        current_index: int,
    ) -> dict[str, float]:
        """ساخت دیکشنری ویژگی‌ها برای یک اندیس مشخص."""
        features: dict[str, float] = {}

        # --- RSI ---
        rsi_value = cache.rsi[current_index]
        rsi_signal, rsi_conf = self._rsi_signal_conf(rsi_value)
        features["rsi_signal"] = rsi_signal.get_score() / 2.0
        features["rsi_confidence"] = rsi_conf
        features["rsi_weighted"] = features["rsi_signal"] * rsi_conf
        features["rsi_divergence"] = self._compute_divergence(
            window_candles,
            cache.rsi[start_index:end_index],
            "RSI",
        )

        # --- Stochastic ---
        k_value = cache.stochastic_k[current_index]
        d_value = cache.stochastic_d[current_index]
        stoch_signal, stoch_conf = self._stochastic_signal_conf(k_value, d_value)
        features["stochastic_signal"] = stoch_signal.get_score() / 2.0
        features["stochastic_confidence"] = stoch_conf
        features["stochastic_weighted"] = features["stochastic_signal"] * stoch_conf
        features["stochastic_divergence"] = self._compute_divergence(
            window_candles,
            cache.stochastic_k[start_index:end_index],
            "Stochastic",
        )

        # --- CCI ---
        cci_value = cache.cci[current_index]
        cci_signal, cci_conf = self._cci_signal_conf(cci_value)
        features["cci_signal"] = cci_signal.get_score() / 2.0
        features["cci_confidence"] = cci_conf
        features["cci_weighted"] = features["cci_signal"] * cci_conf
        features["cci_divergence"] = self._compute_divergence(
            window_candles,
            cache.cci[start_index:end_index],
            "CCI",
        )

        # --- Williams %R ---
        wr_value = cache.williams[current_index]
        wr_signal, wr_conf = self._williams_signal_conf(wr_value)
        features["williams_r_signal"] = wr_signal.get_score() / 2.0
        features["williams_r_confidence"] = wr_conf
        features["williams_r_weighted"] = features["williams_r_signal"] * wr_conf
        features["williams_r_divergence"] = self._compute_divergence(
            window_candles,
            cache.williams[start_index:end_index],
            "Williams%R",
        )

        # --- ROC ---
        roc_value = cache.roc[current_index]
        roc_signal, roc_conf = self._roc_signal_conf(roc_value)
        features["roc_signal"] = roc_signal.get_score() / 2.0
        features["roc_confidence"] = roc_conf
        features["roc_weighted"] = features["roc_signal"] * roc_conf
        features["roc_divergence"] = 0.0

        # --- Momentum ---
        mom_value = cache.momentum[current_index]
        mom_signal, mom_conf = self._momentum_signal_conf(
            mom_value, cache.closes[current_index]
        )
        features["momentum_signal"] = mom_signal.get_score() / 2.0
        features["momentum_confidence"] = mom_conf
        features["momentum_weighted"] = features["momentum_signal"] * mom_conf
        features["momentum_divergence"] = 0.0

        # --- OBV ---
        obv_value = cache.obv[current_index]
        obv_signal, obv_conf = self._obv_signal_conf(
            obv_value,
            cache.obv_sma[current_index],
            cache.obv_trend[current_index],
            cache.price_trend[current_index],
        )
        features["obv_signal"] = obv_signal.get_score() / 2.0
        features["obv_confidence"] = obv_conf
        features["obv_weighted"] = features["obv_signal"] * obv_conf
        features["obv_divergence"] = 0.0

        # --- CMF ---
        cmf_value = cache.cmf[current_index]
        cmf_signal, cmf_conf = self._cmf_signal_conf(cmf_value)
        features["cmf_signal"] = cmf_signal.get_score() / 2.0
        features["cmf_confidence"] = cmf_conf
        features["cmf_weighted"] = features["cmf_signal"] * cmf_conf
        features["cmf_divergence"] = 0.0

        return features

    # ------------------------------------------------------------------ #
    # Precomputation helpers
    # ------------------------------------------------------------------ #
    def _build_series_cache(self, candles: list[Candle]) -> MomentumSeriesCache:
        df = pd.DataFrame(
            [
                {
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                    "typical": c.typical_price,
                }
                for c in candles
            ]
        )

        closes = df["close"].to_numpy(float)
        highs = df["high"].to_numpy(float)
        lows = df["low"].to_numpy(float)
        volumes = df["volume"].to_numpy(float)

        rsi = self._compute_rsi_series(closes)
        stochastic_k, stochastic_d = self._compute_stochastic_series(df)
        cci = self._compute_cci_series(df["typical"].to_numpy(float))
        williams = self._compute_williams_series(df)
        roc = self._compute_roc_series(closes, period=12)
        momentum = self._compute_momentum_series(closes, period=10)
        obv, obv_sma, obv_trend, price_trend = self._compute_obv_series(closes, volumes)
        cmf = self._compute_cmf_series(df, period=20)

        return MomentumSeriesCache(
            closes=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            rsi=rsi,
            stochastic_k=stochastic_k,
            stochastic_d=stochastic_d,
            cci=cci,
            williams=williams,
            roc=roc,
            momentum=momentum,
            obv=obv,
            obv_sma=obv_sma,
            obv_trend=obv_trend,
            price_trend=price_trend,
            cmf=cmf,
        )

    def _compute_return_matrix(self, closes: np.ndarray) -> dict[str, np.ndarray]:
        returns: dict[str, np.ndarray] = {}
        for horizon in self.horizons:
            ret = np.full_like(closes, np.nan, dtype=float)
            ret[:-horizon] = (closes[horizon:] - closes[:-horizon]) / closes[:-horizon]
            returns[f"return_{horizon}d"] = ret
        return returns

    def _collect_targets(
        self,
        return_matrix: dict[str, np.ndarray],
        index: int,
    ) -> dict[str, float] | None:
        targets: dict[str, float] = {}
        for horizon in self.horizons:
            key = f"return_{horizon}d"
            value = return_matrix[key][index]
            if np.isnan(value):
                return None
            targets[key] = float(value)
        return targets

    # ------------------------------------------------------------------ #
    # Indicator series calculations
    # ------------------------------------------------------------------ #
    def _compute_rsi_series(self, closes: np.ndarray, period: int = 14) -> np.ndarray:
        series = pd.Series(closes)
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0).to_numpy()

    def _compute_stochastic_series(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()
        k = 100 * ((df["close"] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        k = k.fillna(50.0)
        d = d.fillna(50.0)
        return k.to_numpy(), d.to_numpy()

    def _compute_cci_series(
        self,
        typical_prices: np.ndarray,
        period: int = 20,
    ) -> np.ndarray:
        series = pd.Series(typical_prices)
        sma = series.rolling(window=period).mean()
        mad = series.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (series - sma) / (0.015 * mad.replace(0, np.nan))
        return cci.fillna(0.0).to_numpy()

    def _compute_williams_series(
        self,
        df: pd.DataFrame,
        period: int = 14,
    ) -> np.ndarray:
        high_max = df["high"].rolling(window=period).max()
        low_min = df["low"].rolling(window=period).min()
        williams = -100 * ((high_max - df["close"]) / (high_max - low_min))
        return williams.fillna(-50.0).to_numpy()

    def _compute_roc_series(
        self,
        closes: np.ndarray,
        period: int = 12,
    ) -> np.ndarray:
        series = pd.Series(closes)
        roc = ((series - series.shift(period)) / series.shift(period)) * 100
        return roc.fillna(0.0).to_numpy()

    def _compute_momentum_series(
        self,
        closes: np.ndarray,
        period: int = 10,
    ) -> np.ndarray:
        series = pd.Series(closes)
        momentum = series - series.shift(period)
        return momentum.fillna(0.0).to_numpy()

    def _compute_obv_series(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
        sma_period: int = 20,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        price_diff = np.diff(closes, prepend=closes[0])
        direction = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))
        obv = np.cumsum(direction * volumes)
        obv_series = pd.Series(obv)
        obv_sma = obv_series.rolling(window=sma_period).mean().fillna(method="bfill")
        obv_trend = obv_series.diff().rolling(window=5).mean().fillna(0.0)
        price_trend = pd.Series(closes).diff().rolling(window=5).mean().fillna(0.0)
        return (
            obv_series.to_numpy(),
            obv_sma.to_numpy(),
            obv_trend.to_numpy(),
            price_trend.to_numpy(),
        )

    def _compute_cmf_series(
        self,
        df: pd.DataFrame,
        period: int = 20,
    ) -> np.ndarray:
        multiplier = (
            ((df["close"] - df["low"]) - (df["high"] - df["close"]))
            / (df["high"] - df["low"]).replace(0, 1)
        )
        mf_volume = multiplier * df["volume"]
        cmf = mf_volume.rolling(window=period).sum() / df["volume"].rolling(
            window=period
        ).sum().replace(0, 1)
        return cmf.fillna(0.0).to_numpy()

    # ------------------------------------------------------------------ #
    # Signal helpers
    # ------------------------------------------------------------------ #
    def _rsi_signal_conf(self, value: float) -> tuple[SignalStrength, float]:
        if value > 80:
            signal = SignalStrength.VERY_BEARISH
        elif value > 70:
            signal = SignalStrength.BEARISH
        elif value > 60:
            signal = SignalStrength.BEARISH_BROKEN
        elif value < 20:
            signal = SignalStrength.VERY_BULLISH
        elif value < 30:
            signal = SignalStrength.BULLISH
        elif value < 40:
            signal = SignalStrength.BULLISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL
        confidence = min(0.95, 0.6 + abs(value - 50) / 100)
        return signal, confidence

    def _stochastic_signal_conf(
        self,
        k: float,
        d: float,
    ) -> tuple[SignalStrength, float]:
        if k > 80 and d > 80:
            signal = SignalStrength.VERY_BEARISH
        elif k > 80:
            signal = SignalStrength.BEARISH
        elif k > 70 and k < d:
            signal = SignalStrength.BEARISH_BROKEN
        elif k < 20 and d < 20:
            signal = SignalStrength.VERY_BULLISH
        elif k < 20:
            signal = SignalStrength.BULLISH
        elif k < 30 and k > d:
            signal = SignalStrength.BULLISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL
        confidence = min(0.95, 0.65 + abs(k - 50) / 200)
        return signal, confidence

    def _cci_signal_conf(self, value: float) -> tuple[SignalStrength, float]:
        if value > 200:
            signal = SignalStrength.VERY_BULLISH
        elif value > 100:
            signal = SignalStrength.BULLISH
        elif value > 50:
            signal = SignalStrength.BULLISH_BROKEN
        elif value < -200:
            signal = SignalStrength.VERY_BEARISH
        elif value < -100:
            signal = SignalStrength.BEARISH
        elif value < -50:
            signal = SignalStrength.BEARISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL
        confidence = min(0.9, 0.6 + abs(value) / 500)
        return signal, confidence

    def _williams_signal_conf(self, value: float) -> tuple[SignalStrength, float]:
        if value > -20:
            signal = SignalStrength.VERY_BEARISH
        elif value > -30:
            signal = SignalStrength.BEARISH
        elif value > -40:
            signal = SignalStrength.BEARISH_BROKEN
        elif value < -80:
            signal = SignalStrength.VERY_BULLISH
        elif value < -70:
            signal = SignalStrength.BULLISH
        elif value < -60:
            signal = SignalStrength.BULLISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL
        confidence = min(0.95, 0.6 + abs(value + 50) / 100)
        return signal, confidence

    def _roc_signal_conf(self, value: float) -> tuple[SignalStrength, float]:
        if value > 10:
            signal = SignalStrength.VERY_BULLISH
        elif value > 5:
            signal = SignalStrength.BULLISH
        elif value > 1:
            signal = SignalStrength.BULLISH_BROKEN
        elif value < -10:
            signal = SignalStrength.VERY_BEARISH
        elif value < -5:
            signal = SignalStrength.BEARISH
        elif value < -1:
            signal = SignalStrength.BEARISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL
        confidence = min(0.85, 0.6 + abs(value) / 50)
        return signal, confidence

    def _momentum_signal_conf(
        self,
        momentum_value: float,
        current_price: float,
    ) -> tuple[SignalStrength, float]:
        if momentum_value >= 0:
            signal = SignalStrength.BULLISH
            confidence = min(0.9, 0.5 + (momentum_value / max(current_price, 1e-9)) * 10)
        else:
            signal = SignalStrength.BEARISH
            confidence = min(
                0.9, 0.5 + (abs(momentum_value) / max(current_price, 1e-9)) * 10
            )
        return signal, confidence

    def _obv_signal_conf(
        self,
        obv_value: float,
        obv_sma: float,
        obv_trend: float,
        price_trend: float,
    ) -> tuple[SignalStrength, float]:
        if obv_trend > 0 and price_trend > 0:
            if obv_value > obv_sma * 1.1:
                signal = SignalStrength.VERY_BULLISH
            else:
                signal = SignalStrength.BULLISH
        elif obv_trend > 0 and price_trend < 0:
            signal = SignalStrength.BULLISH_BROKEN
        elif obv_trend < 0 and price_trend < 0:
            if obv_value < obv_sma * 0.9:
                signal = SignalStrength.VERY_BEARISH
            else:
                signal = SignalStrength.BEARISH
        elif obv_trend < 0 and price_trend > 0:
            signal = SignalStrength.BEARISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL
        confidence = 0.75
        return signal, confidence

    def _cmf_signal_conf(self, value: float) -> tuple[SignalStrength, float]:
        if value > 0.25:
            signal = SignalStrength.VERY_BULLISH
        elif value > 0.1:
            signal = SignalStrength.BULLISH
        elif value > 0.05:
            signal = SignalStrength.BULLISH_BROKEN
        elif value < -0.25:
            signal = SignalStrength.VERY_BEARISH
        elif value < -0.1:
            signal = SignalStrength.BEARISH
        elif value < -0.05:
            signal = SignalStrength.BEARISH_BROKEN
        else:
            signal = SignalStrength.NEUTRAL
        confidence = min(0.95, 0.7 + abs(value) * 0.8)
        return signal, confidence

    # ------------------------------------------------------------------ #
    # Divergence helper
    # ------------------------------------------------------------------ #
    def _compute_divergence(
        self,
        candles: Iterable[Candle],
        indicator_values: Iterable[float],
        name: str,
    ) -> float:
        result = self.divergence_detector.detect(
            list(candles),
            list(indicator_values),
            name,
        )
        return result.get_signal_score() / 2.0
