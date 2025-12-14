"""
Multi-Horizon Weight Learning with Multi-Output Regression

یادگیری وزن برای چندین افق زمانی همزمان با استفاده از:
- MultiOutputRegressor
- LightGBM
- Confidence Metrics (R², MAE, Confidence Intervals)
"""

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

logger = logging.getLogger(__name__)


@dataclass
class HorizonWeights:
    """
    وزن‌های یک افق خاص با معیارهای اعتماد
    """
    horizon: str  # "3d", "7d", "30d"
    weights: dict[str, float]  # {'sma': 0.15, 'ema': 0.12, ...}
    metrics: dict[str, float]  # {'r2': 0.23, 'mae': 0.042, ...}
    confidence: float  # [0, 1] - اعتماد کلی

    def to_dict(self) -> dict:
        return {
            'horizon': self.horizon,
            'weights': self.weights,
            'metrics': self.metrics,
            'confidence': self.confidence
        }


class MultiHorizonWeightLearner:
    """
    یادگیرنده وزن برای چندین افق زمانی
    """

    @staticmethod
    def _normalize_horizon_name(horizon: str) -> str:
        """
        Normalize horizon labels coming from different target column names.

        Accepts variations like 'return_3d', 'target_3d', 'pred_3d' and returns
        the canonical horizon string (e.g., '3d').
        """
        if horizon is None:
            return ''
        cleaned = horizon
        for prefix in ("return_", "target_", "pred_", "horizon_"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
        return cleaned

    def __init__(
        self,
        horizons: list[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        lgbm_params: dict = None
    ):
        """
        Initialize multi-horizon weight learner

        Args:
            horizons: لیست افق‌ها (مثلاً ['3d', '7d', '30d'])
            test_size: نسبت داده آزمایش
            random_state: seed تصادفی
            lgbm_params: پارامترهای LightGBM
        """
        self.horizons = horizons or ['3d', '7d', '30d']
        self.test_size = test_size
        self.random_state = random_state

        # پارامترهای پیش‌فرض LightGBM
        self.lgbm_params = lgbm_params or {
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 5,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state,
            'verbose': -1
        }

        # مدل Multi-Output
        self.model = None

        # وزن‌های آموخته شده برای هر افق
        self.horizon_weights: dict[str, HorizonWeights] = {}

        # feature names
        self.feature_names = None

    def train(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        verbose: bool = True
    ):
        """
        آموزش مدل Multi-Output

        Args:
            X: DataFrame ویژگی‌ها
            Y: DataFrame اهداف (ستون‌های return_3d, return_7d, return_30d)
        """
        if verbose:
            print("\n" + "="*60)
            print("🚀 Multi-Horizon Weight Learning")
            print("="*60)
            print(f"Samples: {len(X)}")
            print(f"Features: {X.shape[1]}")
            print(f"Horizons: {Y.shape[1]}")

        self.feature_names = list(X.columns)

        # تقسیم Train/Test
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        if verbose:
            print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

        # ایجاد و آموزش مدل
        base_estimator = LGBMRegressor(**self.lgbm_params)
        self.model = MultiOutputRegressor(base_estimator, n_jobs=-1)

        if verbose:
            print("\n⏳ Training multi-output model...")

        self.model.fit(X_train, Y_train)

        if verbose:
            print("✅ Training completed!")

        # ارزیابی و استخراج وزن‌ها
        self._evaluate_and_extract_weights(
            X_train, Y_train,
            X_test, Y_test,
            verbose=verbose
        )

    def _evaluate_and_extract_weights(
        self,
        X_train: pd.DataFrame,
        Y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        Y_test: pd.DataFrame,
        verbose: bool = True
    ):
        """
        ارزیابی مدل و استخراج وزن‌ها برای هر افق
        """
        if verbose:
            print("\n" + "="*60)
            print("📊 Evaluation & Weight Extraction")
            print("="*60)

        # پیش‌بینی
        Y_pred_train = self.model.predict(X_train)
        Y_pred_test = self.model.predict(X_test)

        # حلقه روی افق‌ها
        for i, horizon_col in enumerate(Y_train.columns):
            horizon_name = self._normalize_horizon_name(horizon_col)

            if verbose:
                print(f"\n🎯 Horizon: {horizon_name}")
                print("-" * 40)

            # R² و MAE
            r2_train = r2_score(Y_train.iloc[:, i], Y_pred_train[:, i])
            r2_test = r2_score(Y_test.iloc[:, i], Y_pred_test[:, i])
            mae_train = mean_absolute_error(Y_train.iloc[:, i], Y_pred_train[:, i])
            mae_test = mean_absolute_error(Y_test.iloc[:, i], Y_pred_test[:, i])

            if verbose:
                print(f"  R² (train): {r2_train:.4f}")
                print(f"  R² (test):  {r2_test:.4f}")
                print(f"  MAE (train): {mae_train:.4f} ({mae_train*100:.2f}%)")
                print(f"  MAE (test):  {mae_test:.4f} ({mae_test*100:.2f}%)")

            # محاسبه اعتماد
            confidence = self._calculate_confidence(r2_test, mae_test)

            if verbose:
                print(f"  Confidence: {confidence:.2f}")

            # استخراج وزن‌ها از Feature Importance
            feature_importances = self.model.estimators_[i].feature_importances_
            weights = self._normalize_weights(feature_importances)

            # ذخیره
            normalized_horizon = self._normalize_horizon_name(horizon_name)
            self.horizon_weights[normalized_horizon] = HorizonWeights(
                horizon=normalized_horizon,
                weights=dict(zip(self.feature_names, weights, strict=True)),
                metrics={
                    'r2_train': r2_train,
                    'r2_test': r2_test,
                    'mae_train': mae_train,
                    'mae_test': mae_test
                },
                confidence=confidence
            )

            # نمایش Top 5 Features
            if verbose:
                top_features = sorted(
                    zip(self.feature_names, weights, strict=True),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:5]

                print("\n  Top 5 Features:")
                for feat, w in top_features:
                    print(f"    {feat:30s} {w:+.4f}")

    def _calculate_confidence(
        self,
        r2: float,
        mae: float
    ) -> float:
        """
        محاسبه اعتماد بر اساس R² و MAE

        Confidence Formula:
        - R² > 0.5 → full confidence
        - R² < 0 → zero confidence
        - MAE penalty: lower is better
        """
        # R² component [0, 1]
        if r2 > 0.5:
            r2_component = 1.0
        elif r2 > 0:
            r2_component = r2 * 2.0  # scale [0, 0.5] → [0, 1]
        else:
            r2_component = 0.0

        # MAE penalty
        # MAE در بازار رمزارز معمولاً بین 0.02 تا 0.10 است
        mae_component = max(0, 1.0 - mae * 10)  # 0.10 → 0, 0.02 → 0.8

        # ترکیب (70% R², 30% MAE)
        confidence = 0.7 * r2_component + 0.3 * mae_component

        return np.clip(confidence, 0, 1)

    def _normalize_weights(
        self,
        feature_importances: np.ndarray
    ) -> np.ndarray:
        """
        نرمال‌سازی وزن‌ها به [-1, 1] با حفظ علامت
        """
        # نرمال‌سازی به [0, 1]
        total = feature_importances.sum()
        if total > 0:
            normalized = feature_importances / total
        else:
            normalized = np.zeros_like(feature_importances)

        return normalized

    def get_horizon_weights(
        self,
        horizon: str
    ) -> HorizonWeights | None:
        """
        دریافت وزن‌های یک افق خاص

        Args:
            horizon: '3d', '7d', '30d'
        """
        return self.horizon_weights.get(horizon)

    def predict_multi_horizon(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        پیش‌بینی برای همه افق‌ها

        Returns:
            DataFrame با ستون‌های [pred_3d, pred_7d, pred_30d]
        """
        if self.model is not None:
            # Ensure columns are in the correct order
            if self.feature_names:
                X = X[self.feature_names]
            predictions = self.model.predict(X)
            return pd.DataFrame(
                predictions,
                columns=[f'pred_{h}' for h in self.horizons]
            )

        logger.warning(
            "multi_horizon.predict_without_model",
            extra={"horizons": self.horizons}
        )
        return self._predict_with_weights(X)

    def _predict_with_weights(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Lightweight fallback that approximates predictions using stored feature weights.
        """
        def _clean_value(name: str, val: float | int | None) -> float:
            """
            Clamp/normalize feature values so dot-products do not explode to 1.0 after clipping.
            - *_signal, *_weighted, *_normalized are expected to be in [-1, 1] → clip there.
            - *_percentile is roughly [0, 100]; center and scale to [-1, 1].
            - everything else is squashed with tanh to avoid huge magnitudes.
            """
            if val is None:
                return 0.0
            try:
                v = float(val)
            except (TypeError, ValueError):
                return 0.0

            if np.isnan(v):
                return 0.0

            if name.endswith("_percentile"):
                v = (v - 50.0) / 50.0  # 50 -> 0, 0/100 -> -1/+1
            elif name.endswith("_normalized") or name.endswith("_signal") or name.endswith("_weighted"):
                pass  # already in a reasonable range, just clip below
            else:
                # squash arbitrary values to [-1, 1]
                v = np.tanh(v)

            return float(np.clip(v, -1.0, 1.0))

        rows: list[dict[str, float]] = []
        for _, row in X.iterrows():
            horizon_preds: dict[str, float] = {}
            for horizon in self.horizons:
                horizon_weights = self.horizon_weights.get(horizon)
                if not horizon_weights:
                    horizon_preds[horizon] = 0.0
                    continue

                numerator = 0.0
                weight_norm = 0.0
                for feature_name, weight in horizon_weights.weights.items():
                    value = _clean_value(feature_name, row.get(feature_name, 0.0))
                    numerator += value * weight
                    weight_norm += abs(weight)

                score = numerator / weight_norm if weight_norm else numerator
                horizon_preds[horizon] = float(np.clip(score, -1.0, 1.0))

            rows.append({f'pred_{h}': horizon_preds[h] for h in self.horizons})

        return pd.DataFrame(rows)

    def save_weights(
        self,
        filepath: str
    ):
        """
        ذخیره وزن‌ها در فایل JSON
        """
        weights_dict = {
            'horizons': self.horizons,
            'feature_names': self.feature_names,
            'weights': {
                h: w.to_dict()
                for h, w in self.horizon_weights.items()
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(weights_dict, f, indent=2, ensure_ascii=False)

        print(f"✅ Weights saved to: {filepath}")

    def load_weights(
        self,
        filepath: str
    ):
        """
        بارگذاری وزن‌ها از فایل JSON
        """
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)

        self.horizons = [self._normalize_horizon_name(h) for h in data['horizons']]
        self.feature_names = data['feature_names']

        self.horizon_weights = {}
        for horizon, w_dict in data['weights'].items():
            normalized_horizon = self._normalize_horizon_name(horizon)
            self.horizon_weights[normalized_horizon] = HorizonWeights(
                horizon=self._normalize_horizon_name(w_dict.get('horizon', normalized_horizon)),
                weights=w_dict['weights'],
                metrics=w_dict['metrics'],
                confidence=w_dict['confidence']
            )

        print(f"[OK] Weights loaded from: {filepath}")
        print(f"     Horizons: {self.horizons}")

    def save_model_state(
        self,
        filepath: str
    ):
        """
        Persist the trained regression model so it can be reused later.
        """
        if self.model is None:
            raise ValueError("No trained model available to save.")

        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

        logger.info("multi_horizon.model_saved", extra={"path": filepath})

    def load_model_state(
        self,
        filepath: str
    ):
        """
        Load a previously persisted regression model.
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)

        logger.info("multi_horizon.model_loaded", extra={"path": filepath})

    def get_summary(self) -> dict:
        """
        خلاصه‌ای از وزن‌ها و معیارهای اعتماد
        """
        summary = {
            'horizons': self.horizons,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'horizon_details': {}
        }

        for horizon, weights in self.horizon_weights.items():
            summary['horizon_details'][horizon] = {
                'r2_test': weights.metrics['r2_test'],
                'mae_test': weights.metrics['mae_test'],
                'confidence': weights.confidence,
                'top_3_features': sorted(
                    weights.weights.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:3]
            }

        return summary

    @classmethod
    def load(
        cls,
        weights_path: str,
        model_path: str | None = None,
        **kwargs,
    ) -> "MultiHorizonWeightLearner":
        """
        Convenience helper to restore a learner from disk.

        Args:
            weights_path: JSON file produced by `save_weights`.
            model_path: Optional pickle file produced by `save_model_state`.
            kwargs: Optional overrides for constructor arguments (horizons, etc.).
        """
        learner = cls(**kwargs)
        learner.load_weights(weights_path)
        candidate_model = Path(model_path) if model_path else Path(weights_path).with_suffix(".pkl")
        if candidate_model.exists():
            learner.load_model_state(str(candidate_model))
        else:
            logger.info(
                "multi_horizon.model_state_missing",
                extra={"weights_path": weights_path, "model_path": str(candidate_model)},
            )
        return learner


# ═══════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ساخت داده مصنوعی برای تست
    np.random.seed(42)

    n_samples = 1000
    n_features = 21  # 7 indicators × 3 features

    # ویژگی‌ها
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # اهداف (شبیه‌سازی بازدهی)
    Y = pd.DataFrame({
        'return_3d': np.random.randn(n_samples) * 0.05,
        'return_7d': np.random.randn(n_samples) * 0.08,
        'return_30d': np.random.randn(n_samples) * 0.15
    })

    # ایجاد learner
    learner = MultiHorizonWeightLearner(
        horizons=['3d', '7d', '30d']
    )

    # آموزش
    learner.train(X, Y, verbose=True)

    # دریافت وزن‌ها
    weights_3d = learner.get_horizon_weights('3d')
    print("\n3-Day Weights:")
    print(f"  Confidence: {weights_3d.confidence:.2f}")
    print(f"  R² Test: {weights_3d.metrics['r2_test']:.4f}")

    # خلاصه
    print("\n" + "="*60)
    print("📊 Summary")
    print("="*60)
    summary = learner.get_summary()

    for horizon, details in summary['horizon_details'].items():
        print(f"\n{horizon}:")
        print(f"  R²: {details['r2_test']:.4f}")
        print(f"  MAE: {details['mae_test']:.4f}")
        print(f"  Confidence: {details['confidence']:.2f}")
