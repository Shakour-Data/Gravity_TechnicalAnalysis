"""
Training Pipeline for Multi-Horizon Support/Resistance Weights

این ماژول وزن‌های ML را برای تحلیل Support & Resistance آموزش می‌دهد.

سناریوهای تولید داده:
1. Bounce Scenario: قیمت از سطح برمی‌گردد
2. Breakout Scenario: قیمت سطح را می‌شکند
3. Fake-out Scenario: شکست موقت و برگشت
4. Consolidation: نوسان در محدوده
"""

import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from gravity_tech.ml.multi_horizon_support_resistance_analysis import (
    MultiHorizonSupportResistanceAnalyzer,
)
from gravity_tech.ml.multi_horizon_support_resistance_features import (
    MultiHorizonSupportResistanceFeatureExtractor,
)
from gravity_tech.models.schemas import Candle


def create_bounce_scenario(
    base_price: float = 50000, num_candles: int = 100
) -> tuple[list[Candle], float]:
    """
    ایجاد سناریو Bounce (برگشت از حمایت/مقاومت)

    Returns:
        (candles, target_score)
        target_score: +1 for bounce from support, -1 for bounce from resistance
    """
    candles = []
    start_time = datetime.now() - timedelta(hours=num_candles)

    # انتخاب bounce از support یا resistance
    bounce_from_support = np.random.random() > 0.5

    if bounce_from_support:
        # قیمت به سمت پایین می‌رود (به سمت support)
        trend_direction = -1
        target_score = 1.0  # موقعیت خوب برای خرید
    else:
        # قیمت به سمت بالا می‌رود (به سمت resistance)
        trend_direction = 1
        target_score = -1.0  # موقعیت خوب برای فروش

    # مرحله 1: حرکت به سمت سطح (70% از کندل‌ها)
    approach_candles = int(num_candles * 0.7)
    price = base_price

    for i in range(approach_candles):
        # حرکت تدریجی به سمت سطح
        trend = trend_direction * base_price * 0.002 * (1 + i / approach_candles)
        volatility = np.random.normal(0, base_price * 0.005)

        price += trend + volatility

        high = price * (1 + abs(np.random.normal(0, 0.003)))
        low = price * (1 - abs(np.random.normal(0, 0.003)))
        close = np.random.uniform(low, high)

        candle = Candle(
            timestamp=start_time + timedelta(hours=i),
            open=price,
            high=high,
            low=low,
            close=close,
            volume=np.random.uniform(1000, 2000),
        )
        candles.append(candle)
        price = close

    # مرحله 2: bounce از سطح (30% باقیمانده)
    bounce_candles = num_candles - approach_candles
    bounce_strength = 0.015  # 1.5% bounce

    for i in range(bounce_candles):
        # برگشت قوی از سطح
        trend = -trend_direction * base_price * bounce_strength * (1 - i / bounce_candles)
        volatility = np.random.normal(0, base_price * 0.003)

        price += trend + volatility

        high = price * (1 + abs(np.random.normal(0, 0.003)))
        low = price * (1 - abs(np.random.normal(0, 0.003)))
        close = np.random.uniform(low, high)

        # حجم بالا در bounce
        volume = np.random.uniform(1500, 3000)

        candle = Candle(
            timestamp=start_time + timedelta(hours=approach_candles + i),
            open=price,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
        candles.append(candle)
        price = close

    return candles, target_score


def create_breakout_scenario(
    base_price: float = 50000, num_candles: int = 100
) -> tuple[list[Candle], float]:
    """
    ایجاد سناریو Breakout (شکست سطح)

    Returns:
        (candles, target_score)
        target_score: 0.0 (سطح شکسته شده، دیگر معتبر نیست)
    """
    candles = []
    start_time = datetime.now() - timedelta(hours=num_candles)

    # انتخاب breakout به بالا یا پایین
    breakout_up = np.random.random() > 0.5

    if breakout_up:
        trend_direction = 1
        target_score = 0.3  # بعد از شکست مقاومت، ادامه صعود
    else:
        trend_direction = -1
        target_score = -0.3  # بعد از شکست حمایت، ادامه نزول

    # مرحله 1: تست چندباره سطح (60% از کندل‌ها)
    test_candles = int(num_candles * 0.6)
    price = base_price

    for i in range(test_candles):
        # نوسان در نزدیکی سطح (consolidation)
        if i % 10 < 7:  # 70% وقت نزدیک سطح
            movement = np.random.normal(0, base_price * 0.003)
        else:  # 30% وقت تست سطح
            movement = trend_direction * base_price * 0.005

        price += movement

        high = price * (1 + abs(np.random.normal(0, 0.003)))
        low = price * (1 - abs(np.random.normal(0, 0.003)))
        close = np.random.uniform(low, high)

        candle = Candle(
            timestamp=start_time + timedelta(hours=i),
            open=price,
            high=high,
            low=low,
            close=close,
            volume=np.random.uniform(1000, 1500),
        )
        candles.append(candle)
        price = close

    # مرحله 2: شکست سطح (40% باقیمانده)
    breakout_candles = num_candles - test_candles

    for i in range(breakout_candles):
        # حرکت قوی در جهت شکست
        trend = trend_direction * base_price * 0.02 * (1 + i / breakout_candles)
        volatility = np.random.normal(0, base_price * 0.004)

        price += trend + volatility

        high = price * (1 + abs(np.random.normal(0, 0.004)))
        low = price * (1 - abs(np.random.normal(0, 0.004)))
        close = np.random.uniform(low, high)

        # حجم بالا در breakout
        volume = np.random.uniform(2000, 4000)

        candle = Candle(
            timestamp=start_time + timedelta(hours=test_candles + i),
            open=price,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
        candles.append(candle)
        price = close

    return candles, target_score


def create_consolidation_scenario(
    base_price: float = 50000, num_candles: int = 100
) -> tuple[list[Candle], float]:
    """
    ایجاد سناریو Consolidation (نوسان در محدوده)

    Returns:
        (candles, target_score)
        target_score: 0.0 (خنثی)
    """
    candles = []
    start_time = datetime.now() - timedelta(hours=num_candles)

    # تعریف محدوده
    range_size = base_price * 0.04  # 4% range
    support = base_price - range_size / 2
    resistance = base_price + range_size / 2

    price = base_price

    for i in range(num_candles):
        # نوسان در محدوده support-resistance
        # تمایل به برگشت از حد range
        if price < support + range_size * 0.2:
            bias = 1  # فشار به سمت بالا
        elif price > resistance - range_size * 0.2:
            bias = -1  # فشار به سمت پایین
        else:
            bias = 0  # خنثی

        movement = bias * base_price * 0.003 + np.random.normal(0, base_price * 0.005)
        price += movement

        # محدود کردن به range
        price = np.clip(price, support * 0.99, resistance * 1.01)

        high = min(price * (1 + abs(np.random.normal(0, 0.003))), resistance * 1.005)
        low = max(price * (1 - abs(np.random.normal(0, 0.003))), support * 0.995)
        close = np.random.uniform(low, high)

        candle = Candle(
            timestamp=start_time + timedelta(hours=i),
            open=price,
            high=high,
            low=low,
            close=close,
            volume=np.random.uniform(800, 1200),
        )
        candles.append(candle)
        price = close

    target_score = 0.0  # خنثی
    return candles, target_score


def create_fake_out_scenario(
    base_price: float = 50000, num_candles: int = 100
) -> tuple[list[Candle], float]:
    """
    ایجاد سناریو Fake-out (شکست موقت و برگشت)

    Returns:
        (candles, target_score)
        target_score: معکوس جهت fake-out
    """
    candles = []
    start_time = datetime.now() - timedelta(hours=num_candles)

    # انتخاب fake-out به بالا یا پایین
    fake_up = np.random.random() > 0.5

    if fake_up:
        initial_direction = 1
        final_direction = -1
        target_score = -0.7  # fake breakout بالا → فروش
    else:
        initial_direction = -1
        final_direction = 1
        target_score = 0.7  # fake breakdown پایین → خرید

    # مرحله 1: نزدیک شدن به سطح (40%)
    approach_candles = int(num_candles * 0.4)
    price = base_price

    for i in range(approach_candles):
        trend = initial_direction * base_price * 0.001
        volatility = np.random.normal(0, base_price * 0.003)
        price += trend + volatility

        high = price * (1 + abs(np.random.normal(0, 0.003)))
        low = price * (1 - abs(np.random.normal(0, 0.003)))
        close = np.random.uniform(low, high)

        candle = Candle(
            timestamp=start_time + timedelta(hours=i),
            open=price,
            high=high,
            low=low,
            close=close,
            volume=np.random.uniform(1000, 1500),
        )
        candles.append(candle)
        price = close

    # مرحله 2: fake breakout (20%)
    fake_candles = int(num_candles * 0.2)

    for i in range(fake_candles):
        trend = initial_direction * base_price * 0.015
        volatility = np.random.normal(0, base_price * 0.002)
        price += trend + volatility

        high = price * (1 + abs(np.random.normal(0, 0.004)))
        low = price * (1 - abs(np.random.normal(0, 0.004)))
        close = np.random.uniform(low, high)

        # حجم متوسط (نه خیلی بالا - نشانه fake)
        volume = np.random.uniform(1200, 1800)

        candle = Candle(
            timestamp=start_time + timedelta(hours=approach_candles + i),
            open=price,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
        candles.append(candle)
        price = close

    # مرحله 3: برگشت قوی (40%)
    reversal_candles = num_candles - approach_candles - fake_candles

    for i in range(reversal_candles):
        trend = final_direction * base_price * 0.02 * (1 + i / reversal_candles)
        volatility = np.random.normal(0, base_price * 0.004)
        price += trend + volatility

        high = price * (1 + abs(np.random.normal(0, 0.005)))
        low = price * (1 - abs(np.random.normal(0, 0.005)))
        close = np.random.uniform(low, high)

        # حجم بالا در reversal
        volume = np.random.uniform(2000, 3500)

        candle = Candle(
            timestamp=start_time + timedelta(hours=approach_candles + fake_candles + i),
            open=price,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
        candles.append(candle)
        price = close

    return candles, target_score


class MultiHorizonSupportResistanceTrainer:
    """Trainer برای آموزش وزن‌های Support & Resistance"""

    def __init__(self):
        """Initialize trainer"""
        self.feature_extractor = MultiHorizonSupportResistanceFeatureExtractor()

    def prepare_training_data(self, num_samples: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        تولید داده آموزش

        Args:
            num_samples: تعداد نمونه (500 از هر سناریو)

        Returns:
            (features_df, targets_df)
        """
        print(f"🔄 در حال تولید {num_samples} نمونه آموزشی...")

        all_features = []
        all_targets = []

        samples_per_scenario = num_samples // 4

        scenarios = [
            ("bounce", create_bounce_scenario),
            ("breakout", create_breakout_scenario),
            ("consolidation", create_consolidation_scenario),
            ("fake_out", create_fake_out_scenario),
        ]

        for scenario_name, scenario_func in scenarios:
            print(f"  📊 تولید {samples_per_scenario} نمونه {scenario_name}...")

            for i in range(samples_per_scenario):
                # تولید داده
                base_price = np.random.uniform(30000, 70000)
                num_candles = np.random.randint(80, 120)

                candles, target_score = scenario_func(base_price, num_candles)

                # استخراج ویژگی‌ها برای همه افق‌ها
                try:
                    features = self.feature_extractor.extract_all_horizons(candles)

                    # Target برای هر افق
                    targets = {
                        "3d_target": target_score,
                        "7d_target": target_score * 0.9,  # کمی ملایم‌تر
                        "30d_target": target_score * 0.7,  # خیلی ملایم‌تر
                        "scenario": scenario_name,
                    }

                    all_features.append(features)
                    all_targets.append(targets)

                except Exception as e:
                    print(f"    ⚠️  خطا در نمونه {i}: {e}")
                    continue

        features_df = pd.DataFrame(all_features)
        targets_df = pd.DataFrame(all_targets)

        print(f"✅ تولید داده کامل شد: {len(features_df)} نمونه")
        print(f"   ویژگی‌ها: {len(features_df.columns)} ستون")

        return features_df, targets_df

    def train(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        output_path: str = "models/support_resistance/sr_weights.json",
    ):
        """
        آموزش وزن‌ها

        Args:
            features_df: DataFrame ویژگی‌ها
            targets_df: DataFrame target ها
            output_path: مسیر ذخیره وزن‌ها
        """
        print("\n🎯 شروع آموزش وزن‌ها...")

        weights = {}
        model_state: dict[str, dict[str, object]] = {}

        for horizon in ["3d", "7d", "30d"]:
            print(f"\n  📊 آموزش {horizon}...")

            # انتخاب ویژگی‌های مربوط به این افق
            horizon_features = [col for col in features_df.columns if col.startswith(f"{horizon}_")]
            X = features_df[horizon_features].values
            y = targets_df[f"{horizon}_target"].values

            # محاسبه وزن‌ها با Linear Regression ساده
            # X.T @ X @ weights = X.T @ y
            # weights = (X.T @ X)^-1 @ X.T @ y

            try:
                XTX = X.T @ X
                XTy = X.T @ y

                # افزودن regularization کوچک
                reg = 0.01
                XTX_reg = XTX + reg * np.eye(XTX.shape[0])

                w = np.linalg.solve(XTX_reg, XTy)

                # ایجاد dictionary وزن‌ها
                horizon_weights = {}
                clean_names: list[str] = []
                for i, feature_name in enumerate(horizon_features):
                    # حذف prefix افق
                    clean_name = feature_name.replace(f"{horizon}_", "")
                    horizon_weights[clean_name] = float(w[i])
                    clean_names.append(clean_name)

                weights[horizon] = horizon_weights
                model_state[horizon] = {
                    "feature_names": clean_names,
                    "weights": [float(value) for value in w.tolist()],
                    "intercept": 0.0,
                }

                # ارزیابی
                predictions = X @ w
                mae = np.mean(np.abs(predictions - y))
                rmse = np.sqrt(np.mean((predictions - y) ** 2))

                # دقت جهت
                direction_correct = np.sum((predictions > 0) == (y > 0)) / len(y)

                print(f"    MAE: {mae:.4f}")
                print(f"    RMSE: {rmse:.4f}")
                print(f"    Direction Accuracy: {direction_correct:.2%}")

            except Exception as e:
                print(f"    ⚠️  خطا در آموزش {horizon}: {e}")
                # استفاده از وزن‌های پیش‌فرض
                default_weights = self._get_default_weights(horizon)
                weights[horizon] = default_weights
                model_state[horizon] = {
                    "feature_names": list(default_weights.keys()),
                    "weights": list(default_weights.values()),
                    "intercept": 0.0,
                }

        # ذخیره وزن‌ها
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(weights, f, indent=2)

        model_file = output_file.with_suffix(".pkl")
        with open(model_file, "wb") as fh:
            pickle.dump(model_state, fh)

        print(f"\n✅ وزن‌ها ذخیره شد: {output_path}")
        print(f"✅ مدل ذخیره شد: {model_file}")

        return weights

    def _get_default_weights(self, horizon: str) -> dict[str, float]:
        """وزن‌های پیش‌فرض"""
        return {
            "nearest_resistance_dist": -0.3,
            "resistance_strength": -0.2,
            "nearest_support_dist": 0.3,
            "support_strength": 0.2,
            "sr_position": -0.35,
            "sr_bias": 0.25,
            "level_density": 0.15,
            "fib_signal": 0.2,
            "camarilla_signal": 0.15,
        }


# ═══════════════════════════════════════════════════════════════
# اجرا
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Multi-Horizon Support/Resistance Training Pipeline")
    print("=" * 70)

    # ایجاد trainer
    trainer = MultiHorizonSupportResistanceTrainer()

    # تولید داده آموزش
    features_df, targets_df = trainer.prepare_training_data(num_samples=2000)

    # آموزش
    weights = trainer.train(features_df, targets_df)

    # تست با داده validation
    print("\n" + "=" * 70)
    print("🧪 تست با داده Validation")
    print("=" * 70)

    features_val, targets_val = trainer.prepare_training_data(num_samples=600)

    # بارگذاری analyzer با وزن‌های جدید
    analyzer = MultiHorizonSupportResistanceAnalyzer(
        weights_path="models/support_resistance/sr_weights.json",
        model_path="models/support_resistance/sr_weights.pkl",
    )

    # تست روی چند نمونه
    print("\nتست روی نمونه‌های validation:")

    for scenario in ["bounce", "breakout", "consolidation", "fake_out"]:
        scenario_indices = targets_val[targets_val["scenario"] == scenario].index[:3]

        print(f"\n📊 سناریو: {scenario.upper()}")

        for idx in scenario_indices:
            # ایجاد مجدد candles (برای سادگی از یک نمونه استفاده می‌کنیم)
            if scenario == "bounce":
                candles, _ = create_bounce_scenario()
            elif scenario == "breakout":
                candles, _ = create_breakout_scenario()
            elif scenario == "consolidation":
                candles, _ = create_consolidation_scenario()
            else:
                candles, _ = create_fake_out_scenario()

            try:
                analysis = analyzer.analyze(candles)
                print(f"\n  Target: {targets_val.loc[idx, '3d_target']:+.2f}")
                print(f"  Predicted 3d: {analysis.score_3d.score:+.2f}")
                print(f"  Signal: {analysis.score_3d.signal}")
                print(f"  Bounce Prob: {analysis.score_3d.bounce_probability:.1%}")
                print(f"  Breakout Prob: {analysis.score_3d.breakout_probability:.1%}")
            except Exception as e:
                print(f"  ⚠️  خطا: {e}")

            break  # فقط یک نمونه از هر سناریو

    print("\n" + "=" * 70)
    print("✅ آموزش و تست کامل شد!")
    print("=" * 70)
