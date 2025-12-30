"""
Training Pipeline for Volume-Dimension Matrix
==============================================

این ماژول وزن‌های بهینه برای volume interactions را یاد می‌گیرد.

هدف:
یادگیری وزن‌های بهینه برای هر interaction که بهترین adjustment را
برای dimension scores ارائه دهد.

سناریوهای آموزشی:
1. Strong Trend + Confirming Volume → تقویت
2. Strong Trend + Divergent Volume → تضعیف
3. Overbought + High Volume → هشدار
4. BB Squeeze + Volume Spike → شکست قریب‌الوقوع
5. Breakout + High Volume → معتبر
6. Breakout + Low Volume → fake
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gravity_tech.models.schemas import Candle


@dataclass
class TrainingScenario:
    """یک سناریوی آموزشی"""

    name: str
    description: str
    candles: list[Candle]

    # امتیازهای واقعی (ground truth)
    expected_trend_adjustment: float
    expected_momentum_adjustment: float
    expected_volatility_adjustment: float
    expected_cycle_adjustment: float
    expected_sr_adjustment: float

    # توضیحات
    explanation: str


class VolumeMatrixTrainer:
    """
    آموزش‌دهنده Volume-Dimension Matrix

    این کلاس سناریوهای مختلف بازار را شبیه‌سازی کرده و
    وزن‌های بهینه را برای interactions یاد می‌گیرد.
    """

    def __init__(self):
        self.scenarios: list[TrainingScenario] = []
        self.weights = {
            "trend": {},
            "momentum": {},
            "volatility": {},
            "cycle": {},
            "support_resistance": {},
        }

    # ═══════════════════════════════════════════════════════════════════
    # Scenario Generators
    # ═══════════════════════════════════════════════════════════════════

    def create_scenario_1_strong_trend_confirming_volume(self) -> TrainingScenario:
        """
        سناریو 1: روند صعودی قوی + حجم تاییدکننده

        توقع: volume باید trend را تقویت کند (+0.15 تا +0.20)
        """
        candles = []
        base_price = 50000

        for i in range(50):
            # روند صعودی پایدار
            open_price = base_price
            close_price = base_price + np.random.uniform(100, 300)  # کندل‌های صعودی
            high_price = close_price + np.random.uniform(0, 100)
            low_price = open_price - np.random.uniform(0, 50)

            # حجم بالا در کندل‌های صعودی
            volume = np.random.uniform(1800, 2500)  # بالاتر از میانگین

            candles.append(
                Candle(
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timestamp=1700000000 + i * 3600,
                )
            )

            base_price = close_price

        return TrainingScenario(
            name="Strong Uptrend + Confirming Volume",
            description="روند صعودی قوی با حجم بالا در کندل‌های سبز",
            candles=candles,
            expected_trend_adjustment=+0.18,
            expected_momentum_adjustment=+0.10,
            expected_volatility_adjustment=+0.08,
            expected_cycle_adjustment=+0.15,
            expected_sr_adjustment=+0.12,
            explanation="حجم روند صعودی را تایید و تقویت می‌کند",
        )

    def create_scenario_2_strong_trend_divergent_volume(self) -> TrainingScenario:
        """
        سناریو 2: روند صعودی + واگرایی حجم (حجم کاهشی)

        توقع: volume باید هشدار دهد و trend را تضعیف کند (-0.10 تا -0.15)
        """
        candles = []
        base_price = 50000

        for i in range(50):
            # روند صعودی اما با حجم کاهشی
            open_price = base_price
            close_price = base_price + np.random.uniform(50, 150)
            high_price = close_price + np.random.uniform(0, 80)
            low_price = open_price - np.random.uniform(0, 40)

            # حجم کاهشی (واگرایی نزولی)
            volume = 2000 - (i * 20)  # کاهش تدریجی
            volume = max(volume, 800)

            candles.append(
                Candle(
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timestamp=1700000000 + i * 3600,
                )
            )

            base_price = close_price

        return TrainingScenario(
            name="Uptrend + Volume Divergence",
            description="روند صعودی اما حجم کاهشی (هشدار!)",
            candles=candles,
            expected_trend_adjustment=-0.12,
            expected_momentum_adjustment=-0.15,
            expected_volatility_adjustment=-0.05,
            expected_cycle_adjustment=-0.10,
            expected_sr_adjustment=-0.08,
            explanation="واگرایی حجم نشانه ضعف روند است",
        )

    def create_scenario_3_overbought_high_volume(self) -> TrainingScenario:
        """
        سناریو 3: اشباع خرید + حجم بالا

        توقع: momentum adjustment منفی (هشدار exhaustion)
        """
        candles = []
        base_price = 45000

        # مرحله 1: صعود سریع با حجم بالا (30 کندل)
        for i in range(30):
            open_price = base_price
            close_price = base_price + np.random.uniform(200, 400)
            high_price = close_price + np.random.uniform(0, 100)
            low_price = open_price - np.random.uniform(0, 50)
            volume = np.random.uniform(2200, 3000)  # حجم خیلی بالا

            candles.append(
                Candle(
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timestamp=1700000000 + i * 3600,
                )
            )

            base_price = close_price

        # مرحله 2: ادامه صعود با حجم بیشتر (اشباع) - 20 کندل
        for i in range(20):
            open_price = base_price
            close_price = base_price + np.random.uniform(100, 200)  # صعود کمتر
            high_price = close_price + np.random.uniform(0, 150)
            low_price = open_price - np.random.uniform(0, 100)
            volume = np.random.uniform(2500, 3500)  # حجم بیشتر!

            candles.append(
                Candle(
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timestamp=1700000000 + (i + 30) * 3600,
                )
            )

            base_price = close_price

        return TrainingScenario(
            name="Overbought + High Volume",
            description="اشباع خرید با حجم بالا - exhaustion",
            candles=candles,
            expected_trend_adjustment=-0.08,
            expected_momentum_adjustment=-0.18,  # هشدار قوی
            expected_volatility_adjustment=+0.10,  # نوسان افزایش
            expected_cycle_adjustment=-0.12,
            expected_sr_adjustment=-0.10,
            explanation="حجم بالا در اشباع خرید نشانه exhaustion",
        )

    def create_scenario_4_bb_squeeze_volume_spike(self) -> TrainingScenario:
        """
        سناریو 4: BB Squeeze + Volume Spike

        توقع: volatility adjustment مثبت قوی (شکست قریب‌الوقوع)
        """
        candles = []
        base_price = 50000

        # مرحله 1: Consolidation با حجم پایین (30 کندل)
        for i in range(30):
            open_price = base_price + np.random.uniform(-100, 100)
            close_price = base_price + np.random.uniform(-100, 100)
            high_price = max(open_price, close_price) + np.random.uniform(0, 50)
            low_price = min(open_price, close_price) - np.random.uniform(0, 50)
            volume = np.random.uniform(800, 1200)  # حجم پایین

            candles.append(
                Candle(
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timestamp=1700000000 + i * 3600,
                )
            )

        # مرحله 2: Volume Spike (شکست) - 20 کندل
        for i in range(20):
            open_price = base_price
            close_price = base_price + np.random.uniform(200, 400)  # شکست به بالا
            high_price = close_price + np.random.uniform(0, 100)
            low_price = open_price - np.random.uniform(0, 50)
            volume = np.random.uniform(2500, 3500)  # volume spike!

            candles.append(
                Candle(
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timestamp=1700000000 + (i + 30) * 3600,
                )
            )

            base_price = close_price

        return TrainingScenario(
            name="BB Squeeze + Volume Spike",
            description="فشردگی Bollinger Bands با افزایش ناگهانی حجم",
            candles=candles,
            expected_trend_adjustment=+0.15,
            expected_momentum_adjustment=+0.12,
            expected_volatility_adjustment=+0.22,  # تاثیر قوی
            expected_cycle_adjustment=+0.18,
            expected_sr_adjustment=+0.20,
            explanation="Volume spike بعد از squeeze نشانه شکست قوی",
        )

    def create_scenario_5_breakout_high_volume(self) -> TrainingScenario:
        """
        سناریو 5: شکست مقاومت با حجم بالا

        توقع: S/R adjustment مثبت قوی (breakout معتبر)
        """
        candles = []
        resistance_level = 50000

        # مرحله 1: چند بار test مقاومت (25 کندل)
        for i in range(25):
            base_price = resistance_level - np.random.uniform(200, 500)
            open_price = base_price
            close_price = base_price + np.random.uniform(100, 300)
            high_price = min(close_price + np.random.uniform(0, 150), resistance_level + 50)
            low_price = open_price - np.random.uniform(0, 100)
            volume = np.random.uniform(1200, 1600)

            candles.append(
                Candle(
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timestamp=1700000000 + i * 3600,
                )
            )

        # مرحله 2: شکست با حجم بالا (25 کندل)
        base_price = resistance_level
        for i in range(25):
            open_price = base_price
            close_price = base_price + np.random.uniform(200, 400)
            high_price = close_price + np.random.uniform(0, 150)
            low_price = open_price - np.random.uniform(0, 80)
            volume = np.random.uniform(2800, 3800)  # حجم 3× میانگین

            candles.append(
                Candle(
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timestamp=1700000000 + (i + 25) * 3600,
                )
            )

            base_price = close_price

        return TrainingScenario(
            name="Breakout + High Volume",
            description="شکست مقاومت با حجم 3 برابر - معتبر",
            candles=candles,
            expected_trend_adjustment=+0.18,
            expected_momentum_adjustment=+0.15,
            expected_volatility_adjustment=+0.12,
            expected_cycle_adjustment=+0.16,
            expected_sr_adjustment=+0.28,  # تاثیر بسیار قوی
            explanation="حجم بالا breakout را تایید می‌کند",
        )

    def create_scenario_6_breakout_low_volume(self) -> TrainingScenario:
        """
        سناریو 6: شکست با حجم پایین (fake breakout)

        توقع: S/R adjustment منفی (fake breakout)
        """
        candles = []
        resistance_level = 50000

        # مرحله 1: نزدیک شدن به مقاومت (20 کندل)
        for i in range(20):
            base_price = resistance_level - np.random.uniform(200, 500)
            open_price = base_price
            close_price = base_price + np.random.uniform(100, 300)
            high_price = min(close_price + np.random.uniform(0, 100), resistance_level)
            low_price = open_price - np.random.uniform(0, 100)
            volume = np.random.uniform(1400, 1800)

            candles.append(
                Candle(
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timestamp=1700000000 + i * 3600,
                )
            )

        # مرحله 2: fake breakout با حجم پایین (10 کندل)
        base_price = resistance_level
        for i in range(10):
            open_price = base_price
            close_price = base_price + np.random.uniform(50, 150)
            high_price = close_price + np.random.uniform(0, 100)
            low_price = open_price - np.random.uniform(0, 50)
            volume = np.random.uniform(900, 1300)  # حجم پایین!

            candles.append(
                Candle(
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timestamp=1700000000 + (i + 20) * 3600,
                )
            )

            base_price = close_price

        # مرحله 3: بازگشت به زیر مقاومت (20 کندل)
        for i in range(20):
            open_price = base_price
            close_price = base_price - np.random.uniform(100, 250)
            high_price = open_price + np.random.uniform(0, 80)
            low_price = close_price - np.random.uniform(0, 100)
            volume = np.random.uniform(1600, 2200)

            candles.append(
                Candle(
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timestamp=1700000000 + (i + 30) * 3600,
                )
            )

            base_price = close_price

        return TrainingScenario(
            name="Fake Breakout + Low Volume",
            description="شکست با حجم پایین و بازگشت سریع",
            candles=candles,
            expected_trend_adjustment=-0.15,
            expected_momentum_adjustment=-0.12,
            expected_volatility_adjustment=-0.08,
            expected_cycle_adjustment=-0.10,
            expected_sr_adjustment=-0.25,  # تاثیر منفی قوی
            explanation="حجم پایین نشانه fake breakout",
        )

    def create_scenario_7_accumulation_volume_spike(self) -> TrainingScenario:
        """
        سناریو 7: فاز Accumulation + Volume Spike ناگهانی

        توقع: cycle adjustment مثبت (شروع فاز Markup)
        """
        candles = []
        base_price = 48000

        # مرحله 1: Accumulation با حجم پایین (35 کندل)
        for i in range(35):
            open_price = base_price + np.random.uniform(-150, 150)
            close_price = base_price + np.random.uniform(-150, 150)
            high_price = max(open_price, close_price) + np.random.uniform(0, 80)
            low_price = min(open_price, close_price) - np.random.uniform(0, 80)
            volume = np.random.uniform(900, 1300)  # حجم پایین

            candles.append(
                Candle(
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timestamp=1700000000 + i * 3600,
                )
            )

        # مرحله 2: Volume Spike + شروع صعود (15 کندل)
        for i in range(15):
            open_price = base_price
            close_price = base_price + np.random.uniform(200, 400)
            high_price = close_price + np.random.uniform(0, 100)
            low_price = open_price - np.random.uniform(0, 50)
            volume = np.random.uniform(2500, 3500)  # volume spike

            candles.append(
                Candle(
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timestamp=1700000000 + (i + 35) * 3600,
                )
            )

            base_price = close_price

        return TrainingScenario(
            name="Accumulation + Volume Spike",
            description="انباشت طولانی با افزایش ناگهانی حجم",
            candles=candles,
            expected_trend_adjustment=+0.20,
            expected_momentum_adjustment=+0.18,
            expected_volatility_adjustment=+0.15,
            expected_cycle_adjustment=+0.25,  # تاثیر قوی
            expected_sr_adjustment=+0.16,
            explanation="Volume spike در accumulation = شروع markup",
        )

    # ═══════════════════════════════════════════════════════════════════
    # Training Methods
    # ═══════════════════════════════════════════════════════════════════

    def prepare_training_data(self) -> None:
        """تولید همه سناریوهای آموزشی"""
        print("🔄 Generating training scenarios...")

        self.scenarios = [
            self.create_scenario_1_strong_trend_confirming_volume(),
            self.create_scenario_2_strong_trend_divergent_volume(),
            self.create_scenario_3_overbought_high_volume(),
            self.create_scenario_4_bb_squeeze_volume_spike(),
            self.create_scenario_5_breakout_high_volume(),
            self.create_scenario_6_breakout_low_volume(),
            self.create_scenario_7_accumulation_volume_spike(),
        ]

        print(f"✅ Generated {len(self.scenarios)} training scenarios\n")

        for i, scenario in enumerate(self.scenarios, 1):
            print(f"{i}. {scenario.name}")
            print(f"   {scenario.description}")
            print(f"   Candles: {len(scenario.candles)}")
            print()

    def train(self) -> dict:
        """
        آموزش وزن‌ها

        در این نسخه ساده، از میانگین adjustments مورد انتظار استفاده می‌کنیم.
        در نسخه پیشرفته‌تر می‌توان از یادگیری ماشین استفاده کرد.
        """
        print("🎓 Training Volume-Dimension Matrix weights...\n")

        if not self.scenarios:
            self.prepare_training_data()

        # جمع‌آوری adjustments از همه سناریوها
        trend_adjustments = []
        momentum_adjustments = []
        volatility_adjustments = []
        cycle_adjustments = []
        sr_adjustments = []

        for scenario in self.scenarios:
            trend_adjustments.append(scenario.expected_trend_adjustment)
            momentum_adjustments.append(scenario.expected_momentum_adjustment)
            volatility_adjustments.append(scenario.expected_volatility_adjustment)
            cycle_adjustments.append(scenario.expected_cycle_adjustment)
            sr_adjustments.append(scenario.expected_sr_adjustment)

        # محاسبه آمار
        stats = {
            "trend": {
                "mean": float(np.mean(trend_adjustments)),
                "std": float(np.std(trend_adjustments)),
                "min": float(np.min(trend_adjustments)),
                "max": float(np.max(trend_adjustments)),
                "median": float(np.median(trend_adjustments)),
            },
            "momentum": {
                "mean": float(np.mean(momentum_adjustments)),
                "std": float(np.std(momentum_adjustments)),
                "min": float(np.min(momentum_adjustments)),
                "max": float(np.max(momentum_adjustments)),
                "median": float(np.median(momentum_adjustments)),
            },
            "volatility": {
                "mean": float(np.mean(volatility_adjustments)),
                "std": float(np.std(volatility_adjustments)),
                "min": float(np.min(volatility_adjustments)),
                "max": float(np.max(volatility_adjustments)),
                "median": float(np.median(volatility_adjustments)),
            },
            "cycle": {
                "mean": float(np.mean(cycle_adjustments)),
                "std": float(np.std(cycle_adjustments)),
                "min": float(np.min(cycle_adjustments)),
                "max": float(np.max(cycle_adjustments)),
                "median": float(np.median(cycle_adjustments)),
            },
            "support_resistance": {
                "mean": float(np.mean(sr_adjustments)),
                "std": float(np.std(sr_adjustments)),
                "min": float(np.min(sr_adjustments)),
                "max": float(np.max(sr_adjustments)),
                "median": float(np.median(sr_adjustments)),
            },
        }

        # ذخیره وزن‌ها (در اینجا فقط آمار است)
        self.weights = stats

        # نمایش نتایج
        print("📊 Training Statistics:")
        print("=" * 60)

        for dim, stat in stats.items():
            print(f"\n{dim.upper()}:")
            print(f"  Mean Adjustment: {stat['mean']:+.3f}")
            print(f"  Std Dev:         {stat['std']:.3f}")
            print(f"  Range:          [{stat['min']:+.3f}, {stat['max']:+.3f}]")
            print(f"  Median:          {stat['median']:+.3f}")

        print("\n" + "=" * 60)

        return self.weights

    def save_weights(self, filepath: str = "models/volume_matrix/weights.json") -> None:
        """ذخیره وزن‌ها در فایل JSON"""
        filepath_obj = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.weights, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Weights saved to: {filepath}")

    def evaluate(self) -> None:
        """ارزیابی نتایج آموزش"""
        print("\n🔍 Evaluation Summary:")
        print("=" * 60)

        for i, scenario in enumerate(self.scenarios, 1):
            print(f"\n{i}. {scenario.name}")
            print(f"   {scenario.explanation}")

            print("\n   Expected Adjustments:")
            print(f"   - Trend:      {scenario.expected_trend_adjustment:+.3f}")
            print(f"   - Momentum:   {scenario.expected_momentum_adjustment:+.3f}")
            print(f"   - Volatility: {scenario.expected_volatility_adjustment:+.3f}")
            print(f"   - Cycle:      {scenario.expected_cycle_adjustment:+.3f}")
            print(f"   - S/R:        {scenario.expected_sr_adjustment:+.3f}")

        print("\n" + "=" * 60)


# ═══════════════════════════════════════════════════════════════════
# Main Execution
# ═══════════════════════════════════════════════════════════════════


def main():
    """اجرای کامل آموزش"""
    print("=" * 80)
    print("   VOLUME-DIMENSION MATRIX TRAINING PIPELINE")
    print("=" * 80)
    print()

    # ایجاد trainer
    trainer = VolumeMatrixTrainer()

    # تولید داده‌های آموزشی
    trainer.prepare_training_data()

    # آموزش
    weights = trainer.train()

    # ارزیابی
    trainer.evaluate()

    # ذخیره وزن‌ها
    trainer.save_weights()

    print("\n" + "=" * 80)
    print("✅ Training completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
