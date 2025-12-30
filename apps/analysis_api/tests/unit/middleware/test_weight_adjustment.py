"""
Advanced test to show how accuracy affects actual weights distribution
"""

from datetime import UTC, datetime

from gravity_tech.core.domain.entities import CoreSignalStrength as SignalStrength
from gravity_tech.core.domain.entities import IndicatorCategory, IndicatorResult


def create_indicator(
    name: str, category: IndicatorCategory, signal: SignalStrength, confidence: float
) -> IndicatorResult:
    """Helper to create indicator"""
    return IndicatorResult(
        indicator_name=name,
        category=category,
        signal=signal,
        value=0.0,
        confidence=confidence,
        timestamp=datetime.now(UTC),
    )


def calculate_effective_weights(
    trend_conf: float, momentum_conf: float, cycle_conf: float, volume_conf: float
) -> dict:
    """
    Calculate effective weights after accuracy adjustment
    """
    base_weights = {"trend": 0.30, "momentum": 0.25, "cycle": 0.25, "volume": 0.20}

    accuracies = {
        "trend": trend_conf,
        "momentum": momentum_conf,
        "cycle": cycle_conf,
        "volume": volume_conf,
    }

    total_weighted_accuracy = sum(
        base_weights[cat] * accuracies[cat] for cat in base_weights.keys()
    )

    if total_weighted_accuracy > 0:
        adjusted_weights = {
            cat: (base_weights[cat] * accuracies[cat]) / total_weighted_accuracy
            for cat in base_weights.keys()
        }
    else:
        adjusted_weights = base_weights

    return adjusted_weights


def test_weight_distribution():
    """Test various accuracy scenarios and show weight distribution"""

    print("=" * 70)
    print("تأثیر دقت (Accuracy) بر توزیع وزن‌ها")
    print("=" * 70)
    print("\nوزن‌های پایه (قبل از تعدیل دقت):")
    print("  روند: 30% | مومنتوم: 25% | سیکل: 25% | حجم: 20%")
    print("=" * 70)

    scenarios = [
        {
            "name": "همه دسته‌ها دقت یکسان و بالا",
            "trend": 0.9,
            "momentum": 0.9,
            "cycle": 0.9,
            "volume": 0.9,
        },
        {
            "name": "روند دقت بسیار بالا، بقیه متوسط",
            "trend": 0.95,
            "momentum": 0.6,
            "cycle": 0.6,
            "volume": 0.6,
        },
        {
            "name": "مومنتوم دقت بسیار بالا، بقیه متوسط",
            "trend": 0.6,
            "momentum": 0.95,
            "cycle": 0.6,
            "volume": 0.6,
        },
        {
            "name": "سیکل دقت بسیار بالا، بقیه متوسط",
            "trend": 0.6,
            "momentum": 0.6,
            "cycle": 0.95,
            "volume": 0.6,
        },
        {
            "name": "روند و حجم دقت بالا، مومنتوم و سیکل پایین",
            "trend": 0.9,
            "momentum": 0.3,
            "cycle": 0.3,
            "volume": 0.9,
        },
        {
            "name": "مومنتوم و سیکل دقت بالا، روند و حجم پایین",
            "trend": 0.3,
            "momentum": 0.9,
            "cycle": 0.9,
            "volume": 0.3,
        },
        {
            "name": "تنها روند دقت بالا، بقیه بسیار پایین",
            "trend": 0.9,
            "momentum": 0.2,
            "cycle": 0.2,
            "volume": 0.2,
        },
        {"name": "دقت‌های تصادفی", "trend": 0.7, "momentum": 0.4, "cycle": 0.8, "volume": 0.5},
    ]

    for scenario in scenarios:
        print(f"\n{'─' * 70}")
        print(f"📊 {scenario['name']}")
        print(f"{'─' * 70}")

        weights = calculate_effective_weights(
            scenario["trend"], scenario["momentum"], scenario["cycle"], scenario["volume"]
        )

        print("\nدقت‌ها (Confidence):")
        print(f"  روند:    {scenario['trend']:.2f}")
        print(f"  مومنتوم:  {scenario['momentum']:.2f}")
        print(f"  سیکل:    {scenario['cycle']:.2f}")
        print(f"  حجم:     {scenario['volume']:.2f}")

        print("\nوزن‌های تعدیل‌شده:")
        print(f"  روند:    {weights['trend']:.1%} (پایه: 30%)")
        print(f"  مومنتوم:  {weights['momentum']:.1%} (پایه: 25%)")
        print(f"  سیکل:    {weights['cycle']:.1%} (پایه: 25%)")
        print(f"  حجم:     {weights['volume']:.1%} (پایه: 20%)")

        # Calculate changes
        changes = {
            "trend": (weights["trend"] - 0.30) / 0.30 * 100,
            "momentum": (weights["momentum"] - 0.25) / 0.25 * 100,
            "cycle": (weights["cycle"] - 0.25) / 0.25 * 100,
            "volume": (weights["volume"] - 0.20) / 0.20 * 100,
        }

        print("\nتغییرات نسبت به پایه:")
        for cat, change in changes.items():
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            cat_fa = {"trend": "روند", "momentum": "مومنتوم", "cycle": "سیکل", "volume": "حجم"}[cat]
            print(f"  {cat_fa:8s}: {arrow} {abs(change):+.1f}%")

    print(f"\n{'=' * 70}")
    print("💡 نکات کلیدی:")
    print("  1. دسته با دقت بالاتر، وزن بیشتری دریافت می‌کند")
    print("  2. وزن‌های نهایی همواره جمع می‌شوند به 100%")
    print("  3. این امر باعث می‌شود سیگنال‌های با اطمینان بالاتر تأثیر بیشتری داشته باشند")
    print("  4. در صورت دقت یکسان، وزن‌ها برابر با وزن‌های پایه هستند")
    print("=" * 70)


if __name__ == "__main__":
    test_weight_distribution()
