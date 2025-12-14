"""
Quick ML Training Example

This script shows how to train and use ML-based weight optimization.
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.train_weights import train_ml_model, test_ml_model


async def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║        ML-Based Weight Optimization for Technical Analysis      ║
╚══════════════════════════════════════════════════════════════════╝

این سیستم از یادگیری ماشین برای یافتن وزن‌های بهینه استفاده می‌کند.

چگونه کار می‌کند:
1. داده‌های تاریخی را تولید می‌کند (شامل روندهای مختلف بازار)
2. برای هر نمونه، تمام اندیکاتورها را محاسبه می‌کند
3. بازده آینده را به عنوان هدف محاسبه می‌کند
4. مدل ML را آموزش می‌دهد تا وزن‌های بهینه را یاد بگیرد

مزایا:
✓ وزن‌ها بر اساس داده واقعی یاد گرفته می‌شوند
✓ سازگار با شرایط مختلف بازار
✓ بهینه‌سازی خودکار
✓ بهتر از وزن‌های ثابت

گزینه‌ها:
1) آموزش مدل با 100 نمونه (سریع - برای تست)
2) آموزش مدل با 500 نمونه (توصیه شده)
3) آموزش مدل با 1000 نمونه (دقیق - زمان‌بر)
4) تست مدل آموزش دیده
5) خروج

""")
    
    choice = input("انتخاب کنید (1-5): ").strip()
    
    if choice == "1":
        print("\n🚀 آموزش با 100 نمونه...")
        await train_ml_model(num_samples=100, model_type="gradient_boosting")
    
    elif choice == "2":
        print("\n🚀 آموزش با 500 نمونه...")
        await train_ml_model(num_samples=500, model_type="gradient_boosting")
    
    elif choice == "3":
        print("\n🚀 آموزش با 1000 نمونه...")
        await train_ml_model(num_samples=1000, model_type="gradient_boosting")
    
    elif choice == "4":
        print("\n🧪 تست مدل...")
        await test_ml_model()
    
    elif choice == "5":
        print("\n👋 خروج...")
        return
    
    else:
        print("\n❌ انتخاب نامعتبر!")


if __name__ == "__main__":
    asyncio.run(main())
