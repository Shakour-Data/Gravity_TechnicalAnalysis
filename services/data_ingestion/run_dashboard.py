#!/usr/bin/env python3
"""
GravityTseHisPrice Web Dashboard Runner

This script starts the web-based dashboard for the TSE data analysis system.
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'dash', 'plotly', 'dash-bootstrap-components', 'pandas'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ بسته‌های مورد نیاز نصب نیستند:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 لطفاً دستور زیر را اجرا کنید:")
        print("pip install -r requirements.txt")
        return False

    return True

def check_database():
    """Check if database exists and has data"""
    db_path = os.path.join(os.path.dirname(__file__), 'data', 'tse_data.db')
    if not os.path.exists(db_path):
        print("⚠️  پایگاه داده یافت نشد!")
        print("💡 لطفاً ابتدا دیتابیس را راه‌اندازی کنید:")
        print("python main.py create-db")
        print("python main.py create-indices-tables")
        print("python main.py load-initial")
        print("python main.py load-all-prices")
        return False

    return True

def main():
    """Main function to run the dashboard"""
    print("🚀 GravityTseHisPrice Web Dashboard")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        return

    # Check database
    if not check_database():
        return

    # Start the dashboard
    print("📊 Starting dashboard...")
    print("🌐 Dashboard will be available at: http://127.0.0.1:8050/")
    print("❌ Press Ctrl+C to stop the server")
    print()

    try:
        # Run the dashboard
        dashboard_path = os.path.join(os.path.dirname(__file__), 'web', 'dashboard.py')
        subprocess.run([sys.executable, dashboard_path], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dashboard: {e}")
    except FileNotFoundError:
        print("❌ Dashboard file not found!")

if __name__ == '__main__':
    main()
