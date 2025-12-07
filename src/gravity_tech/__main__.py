"""
Gravity Tech CLI Main Entry Point

نقطه ورود اصلی برای دستورات CLI پروژه

Author: Gravity Tech Team
Date: December 5, 2025
Version: 1.0.0
License: MIT
"""

from .cli.db_commands import db_cli

if __name__ == "__main__":
    db_cli()
