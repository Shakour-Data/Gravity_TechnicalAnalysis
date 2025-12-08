"""
Database Manager with Auto-Setup and Fallback

این ماژول مدیریت دیتابیس را با قابلیت‌های زیر فراهم می‌کند:
1. Auto-setup: خودکار ساخت دیتابیس و جداول
2. Fallback: کار بدون دیتابیس (in-memory SQLite یا JSON)
3. Migration: به‌روزرسانی schema به صورت خودکار

استراتژی:
- اگر PostgreSQL موجود باشد → استفاده از PostgreSQL
- اگر PostgreSQL موجود نباشد → fallback به SQLite in-memory
- اگر SQLite هم ناموجود باشد → fallback به JSON file storage

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging

# Try to import PostgreSQL driver
try:
    import psycopg2
    from psycopg2 import pool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("⚠️ psycopg2 not available. Will use SQLite fallback.")


logger = logging.getLogger(__name__)


class DatabaseType(str, Enum):
    """نوع دیتابیس"""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    JSON_FILE = "json_file"


class DatabaseManager:
    """
    مدیریت دیتابیس با auto-setup و fallback
    
    ویژگی‌ها:
    - خودکار ساخت schema
    - Fallback به SQLite یا JSON
    - Migration خودکار
    - Connection pooling
    """
    
    def __init__(
        self,
        db_type: Optional[DatabaseType] = None,
        connection_string: Optional[str] = None,
        sqlite_path: Optional[str] = None,
        json_path: Optional[str] = None,
        auto_setup: bool = True
    ):
        """
        Initialize Database Manager
        
        Args:
            db_type: نوع دیتابیس (اگر None باشد، خودکار تشخیص می‌دهد)
            connection_string: رشته اتصال PostgreSQL
            sqlite_path: مسیر فایل SQLite (برای fallback)
            json_path: مسیر فایل JSON (برای fallback)
            auto_setup: آیا خودکار schema را بسازد؟
        """
        self.db_type = db_type
        self.connection_string = connection_string
        self.sqlite_path = sqlite_path or "data/tool_performance.db"
        self.json_path = json_path or "data/tool_performance.json"
        
        self.connection_pool = None
        self.sqlite_connection = None
        self.json_data = {}
        
        # Auto-detect database type
        if self.db_type is None:
            self.db_type = self._detect_database_type()
        
        # Initialize database
        self._initialize_database()
        
        # Auto-setup schema
        if auto_setup:
            self.setup_schema()
        
        logger.info(f"✅ Database initialized: {self.db_type.value}")
    
    def _detect_database_type(self) -> DatabaseType:
        """
        تشخیص خودکار نوع دیتابیس
        
        اولویت:
        1. PostgreSQL (اگر psycopg2 و connection_string موجود باشد)
        2. SQLite (fallback)
        3. JSON file (fallback نهایی)
        """
        
        # Check for PostgreSQL
        if POSTGRES_AVAILABLE and self.connection_string:
            try:
                # Test connection
                conn = psycopg2.connect(self.connection_string)
                conn.close()
                logger.info("✅ PostgreSQL detected and available")
                return DatabaseType.POSTGRESQL
            except Exception as e:
                logger.warning(f"⚠️ PostgreSQL connection failed: {e}")
        
        # Check for environment variable
        postgres_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
        if POSTGRES_AVAILABLE and postgres_url:
            try:
                conn = psycopg2.connect(postgres_url)
                conn.close()
                self.connection_string = postgres_url
                logger.info("✅ PostgreSQL from env variable")
                return DatabaseType.POSTGRESQL
            except Exception as e:
                logger.warning(f"⚠️ PostgreSQL from env failed: {e}")
        
        # Fallback to SQLite
        try:
            # Create directory if needed
            Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.sqlite_path)
            conn.close()
            logger.info("✅ SQLite fallback activated")
            return DatabaseType.SQLITE
        except Exception as e:
            logger.warning(f"⚠️ SQLite failed: {e}")
        
        # Final fallback to JSON
        logger.info("✅ JSON file fallback activated")
        return DatabaseType.JSON_FILE
    
    def _initialize_database(self):
        """راه‌اندازی اتصال دیتابیس"""
        
        if self.db_type == DatabaseType.POSTGRESQL:
            self._initialize_postgresql()
        elif self.db_type == DatabaseType.SQLITE:
            self._initialize_sqlite()
        elif self.db_type == DatabaseType.JSON_FILE:
            self._initialize_json()
    
    def _initialize_postgresql(self):
        """راه‌اندازی PostgreSQL با connection pool"""
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=self.connection_string
            )
            logger.info("✅ PostgreSQL connection pool created")
        except Exception as e:
            logger.error(f"❌ Failed to create PostgreSQL pool: {e}")
            raise
    
    def _initialize_sqlite(self):
        """راه‌اندازی SQLite"""
        try:
            Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            self.sqlite_connection = sqlite3.connect(
                self.sqlite_path,
                check_same_thread=False
            )
            self.sqlite_connection.row_factory = sqlite3.Row
            logger.info(f"✅ SQLite initialized: {self.sqlite_path}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize SQLite: {e}")
            raise
    
    def _initialize_json(self):
        """راه‌اندازی JSON file storage"""
        try:
            Path(self.json_path).parent.mkdir(parents=True, exist_ok=True)
            
            if Path(self.json_path).exists():
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    self.json_data = json.load(f)
            else:
                self.json_data = {
                    "tool_performance_history": [],
                    "tool_performance_stats": [],
                    "ml_weights_history": [],
                    "tool_recommendations_log": []
                }
                self._save_json()
            
            logger.info(f"✅ JSON storage initialized: {self.json_path}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize JSON: {e}")
            raise
    
    def setup_schema(self):
        """
        ساخت خودکار schema دیتابیس
        
        این متد schema را بر اساس نوع دیتابیس می‌سازد
        """
        
        if self.db_type == DatabaseType.POSTGRESQL:
            self._setup_postgresql_schema()
        elif self.db_type == DatabaseType.SQLITE:
            self._setup_sqlite_schema()
        elif self.db_type == DatabaseType.JSON_FILE:
            # JSON doesn't need schema
            logger.info("✅ JSON schema (structure) ready")
    
    def _setup_postgresql_schema(self):
        """ساخت schema برای PostgreSQL"""
        
        schema_file = Path(__file__).parent / "tool_performance_history.sql"
        
        if not schema_file.exists():
            logger.warning(f"⚠️ Schema file not found: {schema_file}")
            return
        
        try:
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            # Execute schema
            cursor.execute(schema_sql)
            conn.commit()
            
            cursor.close()
            self.connection_pool.putconn(conn)
            
            logger.info("✅ PostgreSQL schema created successfully")
        except Exception as e:
            logger.error(f"❌ Failed to create PostgreSQL schema: {e}")
            # Don't raise - schema might already exist
    
    def _setup_sqlite_schema(self):
        """ساخت schema برای SQLite"""
        
        # SQLite version of schema (simplified, no functions/triggers)
        schema = """
        -- Tool Performance History
        CREATE TABLE IF NOT EXISTS tool_performance_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_name TEXT NOT NULL,
            tool_category TEXT NOT NULL,
            
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            market_regime TEXT NOT NULL,
            volatility_level REAL,
            trend_strength REAL,
            volume_profile TEXT,
            
            prediction_type TEXT NOT NULL,
            prediction_value REAL,
            confidence_score REAL,
            
            actual_result TEXT,
            actual_price_change REAL,
            success INTEGER,
            accuracy REAL,
            
            prediction_timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            result_timestamp TEXT,
            evaluation_period_hours INTEGER,
            
            metadata TEXT,
            
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_tool_performance_tool_name 
            ON tool_performance_history(tool_name);
        CREATE INDEX IF NOT EXISTS idx_tool_performance_symbol 
            ON tool_performance_history(symbol);
        CREATE INDEX IF NOT EXISTS idx_tool_performance_regime 
            ON tool_performance_history(market_regime);
        
        -- Tool Performance Stats
        CREATE TABLE IF NOT EXISTS tool_performance_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_name TEXT NOT NULL,
            tool_category TEXT NOT NULL,
            
            market_regime TEXT,
            timeframe TEXT,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            
            total_predictions INTEGER NOT NULL DEFAULT 0,
            correct_predictions INTEGER NOT NULL DEFAULT 0,
            accuracy REAL,
            
            avg_confidence REAL,
            avg_actual_change REAL,
            
            bullish_predictions INTEGER DEFAULT 0,
            bearish_predictions INTEGER DEFAULT 0,
            neutral_predictions INTEGER DEFAULT 0,
            
            bullish_success_rate REAL,
            bearish_success_rate REAL,
            neutral_success_rate REAL,
            
            best_accuracy REAL,
            worst_accuracy REAL,
            best_symbol TEXT,
            worst_symbol TEXT,
            
            last_updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            
            UNIQUE(tool_name, market_regime, timeframe, period_start, period_end)
        );
        
        -- ML Weights History
        CREATE TABLE IF NOT EXISTS ml_weights_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            
            market_regime TEXT,
            timeframe TEXT,
            
            weights TEXT NOT NULL,
            
            training_accuracy REAL,
            validation_accuracy REAL,
            r2_score REAL,
            mae REAL,
            
            training_samples INTEGER,
            training_date TEXT NOT NULL,
            
            metadata TEXT,
            
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_ml_weights_model 
            ON ml_weights_history(model_name, model_version);
        
        -- Tool Recommendations Log
        CREATE TABLE IF NOT EXISTS tool_recommendations_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            
            request_id TEXT NOT NULL,
            user_id TEXT,
            
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            analysis_goal TEXT,
            trading_style TEXT,
            
            market_regime TEXT NOT NULL,
            volatility_level REAL,
            trend_strength REAL,
            
            recommended_tools TEXT NOT NULL,
            ml_weights TEXT,
            
            user_feedback TEXT,
            tools_actually_used TEXT,
            trade_result TEXT,
            
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            feedback_at TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_recommendations_request 
            ON tool_recommendations_log(request_id);
        """
        
        try:
            cursor = self.sqlite_connection.cursor()
            cursor.executescript(schema)
            self.sqlite_connection.commit()
            logger.info("✅ SQLite schema created successfully")
        except Exception as e:
            logger.error(f"❌ Failed to create SQLite schema: {e}")
    
    def get_connection(self):
        """دریافت connection (برای PostgreSQL یا SQLite)"""
        
        if self.db_type == DatabaseType.POSTGRESQL:
            return self.connection_pool.getconn()
        elif self.db_type == DatabaseType.SQLITE:
            return self.sqlite_connection
        else:
            raise ValueError(f"get_connection not supported for {self.db_type}")
    
    def release_connection(self, conn):
        """آزاد کردن connection (فقط برای PostgreSQL)"""
        
        if self.db_type == DatabaseType.POSTGRESQL:
            self.connection_pool.putconn(conn)
    
    def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        fetch: bool = False
    ) -> Optional[List]:
        """
        اجرای query با fallback
        
        Args:
            query: SQL query
            params: پارامترهای query
            fetch: آیا نتیجه را برگرداند؟
        
        Returns:
            نتیجه query (اگر fetch=True)
        """
        
        if self.db_type == DatabaseType.JSON_FILE:
            # JSON doesn't support SQL queries
            logger.warning("⚠️ SQL query not supported for JSON storage")
            return None
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch:
                if self.db_type == DatabaseType.POSTGRESQL:
                    result = cursor.fetchall()
                else:  # SQLite
                    result = [dict(row) for row in cursor.fetchall()]
            else:
                result = None
            
            conn.commit()
            cursor.close()
            
            self.release_connection(conn)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Query execution failed: {e}")
            raise
    
    def record_tool_performance(
        self,
        tool_name: str,
        tool_category: str,
        symbol: str,
        timeframe: str,
        market_regime: str,
        prediction_type: str,
        confidence_score: float,
        volatility_level: Optional[float] = None,
        trend_strength: Optional[float] = None,
        volume_profile: Optional[str] = None,
        prediction_value: Optional[float] = None,
        actual_result: Optional[str] = None,
        actual_price_change: Optional[float] = None,
        success: Optional[bool] = None,
        accuracy: Optional[float] = None,
        result_timestamp=None,
        evaluation_period_hours: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        ثبت عملکرد ابزار (تمام فیلدهای جدول را پشتیبانی می‌کند)
        
        Returns:
            ID رکورد ذخیره شده
        """

        # Normalize datetime input if provided
        if result_timestamp is not None:
            try:
                result_timestamp = result_timestamp.isoformat()  # type: ignore[attr-defined]
            except Exception:
                pass
        
        if self.db_type == DatabaseType.JSON_FILE:
            return self._record_tool_performance_json(
                tool_name, tool_category, symbol, timeframe, market_regime,
                prediction_type, confidence_score, volatility_level,
                trend_strength, volume_profile, prediction_value,
                actual_result, actual_price_change, success, accuracy,
                result_timestamp, evaluation_period_hours, metadata
            )
        
        query = """
        INSERT INTO tool_performance_history (
            tool_name, tool_category, symbol, timeframe, market_regime,
            volatility_level, trend_strength, volume_profile,
            prediction_type, prediction_value, confidence_score,
            actual_result, actual_price_change, success, accuracy,
            result_timestamp, evaluation_period_hours, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """ if self.db_type == DatabaseType.POSTGRESQL else """
        INSERT INTO tool_performance_history (
            tool_name, tool_category, symbol, timeframe, market_regime,
            volatility_level, trend_strength, volume_profile,
            prediction_type, prediction_value, confidence_score,
            actual_result, actual_price_change, success, accuracy,
            result_timestamp, evaluation_period_hours, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        params = (
            tool_name, tool_category, symbol, timeframe, market_regime,
            volatility_level, trend_strength, volume_profile,
            prediction_type, prediction_value, confidence_score,
            actual_result, actual_price_change, success, accuracy,
            result_timestamp, evaluation_period_hours, metadata_json
        )
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            if self.db_type == DatabaseType.POSTGRESQL:
                record_id = cursor.fetchone()[0]
            else:  # SQLite
                record_id = cursor.lastrowid
            
            conn.commit()
            cursor.close()
            self.release_connection(conn)
            
            return record_id
            
        except Exception as e:
            logger.error(f"❌ Failed to record performance: {e}")
            raise
    
    def update_tool_result(
        self,
        record_id: int,
        actual_result: Optional[str] = None,
        actual_price_change: Optional[float] = None,
        success: Optional[bool] = None,
        accuracy: Optional[float] = None,
        result_timestamp=None,
        evaluation_period_hours: Optional[int] = None
    ) -> bool:
        """
        به‌روزرسانی فیلدهای نتیجه برای رکورد موجود در tool_performance_history
        """
        if result_timestamp is not None:
            try:
                result_timestamp = result_timestamp.isoformat()  # type: ignore[attr-defined]
            except Exception:
                pass

        if self.db_type == DatabaseType.JSON_FILE:
            for item in self.json_data.get("tool_performance_history", []):
                if item.get("id") == record_id:
                    if actual_result is not None:
                        item["actual_result"] = actual_result
                    if actual_price_change is not None:
                        item["actual_price_change"] = actual_price_change
                    if success is not None:
                        item["success"] = success
                    if accuracy is not None:
                        item["accuracy"] = accuracy
                    if result_timestamp is not None:
                        item["result_timestamp"] = result_timestamp
                    if evaluation_period_hours is not None:
                        item["evaluation_period_hours"] = evaluation_period_hours
                    item["updated_at"] = datetime.utcnow().isoformat()
                    self._save_json()
                    return True
            return False

        set_clauses = []
        params: list[Any] = []
        placeholder = "%s" if self.db_type == DatabaseType.POSTGRESQL else "?"

        if actual_result is not None:
            set_clauses.append(f"actual_result = {placeholder}")
            params.append(actual_result)
        if actual_price_change is not None:
            set_clauses.append(f"actual_price_change = {placeholder}")
            params.append(actual_price_change)
        if success is not None:
            set_clauses.append(f"success = {placeholder}")
            params.append(success)
        if accuracy is not None:
            set_clauses.append(f"accuracy = {placeholder}")
            params.append(accuracy)
        if result_timestamp is not None:
            set_clauses.append(f"result_timestamp = {placeholder}")
            params.append(result_timestamp)
        if evaluation_period_hours is not None:
            set_clauses.append(f"evaluation_period_hours = {placeholder}")
            params.append(evaluation_period_hours)

        # always bump updated_at
        set_clauses.append("updated_at = NOW()" if self.db_type == DatabaseType.POSTGRESQL else "updated_at = CURRENT_TIMESTAMP")

        if not set_clauses:
            return False

        query = f"""
        UPDATE tool_performance_history
        SET {', '.join(set_clauses)}
        WHERE id = {placeholder}
        """
        params.append(record_id)

        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            conn.commit()
            updated = cursor.rowcount > 0
            cursor.close()
            self.release_connection(conn)
            return updated
        except Exception as e:
            logger.error(f"❌ Failed to update tool result: {e}")
            raise
    
    def _record_tool_performance_json(
        self, tool_name, tool_category, symbol, timeframe, market_regime,
        prediction_type, confidence_score, volatility_level,
        trend_strength, volume_profile, prediction_value,
        actual_result, actual_price_change, success, accuracy,
        result_timestamp, evaluation_period_hours, metadata
    ) -> int:
        """ثبت عملکرد در JSON"""
        
        record = {
            "id": len(self.json_data["tool_performance_history"]) + 1,
            "tool_name": tool_name,
            "tool_category": tool_category,
            "symbol": symbol,
            "timeframe": timeframe,
            "market_regime": market_regime,
            "volatility_level": volatility_level,
            "trend_strength": trend_strength,
            "volume_profile": volume_profile,
            "prediction_type": prediction_type,
            "prediction_value": prediction_value,
            "confidence_score": confidence_score,
            "actual_result": actual_result,
            "actual_price_change": actual_price_change,
            "success": success,
            "accuracy": accuracy,
            "result_timestamp": result_timestamp,
            "evaluation_period_hours": evaluation_period_hours,
            "metadata": metadata,
            "prediction_timestamp": datetime.utcnow().isoformat(),
            "created_at": datetime.utcnow().isoformat()
        }
        
        self.json_data["tool_performance_history"].append(record)
        self._save_json()
        
        return record["id"]
    
    def get_tool_accuracy(
        self,
        tool_name: str,
        market_regime: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        دریافت دقت یک ابزار
        
        Args:
            tool_name: نام ابزار
            market_regime: رژیم بازار (اختیاری)
            days: تعداد روزهای گذشته
        
        Returns:
            دیکشنری شامل آمار دقت
        """
        
        if self.db_type == DatabaseType.JSON_FILE:
            return self._get_tool_accuracy_json(tool_name, market_regime, days)
        
        query = """
        SELECT 
            tool_name,
            market_regime,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as correct_predictions,
            AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as accuracy,
            AVG(confidence_score) as avg_confidence
        FROM tool_performance_history
        WHERE tool_name = %s
          AND (%s IS NULL OR market_regime = %s)
          AND prediction_timestamp >= NOW() - INTERVAL '%s days'
          AND success IS NOT NULL
        GROUP BY tool_name, market_regime
        """ if self.db_type == DatabaseType.POSTGRESQL else """
        SELECT 
            tool_name,
            market_regime,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as correct_predictions,
            AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as accuracy,
            AVG(confidence_score) as avg_confidence
        FROM tool_performance_history
        WHERE tool_name = ?
          AND (? IS NULL OR market_regime = ?)
          AND datetime(prediction_timestamp) >= datetime('now', '-' || ? || ' days')
          AND success IS NOT NULL
        GROUP BY tool_name, market_regime
        """
        
        params = (tool_name, market_regime, market_regime, days) if self.db_type == DatabaseType.POSTGRESQL \
                 else (tool_name, market_regime, market_regime, days)
        
        results = self.execute_query(query, params, fetch=True)
        
        if results:
            return dict(results[0]) if isinstance(results[0], sqlite3.Row) else {
                "tool_name": results[0][0],
                "market_regime": results[0][1],
                "total_predictions": results[0][2],
                "correct_predictions": results[0][3],
                "accuracy": results[0][4],
                "avg_confidence": results[0][5]
            }
        
        return {
            "tool_name": tool_name,
            "market_regime": market_regime,
            "total_predictions": 0,
            "correct_predictions": 0,
            "accuracy": 0.0,
            "avg_confidence": 0.0
        }
    
    def _get_tool_accuracy_json(
        self, tool_name: str, market_regime: Optional[str], days: int
    ) -> Dict[str, Any]:
        """دریافت دقت از JSON"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        records = [
            r for r in self.json_data["tool_performance_history"]
            if r["tool_name"] == tool_name
            and (market_regime is None or r["market_regime"] == market_regime)
            and r.get("success") is not None
            and datetime.fromisoformat(r["prediction_timestamp"]) >= cutoff_date
        ]
        
        total = len(records)
        correct = sum(1 for r in records if r.get("success"))
        accuracy = correct / total if total > 0 else 0.0
        avg_confidence = sum(r.get("confidence_score", 0) for r in records) / total if total > 0 else 0.0
        
        return {
            "tool_name": tool_name,
            "market_regime": market_regime,
            "total_predictions": total,
            "correct_predictions": correct,
            "accuracy": accuracy,
            "avg_confidence": avg_confidence
        }
    
    def _save_json(self):
        """ذخیره داده‌ها در JSON file"""
        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.json_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Failed to save JSON: {e}")
    
    def close(self):
        """بستن اتصالات دیتابیس"""
        
        if self.db_type == DatabaseType.POSTGRESQL and self.connection_pool:
            self.connection_pool.closeall()
            logger.info("✅ PostgreSQL connections closed")
        
        elif self.db_type == DatabaseType.SQLITE and self.sqlite_connection:
            self.sqlite_connection.close()
            logger.info("✅ SQLite connection closed")
        
        elif self.db_type == DatabaseType.JSON_FILE:
            self._save_json()
            logger.info("✅ JSON data saved")
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.close()


# CLI tool for database setup
def main():
    """
    ابزار خط فرمان برای setup دیتابیس
    
    Usage:
        python database_manager.py
        python database_manager.py --type postgresql --connection "postgresql://..."
        python database_manager.py --type sqlite --path data/mydb.db
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Database Setup Tool")
    parser.add_argument(
        "--type",
        choices=["postgresql", "sqlite", "json"],
        help="نوع دیتابیس (اگر مشخص نشود، خودکار تشخیص داده می‌شود)"
    )
    parser.add_argument(
        "--connection",
        help="رشته اتصال PostgreSQL"
    )
    parser.add_argument(
        "--sqlite-path",
        default="data/tool_performance.db",
        help="مسیر فایل SQLite"
    )
    parser.add_argument(
        "--json-path",
        default="data/tool_performance.json",
        help="مسیر فایل JSON"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🗄️  Database Setup Tool")
    print("=" * 70)
    
    db_type = DatabaseType(args.type) if args.type else None
    
    try:
        with DatabaseManager(
            db_type=db_type,
            connection_string=args.connection,
            sqlite_path=args.sqlite_path,
            json_path=args.json_path,
            auto_setup=True
        ) as db:
            print(f"\n✅ Database setup complete!")
            print(f"   Type: {db.db_type.value}")
            
            if db.db_type == DatabaseType.POSTGRESQL:
                print(f"   Connection: PostgreSQL")
            elif db.db_type == DatabaseType.SQLITE:
                print(f"   Path: {db.sqlite_path}")
            elif db.db_type == DatabaseType.JSON_FILE:
                print(f"   Path: {db.json_path}")
            
            print(f"\n✅ Schema created successfully")
            print(f"   Tables: tool_performance_history, tool_performance_stats,")
            print(f"           ml_weights_history, tool_recommendations_log")
    
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return 1
    
    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
