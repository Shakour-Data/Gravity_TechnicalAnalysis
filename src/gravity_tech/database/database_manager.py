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

import json
import logging
import os
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from gravity_tech.config.settings import settings

# Try to import PostgreSQL driver
try:
    import psycopg2
    from psycopg2 import pool
    POSTGRES_AVAILABLE = True
except ImportError:
    psycopg2 = None  # type: ignore
    pool = None  # type: ignore
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
        db_type: DatabaseType | None = None,
        connection_string: str | None = None,
        sqlite_path: str | None = None,
        json_path: str | None = None,
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
        default_sqlite = settings.sqlite_path or "data/tool_performance.db"
        default_json = settings.json_storage_path or "data/tool_performance.json"
        self.sqlite_path = sqlite_path or default_sqlite
        self.json_path = json_path or default_json

        self.connection_pool: Any = None  # psycopg2 pool or None
        self.sqlite_connection: sqlite3.Connection | None = None
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
        if POSTGRES_AVAILABLE and psycopg2 and self.connection_string:
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
        if POSTGRES_AVAILABLE and psycopg2 and postgres_url:
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
        if not POSTGRES_AVAILABLE or not pool:
            raise RuntimeError("PostgreSQL not available")
        try:
            self.connection_pool = pool.SimpleConnectionPool(
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
                with open(self.json_path, encoding='utf-8') as f:
                    self.json_data = json.load(f)
            else:
                self.json_data = {
                    "tool_performance_history": [],
                    "tool_performance_stats": [],
                    "ml_weights_history": [],
                    "tool_recommendations_log": [],
                    "historical_scores": [],
                    "backtest_runs": [],
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

        # Load main schema
        schema_file = Path(__file__).parent / "tool_performance_history.sql"
        if schema_file.exists():
            try:
                with open(schema_file, encoding='utf-8') as f:
                    schema_sql = f.read()

                conn = self.connection_pool.getconn()
                cursor = conn.cursor()
                cursor.execute(schema_sql)
                conn.commit()
                cursor.close()
                self.connection_pool.putconn(conn)

                logger.info("✅ PostgreSQL tool_performance schema created")
            except Exception as e:
                logger.warning(f"⚠️ Failed to create tool_performance schema: {e}")

        # Load historical schema
        historical_schema_file = Path(__file__).parent / "historical_schemas.sql"
        if historical_schema_file.exists():
            try:
                with open(historical_schema_file, encoding='utf-8') as f:
                    schema_sql = f.read()

                conn = self.connection_pool.getconn()
                cursor = conn.cursor()
                cursor.execute(schema_sql)
                conn.commit()
                cursor.close()
                self.connection_pool.putconn(conn)

                logger.info("✅ PostgreSQL historical schema created")
            except Exception as e:
                logger.warning(f"⚠️ Failed to create historical schema: {e}")
        else:
            logger.warning(f"⚠️ Historical schema file not found: {historical_schema_file}")

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

        -- Historical Scores (Hybrid Architecture)
        CREATE TABLE IF NOT EXISTS historical_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            timeframe TEXT NOT NULL,

            trend_score REAL DEFAULT 0.0,
            trend_confidence REAL DEFAULT 0.0,
            momentum_score REAL DEFAULT 0.0,
            momentum_confidence REAL DEFAULT 0.0,
            combined_score REAL DEFAULT 0.0,
            combined_confidence REAL DEFAULT 0.0,

            trend_weight REAL DEFAULT 0.5,
            momentum_weight REAL DEFAULT 0.5,

            trend_signal TEXT DEFAULT 'NEUTRAL',
            momentum_signal TEXT DEFAULT 'NEUTRAL',
            combined_signal TEXT DEFAULT 'NEUTRAL',

            volume_score REAL DEFAULT 0.0,
            volatility_score REAL DEFAULT 0.0,
            cycle_score REAL DEFAULT 0.0,
            support_resistance_score REAL DEFAULT 0.0,

            recommendation TEXT,
            action TEXT,
            price_at_analysis REAL,

            raw_data TEXT,

            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_historical_scores_symbol
            ON historical_scores(symbol);
        CREATE INDEX IF NOT EXISTS idx_historical_scores_timeframe
            ON historical_scores(timeframe);
        CREATE INDEX IF NOT EXISTS idx_historical_scores_timestamp
            ON historical_scores(timestamp);
        CREATE INDEX IF NOT EXISTS idx_historical_scores_symbol_timeframe
            ON historical_scores(symbol, timeframe);
        CREATE INDEX IF NOT EXISTS idx_historical_scores_combined_score
            ON historical_scores(combined_score);
        CREATE INDEX IF NOT EXISTS idx_historical_scores_created_at
            ON historical_scores(created_at);

        -- Historical Indicator Scores
        CREATE TABLE IF NOT EXISTS historical_indicator_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            score_id INTEGER,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            timeframe TEXT NOT NULL,

            indicator_name TEXT NOT NULL,
            indicator_category TEXT,
            indicator_params TEXT,

            value REAL NOT NULL,
            signal TEXT,
            confidence REAL,

            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (score_id) REFERENCES historical_scores(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_indicator_symbol_timestamp
            ON historical_indicator_scores(symbol, timestamp);
        CREATE INDEX IF NOT EXISTS idx_indicator_name
            ON historical_indicator_scores(indicator_name);
        CREATE INDEX IF NOT EXISTS idx_indicator_category
            ON historical_indicator_scores(indicator_category);
        CREATE INDEX IF NOT EXISTS idx_indicator_score_id
            ON historical_indicator_scores(score_id);

        -- Backtest runs (summary-level)
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            source TEXT NOT NULL,
            interval TEXT,
            params TEXT,
            metrics TEXT,
            period_start TEXT,
            period_end TEXT,
            model_version TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """

        try:
            if self.sqlite_connection is None:
                raise RuntimeError("SQLite connection not initialized")
            cursor = self.sqlite_connection.cursor()
            cursor.executescript(schema)
            self.sqlite_connection.commit()
            logger.info("✅ SQLite schema created successfully")
        except Exception as e:
            logger.error(f"❌ Failed to create SQLite schema: {e}")


    def save_backtest_run(
        self,
        *,
        symbol: str,
        source: str,
        interval: str | None,
        params: dict[str, Any] | None,
        metrics: dict[str, Any] | None,
        period_start: datetime | None,
        period_end: datetime | None,
        model_version: str | None = None,
    ) -> int | None:
        """
        Persist a backtest summary row.

        Returns the inserted id when available (None for JSON fallback).
        """
        params_json = json.dumps(params or {})
        metrics_json = json.dumps(metrics or {})
        start_ts = period_start.isoformat() if period_start else None
        end_ts = period_end.isoformat() if period_end else None

        if self.db_type == DatabaseType.POSTGRESQL:
            query = """
                INSERT INTO backtest_runs
                    (symbol, source, interval, params, metrics, period_start, period_end, model_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """
            conn = self.connection_pool.getconn()
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        query,
                        (
                            symbol,
                            source,
                            interval,
                            params_json,
                            metrics_json,
                            start_ts,
                            end_ts,
                            model_version,
                        ),
                    )
                    inserted_id = cursor.fetchone()[0]
                    conn.commit()
                    return inserted_id
            finally:
                self.connection_pool.putconn(conn)

        if self.db_type == DatabaseType.SQLITE:
            query = """
                INSERT INTO backtest_runs
                    (symbol, source, interval, params, metrics, period_start, period_end, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            if self.sqlite_connection is None:
                raise RuntimeError("SQLite connection not initialized")
            cursor = self.sqlite_connection.cursor()
            cursor.execute(
                query,
                (
                    symbol,
                    source,
                    interval,
                    params_json,
                    metrics_json,
                    start_ts,
                    end_ts,
                    model_version,
                ),
            )
            self.sqlite_connection.commit()
            return cursor.lastrowid

        if self.db_type == DatabaseType.JSON_FILE:
            record = {
                "id": len(self.json_data.get("backtest_runs", [])) + 1,
                "symbol": symbol,
                "source": source,
                "interval": interval,
                "params": params or {},
                "metrics": metrics or {},
                "period_start": start_ts,
                "period_end": end_ts,
                "model_version": model_version,
                "created_at": datetime.utcnow().isoformat(),
            }
            self.json_data.setdefault("backtest_runs", []).append(record)
            self._save_json()
            return record["id"]

        return None

    def get_connection(self):
        """دریافت connection (برای PostgreSQL یا SQLite)"""

        if self.db_type == DatabaseType.POSTGRESQL:
            if self.connection_pool is None:
                raise RuntimeError("PostgreSQL connection pool not initialized")
            return self.connection_pool.getconn()
        elif self.db_type == DatabaseType.SQLITE:
            if self.sqlite_connection is None:
                raise RuntimeError("SQLite connection not initialized")
            return self.sqlite_connection
        else:
            raise ValueError(f"get_connection not supported for {self.db_type}")

    def release_connection(self, conn):
        """آزاد کردن connection (فقط برای PostgreSQL)"""

        if self.db_type == DatabaseType.POSTGRESQL:
            self.connection_pool.putconn(conn)

    def get_sql_placeholder(self) -> str:
        """دریافت placeholder مناسب برای پارامترهای SQL"""
        if self.db_type == DatabaseType.POSTGRESQL:
            return "%s"
        if self.db_type == DatabaseType.SQLITE:
            return "?"
        raise ValueError("SQL placeholders not supported for JSON storage")

    def execute_query(
        self,
        query: str,
        params: tuple | None = None,
        fetch: bool = False
    ) -> list[dict[str, Any]] | None:
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

        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            result: list[dict[str, Any]] | None = None
            if fetch:
                rows = cursor.fetchall()
                if self.db_type == DatabaseType.POSTGRESQL:
                    columns = [desc[0] for desc in cursor.description or []]
                    result = [
                        dict(zip(columns, row, strict=True))
                        for row in rows
                    ]
                else:  # SQLite
                    result = [dict(row) for row in rows]

            if conn:
                conn.commit()

            return result

        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            logger.error(f"❌ Query execution failed: {e}")
            raise
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn and self.db_type == DatabaseType.POSTGRESQL:
                try:
                    self.release_connection(conn)
                except Exception:
                    pass

    def record_tool_performance(
        self,
        tool_name: str,
        tool_category: str,
        symbol: str,
        timeframe: str,
        market_regime: str,
        prediction_type: str,
        confidence_score: float,
        volatility_level: float | None = None,
        trend_strength: float | None = None,
        volume_profile: str | None = None,
        metadata: dict | None = None
    ) -> int:
        """
        ثبت عملکرد ابزار

        Returns:
            ID رکورد ذخیره شده
        """

        if self.db_type == DatabaseType.JSON_FILE:
            return self._record_tool_performance_json(
                tool_name, tool_category, symbol, timeframe, market_regime,
                prediction_type, confidence_score, volatility_level,
                trend_strength, volume_profile, metadata
            )

        query = """
        INSERT INTO tool_performance_history (
            tool_name, tool_category, symbol, timeframe, market_regime,
            volatility_level, trend_strength, volume_profile,
            prediction_type, confidence_score, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """ if self.db_type == DatabaseType.POSTGRESQL else """
        INSERT INTO tool_performance_history (
            tool_name, tool_category, symbol, timeframe, market_regime,
            volatility_level, trend_strength, volume_profile,
            prediction_type, confidence_score, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        metadata_json = json.dumps(metadata) if metadata else None

        params = (
            tool_name, tool_category, symbol, timeframe, market_regime,
            volatility_level, trend_strength, volume_profile,
            prediction_type, confidence_score, metadata_json
        )

        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)

            if self.db_type == DatabaseType.POSTGRESQL:
                result = cursor.fetchone()
                if result is None:
                    raise RuntimeError("Failed to get inserted record ID")
                record_id = result[0]
            else:  # SQLite
                record_id = cursor.lastrowid
                if record_id is None:
                    raise RuntimeError("Failed to get inserted record ID")

            if not isinstance(record_id, int):
                raise RuntimeError(f"Invalid record ID type: {type(record_id)}")

            conn.commit()
            cursor.close()
            self.release_connection(conn)

            return record_id

        except Exception as e:
            logger.error(f"❌ Failed to record performance: {e}")
            raise

    def _record_tool_performance_json(
        self, tool_name, tool_category, symbol, timeframe, market_regime,
        prediction_type, confidence_score, volatility_level,
        trend_strength, volume_profile, metadata
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
            "confidence_score": confidence_score,
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
        market_regime: str | None = None,
        days: int = 30
    ) -> dict[str, Any]:
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
          AND prediction_timestamp >= NOW() - make_interval(days => %s)
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
            row = results[0]
            if isinstance(row, dict):
                return {
                    "tool_name": row.get("tool_name"),
                    "market_regime": row.get("market_regime"),
                    "total_predictions": row.get("total_predictions", 0),
                    "correct_predictions": row.get("correct_predictions", 0),
                    "accuracy": row.get("accuracy", 0.0),
                    "avg_confidence": row.get("avg_confidence", 0.0)
                }
            # Fallback for unexpected cursor return types
            return {
                "tool_name": row[0] if len(row) > 0 else tool_name,
                "market_regime": row[1] if len(row) > 1 else market_regime,
                "total_predictions": row[2] if len(row) > 2 else 0,
                "correct_predictions": row[3] if len(row) > 3 else 0,
                "accuracy": row[4] if len(row) > 4 else 0.0,
                "avg_confidence": row[5] if len(row) > 5 else 0.0
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
        self, tool_name: str, market_regime: str | None, days: int
    ) -> dict[str, Any]:
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

    def _check_database_exists(self) -> bool:
        """بررسی وجود دیتابیس"""
        if self.db_type == DatabaseType.SQLITE:
            return Path(self.sqlite_path).exists()
        if self.db_type == DatabaseType.JSON_FILE:
            return Path(self.json_path).exists()
        if self.db_type == DatabaseType.POSTGRESQL:
            try:
                conn = self.get_connection()
                self.release_connection(conn)
                return True
            except Exception:
                return False
        return False

    def setup_database(self):
        """ساخت دیتابیس و جداول (alias for setup_schema)"""
        self.setup_schema()

    def get_database_info(self) -> dict[str, Any]:
        """دریافت اطلاعات دیتابیس"""
        if self.db_type is None:
            return {}

        info: dict[str, Any] = {
            "type": self.db_type.value,
            "connected": self._check_database_exists()
        }

        if self.db_type == DatabaseType.SQLITE:
            info["path"] = self.sqlite_path
            if Path(self.sqlite_path).exists():
                info["size"] = Path(self.sqlite_path).stat().st_size
        elif self.db_type == DatabaseType.JSON_FILE:
            info["path"] = self.json_path
        elif self.db_type == DatabaseType.POSTGRESQL:
            info["connection"] = self.connection_string

        try:
            tables = self.get_tables()
            info["table_count"] = len(tables)
        except Exception:
            info["table_count"] = 0

        return info

    def get_tables(self) -> list[str]:
        """دریافت لیست جداول"""
        if self.db_type == DatabaseType.JSON_FILE:
            return list(self.json_data.keys())

        if self.db_type == DatabaseType.SQLITE:
            query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        else:  # PostgreSQL
            query = "SELECT tablename FROM pg_tables WHERE schemaname='public'"

        try:
            results = self.execute_query(query, fetch=True)
            if results:
                return [r["name" if self.db_type == DatabaseType.SQLITE else "tablename"] for r in results]
        except Exception:
            pass

        return []

    def get_table_schema(self, table_name: str) -> list[dict[str, Any]]:
        """دریافت schema یک جدول"""
        if self.db_type == DatabaseType.JSON_FILE:
            return [{"name": "data", "type": "json"}]

        if self.db_type == DatabaseType.SQLITE:
            query = f"PRAGMA table_info({table_name})"
            results = self.execute_query(query, fetch=True)
            if results:
                return [
                    {
                        "name": r["name"],
                        "type": r["type"],
                        "nullable": not r["notnull"],
                        "default": r["dflt_value"],
                    }
                    for r in results
                ]
        else:  # PostgreSQL
            query = (
                """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
            """
            )
            results = self.execute_query(query, (table_name,), fetch=True)
            if results:
                return [
                    {
                        "name": r.get("column_name"),
                        "type": r.get("data_type"),
                        "nullable": (r.get("is_nullable") == "YES"),
                        "default": r.get("column_default"),
                    }
                    for r in results
                ]

        return []

    def get_statistics(self) -> dict[str, int]:
        """آمار تعداد رکوردهای هر جدول"""
        stats: dict[str, int] = {}

        if self.db_type == DatabaseType.JSON_FILE:
            for table, records in self.json_data.items():
                if isinstance(records, list):
                    stats[table] = len(records)
                elif isinstance(records, dict):
                    stats[table] = len(records.keys())
                else:
                    stats[table] = 0
            return stats

        try:
            tables = self.get_tables()
            for table in tables:
                query = f"SELECT COUNT(*) FROM {table}"
                result = self.execute_query(query, fetch=True)
                if result:
                    if isinstance(result[0], dict):
                        count = (
                            result[0].get("COUNT(*)")
                            or result[0].get("count")
                            or next(iter(result[0].values()), 0)
                        )
                    else:
                        count = result[0][0] if result[0] else 0
                    stats[table] = int(count)
        except Exception as e:
            logger.warning(f"⚠️ Could not get statistics: {e}")
        return stats

    def reset_table(self, table_name: str) -> None:
        """حذف تمام رکوردهای یک جدول"""
        try:
            if self.db_type == DatabaseType.JSON_FILE:
                self.json_data[table_name] = []
                self._save_json()
                logger.info(f"✅ JSON table {table_name} reset successfully")
                return

            query = f"DELETE FROM {table_name}"
            self.execute_query(query)
            logger.info(f"✅ Table {table_name} reset successfully")
        except Exception as e:
            logger.error(f"❌ Failed to reset table {table_name}: {e}")
            raise

    def run_migrations(self) -> list[str]:
        """اجرای migration (placeholder)"""
        logger.info("✅ No pending migrations")
        return []

    def create_backup(self, tables: list[str] | None = None) -> dict[str, Any]:
        """ایجاد backup از دیتابیس"""
        backup_data: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "database_type": self.db_type.value if self.db_type else None,
            "data": {},
        }

        try:
            table_list = tables if tables else self.get_tables()

            if self.db_type == DatabaseType.JSON_FILE:
                for table in table_list:
                    backup_data["data"][table] = json.loads(
                        json.dumps(self.json_data.get(table, []), default=str)
                    )
                logger.info("✅ Backup created successfully (JSON storage)")
                return backup_data

            for table in table_list:
                backup_data["data"][table] = list(self.stream_table_records(table))
            logger.info("✅ Backup created successfully")
        except Exception as e:
            logger.error(f"❌ Failed to create backup: {e}")
            raise

        return backup_data

    def restore_backup(self, backup_data: dict[str, Any]) -> dict[str, int]:
        """بازیابی دیتابیس از backup"""
        result: dict[str, int] = {}

        try:
            data = backup_data.get("data", {})
            for table, records in data.items():
                self.reset_table(table)
                if records:
                    self.bulk_insert(table, records)
                result[table] = len(records)
            logger.info("✅ Backup restored successfully")
        except Exception as e:
            logger.error(f"❌ Failed to restore backup: {e}")
            raise

        return result

    def stream_table_records(
        self,
        table_name: str,
        chunk_size: int = 1000
    ) -> Iterator[dict[str, Any]]:
        """بازگرداندن رکوردهای یک جدول به صورت chunk برای کاهش مصرف حافظه"""

        if self.db_type == DatabaseType.JSON_FILE:
            table_data = self.json_data.get(table_name, [])
            if isinstance(table_data, list):
                yield from table_data
            elif isinstance(table_data, dict):
                for key, value in table_data.items():
                    yield {"key": key, "value": value}
            return

        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table_name}")
            columns = [desc[0] for desc in cursor.description or []]

            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break

                if self.db_type == DatabaseType.POSTGRESQL:
                    for row in rows:
                        yield dict(zip(columns, row, strict=True))
                else:
                    for row in rows:
                        yield dict(row)
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn and self.db_type == DatabaseType.POSTGRESQL:
                try:
                    self.release_connection(conn)
                except Exception:
                    pass

    def bulk_insert(self, table_name: str, records: list[dict[str, Any]]) -> None:
        """درج چندین رکورد"""
        if not records:
            return

        try:
            if self.db_type == DatabaseType.JSON_FILE:
                table = self.json_data.setdefault(table_name, [])
                if not isinstance(table, list):
                    raise ValueError(f"Table {table_name} is not list-based in JSON storage")
                table.extend(records)
                self._save_json()
                logger.info(f"✅ {len(records)} records inserted into JSON table {table_name}")
                return

            for record in records:
                columns = ", ".join(record.keys())
                placeholders = ", ".join(
                    ["%s" if self.db_type == DatabaseType.POSTGRESQL else "?"] * len(record)
                )
                query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                self.execute_query(query, tuple(record.values()))
            logger.info(f"✅ {len(records)} records inserted into {table_name}")
        except Exception as e:
            logger.error(f"❌ Failed to insert records: {e}")
            raise

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
            print("\n✅ Database setup complete!")
            if db.db_type is not None:
                print(f"   Type: {db.db_type.value}")
            else:
                print("   Type: Unknown")

            if db.db_type == DatabaseType.POSTGRESQL:
                print("   Connection: PostgreSQL")
            elif db.db_type == DatabaseType.SQLITE:
                print(f"   Path: {db.sqlite_path}")
            elif db.db_type == DatabaseType.JSON_FILE:
                print(f"   Path: {db.json_path}")

            print("\n✅ Schema created successfully")
            print("   Tables: tool_performance_history, tool_performance_stats,")
            print("           ml_weights_history, tool_recommendations_log")

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return 1

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
