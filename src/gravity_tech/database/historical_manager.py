"""
Historical Score Manager

این ماژول برای ذخیره و بازیابی امتیازهای تاریخی استفاده می‌شود.

هر بار که تحلیل انجام می‌شود، نتایج در دیتابیس ذخیره می‌شوند تا:
1. کاربر بتواند امتیاز هر تاریخی را ببیند
2. نمودارهای تاریخی ترسیم شوند
3. Backtesting انجام شود

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

from gravity_tech.config.settings import settings
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Try to import psycopg2
try:
    import psycopg2
    from psycopg2 import pool
    from psycopg2.extras import RealDictCursor, execute_values
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    RealDictCursor = None
    execute_values = None
    print("⚠️ psycopg2 not available. Will use SQLite fallback.")

import json


@dataclass
class HistoricalScoreEntry:
    """
    ورودی کامل یک تحلیل برای ذخیره در دیتابیس
    """
    # شناسایی
    symbol: str
    timestamp: datetime
    timeframe: str
    
    # امتیازهای کلی
    trend_score: float
    trend_confidence: float
    momentum_score: float
    momentum_confidence: float
    combined_score: float
    combined_confidence: float
    
    # وزن‌ها
    trend_weight: float
    momentum_weight: float
    
    # سیگنال‌ها
    trend_signal: str
    momentum_signal: str
    combined_signal: str
    
    # توصیه
    recommendation: str
    action: str
    
    # قیمت
    price_at_analysis: float
    
    # اختیاری
    id: Optional[int] = None
    created_at: Optional[datetime] = None


class HistoricalScoreManager:
    """
    مدیریت ذخیره و بازیابی امتیازهای تاریخی
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Args:
            connection_string: مثل "postgresql://user:pass@localhost:5432/dbname"
                             اگر None باشد، از settings استفاده می‌شود
        """
        self.connection_string = connection_string or settings.database_url
        self._connection = None
    
    def connect(self):
        """اتصال به دیتابیس"""
        if self._connection is None:
            if HAS_PSYCOPG2 and self.connection_string.startswith('postgresql://'):
                try:
                    import psycopg2.extras
                    self._connection = psycopg2.connect(
                        self.connection_string,
                        cursor_factory=psycopg2.extras.RealDictCursor
                    )
                except Exception as e:
                    print(f"⚠️ PostgreSQL connection failed: {e}")
                    print("🔄 Falling back to SQLite...")
                    import sqlite3
                    self._connection = sqlite3.connect('data/tool_performance.db')
            else:
                # SQLite fallback
                import sqlite3
                self._connection = sqlite3.connect('data/tool_performance.db')
        return self._connection
    
    def close(self):
        """بستن اتصال"""
        if self._connection and not self._connection.closed:
            self._connection.close()
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # ═══════════════════════════════════════════════════════════════════
    # ذخیره امتیازها (Save)
    # ═══════════════════════════════════════════════════════════════════
    
    def save_score(
        self,
        score_entry: HistoricalScoreEntry,
        horizon_scores: Optional[List[Dict]] = None,
        indicator_scores: Optional[List[Dict]] = None,
        patterns: Optional[List[Dict]] = None,
        volume_analysis: Optional[Dict] = None,
        price_targets: Optional[List[Dict]] = None
    ) -> int:
        """
        ذخیره کامل یک تحلیل
        
        Args:
            score_entry: امتیازهای اصلی
            horizon_scores: لیست امتیازهای multi-horizon
            indicator_scores: لیست امتیازهای تک تک اندیکاتورها
            patterns: لیست الگوهای تشخیص داده شده
            volume_analysis: تحلیل حجم
            price_targets: اهداف قیمتی
            
        Returns:
            score_id: شناسه رکورد ذخیره شده
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # 1. ذخیره امتیاز اصلی
            cursor.execute("""
                INSERT OR REPLACE INTO historical_scores (
                    symbol, timestamp, timeframe,
                    trend_score, trend_confidence,
                    momentum_score, momentum_confidence,
                    combined_score, combined_confidence,
                    trend_weight, momentum_weight,
                    trend_signal, momentum_signal, combined_signal,
                    volume_score, volatility_score, cycle_score, support_resistance_score,
                    recommendation, action,
                    price_at_analysis, raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                score_entry.symbol, score_entry.timestamp.isoformat(), score_entry.timeframe,
                score_entry.trend_score, score_entry.trend_confidence,
                score_entry.momentum_score, score_entry.momentum_confidence,
                score_entry.combined_score, score_entry.combined_confidence,
                score_entry.trend_weight, score_entry.momentum_weight,
                score_entry.trend_signal, score_entry.momentum_signal, score_entry.combined_signal,
                0.0, 0.0, 0.0, 0.0,  # volume_score, volatility_score, cycle_score, support_resistance_score
                score_entry.recommendation, score_entry.action,
                score_entry.price_at_analysis, None  # raw_data
            ))
            
            score_id = cursor.lastrowid
            
            # 2. ذخیره horizon scores
            if horizon_scores:
                self._save_horizon_scores(cursor, score_id, horizon_scores)
            
            # 3. ذخیره indicator scores
            if indicator_scores:
                self._save_indicator_scores(cursor, score_id, indicator_scores)
            
            # 4. ذخیره patterns
            if patterns:
                self._save_patterns(cursor, score_id, patterns)
            
            # 5. ذخیره volume analysis
            if volume_analysis:
                self._save_volume_analysis(cursor, score_id, volume_analysis)
            
            # 6. ذخیره price targets
            if price_targets:
                self._save_price_targets(cursor, score_id, price_targets)
            
            # 7. به‌روزرسانی metadata
            # self._update_metadata(cursor, score_entry.symbol, score_entry.timeframe, score_id)
            
            conn.commit()
            return score_id
            
        except Exception as e:
            conn.rollback()
            raise Exception(f"Error saving score: {str(e)}")
        finally:
            cursor.close()
    
    def _save_horizon_scores(self, cursor, score_id: int, horizon_scores: List[Dict]):
        """ذخیره امتیازهای multi-horizon"""
        data = [
            (
                score_id,
                h['horizon'],
                h['analysis_type'],
                h['score'],
                h['confidence'],
                h['signal']
            )
            for h in horizon_scores
        ]
        
        execute_values(
            cursor,
            """
            INSERT INTO historical_horizon_scores 
                (score_id, horizon, analysis_type, score, confidence, signal)
            VALUES %s
            ON CONFLICT (score_id, horizon, analysis_type) 
            DO UPDATE SET score = EXCLUDED.score, confidence = EXCLUDED.confidence
            """,
            data
        )
    
    def _save_indicator_scores(self, cursor, score_id: int, indicator_scores: List[Dict]):
        """ذخیره امتیازهای تک تک اندیکاتورها"""
        data = [
            (
                score_id,
                ind['name'],
                ind['category'],
                json.dumps(ind.get('params', {})),
                ind['score'],
                ind['confidence'],
                ind['signal'],
                ind.get('raw_value')
            )
            for ind in indicator_scores
        ]
        
        execute_values(
            cursor,
            """
            INSERT INTO historical_indicator_scores 
                (score_id, indicator_name, indicator_category, indicator_params,
                 score, confidence, signal, raw_value)
            VALUES %s
            """,
            data
        )
    
    def _save_patterns(self, cursor, score_id: int, patterns: List[Dict]):
        """ذخیره الگوهای تشخیص داده شده"""
        data = [
            (
                score_id,
                p['type'],
                p['name'],
                p['score'],
                p['confidence'],
                p['signal'],
                p.get('description'),
                json.dumps(p.get('candle_indices', [])),
                json.dumps(p.get('price_levels', {})),
                p.get('projected_target')
            )
            for p in patterns
        ]
        
        execute_values(
            cursor,
            """
            INSERT INTO historical_patterns 
                (score_id, pattern_type, pattern_name, score, confidence, signal,
                 description, candle_indices, price_levels, projected_target)
            VALUES %s
            """,
            data
        )
    
    def _save_volume_analysis(self, cursor, score_id: int, volume: Dict):
        """ذخیره تحلیل حجم"""
        cursor.execute("""
            INSERT INTO historical_volume_analysis 
                (score_id, volume_score, volume_confidence, avg_volume, 
                 current_volume, volume_ratio, confirms_trend)
            VALUES 
                (%(score_id)s, %(volume_score)s, %(volume_confidence)s, %(avg_volume)s,
                 %(current_volume)s, %(volume_ratio)s, %(confirms_trend)s)
            ON CONFLICT (score_id) 
            DO UPDATE SET 
                volume_score = EXCLUDED.volume_score,
                volume_confidence = EXCLUDED.volume_confidence
        """, {
            'score_id': score_id,
            **volume
        })
    
    def _save_price_targets(self, cursor, score_id: int, targets: List[Dict]):
        """ذخیره اهداف قیمتی"""
        data = [
            (
                score_id,
                t['target_type'],
                t['target_price'],
                t.get('stop_loss'),
                t.get('expected_timeframe'),
                t.get('confidence')
            )
            for t in targets
        ]
        
        execute_values(
            cursor,
            """
            INSERT INTO historical_price_targets 
                (score_id, target_type, target_price, stop_loss, expected_timeframe, confidence)
            VALUES %s
            """,
            data
        )
    
    def _update_metadata(self, cursor, symbol: str, timeframe: str, score_id: int):
        """به‌روزرسانی metadata"""
        # Check if we're using PostgreSQL or SQLite
        is_postgres = hasattr(cursor.connection, 'server_version') or str(type(cursor.connection)).find('psycopg2') >= 0
        
        if is_postgres:
            cursor.execute("""
                INSERT INTO analysis_metadata (symbol, timeframe, last_analysis_at, last_score_id, total_analyses)
                VALUES (%(symbol)s, %(timeframe)s, NOW(), %(score_id)s, 1)
                ON CONFLICT (symbol, timeframe)
                DO UPDATE SET
                    last_analysis_at = NOW(),
                    last_score_id = %(score_id)s,
                    total_analyses = analysis_metadata.total_analyses + 1,
                    updated_at = NOW()
            """, {'symbol': symbol, 'timeframe': timeframe, 'score_id': score_id})
        else:
            # SQLite version
            cursor.execute("""
                INSERT OR REPLACE INTO analysis_metadata 
                (symbol, timeframe, last_analysis_at, last_score_id, total_analyses, updated_at)
                VALUES (?, ?, datetime('now'), ?, 
                    COALESCE((SELECT total_analyses + 1 FROM analysis_metadata WHERE symbol = ? AND timeframe = ?), 1),
                    datetime('now'))
            """, (symbol, timeframe, score_id, symbol, timeframe))
    
    # ═══════════════════════════════════════════════════════════════════
    # بازیابی امتیازها (Retrieve)
    # ═══════════════════════════════════════════════════════════════════
    
    def get_latest_score(self, symbol: str, timeframe: str = '1h') -> Optional[Dict]:
        """
        دریافت آخرین امتیاز یک symbol
        
        Returns:
            دیکشنری شامل تمام اطلاعات یا None
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM v_latest_scores
                WHERE symbol = %s AND timeframe = %s
            """, (symbol, timeframe))
            
            return cursor.fetchone()
        finally:
            cursor.close()
    
    def get_score_at_date(
        self,
        symbol: str,
        date: datetime,
        timeframe: str = '1h'
    ) -> Optional[Dict]:
        """
        دریافت امتیاز در یک تاریخ خاص (آخرین تحلیل قبل از آن تاریخ)
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT score_data FROM get_score_at_date(%s, %s, %s)
            """, (symbol, timeframe, date))
            
            result = cursor.fetchone()
            return result['score_data'] if result else None
        finally:
            cursor.close()
    
    def get_score_timeseries(
        self,
        symbol: str,
        from_date: datetime,
        to_date: datetime,
        timeframe: str = '1h'
    ) -> List[Dict]:
        """
        دریافت سری زمانی امتیازها (برای نمودار)
        
        Returns:
            لیست دیکشنری‌ها با timestamp, trend_score, momentum_score, ...
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM get_score_timeseries(%s, %s, %s, %s)
            """, (symbol, timeframe, from_date, to_date))
            
            return cursor.fetchall()
        finally:
            cursor.close()
    
    def get_score_with_details(self, score_id: int) -> Dict:
        """
        دریافت کامل یک تحلیل با تمام جزئیات
        (horizons, indicators, patterns, volume)
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # امتیاز اصلی
            cursor.execute("SELECT * FROM historical_scores WHERE id = %s", (score_id,))
            score = cursor.fetchone()
            
            if not score:
                return None
            
            # Horizons
            cursor.execute("""
                SELECT * FROM historical_horizon_scores WHERE score_id = %s
            """, (score_id,))
            score['horizons'] = cursor.fetchall()
            
            # Indicators
            cursor.execute("""
                SELECT * FROM historical_indicator_scores WHERE score_id = %s
            """, (score_id,))
            score['indicators'] = cursor.fetchall()
            
            # Patterns
            cursor.execute("""
                SELECT * FROM historical_patterns WHERE score_id = %s
            """, (score_id,))
            score['patterns'] = cursor.fetchall()
            
            # Volume
            cursor.execute("""
                SELECT * FROM historical_volume_analysis WHERE score_id = %s
            """, (score_id,))
            score['volume'] = cursor.fetchone()
            
            # Targets
            cursor.execute("""
                SELECT * FROM historical_price_targets WHERE score_id = %s
            """, (score_id,))
            score['targets'] = cursor.fetchall()
            
            return score
            
        finally:
            cursor.close()
    
    # ═══════════════════════════════════════════════════════════════════
    # آمار و تحلیل (Statistics)
    # ═══════════════════════════════════════════════════════════════════
    
    def get_indicator_performance(
        self,
        symbol: str,
        days: int = 30
    ) -> List[Dict]:
        """
        عملکرد اندیکاتورها در X روز گذشته
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    indicator_name,
                    indicator_category,
                    AVG(confidence) as avg_confidence,
                    STDDEV(confidence) as std_confidence,
                    COUNT(*) as usage_count,
                    AVG(score) as avg_score
                FROM historical_indicator_scores
                WHERE score_id IN (
                    SELECT id FROM historical_scores 
                    WHERE symbol = %s 
                    AND timestamp > NOW() - INTERVAL '%s days'
                )
                GROUP BY indicator_name, indicator_category
                ORDER BY avg_confidence DESC
            """, (symbol, days))
            
            return cursor.fetchall()
        finally:
            cursor.close()
    
    def get_pattern_success_rate(
        self,
        pattern_name: Optional[str] = None,
        days: int = 90
    ) -> List[Dict]:
        """
        نرخ موفقیت الگوها
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            where_clause = "WHERE hp.detected_at > NOW() - INTERVAL '%s days'"
            params = [days]
            
            if pattern_name:
                where_clause += " AND hp.pattern_name = %s"
                params.append(pattern_name)
            
            cursor.execute(f"""
                SELECT 
                    hp.pattern_name,
                    hp.pattern_type,
                    COUNT(*) as detected_count,
                    AVG(hp.confidence) as avg_confidence,
                    COUNT(CASE WHEN hpt.actual_reached THEN 1 END) as success_count,
                    COUNT(CASE WHEN hpt.actual_reached THEN 1 END)::FLOAT / COUNT(*) as success_rate
                FROM historical_patterns hp
                LEFT JOIN historical_price_targets hpt ON hp.score_id = hpt.score_id
                {where_clause}
                GROUP BY hp.pattern_name, hp.pattern_type
                ORDER BY success_rate DESC NULLS LAST
            """, params)
            
            return cursor.fetchall()
        finally:
            cursor.close()
    
    # ═══════════════════════════════════════════════════════════════════
    # Cleanup
    # ═══════════════════════════════════════════════════════════════════
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> int:
        """
        حذف داده‌های قدیمی‌تر از X روز
        
        Returns:
            تعداد رکوردهای حذف شده
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT cleanup_old_scores(%s)", (days_to_keep,))
            deleted_count = cursor.fetchone()['cleanup_old_scores']
            conn.commit()
            return deleted_count
        finally:
            cursor.close()
    
    # ═══════════════════════════════════════════════════════════════════
    # متدهای جدید برای API Historical
    # ═══════════════════════════════════════════════════════════════════
    
    def get_scores_by_symbol_timeframe(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[HistoricalScoreEntry]:
        """
        دریافت امتیازهای historical برای یک نماد و تایم‌فریم
        
        Args:
            symbol: نماد معاملاتی
            timeframe: تایم‌فریم
            start_date: تاریخ شروع
            end_date: تاریخ پایان
            limit: حداکثر تعداد نتایج
            
        Returns:
            لیست HistoricalScoreEntry
        """
        conn = self.connect()
        # Check if we're using PostgreSQL or SQLite
        is_postgres = hasattr(conn, 'server_version') or str(type(conn)).find('psycopg2') >= 0
        if is_postgres:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = conn.cursor()
        
        try:
            query = """
                SELECT * FROM historical_scores
                WHERE symbol = %s AND timeframe = %s
            """
            params = [symbol, timeframe]
            
            if start_date:
                query += " AND timestamp >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= %s"
                params.append(end_date)
            
            query += " ORDER BY timestamp DESC LIMIT %s"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                # تبدیل row به HistoricalScoreEntry
                entry = HistoricalScoreEntry(
                    symbol=row['symbol'],
                    timestamp=row['timestamp'],
                    timeframe=row['timeframe'],
                    trend_score=row['trend_score'],
                    trend_confidence=row['trend_confidence'],
                    momentum_score=row['momentum_score'],
                    momentum_confidence=row['momentum_confidence'],
                    combined_score=row['combined_score'],
                    combined_confidence=row['combined_confidence'],
                    trend_weight=row['trend_weight'],
                    momentum_weight=row['momentum_weight'],
                    trend_signal=row['trend_signal'],
                    momentum_signal=row['momentum_signal'],
                    combined_signal=row['combined_signal'],
                    recommendation=row.get('recommendation'),
                    action=row.get('action'),
                    price_at_analysis=row.get('price_at_analysis')
                )
                results.append(entry)
            
            return results
        
        finally:
            cursor.close()
            conn.close()
    
    def get_available_symbols(self) -> List[str]:
        """دریافت لیست نمادهای موجود"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT DISTINCT symbol FROM historical_scores ORDER BY symbol")
            rows = cursor.fetchall()
            return [row[0] for row in rows]
        
        finally:
            cursor.close()
            conn.close()
    
    def get_available_timeframes(self, symbol: Optional[str] = None) -> List[str]:
        """دریافت لیست تایم‌فریم‌های موجود"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            if symbol:
                cursor.execute(
                    "SELECT DISTINCT timeframe FROM historical_scores WHERE symbol = %s ORDER BY timeframe",
                    (symbol,)
                )
            else:
                cursor.execute("SELECT DISTINCT timeframe FROM historical_scores ORDER BY timeframe")
            
            rows = cursor.fetchall()
            return [row[0] for row in rows]
        
        finally:
            cursor.close()
            conn.close()
    
    def get_symbol_statistics(self, symbol: str, timeframe: Optional[str] = None) -> Dict[str, Any]:
        """دریافت آمار historical برای یک نماد"""
        conn = self.connect()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            query = """
                SELECT
                    COUNT(*) as total_records,
                    AVG(combined_score) as avg_score,
                    MIN(combined_score) as min_score,
                    MAX(combined_score) as max_score,
                    AVG(combined_confidence) as avg_confidence,
                    MIN(timestamp) as first_date,
                    MAX(timestamp) as last_date
                FROM historical_scores
                WHERE symbol = %s
            """
            params = [symbol]
            
            if timeframe:
                query += " AND timeframe = %s"
                params.append(timeframe)
            
            cursor.execute(query, params)
            result = cursor.fetchone()
            
            if result:
                return dict(result)
            else:
                return {}
        
        finally:
            cursor.close()
            conn.close()
    
    def cleanup_old_data(self, cutoff_date: datetime) -> int:
        """
        پاک کردن داده‌های قدیمی‌تر از تاریخ مشخص
        
        Args:
            cutoff_date: تاریخ cutoff
            
        Returns:
            تعداد رکوردهای حذف شده
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "DELETE FROM historical_scores WHERE timestamp < %s",
                (cutoff_date,)
            )
            deleted_count = cursor.rowcount
            conn.commit()
            
            return deleted_count
        
        finally:
            cursor.close()
            conn.close()
    
    # ═══════════════════════════════════════════════════════════════════
    # متدهای بهینه‌سازی شده برای عملکرد بهتر
    # ═══════════════════════════════════════════════════════════════════
    
    async def get_paginated_scores(
        self,
        symbol: str,
        timeframe: str,
        page: int = 1,
        page_size: int = 50,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        دریافت امتیازهای paginated با فیلتر پیشرفته
        
        Args:
            symbol: نماد معاملاتی
            timeframe: تایم‌فریم
            page: شماره صفحه (1-based)
            page_size: تعداد رکورد در هر صفحه
            start_date: فیلتر تاریخ شروع
            end_date: فیلتر تاریخ پایان
            min_score: حداقل امتیاز ترکیبی
            max_score: حداکثر امتیاز ترکیبی
            
        Returns:
            دیکشنری شامل نتایج و اطلاعات pagination
        """
        conn = self.connect()
        # بررسی نوع دیتابیس
        is_postgres = hasattr(conn, 'server_version') or str(type(conn)).find('psycopg2') >= 0
        if is_postgres:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = conn.cursor()
        
        try:
            # ساخت query با فیلترها
            query = """
                SELECT * FROM historical_scores
                WHERE symbol = ? AND timeframe = ?
            """
            params = [symbol, timeframe]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            if min_score is not None:
                query += " AND combined_score >= ?"
                params.append(min_score)
            
            if max_score is not None:
                query += " AND combined_score <= ?"
                params.append(max_score)
            
            # دریافت تعداد کل
            count_query = f"SELECT COUNT(*) as total FROM ({query})"
            cursor.execute(count_query, params)
            total_count = cursor.fetchone()[0] if is_postgres else cursor.fetchone()['total']
            
            # اضافه کردن pagination
            offset = (page - 1) * page_size
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([page_size, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # تبدیل به فرمت دیکشنری
            results = []
            for row in rows:
                if is_postgres:
                    result = dict(row)
                else:
                    # SQLite برمی‌گرداند tuple، تبدیل به dict
                    columns = [desc[0] for desc in cursor.description]
                    result = dict(zip(columns, row))
                # تبدیل timestamp string به datetime اگر لازم باشد
                if isinstance(result['timestamp'], str):
                    result['timestamp'] = datetime.fromisoformat(result['timestamp'])
                results.append(result)
            
            total_pages = (total_count + page_size - 1) // page_size
            
            return {
                "results": results,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_count": total_count,
                    "total_pages": total_pages,
                    "has_next": page < total_pages,
                    "has_prev": page > 1
                }
            }
        
        finally:
            cursor.close()
            conn.close()
    
    async def get_score_statistics(
        self,
        symbol: str,
        timeframe: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        دریافت آمار خلاصه امتیازها برای تحلیل
        
        Args:
            symbol: نماد معاملاتی
            timeframe: تایم‌فریم
            days: تعداد روزهای تحلیل
            
        Returns:
            آمار خلاصه
        """
        conn = self.connect()
        is_postgres = hasattr(conn, 'server_version') or str(type(conn)).find('psycopg2') >= 0
        if is_postgres:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = conn.cursor()
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            query = """
                SELECT
                    COUNT(*) as total_scores,
                    AVG(combined_score) as avg_score,
                    MIN(combined_score) as min_score,
                    MAX(combined_score) as max_score,
                    AVG(combined_confidence) as avg_confidence,
                    COUNT(CASE WHEN combined_score > 0 THEN 1 END) as bullish_count,
                    COUNT(CASE WHEN combined_score < 0 THEN 1 END) as bearish_count,
                    COUNT(CASE WHEN combined_score = 0 THEN 1 END) as neutral_count
                FROM historical_scores
                WHERE symbol = ? AND timeframe = ? AND timestamp >= ?
            """
            
            cursor.execute(query, [symbol, timeframe, cutoff_date])
            row = cursor.fetchone()
            
            if is_postgres:
                stats = dict(row)
            else:
                columns = [desc[0] for desc in cursor.description]
                stats = dict(zip(columns, row))
            
            # محاسبه معیارهای اضافی
            total = stats['total_scores']
            if total > 0:
                stats['bullish_percentage'] = (stats['bullish_count'] / total) * 100
                stats['bearish_percentage'] = (stats['bearish_count'] / total) * 100
                stats['neutral_percentage'] = (stats['neutral_count'] / total) * 100
                
                # قدرت روند (میانگین مطلق امتیاز)
                stats['trend_strength'] = abs(stats['avg_score'])
                
                # نوسان امتیاز (انحراف معیار بهتر است اما برای سادگی از محدوده استفاده می‌کنیم)
                stats['score_range'] = stats['max_score'] - stats['min_score']
            
            return stats
        
        finally:
            cursor.close()
            conn.close()
    
    async def get_score_trends(
        self,
        symbol: str,
        timeframe: str,
        days: int = 30,
        interval: str = "1d"
    ) -> List[Dict[str, Any]]:
        """
        دریافت روند امتیازها در طول زمان برای نمودار
        
        Args:
            symbol: نماد معاملاتی
            timeframe: تایم‌فریم تحلیل
            days: تعداد روزهای تحلیل
            interval: فاصله گروه‌بندی ('1d', '4h', '1h')
            
        Returns:
            داده‌های سری زمانی برای نمودار
        """
        conn = self.connect()
        is_postgres = hasattr(conn, 'server_version') or str(type(conn)).find('psycopg2') >= 0
        if is_postgres:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = conn.cursor()
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # ساخت گروه‌بندی تاریخ بر اساس نوع دیتابیس
            if is_postgres:
                if interval == "1d":
                    date_group = "DATE(timestamp)"
                elif interval == "4h":
                    date_group = "DATE_TRUNC('hour', timestamp) + INTERVAL '4 hours' * (EXTRACT(hour FROM timestamp)::int / 4)"
                else:  # 1h
                    date_group = "DATE_TRUNC('hour', timestamp)"
            else:
                # SQLite توابع تاریخ پیشرفته ندارد، از strftime استفاده می‌کنیم
                if interval == "1d":
                    date_group = "DATE(timestamp)"
                else:
                    date_group = "strftime('%Y-%m-%d %H:00:00', timestamp)"
            
            query = f"""
                SELECT
                    {date_group} as period,
                    AVG(combined_score) as avg_score,
                    AVG(combined_confidence) as avg_confidence,
                    COUNT(*) as sample_count,
                    MIN(combined_score) as min_score,
                    MAX(combined_score) as max_score
                FROM historical_scores
                WHERE symbol = ? AND timeframe = ? AND timestamp >= ?
                GROUP BY {date_group}
                ORDER BY period
            """
            
            cursor.execute(query, [symbol, timeframe, cutoff_date])
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                if is_postgres:
                    result = dict(row)
                else:
                    columns = [desc[0] for desc in cursor.description]
                    result = dict(zip(columns, row))
                
                # تبدیل period به datetime اگر string باشد
                if isinstance(result['period'], str):
                    try:
                        result['period'] = datetime.fromisoformat(result['period'])
                    except:
                        pass
                
                results.append(result)
            
            return results
        
        finally:
            cursor.close()
            conn.close()


# ═══════════════════════════════════════════════════════════════════
# مثال استفاده
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اتصال به دیتابیس
    manager = HistoricalScoreManager(
        "postgresql://user:pass@localhost:5432/trading_db"
    )
    
    # مثال ذخیره
    score_entry = HistoricalScoreEntry(
        symbol="BTCUSDT",
        timestamp=datetime.now(),
        timeframe="1h",
        trend_score=0.85,
        trend_confidence=0.82,
        momentum_score=0.55,
        momentum_confidence=0.70,
        combined_score=0.72,
        combined_confidence=0.76,
        trend_weight=0.6,
        momentum_weight=0.4,
        trend_signal="VERY_BULLISH",
        momentum_signal="BULLISH",
        combined_signal="BULLISH",
        recommendation="BUY",
        action="ACCUMULATE",
        price_at_analysis=50000.00
    )
    
    horizon_scores = [
        {'horizon': '3d', 'analysis_type': 'TREND', 'score': 0.85, 'confidence': 0.82, 'signal': 'VERY_BULLISH'},
        {'horizon': '7d', 'analysis_type': 'TREND', 'score': 0.75, 'confidence': 0.78, 'signal': 'BULLISH'},
        {'horizon': '30d', 'analysis_type': 'TREND', 'score': 0.60, 'confidence': 0.75, 'signal': 'BULLISH'}
    ]
    
    with manager:
        score_id = manager.save_score(score_entry, horizon_scores=horizon_scores)
        print(f"✅ Saved score with ID: {score_id}")
        
        # بازیابی
        latest = manager.get_latest_score("BTCUSDT", "1h")
        print(f"📊 Latest score: {latest}")
