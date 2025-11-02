"""
Historical Score Manager
========================

این ماژول برای ذخیره و بازیابی امتیازهای تاریخی استفاده می‌شود.

هر بار که تحلیل انجام می‌شود، نتایج در دیتابیس ذخیره می‌شوند تا:
1. کاربر بتواند امتیاز هر تاریخی را ببیند
2. نمودارهای تاریخی ترسیم شوند
3. Backtesting انجام شود
4. مقایسه عملکرد اندیکاتورها
"""

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
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
    
    def __init__(self, connection_string: str):
        """
        Args:
            connection_string: مثل "postgresql://user:pass@localhost:5432/dbname"
        """
        self.connection_string = connection_string
        self._connection = None
    
    def connect(self):
        """اتصال به دیتابیس"""
        if self._connection is None or self._connection.closed:
            self._connection = psycopg2.connect(
                self.connection_string,
                cursor_factory=RealDictCursor
            )
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
                INSERT INTO historical_scores (
                    symbol, timestamp, timeframe,
                    trend_score, trend_confidence,
                    momentum_score, momentum_confidence,
                    combined_score, combined_confidence,
                    trend_weight, momentum_weight,
                    trend_signal, momentum_signal, combined_signal,
                    recommendation, action,
                    price_at_analysis
                ) VALUES (
                    %(symbol)s, %(timestamp)s, %(timeframe)s,
                    %(trend_score)s, %(trend_confidence)s,
                    %(momentum_score)s, %(momentum_confidence)s,
                    %(combined_score)s, %(combined_confidence)s,
                    %(trend_weight)s, %(momentum_weight)s,
                    %(trend_signal)s, %(momentum_signal)s, %(combined_signal)s,
                    %(recommendation)s, %(action)s,
                    %(price_at_analysis)s
                )
                ON CONFLICT (symbol, timestamp, timeframe) 
                DO UPDATE SET
                    trend_score = EXCLUDED.trend_score,
                    trend_confidence = EXCLUDED.trend_confidence,
                    momentum_score = EXCLUDED.momentum_score,
                    momentum_confidence = EXCLUDED.momentum_confidence,
                    combined_score = EXCLUDED.combined_score,
                    combined_confidence = EXCLUDED.combined_confidence
                RETURNING id
            """, asdict(score_entry))
            
            score_id = cursor.fetchone()['id']
            
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
            self._update_metadata(cursor, score_entry.symbol, score_entry.timeframe, score_id)
            
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
