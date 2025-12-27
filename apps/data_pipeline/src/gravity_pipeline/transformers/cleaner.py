"""
Data cleaner transformer

Handles missing values, outliers, and data normalization.
"""

import math
from typing import Any, Dict, List, Optional

import structlog

from gravity_pipeline.transformers.base import Transformer

logger = structlog.get_logger()


class DataCleaner(Transformer):
    """Clean and normalize OHLCV data"""
    
    def __init__(
        self,
        remove_outliers: bool = True,
        fill_missing_with: str = "previous",  # "previous", "next", "skip", "zero"
        outlier_std_dev: float = 3.0,
    ):
        """
        Initialize cleaner
        
        Args:
            remove_outliers: Remove price outliers beyond N std devs
            fill_missing_with: How to handle missing values
            outlier_std_dev: Number of standard deviations for outlier detection
        """
        super().__init__()
        self.remove_outliers = remove_outliers
        self.fill_missing_with = fill_missing_with
        self.outlier_std_dev = outlier_std_dev
    
    async def transform(self, candles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean candle data
        
        Steps:
        1. Normalize field names
        2. Convert types
        3. Handle missing values
        4. Remove outliers
        5. Validate OHLC relationships
        """
        
        logger.info("cleaner_starting", count=len(candles))
        
        cleaned = []
        
        for candle in candles:
            try:
                # Step 1: Normalize field names and types
                normalized = {
                    "timestamp": str(candle.get("timestamp") or ""),
                    "open": self._to_float(candle.get("open"), "open"),
                    "high": self._to_float(candle.get("high"), "high"),
                    "low": self._to_float(candle.get("low"), "low"),
                    "close": self._to_float(candle.get("close"), "close"),
                    "volume": self._to_float(candle.get("volume"), "volume"),
                }
                
                # Step 2: Handle missing values
                if self._has_missing_values(normalized):
                    if self.fill_missing_with == "skip":
                        logger.warning("candle_skipped_missing", candle=normalized)
                        continue
                    elif self.fill_missing_with == "zero":
                        normalized = self._fill_missing_zero(normalized)
                    # else: "previous" and "next" handled after all candles
                
                # Step 3: Validate OHLC relationships
                if not self._validate_ohlc(normalized):
                    logger.warning("invalid_ohlc", candle=normalized)
                    continue
                
                # Step 4: Check for outliers (if removing)
                if self.remove_outliers and self._is_outlier(normalized):
                    logger.warning("outlier_detected", candle=normalized)
                    if self.fill_missing_with == "skip":
                        continue
                
                cleaned.append(normalized)
                self.processed_count += 1
            
            except Exception as e:
                self.error_count += 1
                logger.warning("cleaning_error", error=str(e), candle=candle)
                continue
        
        # Handle "previous" and "next" filling after all candles
        if self.fill_missing_with in ["previous", "next"]:
            cleaned = self._fill_missing_forward_backward(cleaned)
        
        logger.info("cleaner_complete", output=len(cleaned), errors=self.error_count)
        return cleaned
    
    def _to_float(self, value: Any, field_name: str) -> Optional[float]:
        """Convert value to float, return None if invalid"""
        if value is None:
            return None
        
        try:
            f = float(value)
            if math.isnan(f) or math.isinf(f):
                logger.warning("invalid_number", field=field_name, value=value)
                return None
            return f
        except (ValueError, TypeError):
            logger.warning("conversion_error", field=field_name, value=value)
            return None
    
    def _has_missing_values(self, candle: Dict) -> bool:
        """Check if candle has any None values"""
        required_fields = ["open", "high", "low", "close", "volume"]
        return any(candle.get(field) is None for field in required_fields)
    
    def _validate_ohlc(self, candle: Dict) -> bool:
        """Validate OHLC relationships"""
        o, h, l, c, v = (
            candle.get("open"),
            candle.get("high"),
            candle.get("low"),
            candle.get("close"),
            candle.get("volume"),
        )
        
        # Check if any are missing
        if any(x is None for x in [o, h, l, c]):
            return False
        
        # High >= Low
        if h < l:
            logger.debug("invalid_high_low", high=h, low=l)
            return False
        
        # Volume >= 0
        if v is not None and v < 0:
            logger.debug("negative_volume", volume=v)
            return False
        
        # Open and Close should be between Low and High (relaxed)
        if o is not None and (o < l * 0.95 or o > h * 1.05):
            logger.debug("open_outside_range", open=o, high=h, low=l)
            # Don't reject, just log
        
        if c is not None and (c < l * 0.95 or c > h * 1.05):
            logger.debug("close_outside_range", close=c, high=h, low=l)
            # Don't reject, just log
        
        return True
    
    def _is_outlier(self, candle: Dict) -> bool:
        """Check if candle is statistical outlier (simplified)"""
        # For now, just check for extreme price changes
        o = candle.get("open")
        c = candle.get("close")
        
        if o is None or c is None or o == 0:
            return False
        
        # Check if price change > 50% in single candle
        pct_change = abs((c - o) / o) * 100
        if pct_change > 50:
            logger.debug("large_price_change", pct=pct_change)
            return True
        
        return False
    
    def _fill_missing_zero(self, candle: Dict) -> Dict:
        """Fill missing OHLCV with zeros"""
        for field in ["open", "high", "low", "close", "volume"]:
            if candle.get(field) is None:
                candle[field] = 0.0
        return candle
    
    def _fill_missing_forward_backward(self, candles: List[Dict]) -> List[Dict]:
        """
        Fill missing values using forward/backward fill
        
        For "previous": use previous candle's value
        For "next": use next candle's value
        """
        if not candles or self.fill_missing_with not in ["previous", "next"]:
            return candles
        
        filled = candles.copy()
        
        for i, candle in enumerate(filled):
            for field in ["open", "high", "low", "close", "volume"]:
                if candle.get(field) is None:
                    if self.fill_missing_with == "previous" and i > 0:
                        candle[field] = filled[i - 1].get(field)
                    elif self.fill_missing_with == "next" and i < len(filled) - 1:
                        candle[field] = filled[i + 1].get(field)
                    else:
                        candle[field] = 0.0  # Fallback to zero
        
        return filled
