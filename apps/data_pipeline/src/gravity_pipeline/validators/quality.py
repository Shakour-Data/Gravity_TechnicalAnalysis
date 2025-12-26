"""
Data quality validator

Validates OHLCV data quality and consistency.
"""

from typing import List, Dict, Any, Tuple, Optional
import structlog
import math
from gravity_pipeline.validators.base import Validator

logger = structlog.get_logger()


class DataQualityValidator(Validator):
    """Validate OHLCV data quality"""
    
    def __init__(
        self,
        check_ohlc: bool = True,
        check_volume: bool = True,
        check_timestamps: bool = True,
        check_nan_inf: bool = True,
        check_duplicates: bool = False,
    ):
        """
        Initialize validator
        
        Args:
            check_ohlc: Validate high >= low, etc
            check_volume: Check volume >= 0
            check_timestamps: Check timestamp ordering
            check_nan_inf: Check for NaN/Inf values
            check_duplicates: Check for duplicate timestamps
        """
        super().__init__()
        self.check_ohlc = check_ohlc
        self.check_volume = check_volume
        self.check_timestamps = check_timestamps
        self.check_nan_inf = check_nan_inf
        self.check_duplicates = check_duplicates
    
    async def validate(self, candles: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Validate candles
        
        Returns:
            (valid_count, invalid_count)
        """
        
        logger.info("validation_starting", count=len(candles))
        
        valid = []
        invalid = 0
        seen_timestamps = set()
        
        for candle in candles:
            try:
                # Check for NaN/Inf values
                if self.check_nan_inf:
                    if self._has_nan_inf(candle):
                        raise ValueError("Contains NaN or Inf values")
                
                # Validate OHLC relationships
                if self.check_ohlc:
                    if not self._validate_ohlc(candle):
                        raise ValueError("Invalid OHLC relationship")
                
                # Check volume
                if self.check_volume:
                    if not self._validate_volume(candle):
                        raise ValueError("Invalid volume")
                
                # Check timestamps
                if self.check_timestamps:
                    ts = candle.get("timestamp")
                    if self.check_duplicates and ts in seen_timestamps:
                        raise ValueError(f"Duplicate timestamp: {ts}")
                    if ts:
                        seen_timestamps.add(ts)
                
                valid.append(candle)
                self.checked_count += 1
            
            except Exception as e:
                invalid += 1
                self.invalid_count += 1
                logger.debug("validation_error", error=str(e), candle=candle)
                continue
        
        logger.info("validation_complete", valid=len(valid), invalid=invalid)
        return len(valid), invalid
    
    def _has_nan_inf(self, candle: Dict) -> bool:
        """Check if candle has NaN or Inf values"""
        for field in ["open", "high", "low", "close", "volume"]:
            val = candle.get(field)
            if val is None:
                continue
            
            try:
                f = float(val)
                if math.isnan(f) or math.isinf(f):
                    logger.debug("nan_inf_detected", field=field, value=val)
                    return True
            except (ValueError, TypeError):
                pass
        
        return False
    
    def _validate_ohlc(self, candle: Dict) -> bool:
        """Validate OHLC relationships"""
        
        try:
            o = float(candle.get("open", 0))
            h = float(candle.get("high", 0))
            l = float(candle.get("low", 0))
            c = float(candle.get("close", 0))
        except (ValueError, TypeError):
            return False
        
        # High must be >= Low
        if h < l:
            logger.debug("high_less_than_low", high=h, low=l)
            return False
        
        # Open and Close should be within High-Low range (with 5% tolerance)
        tolerance = (h - l) * 0.05 if (h - l) > 0 else 0
        
        if o is not None and (o < l - tolerance or o > h + tolerance):
            logger.debug("open_outside_range", open=o, high=h, low=l)
            # Don't fail, but warn
        
        if c is not None and (c < l - tolerance or c > h + tolerance):
            logger.debug("close_outside_range", close=c, high=h, low=l)
            # Don't fail, but warn
        
        return True
    
    def _validate_volume(self, candle: Dict) -> bool:
        """Validate volume"""
        try:
            volume = float(candle.get("volume", 0))
            
            if volume < 0:
                logger.debug("negative_volume", volume=volume)
                return False
            
            # Warn if zero volume (but don't fail)
            if volume == 0:
                logger.debug("zero_volume")
            
            return True
        
        except (ValueError, TypeError):
            return False
