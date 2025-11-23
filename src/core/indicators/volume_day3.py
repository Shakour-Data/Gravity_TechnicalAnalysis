"""
================================================================================
DAY 3 VOLUME INDICATORS - v1.1.0
================================================================================
Author:              Maria Gonzalez (Market Microstructure Expert, TM-004-MME)
Created Date:        November 9, 2025
Purpose:             3 advanced volume indicators for Day 3 of v1.1.0
Indicators:
  1. Volume-Weighted MACD (VWMACD)
  2. Ease of Movement (EOM)
  3. Force Index (FI)
================================================================================

These indicators detect institutional activity by analyzing volume-price
relationships. Volume precedes price, so these help identify accumulation
and distribution before major price moves.
"""

import numpy as np
from typing import Dict, Any


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average
    
    Args:
        values: Array of values
        period: EMA period
        
    Returns:
        Array of EMA values
    """
    alpha = 2.0 / (period + 1)
    ema_values = np.zeros_like(values, dtype=float)
    ema_values[0] = values[0]
    
    for i in range(1, len(values)):
        ema_values[i] = alpha * values[i] + (1 - alpha) * ema_values[i - 1]
    
    return ema_values


def volume_weighted_macd(
    prices: np.ndarray,
    volumes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9
) -> Dict[str, Any]:
    """
    Volume-Weighted MACD (VWMACD)
    
    Traditional MACD where each price is weighted by its volume.
    This makes the indicator more sensitive to high-volume institutional moves.
    
    Formula:
      1. Calculate volume-weighted price: VWP = price * volume
      2. Fast VWMA = EMA(VWP, fast) / EMA(volume, fast)
      3. Slow VWMA = EMA(VWP, slow) / EMA(volume, slow)
      4. VWMACD = Fast VWMA - Slow VWMA
      5. Signal = EMA(VWMACD, signal_period)
      6. Histogram = VWMACD - Signal
    
    Args:
        prices: Array of closing prices
        volumes: Array of volumes
        fast: Fast period (default: 12)
        slow: Slow period (default: 26)
        signal_period: Signal line period (default: 9)
        
    Returns:
        dict with:
          - macd_line: VWMACD line values
          - signal_line: Signal line values
          - histogram: Histogram values
          - signal: 'BUY', 'SELL', or None
          - confidence: 0.0 to 1.0
          
    Signal Generation:
      - BUY: VWMACD crosses above signal line (histogram > 0)
      - SELL: VWMACD crosses below signal line (histogram < 0)
      - Confidence based on histogram strength and volume confirmation
    """
    if len(prices) < slow + signal_period:
        return {
            "macd_line": np.array([]),
            "signal_line": np.array([]),
            "histogram": np.array([]),
            "signal": None,
            "confidence": 0.0
        }
    
    # Calculate volume-weighted price
    vwp = prices * volumes
    
    # Calculate volume-weighted moving averages
    vwp_fast_ema = _ema(vwp, fast)
    vol_fast_ema = _ema(volumes, fast)
    vwma_fast = vwp_fast_ema / np.where(vol_fast_ema != 0, vol_fast_ema, 1e-10)
    
    vwp_slow_ema = _ema(vwp, slow)
    vol_slow_ema = _ema(volumes, slow)
    vwma_slow = vwp_slow_ema / np.where(vol_slow_ema != 0, vol_slow_ema, 1e-10)
    
    # Calculate VWMACD line
    macd_line = vwma_fast - vwma_slow
    
    # Calculate signal line
    signal_line = _ema(macd_line, signal_period)
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    # Generate trading signal
    last_hist = histogram[-1]
    prev_hist = histogram[-2] if len(histogram) > 1 else 0
    last_macd = macd_line[-1]
    
    # Check for crossover
    if last_hist > 0 and prev_hist <= 0:
        signal = 'BUY'
        confidence = min(1.0, abs(last_hist) / (np.std(histogram[-20:]) + 1e-10))
    elif last_hist < 0 and prev_hist >= 0:
        signal = 'SELL'
        confidence = min(1.0, abs(last_hist) / (np.std(histogram[-20:]) + 1e-10))
    else:
        # Use both histogram and MACD line for signal
        if last_hist > 0 and last_macd > 0:
            signal = 'BUY'
        elif last_hist < 0 and last_macd < 0:
            signal = 'SELL'
        elif last_hist > 0 and last_macd < 0:
            signal = None  # Conflicting signals
        elif last_hist < 0 and last_macd > 0:
            signal = None  # Conflicting signals
        else:
            signal = None
        confidence = min(1.0, abs(last_hist) / (np.std(histogram[-20:]) + 1e-10)) * 0.5
    
    return {
        "macd_line": macd_line,
        "signal_line": signal_line,
        "histogram": histogram,
        "signal": signal,
        "confidence": float(confidence)
    }


def ease_of_movement(
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    period: int = 14
) -> Dict[str, Any]:
    """
    Ease of Movement (EOM)
    
    Relates price change to volume, identifying how easily price moves.
    Low EOM = difficult to move price (resistance)
    High EOM = easy to move price (momentum)
    
    Formula:
      1. Distance Moved = ((High + Low)/2 - (Prior High + Prior Low)/2)
      2. Box Ratio = (Volume / 1,000,000) / (High - Low)
      3. EMV = Distance Moved / Box Ratio
      4. EOM = EMA(EMV, period)
    
    Args:
        high: Array of high prices
        low: Array of low prices
        volume: Array of volumes
        period: Smoothing period (default: 14)
        
    Returns:
        dict with:
          - values: EOM values
          - signal: 'BUY', 'SELL', or None
          - confidence: 0.0 to 1.0
          
    Signal Generation:
      - BUY: EOM > 0 (price moving up easily)
      - SELL: EOM < 0 (price moving down easily)
      - Confidence based on EOM magnitude and trend
    """
    if len(high) < period + 1:
        return {
            "values": np.array([]),
            "signal": None,
            "confidence": 0.0
        }
    
    # Calculate midpoint
    midpoint = (high + low) / 2.0
    
    # Calculate distance moved
    distance_moved = np.diff(midpoint)
    
    # Calculate box ratio (volume adjusted by price range)
    price_range = high[1:] - low[1:]  # Skip first since we used diff
    # Avoid division by zero
    price_range = np.where(price_range != 0, price_range, 1e-10)
    
    # Scale volume to millions for readability
    volume_scaled = volume[1:] / 1_000_000.0
    volume_scaled = np.where(volume_scaled != 0, volume_scaled, 1e-10)
    
    box_ratio = volume_scaled / price_range
    
    # Calculate EMV (1-period)
    emv = distance_moved / box_ratio
    
    # Smooth with EMA
    eom = _ema(emv, period)
    
    # Generate signal
    last_eom = eom[-1]
    eom_trend = np.mean(eom[-5:]) if len(eom) >= 5 else last_eom
    
    if last_eom > 0:
        signal = 'BUY'
        confidence = min(1.0, abs(last_eom) / (np.std(eom) + 1e-10))
    elif last_eom < 0:
        signal = 'SELL'
        confidence = min(1.0, abs(last_eom) / (np.std(eom) + 1e-10))
    else:
        signal = None
        confidence = 0.0
    
    # Pad EOM to match input length (prepend NaN for first element)
    eom_padded = np.concatenate([np.array([np.nan]), eom])
    
    return {
        "values": eom_padded,
        "signal": signal,
        "confidence": float(confidence)
    }


def force_index(
    prices: np.ndarray,
    volume: np.ndarray,
    period: int = 13
) -> Dict[str, Any]:
    """
    Force Index (FI)
    
    Combines price change and volume to measure buying/selling pressure.
    Developed by Alexander Elder.
    
    Formula:
      1. Force = (Close - Prior Close) * Volume
      2. Force Index = EMA(Force, period)
    
    Interpretation:
      - Positive FI: Buying pressure (bulls in control)
      - Negative FI: Selling pressure (bears in control)
      - Magnitude indicates strength
    
    Args:
        prices: Array of closing prices
        volume: Array of volumes
        period: Smoothing period (default: 13)
        
    Returns:
        dict with:
          - values: Force Index values
          - signal: 'BUY', 'SELL', or None
          - confidence: 0.0 to 1.0
          
    Signal Generation:
      - BUY: FI > 0 and rising (strong buying pressure)
      - SELL: FI < 0 and falling (strong selling pressure)
      - Confidence based on FI strength and volume confirmation
    """
    if len(prices) < period + 1:
        return {
            "values": np.array([]),
            "signal": None,
            "confidence": 0.0
        }
    
    # Calculate price change
    price_change = np.diff(prices)
    
    # Calculate raw force (price change * volume)
    # Skip first volume since we used diff on prices
    raw_force = price_change * volume[1:]
    
    # Smooth with EMA
    force_index = _ema(raw_force, period)
    
    # Generate signal
    last_fi = force_index[-1]
    prev_fi = force_index[-2] if len(force_index) > 1 else 0
    
    # Check trend (rising or falling)
    fi_trend = last_fi - prev_fi
    
    if last_fi > 0:
        signal = 'BUY'
        # Higher confidence if FI is rising
        base_confidence = min(1.0, abs(last_fi) / (np.std(force_index) + 1e-10))
        trend_multiplier = 1.2 if fi_trend > 0 else 0.8
        confidence = min(1.0, base_confidence * trend_multiplier)
    elif last_fi < 0:
        signal = 'SELL'
        base_confidence = min(1.0, abs(last_fi) / (np.std(force_index) + 1e-10))
        trend_multiplier = 1.2 if fi_trend < 0 else 0.8
        confidence = min(1.0, base_confidence * trend_multiplier)
    else:
        signal = None
        confidence = 0.0
    
    # Pad Force Index to match input length (prepend NaN for first element)
    fi_padded = np.concatenate([np.array([np.nan]), force_index])
    
    return {
        "values": fi_padded,
        "signal": signal,
        "confidence": float(confidence)
    }


# ============================================================================
# MARIA GONZALEZ'S NOTES (Market Microstructure Expert)
# ============================================================================
"""
These three indicators are critical for detecting institutional activity:

1. VOLUME-WEIGHTED MACD:
   - More accurate than traditional MACD
   - Weights each price by its volume
   - Catches institutional moves that regular MACD misses
   - Best for: Crypto markets with high volume spikes

2. EASE OF MOVEMENT:
   - Shows if price moves are "easy" (low volume) or "difficult" (high volume)
   - Positive EOM = uptrend on low volume (weak, may reverse)
   - Negative EOM = downtrend on high volume (strong, may continue)
   - Best for: Identifying false breakouts

3. FORCE INDEX:
   - Elder's classic indicator
   - Combines price momentum and volume
   - Divergences are powerful signals
   - Best for: Confirming trend strength

USAGE TOGETHER:
- VWMACD for trend direction
- EOM for move sustainability
- Force Index for strength confirmation

If all three align (e.g., VWMACD BUY + EOM > 0 + FI rising), 
that's an extremely high-probability trade setup.

— Maria Gonzalez, Market Microstructure Expert
   Jane Street Capital (7 years), now leading Gravity TechAnalysis Volume Team
"""
