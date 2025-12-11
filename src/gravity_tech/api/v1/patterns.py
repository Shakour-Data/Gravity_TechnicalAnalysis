"""
================================================================================
FILE IDENTITY CARD (شناسنامه فایل)
================================================================================
File Path:           api/v1/patterns.py
Author:              Dmitry Volkov (API Architect)
Team ID:             BA-003-API
Created Date:        2025-11-12
Last Modified:       2025-11-12
Version:             1.0.0
Purpose:             Pattern Recognition API Endpoints (Harmonic Patterns + ML)
Dependencies:        fastapi, patterns.harmonic, ml.pattern_classifier
Related Files:       api/v1/__init__.py, patterns/harmonic.py, ml/pattern_classifier.py
Complexity:          7/10
Lines of Code:       ~300
Test Coverage:       TBD
Performance Impact:  HIGH (ML inference + pattern detection)
Time Spent:          4 hours (Day 6)
Cost:                $1,920 (4 × $480/hr)
Review Status:       Development
Notes:               Harmonic pattern detection with ML confidence scoring
================================================================================

Pattern Recognition API Endpoints

Provides RESTful endpoints for:
- Harmonic pattern detection (Gartley, Butterfly, Bat, Crab)
- ML-based pattern confidence scoring
- Batch pattern analysis
- Real-time pattern monitoring
"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = structlog.get_logger()

router = APIRouter(tags=["Pattern Recognition"], prefix="/patterns")
MODEL_CACHE: dict[str, object] = {}
MODEL_META: dict[str, str] = {}
MAX_CANDLES = 5000


# ============================================================================
# Request/Response Models
# ============================================================================

class CandleData(BaseModel):
    """Single candle/bar data"""
    timestamp: int = Field(..., description="Unix timestamp")
    open: float = Field(..., gt=0, description="Open price")
    high: float = Field(..., gt=0, description="High price")
    low: float = Field(..., gt=0, description="Low price")
    close: float = Field(..., gt=0, description="Close price")
    volume: float = Field(..., ge=0, description="Volume")


class PatternDetectionRequest(BaseModel):
    """Request for pattern detection"""
    symbol: str = Field(..., description="Trading pair symbol (e.g., BTCUSDT)")
    timeframe: str = Field(
        ...,
        description="Timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)",
        pattern="^(1m|5m|15m|30m|1h|4h|1d|1w)$"
    )
    candles: list[CandleData] = Field(..., min_items=60, description="OHLCV candle data (minimum 60)")  # type: ignore[call-overload]
    pattern_types: list[str] | None = Field(
        default=None,
        description="Specific patterns to detect (if None, detect all). Options: gartley, butterfly, bat, crab"
    )
    use_ml: bool = Field(default=True, description="Use ML confidence scoring")
    min_confidence: float = Field(default=0.5, ge=0, le=1, description="Minimum ML confidence threshold")
    tolerance: float = Field(default=0.05, ge=0, le=0.2, description="Pattern ratio tolerance (0.05 = 5%)")


class PatternPoint(BaseModel):
    """Pattern point (X, A, B, C, D)"""
    label: str
    index: int
    price: float
    timestamp: int


class PatternResult(BaseModel):
    """Detected pattern result"""
    pattern_type: str = Field(..., description="Pattern type (gartley, butterfly, bat, crab)")
    direction: str = Field(..., description="bullish or bearish")
    points: dict[str, PatternPoint] = Field(..., description="Pattern points (X, A, B, C, D)")
    ratios: dict[str, float] = Field(..., description="Fibonacci ratios")
    completion_price: float = Field(..., description="Pattern completion price (D point)")
    confidence: float | None = Field(None, description="ML confidence score (0-1)")
    targets: dict[str, float] | None = Field(None, description="Price targets")
    stop_loss: float | None = Field(None, description="Suggested stop-loss")
    detected_at: str = Field(..., description="Detection timestamp")


class PatternDetectionResponse(BaseModel):
    """Pattern detection response"""
    symbol: str
    timeframe: str
    patterns_found: int = Field(..., description="Number of patterns detected")
    patterns: list[PatternResult] = Field(..., description="Detected patterns")
    analysis_time_ms: float = Field(..., description="Analysis duration in milliseconds")
    ml_enabled: bool = Field(..., description="Whether ML scoring was used")
    ml_status: str | None = Field(default=None, description="Model status: enabled/disabled/not_available/failed")


class PatternStatsResponse(BaseModel):
    """Pattern statistics response"""
    total_patterns: int
    by_type: dict[str, int]
    by_direction: dict[str, int]
    average_confidence: float | None
    high_confidence_patterns: int


# ============================================================================
# Endpoints
# ============================================================================

@router.post(
    "/detect",
    response_model=PatternDetectionResponse,
    summary="Detect Harmonic Patterns",
    description="Detect harmonic patterns (Gartley, Butterfly, Bat, Crab) with optional ML scoring"
)
async def detect_patterns(request: PatternDetectionRequest) -> PatternDetectionResponse:
    """
    Detect harmonic patterns in price data

    **Supported Patterns:**
    - **Gartley**: Classic harmonic pattern with 0.618 retracement
    - **Butterfly**: Extended pattern with 1.27-1.618 projection
    - **Bat**: Pattern with 0.886 XA retracement
    - **Crab**: Extreme pattern with 1.618 projection

    **ML Confidence Scoring:**
    - Uses trained XGBoost classifier
    - Analyzes 21 pattern features
    - Returns confidence score (0-1)
    - Filters patterns below min_confidence

    **Example Request:**
    ```json
    {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "candles": [...],
        "use_ml": true,
        "min_confidence": 0.6,
        "tolerance": 0.05
    }
    ```
    """
    detected_patterns = []
    start_time = 0.0
    highs = lows = closes = volumes = timestamps = np.array([])
    ml_status = "enabled" if request.use_ml else "disabled"
    import time

    from gravity_tech.ml.pattern_features import PatternFeatureExtractor
    from gravity_tech.patterns.harmonic import HarmonicPatternDetector

    start_time = time.time()

    if len(request.candles) > MAX_CANDLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many candles (max {MAX_CANDLES})"
        )

    # Initialize detector
    detector = HarmonicPatternDetector(tolerance=request.tolerance)

    # Prepare data
    highs = np.array([c.high for c in request.candles])
    lows = np.array([c.low for c in request.candles])
    closes = np.array([c.close for c in request.candles])
    volumes = np.array([c.volume for c in request.candles])
    timestamps = np.array([c.timestamp for c in request.candles])

    try:
        # Detect patterns
        detected_patterns = detector.detect_patterns(highs, lows, closes)

        # Filter by pattern type if specified
        if request.pattern_types:
            detected_patterns = [
                p for p in detected_patterns
                if p.pattern_type in request.pattern_types
            ]

        # Validate chronological order
        if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Candles must be strictly increasing in time",
            )

        # Apply ML scoring if enabled
        if request.use_ml:
            try:
                classifier, version = _load_pattern_model()
                extractor = PatternFeatureExtractor()
                filtered_patterns = []
                clf: Any = classifier

                for pattern in detected_patterns:
                    features = extractor.extract_features(
                        pattern, highs, lows, closes, volumes
                    )
                    feature_array = extractor.features_to_array(features)

                    if hasattr(clf, 'predict_single'):
                        prediction = clf.predict_single(feature_array)
                        confidence = prediction['confidence']
                    else:
                        probas = clf.predict_proba(feature_array.reshape(1, -1))[0]
                        confidence = float(np.max(probas))

                    if confidence >= request.min_confidence:
                        pattern.confidence = confidence
                        filtered_patterns.append(pattern)

                detected_patterns = filtered_patterns
                logger.info(
                    "ml_scoring_applied",
                    version=version,
                    patterns_before=len(request.candles),
                    patterns_after=len(filtered_patterns),
                )

            except FileNotFoundError as e:
                logger.warning("ml_model_not_found", error=str(e))
                request.use_ml = False  # reflect actual usage
                ml_status = "not_available"
            except Exception as e:
                logger.warning("ml_scoring_failed", error=str(e))
                request.use_ml = False  # reflect actual usage
                ml_status = "failed"

    except Exception as e:
        logger.error("pattern_detection_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pattern detection failed: {str(e)}"
        ) from e

    # Format response
    patterns_list = []
    for pattern in detected_patterns:
        # Get pattern points with timestamps
        points_dict = {}
        for label, point in pattern.points.items():
            points_dict[label] = PatternPoint(
                label=label,
                index=point.index,
                price=point.price,
                timestamp=int(timestamps[point.index])
            )

        # Calculate targets and stop-loss (ATR-like dynamic)
        d_price = pattern.points['D'].price
        target1, target2, stop_loss = _dynamic_targets(
            highs=highs,
            lows=lows,
            completion_price=d_price,
            is_bullish=(pattern.direction == 'bullish')
        )

        pattern_result = PatternResult(
            pattern_type=pattern.pattern_type.value if hasattr(pattern.pattern_type, 'value') else str(pattern.pattern_type),
            direction=pattern.direction.value if hasattr(pattern.direction, 'value') else str(pattern.direction),
            points=points_dict,
            ratios=pattern.ratios,
            completion_price=d_price,
            confidence=getattr(pattern, 'confidence', None),
            targets={'target1': target1, 'target2': target2},
            stop_loss=stop_loss,
            detected_at=datetime.now(UTC).isoformat()
        )
        patterns_list.append(pattern_result)

    analysis_time = (time.time() - start_time) * 1000  # Convert to ms

    response = PatternDetectionResponse(
        symbol=request.symbol,
        timeframe=request.timeframe,
        patterns_found=len(patterns_list),
        patterns=patterns_list,
        analysis_time_ms=round(analysis_time, 2),
        ml_enabled=request.use_ml,
        ml_status=ml_status
    )

    logger.info(
        "patterns_detected",
        symbol=request.symbol,
        timeframe=request.timeframe,
        patterns_found=len(patterns_list),
        analysis_time_ms=round(analysis_time, 2)
    )

    return response

@router.get(
    "/types",
    summary="List Pattern Types",
    description="Get list of available harmonic pattern types"
)
async def list_pattern_types():
    """
    List all available harmonic pattern types

    Returns pattern types with descriptions and characteristics
    """
    return {
        "patterns": [
            {
                "type": "gartley",
                "name": "Gartley Pattern",
                "description": "Classic harmonic pattern discovered by H.M. Gartley",
                "key_ratios": {
                    "XAB": "0.618",
                    "ABC": "0.382-0.886",
                    "BCD": "1.13-1.618",
                    "XAD": "0.786"
                },
                "reliability": "High"
            },
            {
                "type": "butterfly",
                "name": "Butterfly Pattern",
                "description": "Extended harmonic pattern with 1.27-1.618 projection",
                "key_ratios": {
                    "XAB": "0.786",
                    "ABC": "0.382-0.886",
                    "BCD": "1.618-2.24",
                    "XAD": "1.27-1.618"
                },
                "reliability": "Medium-High"
            },
            {
                "type": "bat",
                "name": "Bat Pattern",
                "description": "Pattern with precise 0.886 XA retracement",
                "key_ratios": {
                    "XAB": "0.382-0.50",
                    "ABC": "0.382-0.886",
                    "BCD": "1.618-2.618",
                    "XAD": "0.886"
                },
                "reliability": "Very High"
            },
            {
                "type": "crab",
                "name": "Crab Pattern",
                "description": "Extreme harmonic pattern with 1.618 XAD projection",
                "key_ratios": {
                    "XAB": "0.382-0.618",
                    "ABC": "0.382-0.886",
                    "BCD": "2.24-3.618",
                    "XAD": "1.618"
                },
                "reliability": "High"
            }
        ],
        "total": 4,
        "ml_confidence_available": True
    }


@router.get(
    "/health",
    summary="Pattern Detection Service Health",
    description="Check pattern detection service health and ML model availability"
)
async def pattern_service_health():
    """Check pattern detection service health"""
    # Check if ML models are available
    model_v2_path = Path(__file__).parent.parent.parent / "ml_models" / "pattern_classifier_advanced_v2.pkl"
    model_v1_path = Path(__file__).parent.parent.parent / "ml_models" / "pattern_classifier_v1.pkl"

    ml_available = model_v2_path.exists() or model_v1_path.exists()
    ml_model_version = "v2" if model_v2_path.exists() else ("v1" if model_v1_path.exists() else None)

    return {
        "status": "healthy",
        "service": "pattern-detection",
        "version": "1.0.0",
        "ml_available": ml_available,
        "ml_model_version": ml_model_version,
        "supported_patterns": ["gartley", "butterfly", "bat", "crab"],
        "features": {
            "harmonic_detection": True,
            "ml_confidence_scoring": ml_available,
            "batch_analysis": True,
            "real_time_monitoring": True
        }
    }


# ============================================================================
# Helpers
# ============================================================================

def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_pattern_model():
    """Load harmonic pattern classifier with caching and hash check."""
    candidates = [
        ("v2", Path(__file__).parent.parent.parent / "ml_models" / "pattern_classifier_advanced_v2.pkl"),
        ("v1", Path(__file__).parent.parent.parent / "ml_models" / "pattern_classifier_v1.pkl"),
    ]

    for version, path in candidates:
        if not path.exists():
            continue
        file_hash = _hash_file(path)
        cached = MODEL_CACHE.get(version)
        cached_hash = MODEL_META.get(version)
        if cached and cached_hash == file_hash:
            return cached, version

        import pickle

        with path.open("rb") as f:
            model = pickle.load(f)

        MODEL_CACHE.clear()
        MODEL_META.clear()
        MODEL_CACHE[version] = model
        MODEL_META[version] = file_hash
        return model, version

    raise FileNotFoundError("No pattern classifier model found in ml_models/")


def _dynamic_targets(highs: np.ndarray, lows: np.ndarray, completion_price: float, is_bullish: bool):
    """Compute dynamic targets using an ATR-like range."""
    lookback = min(14, len(highs))
    if lookback < 2:
        return (
            completion_price * (1.03 if is_bullish else 0.97),
            completion_price * (1.05 if is_bullish else 0.95),
            completion_price * (0.98 if is_bullish else 1.02),
        )

    high_tail = highs[-lookback:]
    low_tail = lows[-lookback:]
    prev_close = (high_tail[:-1] + low_tail[:-1]) / 2
    tr = np.maximum.reduce([
        high_tail[1:] - low_tail[1:],
        np.abs(high_tail[1:] - prev_close),
        np.abs(low_tail[1:] - prev_close),
    ])
    atr = float(np.mean(tr)) if len(tr) else float(np.std(high_tail - low_tail))
    if atr <= 0:
        atr = float(abs(completion_price) * 0.02)  # fallback 2%

    direction = 1 if is_bullish else -1
    target1 = completion_price + direction * atr
    target2 = completion_price + direction * 2 * atr
    stop_loss = completion_price - direction * 0.8 * atr
    return target1, target2, stop_loss
