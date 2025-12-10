"""
================================================================================
FILE IDENTITY CARD (شناسنامه فایل)
================================================================================
File Path:           api/v1/ml.py
Author:              Dmitry Volkov (API Architect) & Yuki Tanaka (ML Scientist)
Team ID:             BA-003-API / AI-008-ML
Created Date:        2025-11-12
Last Modified:       2025-11-12
Version:             1.0.0
Purpose:             Machine Learning API Endpoints (Pattern Classification, Backtesting)
Dependencies:        fastapi, ml.pattern_classifier, ml.backtesting
Related Files:       api/v1/__init__.py, ml/pattern_classifier.py, ml/backtesting.py
Complexity:          8/10
Lines of Code:       ~400
Test Coverage:       TBD
Performance Impact:  HIGH (ML inference)
Time Spent:          5 hours (Day 6)
Cost:                $2,400 (5 × $480/hr)
Review Status:       Development
Notes:               ML prediction endpoints with model info and backtesting API
================================================================================

Machine Learning API Endpoints

Provides RESTful endpoints for:
- Pattern classification with confidence scoring
- Model information and statistics
- Batch predictions
- Backtesting API
- Model performance metrics
"""

import pickle
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Any

import numpy as np
import structlog
from fastapi import APIRouter, HTTPException, status
from gravity_tech.database.database_manager import DatabaseManager
from pydantic import BaseModel, Field

try:
    from prometheus_client import Counter, Histogram
except Exception:  # pragma: no cover - fallback when prometheus_client missing
    class _Noop:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            return self

        def observe(self, *args, **kwargs):
            return self

    Counter = Histogram = lambda *args, **kwargs: _Noop()

logger = structlog.get_logger()

router = APIRouter(tags=["Machine Learning"], prefix="/ml")
MODEL_CACHE: dict[str, Any] = {}
MODEL_META: dict[str, Any] = {}
MODEL_LOCK = threading.Lock()
MAX_BATCH = 256
PREDICTION_TIMEOUT_SECONDS = 2.0
BATCH_TIMEOUT_SECONDS = 5.0

_EXECUTOR = ThreadPoolExecutor(max_workers=4)

# Prometheus metrics
MODEL_CACHE_HITS = Counter("ml_model_cache_hits_total", "ML model cache hits", ["version"])
MODEL_CACHE_LOADS = Counter("ml_model_loads_total", "ML model loads from disk", ["version"])
PREDICTION_REQUESTS = Counter(
    "ml_prediction_requests_total", "Total ML prediction requests", ["endpoint", "status", "model_version"]
)
PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_seconds", "Prediction latency (seconds)", ["endpoint", "model_version"]
)
BACKTEST_REQUESTS = Counter(
    "ml_backtest_requests_total", "Total backtest requests", ["status"]
)
BACKTEST_LATENCY = Histogram(
    "ml_backtest_latency_seconds", "Backtest latency (seconds)"
)


# ============================================================================
# Request/Response Models
# ============================================================================

class PatternFeatures(BaseModel):
    """Pattern features for ML classification"""
    xab_ratio_accuracy: float
    abc_ratio_accuracy: float
    bcd_ratio_accuracy: float
    xad_ratio_accuracy: float
    pattern_symmetry: float
    pattern_slope: float
    xa_angle: float
    ab_angle: float
    bc_angle: float
    cd_angle: float
    pattern_duration: float
    xa_magnitude: float
    ab_magnitude: float
    bc_magnitude: float
    cd_magnitude: float
    volume_at_d: float
    volume_trend: float
    volume_confirmation: float
    rsi_at_d: float
    macd_at_d: float
    momentum_divergence: float

    @classmethod
    def _validate_finite(cls, v: float, field_name: str) -> float:
        if not np.isfinite(v):
            raise ValueError(f"{field_name} must be finite")
        return float(v)

    @classmethod
    def _validate_range(cls, v: float, field_name: str) -> float:
        # Soft clamp check to catch extreme/unscaled inputs
        if abs(v) > 1e6:
            raise ValueError(f"{field_name} value too large (>|1e6|)")
        return v

    def model_post_init(self, __context):
        for fname, value in self.__dict__.items():
            self._validate_finite(value, fname)
            self._validate_range(value, fname)


class PredictionRequest(BaseModel):
    """Request for pattern prediction"""
    features: PatternFeatures = Field(..., description="21-dimensional feature vector")
    timeout_seconds: float | None = Field(
        default=None,
        ge=0.1,
        le=30.0,
        description="Optional per-request timeout; defaults to 2s"
    )


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""
    features_list: list[PatternFeatures] = Field(
        ...,
        min_items=1,
        max_items=MAX_BATCH,
        description=f"List of feature vectors (max {MAX_BATCH})"
    )
    timeout_seconds: float | None = Field(
        default=None,
        ge=0.1,
        le=60.0,
        description="Optional timeout for entire batch request; defaults to 5s"
    )


class PredictionResponse(BaseModel):
    """ML prediction response"""
    predicted_pattern: str = Field(..., description="Predicted pattern type")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    probabilities: dict[str, float] = Field(..., description="Probability for each pattern class")
    model_version: str = Field(..., description="Model version used")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    model_hash: str | None = Field(None, description="SHA256 of model artifact")
    model_loaded_at: str | None = Field(None, description="UTC time when model was loaded into cache")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: list[PredictionResponse]
    total_predictions: int
    average_confidence: float
    total_inference_time_ms: float


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_name: str
    model_version: str
    model_type: str
    training_date: str | None
    accuracy: float | None
    features_count: int
    supported_patterns: list[str]
    hyperparameters: dict[str, Any] | None


class BacktestRequest(BaseModel):
    """Request for backtesting"""
    highs: list[float] = Field(..., min_items=300, description="High prices (minimum 300 bars)")
    lows: list[float] = Field(..., min_items=300, description="Low prices")
    closes: list[float] = Field(..., min_items=300, description="Close prices")
    volumes: list[float] = Field(..., min_items=300, description="Volume data")
    dates: list[int] = Field(..., min_items=300, description="Timestamps")
    min_confidence: float = Field(default=0.6, ge=0, le=1, description="Minimum confidence for trades")
    window_size: int = Field(default=200, ge=100, le=500, description="Analysis window size")
    step_size: int = Field(default=50, ge=10, le=100, description="Window step size")


class BacktestMetrics(BaseModel):
    """Backtesting metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    average_pnl: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    target1_hits: int
    target2_hits: int


class BacktestResponse(BaseModel):
    """Backtesting response"""
    metrics: BacktestMetrics
    trade_count: int
    backtest_period: dict[str, str]
    analysis_time_ms: float
    model_version: str | None = Field(None, description="Model version used (if classifier involved)")


# ============================================================================
# Helper Functions
# ============================================================================

def _hash_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


async def _predict_with_timeout(model: Any, feature_array: np.ndarray, timeout: float) -> dict[str, Any]:
    """Run model prediction in a thread with timeout."""

    def _predict_blocking():
        if hasattr(model, 'predict_single'):
            return model.predict_single(feature_array)
        pred = model.predict(feature_array.reshape(1, -1))[0]
        probas = model.predict_proba(feature_array.reshape(1, -1))[0]
        class_names = ['gartley', 'butterfly', 'bat', 'crab']
        predicted_class = class_names[pred]
        confidence = float(np.max(probas))
        probabilities = {name: float(prob) for name, prob in zip(class_names, probas)}
        return {
            "pattern_type": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_EXECUTOR, _predict_blocking),
            timeout=timeout
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Prediction exceeded timeout ({timeout}s)"
        ) from exc


class _DummyPatternModel:
    """Fallback model when no classifier is available."""

    def predict_single(self, features: np.ndarray) -> dict[str, Any]:
        probs = {"gartley": 0.25, "butterfly": 0.25, "bat": 0.25, "crab": 0.25}
        return {
            "pattern_type": "unknown",
            "confidence": 0.0,
            "probabilities": probs,
        }


def load_ml_model():
    """Load the latest ML model (cached). Falls back to dummy model if missing."""
    global MODEL_CACHE, MODEL_META

    model_v2_path = Path(__file__).parent.parent.parent / "ml_models" / "pattern_classifier_advanced_v2.pkl"
    model_v1_path = Path(__file__).parent.parent.parent / "ml_models" / "pattern_classifier_v1.pkl"

    candidates = [("v2", model_v2_path), ("v1", model_v1_path)]

    with MODEL_LOCK:
        for version, path in candidates:
            if not path.exists():
                continue
            file_hash = _hash_file(path)
            cached = MODEL_CACHE.get(version)
            cached_meta = MODEL_META.get(version)
            if cached and cached_meta and cached_meta.get("hash") == file_hash:
                MODEL_CACHE_HITS.labels(version).inc()
                return cached, version

            with open(path, "rb") as f:
                data = pickle.load(f)
                model = data["model"] if isinstance(data, dict) else data
                MODEL_CACHE.clear()
                MODEL_META.clear()
                MODEL_CACHE[version] = model
                MODEL_META[version] = {
                    "path": str(path),
                    "hash": file_hash,
                    "loaded_at": datetime.now(timezone.utc).isoformat(),
                }
                MODEL_CACHE_LOADS.labels(version).inc()
                return model, version

        # Fallback dummy model (uniform probabilities) to keep service responsive
        dummy = _DummyPatternModel()
        MODEL_CACHE.clear()
        MODEL_META.clear()
        MODEL_CACHE["fallback"] = dummy
        MODEL_META["fallback"] = {
            "path": "fallback",
            "hash": "fallback",
            "loaded_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.warning("ml_model_fallback_used")
        return dummy, "fallback"


# ============================================================================
# Endpoints
# ============================================================================

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Pattern Type",
    description="Classify pattern using ML model and return confidence scores"
)
async def predict_pattern(request: PredictionRequest) -> PredictionResponse:
    """
    Classify a harmonic pattern using trained ML model

    **Input Features (21 dimensions):**
    - Ratio accuracies (4): XAB, ABC, BCD, XAD
    - Geometry (6): symmetry, slope, 4 angles
    - Magnitude (5): duration + 4 leg magnitudes
    - Volume (3): at D, trend, confirmation
    - Technical indicators (3): RSI, MACD, momentum divergence

    **Returns:**
    - Predicted pattern type (gartley, butterfly, bat, crab)
    - Confidence score (0-1)
    - Probability distribution across all classes

    **Example:**
    ```json
    {
        "features": {
            "xab_ratio_accuracy": 0.95,
            "abc_ratio_accuracy": 0.87,
            ...
        }
    }
    ```
    """
    try:
        start_ts = time.perf_counter()

        # Load model
        model, version = load_ml_model()
        meta = MODEL_META.get(version, {})

        # Prepare features
        feature_dict = request.features.model_dump()
        feature_array = np.array([
            feature_dict['xab_ratio_accuracy'],
            feature_dict['abc_ratio_accuracy'],
            feature_dict['bcd_ratio_accuracy'],
            feature_dict['xad_ratio_accuracy'],
            feature_dict['pattern_symmetry'],
            feature_dict['pattern_slope'],
            feature_dict['xa_angle'],
            feature_dict['ab_angle'],
            feature_dict['bc_angle'],
            feature_dict['cd_angle'],
            feature_dict['pattern_duration'],
            feature_dict['xa_magnitude'],
            feature_dict['ab_magnitude'],
            feature_dict['bc_magnitude'],
            feature_dict['cd_magnitude'],
            feature_dict['volume_at_d'],
            feature_dict['volume_trend'],
            feature_dict['volume_confirmation'],
            feature_dict['rsi_at_d'],
            feature_dict['macd_at_d'],
            feature_dict['momentum_divergence']
        ])

        # Make prediction (with timeout)
        timeout = request.timeout_seconds or PREDICTION_TIMEOUT_SECONDS
        prediction = await _predict_with_timeout(model, feature_array, timeout)
        predicted_class = prediction['pattern_type']
        confidence = prediction['confidence']
        probabilities = prediction['probabilities']

        duration_seconds = time.perf_counter() - start_ts
        inference_time = duration_seconds * 1000

        response = PredictionResponse(
            predicted_pattern=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            model_version=version,
            inference_time_ms=round(inference_time, 2),
            model_hash=meta.get("hash"),
            model_loaded_at=meta.get("loaded_at"),
        )

        logger.info("ml_prediction",
                   pattern=predicted_class,
                   confidence=confidence,
                   inference_time_ms=round(inference_time, 2))
        PREDICTION_REQUESTS.labels("predict", "success", version).inc()
        PREDICTION_LATENCY.labels("predict", version).observe(duration_seconds)

        return response

    except FileNotFoundError as e:
        PREDICTION_REQUESTS.labels("predict", "model_missing", "unknown").inc()
        # fallback response with ml_enabled=false semantic via model_version=fallback
        return PredictionResponse(
            predicted_pattern="unknown",
            confidence=0.0,
            probabilities={"gartley": 0.25, "butterfly": 0.25, "bat": 0.25, "crab": 0.25},
            model_version="fallback",
            inference_time_ms=0.0,
            model_hash=None,
            model_loaded_at=None,
        )
    except Exception as e:
        logger.error("ml_prediction_error", error=str(e))
        PREDICTION_REQUESTS.labels("predict", "error", version if "version" in locals() else "unknown").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        ) from e


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Pattern Predictions",
    description="Classify multiple patterns in a single request"
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Classify multiple patterns in batch mode for improved efficiency

    **Benefits:**
    - Reduced network overhead
    - Faster processing (batch inference)
    - Lower latency per prediction

    **Use Cases:**
    - Historical pattern analysis
    - Multi-symbol scanning
    - Backtesting validation
    """
    try:
        start_ts = time.perf_counter()

        # Load model
        model, version = load_ml_model()
        meta = MODEL_META.get(version, {})

        predictions = []
        confidences = []

        errors = 0
        for features in request.features_list:
            feature_start = time.perf_counter()
            try:
                feature_dict = features.model_dump()
                feature_array = np.array([
                    feature_dict['xab_ratio_accuracy'],
                    feature_dict['abc_ratio_accuracy'],
                    feature_dict['bcd_ratio_accuracy'],
                    feature_dict['xad_ratio_accuracy'],
                    feature_dict['pattern_symmetry'],
                    feature_dict['pattern_slope'],
                    feature_dict['xa_angle'],
                    feature_dict['ab_angle'],
                    feature_dict['bc_angle'],
                    feature_dict['cd_angle'],
                    feature_dict['pattern_duration'],
                    feature_dict['xa_magnitude'],
                    feature_dict['ab_magnitude'],
                    feature_dict['bc_magnitude'],
                    feature_dict['cd_magnitude'],
                    feature_dict['volume_at_d'],
                    feature_dict['volume_trend'],
                    feature_dict['volume_confirmation'],
                    feature_dict['rsi_at_d'],
                    feature_dict['macd_at_d'],
                    feature_dict['momentum_divergence']
                ])

                # Make prediction
                if hasattr(model, 'predict_single'):
                    prediction = model.predict_single(feature_array)
                    predicted_class = prediction['pattern_type']
                    confidence = prediction['confidence']
                    probabilities = prediction['probabilities']
                else:
                    pred = model.predict(feature_array.reshape(1, -1))[0]
                    probas = model.predict_proba(feature_array.reshape(1, -1))[0]

                    class_names = ['gartley', 'butterfly', 'bat', 'crab']
                    predicted_class = class_names[pred]
                    confidence = float(np.max(probas))
                    probabilities = {name: float(prob) for name, prob in zip(class_names, probas)}

                feature_inference_time = (time.perf_counter() - feature_start) * 1000

                predictions.append(PredictionResponse(
                    predicted_pattern=predicted_class,
                    confidence=confidence,
                    probabilities=probabilities,
                    model_version=version,
                    inference_time_ms=round(feature_inference_time, 2),
                    model_hash=meta.get("hash") if meta else None,
                    model_loaded_at=meta.get("loaded_at") if meta else None,
                ))
                confidences.append(confidence)
            except Exception as exc:
                errors += 1
                logger.warning("batch_prediction_item_failed", error=str(exc), error_type=type(exc).__name__)

        duration_seconds = time.perf_counter() - start_ts
        total_time = duration_seconds * 1000
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        response = BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            average_confidence=round(avg_confidence, 4),
            total_inference_time_ms=round(total_time, 2)
        )

        logger.info("batch_prediction",
                   count=len(predictions),
                   avg_confidence=avg_confidence,
                   total_time_ms=round(total_time, 2),
                   errors=errors)
        PREDICTION_REQUESTS.labels("predict_batch", "success", version).inc()
        PREDICTION_LATENCY.labels("predict_batch", version).observe(duration_seconds)

        return response

    except FileNotFoundError as e:
        PREDICTION_REQUESTS.labels("predict_batch", "model_missing", "unknown").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        ) from e
    except Exception as e:
        logger.error("batch_prediction_error", error=str(e))
        PREDICTION_REQUESTS.labels("predict_batch", "error", version if "version" in locals() else "unknown").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        ) from e


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Get Model Information",
    description="Get information about the loaded ML model"
)
async def get_model_info() -> ModelInfoResponse:
    """
    Get comprehensive information about the loaded ML model

    **Includes:**
    - Model version and type
    - Training accuracy
    - Feature count
    - Supported pattern types
    - Hyperparameters
    """
    try:
        model, version = load_ml_model()

        # Get model type
        model_type = type(model).__name__
        meta = MODEL_META.get(version, {})

        # Default info
        info = ModelInfoResponse(
            model_name=f"Pattern Classifier {version}",
            model_version=version,
            model_type=model_type,
            training_date=None,
            accuracy=None,
            features_count=21,
            supported_patterns=["gartley", "butterfly", "bat", "crab"],
            hyperparameters=None
        )

        # Try to get additional info from model
        if version == "v2":
            info.accuracy = 0.6495  # From Day 5 training
            info.training_date = "2025-11-12"
            if hasattr(model, 'get_params'):
                info.hyperparameters = model.get_params()
        elif version == "v1":
            info.accuracy = 0.4825  # From Day 4 training
            info.training_date = "2025-11-11"

        logger.info("model_info_retrieved", version=version, model_type=model_type)

        return info.copy(update={"hyperparameters": info.hyperparameters, **meta})

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        ) from e
    except Exception as e:
        logger.error("model_info_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model info: {str(e)}"
        ) from e


@router.get(
    "/health",
    summary="ML Service Health",
    description="Check ML service health and model availability"
)
async def ml_service_health():
    """Check ML service health"""
    try:
        model, version = load_ml_model()
        meta = MODEL_META.get(version, {})

        return {
            "status": "healthy",
            "service": "ml-prediction",
            "version": "1.0.0",
            "model_loaded": True,
            "model_version": version,
            "model_type": type(model).__name__,
            "model_hash": meta.get("hash"),
            "features": {
                "single_prediction": True,
                "batch_prediction": True,
                "model_info": True,
                "backtesting": True
            }
        }
    except FileNotFoundError:
        return {
            "status": "degraded",
            "service": "ml-prediction",
            "version": "1.0.0",
            "model_loaded": False,
            "error": "ML model not found"
        }
    except Exception as e:
        logger.error("ml_health_check_error", error=str(e))
        return {
            "status": "unhealthy",
            "service": "ml-prediction",
            "version": "1.0.0",
            "error": str(e)
        }
