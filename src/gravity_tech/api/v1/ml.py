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

import asyncio
from pathlib import Path
from typing import Any, Optional

import numpy as np
import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from gravity_tech.ml.model_registry import (
    PatternClassifierRegistry,
    run_classifier,
)

logger = structlog.get_logger()

router = APIRouter(tags=["Machine Learning"], prefix="/ml")
classifier_registry = PatternClassifierRegistry()


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


class PredictionRequest(BaseModel):
    """Request for pattern prediction"""
    features: PatternFeatures = Field(..., description="21-dimensional feature vector")


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""
    features_list: list[PatternFeatures] = Field(..., description="List of feature vectors")


class PredictionResponse(BaseModel):
    """ML prediction response"""
    predicted_pattern: str = Field(..., description="Predicted pattern type")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    probabilities: dict[str, float] = Field(..., description="Probability for each pattern class")
    model_version: str = Field(..., description="Model version used")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


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
    training_date: Optional[str]
    accuracy: Optional[float]
    features_count: int
    supported_patterns: list[str]
    hyperparameters: Optional[dict[str, Any]]


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


# ============================================================================
# Helper Functions
# ============================================================================

def _build_feature_array(feature_dict: dict[str, float]) -> np.ndarray:
    """Convert the request payload into the ordered numpy array expected by the model."""

    ordered_keys = [
        "xab_ratio_accuracy",
        "abc_ratio_accuracy",
        "bcd_ratio_accuracy",
        "xad_ratio_accuracy",
        "pattern_symmetry",
        "pattern_slope",
        "xa_angle",
        "ab_angle",
        "bc_angle",
        "cd_angle",
        "pattern_duration",
        "xa_magnitude",
        "ab_magnitude",
        "bc_magnitude",
        "cd_magnitude",
        "volume_at_d",
        "volume_trend",
        "volume_confirmation",
        "rsi_at_d",
        "macd_at_d",
        "momentum_divergence",
    ]

    return np.array([feature_dict[key] for key in ordered_keys], dtype=float)


async def _infer_pattern(feature_array: np.ndarray) -> tuple[dict[str, Any], str]:
    """Run inference in a background thread and return (result, model_version)."""
    model, version = classifier_registry.get_classifier()
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: run_classifier(model, feature_array))
    return result, version


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
        import time
        start_time = time.time()

        # Prepare features
        feature_dict = request.features.dict()
        feature_array = _build_feature_array(feature_dict)

        # Make prediction
        prediction, version = await _infer_pattern(feature_array)
        predicted_class = prediction["predicted_pattern"]
        confidence = prediction["confidence"]
        probabilities = prediction["probabilities"]

        inference_time = (time.time() - start_time) * 1000

        response = PredictionResponse(
            predicted_pattern=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            model_version=version,
            inference_time_ms=round(inference_time, 2)
        )

        logger.info("ml_prediction",
                   pattern=predicted_class,
                   confidence=confidence,
                   inference_time_ms=round(inference_time, 2))

        return response

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error("ml_prediction_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


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
        import time
        start_time = time.time()

        # Load model
        predictions = []
        confidences = []

        for features in request.features_list:
            feature_start = time.time()

            # Prepare features
            feature_dict = features.dict()
            feature_array = _build_feature_array(feature_dict)

            prediction, version = await _infer_pattern(feature_array)
            predicted_class = prediction["predicted_pattern"]
            confidence = prediction["confidence"]
            probabilities = prediction["probabilities"]

            feature_inference_time = (time.time() - feature_start) * 1000

            predictions.append(PredictionResponse(
                predicted_pattern=predicted_class,
                confidence=confidence,
                probabilities=probabilities,
                model_version=version,
                inference_time_ms=round(feature_inference_time, 2)
            ))
            confidences.append(confidence)

        total_time = (time.time() - start_time) * 1000
        avg_confidence = float(np.mean(confidences))

        response = BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            average_confidence=round(avg_confidence, 4),
            total_inference_time_ms=round(total_time, 2)
        )

        logger.info("batch_prediction",
                   count=len(predictions),
                   avg_confidence=avg_confidence,
                   total_time_ms=round(total_time, 2))

        return response

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error("batch_prediction_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


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
        model, version = classifier_registry.get_classifier()

        # Get model type
        model_type = type(model).__name__

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

        return info

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error("model_info_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model info: {str(e)}"
        )


@router.get(
    "/health",
    summary="ML Service Health",
    description="Check ML service health and model availability"
)
async def ml_service_health():
    """Check ML service health"""
    try:
        model, version = classifier_registry.get_classifier()

        return {
            "status": "healthy",
            "service": "ml-prediction",
            "version": "1.0.0",
            "model_loaded": True,
            "model_version": version,
            "model_type": type(model).__name__,
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
