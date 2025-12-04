"""
Prediction Result Entity

Clean Architecture - Domain Layer
Immutable entity for ML model predictions.

Author: Gravity Tech Team
Date: December 4, 2025
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class PredictionSignal(str, Enum):
    """Prediction signal enumeration"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class PredictionResult:
    """
    Result of ML model prediction

    Attributes:
        predictions: List of predicted values
        confidence: Prediction confidence score (0.0 to 1.0)
        signal: Prediction signal (BULLISH/BEARISH/NEUTRAL)
        model_type: Type of model used (LSTM, Transformer, etc.)
        prediction_timestamp: When prediction was made
        input_features: Features used for prediction
        metadata: Additional prediction metadata
        description: Human-readable description
    """

    predictions: List[float]
    confidence: float
    signal: PredictionSignal
    model_type: str
    prediction_timestamp: datetime
    input_features: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]
    description: str

    def __post_init__(self):
        """Validate prediction result data"""
        if not isinstance(self.confidence, (int, float)) or not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be a float between 0.0 and 1.0")

        if not self.predictions:
            raise ValueError("predictions list cannot be empty")

        if not isinstance(self.signal, PredictionSignal):
            raise ValueError("signal must be a PredictionSignal enum value")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "predictions": self.predictions,
            "confidence": self.confidence,
            "signal": self.signal.value,
            "model_type": self.model_type,
            "prediction_timestamp": self.prediction_timestamp.isoformat(),
            "input_features": self.input_features,
            "metadata": self.metadata,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionResult':
        """Create from dictionary representation"""
        return cls(
            predictions=data["predictions"],
            confidence=data["confidence"],
            signal=PredictionSignal(data["signal"]),
            model_type=data["model_type"],
            prediction_timestamp=datetime.fromisoformat(data["prediction_timestamp"]),
            input_features=data.get("input_features"),
            metadata=data.get("metadata"),
            description=data["description"]
        )

    def get_primary_prediction(self) -> float:
        """Get the primary (first) prediction value"""
        return self.predictions[0] if self.predictions else 0.0

    def is_confident(self, threshold: float = 0.7) -> bool:
        """Check if prediction confidence meets threshold"""
        return self.confidence >= threshold

    def is_bullish(self) -> bool:
        """Check if prediction is bullish"""
        return self.signal == PredictionSignal.BULLISH

    def is_bearish(self) -> bool:
        """Check if prediction is bearish"""
        return self.signal == PredictionSignal.BEARISH

    def is_neutral(self) -> bool:
        """Check if prediction is neutral"""
        return self.signal == PredictionSignal.NEUTRAL

    def get_signal_strength(self) -> str:
        """Get signal strength description"""
        if self.confidence >= 0.8:
            return "STRONG"
        elif self.confidence >= 0.6:
            return "MODERATE"
        else:
            return "WEAK"

    def get_summary(self) -> str:
        """Get human-readable prediction summary"""
        signal_desc = {
            PredictionSignal.BULLISH: "Bullish",
            PredictionSignal.BEARISH: "Bearish",
            PredictionSignal.NEUTRAL: "Neutral"
        }.get(self.signal, "Unknown")

        strength = self.get_signal_strength()

        return f"Prediction {self.model_type}: {signal_desc} ({strength}, confidence: {self.confidence:.1%})"
