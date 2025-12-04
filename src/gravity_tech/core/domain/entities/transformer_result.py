"""
Transformer Result Entity

Clean Architecture - Domain Layer
Immutable entity for Transformer model results.

Author: Gravity Tech Team
Date: December 4, 2025
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass(frozen=True)
class TransformerResult:
    """
    Result of Transformer model training/evaluation

    Attributes:
        model_id: Unique identifier for the model
        training_accuracy: Training accuracy score
        validation_accuracy: Validation accuracy score (if available)
        test_accuracy: Test accuracy score (if available)
        loss_history: Training loss history
        epochs_trained: Number of epochs trained
        training_time_seconds: Total training time
        model_parameters: Model hyperparameters used
        attention_metrics: Attention mechanism metrics (if calculated)
        feature_importance: Feature importance scores (if calculated)
        created_at: Timestamp when model was created
        description: Human-readable description
    """

    model_id: str
    training_accuracy: float
    validation_accuracy: Optional[float]
    test_accuracy: Optional[float]
    loss_history: List[float]
    epochs_trained: int
    training_time_seconds: float
    model_parameters: Dict[str, Any]
    attention_metrics: Optional[Dict[str, Any]]
    feature_importance: Optional[Dict[str, float]]
    created_at: datetime
    description: str

    def __post_init__(self):
        """Validate Transformer result data"""
        if not isinstance(self.training_accuracy, (int, float)) or not (0.0 <= self.training_accuracy <= 1.0):
            raise ValueError("training_accuracy must be a float between 0.0 and 1.0")

        if self.validation_accuracy is not None:
            if not isinstance(self.validation_accuracy, (int, float)) or not (0.0 <= self.validation_accuracy <= 1.0):
                raise ValueError("validation_accuracy must be a float between 0.0 and 1.0")

        if self.test_accuracy is not None:
            if not isinstance(self.test_accuracy, (int, float)) or not (0.0 <= self.test_accuracy <= 1.0):
                raise ValueError("test_accuracy must be a float between 0.0 and 1.0")

        if self.epochs_trained < 0:
            raise ValueError("epochs_trained must be non-negative")

        if self.training_time_seconds < 0:
            raise ValueError("training_time_seconds must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "model_id": self.model_id,
            "training_accuracy": self.training_accuracy,
            "validation_accuracy": self.validation_accuracy,
            "test_accuracy": self.test_accuracy,
            "loss_history": self.loss_history,
            "epochs_trained": self.epochs_trained,
            "training_time_seconds": self.training_time_seconds,
            "model_parameters": self.model_parameters,
            "attention_metrics": self.attention_metrics,
            "feature_importance": self.feature_importance,
            "created_at": self.created_at.isoformat(),
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransformerResult':
        """Create from dictionary representation"""
        return cls(
            model_id=data["model_id"],
            training_accuracy=data["training_accuracy"],
            validation_accuracy=data.get("validation_accuracy"),
            test_accuracy=data.get("test_accuracy"),
            loss_history=data["loss_history"],
            epochs_trained=data["epochs_trained"],
            training_time_seconds=data["training_time_seconds"],
            model_parameters=data["model_parameters"],
            attention_metrics=data.get("attention_metrics"),
            feature_importance=data.get("feature_importance"),
            created_at=datetime.fromisoformat(data["created_at"]),
            description=data["description"]
        )

    def is_model_ready(self) -> bool:
        """Check if model is ready for production use"""
        return (
            self.training_accuracy >= 0.7 and
            (self.validation_accuracy is None or self.validation_accuracy >= 0.65) and
            self.epochs_trained >= 10
        )

    def get_attention_summary(self) -> Optional[str]:
        """Get human-readable attention metrics summary"""
        if self.attention_metrics is None:
            return None

        summary = "Attention Metrics: "
        if "mean_attention_entropy" in self.attention_metrics:
            summary += f"Entropy={self.attention_metrics['mean_attention_entropy']:.4f}, "
        if "num_heads" in self.attention_metrics:
            summary += f"Heads={self.attention_metrics['num_heads']}, "
        if "max_attention_concentration" in self.attention_metrics:
            summary += f"Concentration={self.attention_metrics['max_attention_concentration']:.4f}"

        return summary.rstrip(", ")

    def get_performance_summary(self) -> str:
        """Get human-readable performance summary"""
        summary = f"Transformer Model {self.model_id}: "
        summary += f"Training Acc: {self.training_accuracy:.3f}"

        if self.validation_accuracy is not None:
            summary += f", Validation Acc: {self.validation_accuracy:.3f}"

        if self.test_accuracy is not None:
            summary += f", Test Acc: {self.test_accuracy:.3f}"

        summary += f" ({self.epochs_trained} epochs, {self.training_time_seconds:.1f}s)"

        return summary
