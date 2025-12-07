"""
Thread-safe registry for ML models used by the public APIs.

Models are loaded once per process and reused for every request so that
we avoid the repeated `pickle.load` cost and the associated security risks.
"""

from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import Any, Tuple

import numpy as np


class PatternClassifierRegistry:
    """Singleton-style loader for the harmonic pattern classifier."""

    _instance: "PatternClassifierRegistry | None" = None

    def __new__(cls, models_dir: Path | None = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models_dir = models_dir or Path(__file__).resolve().parents[2] / "ml_models"
            cls._instance._model = None
            cls._instance._version = None
            cls._instance._lock = threading.Lock()
        return cls._instance

    def get_classifier(self) -> Tuple[Any, str]:
        """Return the cached classifier, loading it if necessary."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model, self._version = self._load_classifier()
        return self._model, self._version  # type: ignore[return-value]

    def _load_classifier(self) -> Tuple[Any, str]:
        preferred = [
            ("pattern_classifier_advanced_v2.pkl", "v2"),
            ("pattern_classifier_v1.pkl", "v1"),
        ]

        for filename, version in preferred:
            path = self._models_dir / filename
            if path.exists():
                with open(path, "rb") as f:
                    payload = pickle.load(f)

                if isinstance(payload, dict) and "model" in payload:
                    return payload["model"], version
                return payload, version

        raise FileNotFoundError(
            f"No ML model found in {self._models_dir}. Train or place a classifier pickle."
        )


def run_classifier(model: Any, feature_array: np.ndarray) -> dict[str, Any]:
    """
    Execute the classifier synchronously and return prediction details.

    This helper is intentionally synchronous so that callers can wrap it in
    `run_in_executor` and keep the FastAPI event loop responsive.
    """
    if hasattr(model, "predict_single"):
        result = model.predict_single(feature_array)
        return {
            "predicted_pattern": result.get("pattern_type"),
            "confidence": float(result.get("confidence", 0.0)),
            "probabilities": result.get("probabilities", {}),
        }

    prediction = model.predict(feature_array.reshape(1, -1))[0]
    probabilities = model.predict_proba(feature_array.reshape(1, -1))[0]

    # Try to read class names from the model; otherwise fall back to defaults.
    if hasattr(model, "classes_"):
        class_names = list(model.classes_)
    else:
        class_names = ["gartley", "butterfly", "bat", "crab"]

    predicted_pattern = class_names[prediction] if isinstance(prediction, (int, np.integer)) else prediction
    prob_dict = {name: float(prob) for name, prob in zip(class_names, probabilities)}
    confidence = max(prob_dict.values()) if prob_dict else 0.0

    return {
        "predicted_pattern": predicted_pattern,
        "confidence": confidence,
        "probabilities": prob_dict,
    }
