"""
Prediction service layer.
Handles ML model inference, input validation, and result formatting.
"""
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from app.core.config import config

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Service class for making predictions with the loaded ML regression model.
    """

    @staticmethod
    def validate_features(features: Dict[str, float], expected_features: List[str]) -> bool:
        """
        Validate that input features match the expected format.

        Args:
            features: Dictionary of feature name -> value.
            expected_features: List of expected feature names.

        Returns:
            True if valid, raises ValueError otherwise.
        """
        if not isinstance(features, dict):
            raise ValueError("Features must be provided as a dictionary")

        missing = [f for f in expected_features if f not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        extra = [f for f in features if f not in expected_features]
        if extra:
            logger.warning(f"Extra features ignored: {extra}")

        for name, value in features.items():
            if name in expected_features:
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Feature '{name}' must be numeric, got {type(value).__name__}")
                if np.isnan(value) or np.isinf(value):
                    raise ValueError(f"Feature '{name}' contains invalid value: {value}")

        return True

    @staticmethod
    def predict(features: Dict[str, float]) -> Dict[str, Any]:
        """
        Make a single inflation prediction.

        Args:
            features: Dictionary of feature name -> numeric value.

        Returns:
            Dictionary with prediction results.
        """
        model = config.get_model()
        metadata = config.get_metadata()

        if model is None:
            raise RuntimeError("Model not loaded. Please check server configuration.")

        # Get expected features from metadata
        expected_features = []
        if metadata and "features" in metadata:
            expected_features = metadata["features"].get("numeric", []) + metadata["features"].get("text", [])

        if not expected_features:
            # Fallback: try to infer from model
            logger.warning("No expected features in metadata, attempting inference")
            expected_features = list(features.keys())

        # Validate input
        PredictionService.validate_features(features, expected_features)

        # Build feature vector in correct order
        feature_vector = [float(features.get(f, 0.0)) for f in expected_features]
        input_array = np.array(feature_vector).reshape(1, -1)

        # Make prediction
        prediction = float(model.predict(input_array)[0])

        # Build response
        result = {
            "prediction": round(prediction, 4),
            "model_version": metadata.get("model_version", "unknown") if metadata else "unknown",
            "input_features": {k: round(v, 4) for k, v in features.items() if k in expected_features},
        }

        logger.info(f"Prediction made: {prediction:.4f}")
        return result

    @staticmethod
    def get_model_info() -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata and performance metrics.
        """
        metadata = config.get_metadata()
        metrics = config.get_metrics()

        info = {
            "model_loaded": config.is_model_loaded(),
            "project_name": config.PROJECT_NAME,
            "api_version": config.VERSION,
        }

        if metadata:
            info["model"] = {
                "version": metadata.get("model_version", "unknown"),
                "algorithm": metadata.get("algorithm", "unknown"),
                "experiment_name": metadata.get("experiment_name", "unknown"),
                "features": metadata.get("features", {}),
                "target": metadata.get("target", {}),
                "preprocessing": metadata.get("preprocessing", {}),
                "train_samples": metadata.get("train_samples", 0),
                "test_samples": metadata.get("test_samples", 0),
                "feature_count": metadata.get("feature_count", 0),
            }

        if metrics:
            info["performance"] = {
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "mape": metrics.get("mape"),
                "r2": metrics.get("r2"),
                "explained_variance": metrics.get("explained_variance"),
                "directional_accuracy": metrics.get("directional_accuracy"),
            }

        return info


# Service instance
prediction_service = PredictionService()
