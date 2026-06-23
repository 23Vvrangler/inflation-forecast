"""
Application configuration and settings.
Loads model metadata, datasets, and manages shared state.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


class AppConfig:
    """
    Application configuration singleton.
    Manages model loading, dataset discovery, metadata, and API settings.
    """

    PROJECT_NAME: str = "Inflation Forecast API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "REST API for inflation forecasting using ML and web data"

    # Paths (relative to backend directory)
    MODEL_PATH: str = os.getenv("MODEL_PATH", "model/artifacts/model_v1.pkl")
    METADATA_PATH: str = os.getenv("METADATA_PATH", "model/artifacts/metadata.json")
    METRICS_PATH: str = os.getenv("METRICS_PATH", "model/artifacts/metrics.json")

    # Data directories
    RAW_DATA_DIR: Path = Path("data/raw")
    UPLOADS_DIR: Path = Path("data/uploads")

    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list = ["*"]

    # Pre-loaded countries (ISO codes)
    COUNTRIES: Dict[str, str] = {
        "PER": "Peru",
        "ARG": "Argentina",
        "BRA": "Brazil",
        "CHL": "Chile",
        "COL": "Colombia",
        "MEX": "Mexico",
    }

    # Model state (loaded at startup)
    _model: Any = None
    _metadata: Optional[Dict[str, Any]] = None
    _metrics: Optional[Dict[str, Any]] = None
    _model_loaded: bool = False

    @classmethod
    def load_model(cls) -> bool:
        """
        Load the ML model and metadata into memory.
        Called once at application startup.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            # Load model
            if Path(cls.MODEL_PATH).exists():
                cls._model = joblib.load(cls.MODEL_PATH)
                logger.info(f"Model loaded successfully from {cls.MODEL_PATH}")
            else:
                logger.warning(f"Model file not found at {cls.MODEL_PATH}")
                cls._model_loaded = False
                return False

            # Load metadata
            if Path(cls.METADATA_PATH).exists():
                with open(cls.METADATA_PATH, "r") as f:
                    cls._metadata = json.load(f)
                logger.info(f"Metadata loaded from {cls.METADATA_PATH}")
            else:
                logger.warning(f"Metadata file not found at {cls.METADATA_PATH}")

            # Load metrics
            if Path(cls.METRICS_PATH).exists():
                with open(cls.METRICS_PATH, "r") as f:
                    cls._metrics = json.load(f)
                logger.info(f"Metrics loaded from {cls.METRICS_PATH}")
            else:
                logger.warning(f"Metrics file not found at {cls.METRICS_PATH}")

            cls._model_loaded = True
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            cls._model_loaded = False
            return False

    @classmethod
    def get_model(cls) -> Any:
        """Get the loaded ML model."""
        return cls._model

    @classmethod
    def get_metadata(cls) -> Optional[Dict[str, Any]]:
        """Get model metadata."""
        return cls._metadata

    @classmethod
    def get_metrics(cls) -> Optional[Dict[str, Any]]:
        """Get model evaluation metrics."""
        return cls._metrics

    @classmethod
    def is_model_loaded(cls) -> bool:
        """Check if model is loaded and ready."""
        return cls._model_loaded

    @classmethod
    def discover_datasets(cls) -> List[Dict[str, Any]]:
        """
        Discover all available datasets: pre-loaded countries + user uploads.

        Returns:
            List of dataset dictionaries with id, name, source, records.
        """
        datasets = []

        # Pre-loaded country datasets (merged CSVs from World Bank)
        for code, name in cls.COUNTRIES.items():
            merged_file = cls.RAW_DATA_DIR / f"merged_{code}.csv"
            if merged_file.exists():
                try:
                    df = pd.read_csv(merged_file)
                    datasets.append({
                        "id": code,
                        "name": name,
                        "source": "worldbank",
                        "records": len(df),
                        "file": str(merged_file),
                    })
                except Exception as e:
                    logger.warning(f"Could not read {merged_file}: {e}")

        # User-uploaded datasets
        if cls.UPLOADS_DIR.exists():
            for csv_file in cls.UPLOADS_DIR.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file, parse_dates=["date"])
                    datasets.append({
                        "id": f"upload_{csv_file.stem}",
                        "name": csv_file.name,
                        "source": "user_upload",
                        "records": len(df),
                        "file": str(csv_file),
                    })
                except Exception as e:
                    logger.warning(f"Could not read upload {csv_file}: {e}")

        return datasets

    @classmethod
    def get_dataset_history(cls, dataset_id: str) -> Optional[pd.DataFrame]:
        """
        Load historical data for a specific dataset.

        Args:
            dataset_id: Dataset ID (country code or upload ID).

        Returns:
            DataFrame with historical data, or None if not found.
        """
        # Check pre-loaded countries
        if dataset_id in cls.COUNTRIES:
            file_path = cls.RAW_DATA_DIR / f"merged_{dataset_id}.csv"
            if file_path.exists():
                return pd.read_csv(file_path, parse_dates=["date"])

        # Check uploads
        upload_file = cls.UPLOADS_DIR / f"{dataset_id.replace('upload_', '')}.csv"
        if upload_file.exists():
            return pd.read_csv(upload_file, parse_dates=["date"])

        # Try generic lookup
        for csv_file in cls.UPLOADS_DIR.glob("*.csv"):
            if f"upload_{csv_file.stem}" == dataset_id:
                return pd.read_csv(csv_file, parse_dates=["date"])

        return None


# Global config instance
config = AppConfig()
