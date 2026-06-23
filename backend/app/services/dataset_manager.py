"""
Dataset management service.
Handles discovery, validation, and loading of datasets (pre-loaded + user uploads).
"""
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

from app.core.config import config

logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Service class for managing multi-dataset support.
    Handles pre-loaded country datasets and user-uploaded CSV files.
    """

    # Required columns for uploaded CSV files
    REQUIRED_COLUMNS = ["date", "inflation_rate"]
    OPTIONAL_COLUMNS = ["gdp_growth", "interest_rate"]
    ALLOWED_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS + [
        "inflation_lag_1", "inflation_lag_3", "inflation_lag_6", "inflation_lag_12",
        "rolling_mean_3", "rolling_mean_6", "rolling_mean_12", "rolling_std_6",
        "text_sentiment_score", "text_tfidf_mean", "text_keyword_inflation",
        "text_keyword_crisis", "text_keyword_growth", "text_keyword_interest",
        "text_keyword_gdp", "text_keyword_currency", "text_keyword_market",
        "text_article_count", "month_sin", "month_cos",
    ]

    @staticmethod
    def list_datasets() -> List[Dict[str, Any]]:
        """
        List all available datasets.

        Returns:
            List of dataset dictionaries.
        """
        return config.discover_datasets()

    @staticmethod
    def get_dataset_history(dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get historical data for a dataset.

        Args:
            dataset_id: Dataset ID (country code or upload ID).

        Returns:
            Dictionary with dataset info and historical records, or None.
        """
        df = config.get_dataset_history(dataset_id)
        if df is None or df.empty:
            return None

        # Convert to records
        records = df.to_dict("records")

        # Convert dates to strings for JSON serialization
        for record in records:
            if "date" in record and pd.notna(record["date"]):
                if isinstance(record["date"], pd.Timestamp):
                    record["date"] = record["date"].strftime("%Y-%m-%d")
                else:
                    record["date"] = str(record["date"])

        # Determine dataset name
        name = dataset_id
        source = "unknown"
        if dataset_id in config.COUNTRIES:
            name = config.COUNTRIES[dataset_id]
            source = "worldbank"
        elif dataset_id.startswith("upload_"):
            name = dataset_id.replace("upload_", "") + ".csv"
            source = "user_upload"

        return {
            "dataset_id": dataset_id,
            "name": name,
            "source": source,
            "records": len(records),
            "columns": list(df.columns),
            "date_min": str(df["date"].min()) if "date" in df.columns else None,
            "date_max": str(df["date"].max()) if "date" in df.columns else None,
            "data": records,
        }

    @staticmethod
    def validate_uploaded_csv(df: pd.DataFrame) -> tuple:
        """
        Validate a user-uploaded CSV DataFrame.

        Args:
            df: DataFrame from uploaded CSV.

        Returns:
            Tuple of (is_valid: bool, message: str, columns: list).
        """
        # Check required columns
        missing_required = [c for c in DatasetManager.REQUIRED_COLUMNS if c not in df.columns]
        if missing_required:
            return False, f"Missing required columns: {missing_required}", []

        # Check for unknown columns
        unknown = [c for c in df.columns if c not in DatasetManager.ALLOWED_COLUMNS and c not in ["date", "country_code", "country_name"]]
        if unknown:
            logger.warning(f"Unknown columns in uploaded CSV (will be ignored): {unknown}")

        # Validate date column
        if "date" in df.columns:
            try:
                pd.to_datetime(df["date"])
            except Exception:
                return False, "Column 'date' must be parseable as datetime", []

        # Validate numeric columns
        numeric_cols = [c for c in df.columns if c != "date"]
        for col in numeric_cols:
            try:
                pd.to_numeric(df[col], errors="raise")
            except Exception:
                return False, f"Column '{col}' must be numeric", []

        valid_columns = [c for c in df.columns if c in DatasetManager.ALLOWED_COLUMNS or c in ["date", "country_code", "country_name"]]
        return True, "Valid CSV", valid_columns

    @staticmethod
    def save_uploaded_csv(file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Save and validate an uploaded CSV file.

        Args:
            file_content: Raw bytes of the uploaded file.
            filename: Original filename.

        Returns:
            Dictionary with upload result info.
        """
        # Ensure uploads directory exists
        config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

        # Clean filename
        safe_filename = Path(filename).name
        if not safe_filename.endswith(".csv"):
            safe_filename += ".csv"

        # Save file
        file_path = config.UPLOADS_DIR / safe_filename
        with open(file_path, "wb") as f:
            f.write(file_content)

        logger.info(f"Uploaded file saved: {file_path}")

        # Validate
        try:
            df = pd.read_csv(file_path, parse_dates=["date"])
        except Exception as e:
            file_path.unlink(missing_ok=True)
            raise ValueError(f"Could not parse CSV: {str(e)}")

        is_valid, message, valid_columns = DatasetManager.validate_uploaded_csv(df)

        if not is_valid:
            file_path.unlink(missing_ok=True)
            raise ValueError(message)

        # Generate dataset ID
        dataset_id = f"upload_{Path(safe_filename).stem}"

        return {
            "dataset_id": dataset_id,
            "name": safe_filename,
            "records": len(df),
            "columns": valid_columns,
            "date_min": str(df["date"].min()),
            "date_max": str(df["date"].max()),
            "status": "success",
            "message": message,
        }


# Service instance
dataset_manager = DatasetManager()
