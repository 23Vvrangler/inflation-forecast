"""
Training script for the inflation forecasting regression model.

Executes the full training pipeline:
    1. Load processed features (numeric + text)
    2. Temporal train/test split (NO random shuffle - time series)
    3. Train regression model with sklearn Pipeline
    4. Evaluate on test set
    5. Save model artifact, metadata, and metrics

Usage:
    python -m model.train

Output:
    model/artifacts/model_v1.pkl
    model/artifacts/metadata.json
"""
import os
import sys
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.utils import load_config, setup_logging, ensure_directories
from model.pipeline import build_pipeline


def load_features(config: dict) -> pd.DataFrame:
    """
    Load the processed features dataset.

    Args:
        config: Configuration dictionary.

    Returns:
        DataFrame with all features and target column.
    """
    features_file = config["paths"]["features_file"]

    if not Path(features_file).exists():
        raise FileNotFoundError(
            f"Features file not found: {features_file}. "
            "Please run the feature engineering pipeline first."
        )

    df = pd.read_csv(features_file, parse_dates=["date"])
    logger.info(f"Loaded features dataset: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")
    return df


def temporal_split(df: pd.DataFrame, config: dict) -> tuple:
    """
    Split data temporally for time series.

    Uses the last N months as test set to avoid data leakage.
    Does NOT shuffle - preserves temporal order.

    Args:
        df: Full features DataFrame sorted by date.
        config: Configuration dictionary with split settings.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, train_df, test_df).
    """
    split_config = config["split"]
    test_months = split_config["test_months"]
    target_col = config["target"]["column"]

    # Ensure sorted by date
    df = df.sort_values("date").reset_index(drop=True)

    # Temporal split: last N months as test
    cutoff_date = df["date"].max() - pd.DateOffset(months=test_months)

    train_df = df[df["date"] <= cutoff_date].copy()
    test_df = df[df["date"] > cutoff_date].copy()

    # Define feature columns
    numeric_features = config["features"]["numeric"]
    text_features = config["features"]["text"]
    feature_cols = numeric_features + text_features

    # Filter to available columns
    available_features = [c for c in feature_cols if c in df.columns]
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        logger.warning(f"Missing features (will be ignored): {missing_features}")

    X_train = train_df[available_features]
    y_train = train_df[target_col]
    X_test = test_df[available_features]
    y_test = test_df[target_col]

    logger.info(f"Temporal split: train={len(train_df)} rows (up to {cutoff_date.date()}), test={len(test_df)} rows")
    logger.info(f"Features used: {available_features}")
    logger.info(f"Target: {target_col}")

    return X_train, X_test, y_train, y_test, train_df, test_df


def train_model(X_train: pd.DataFrame, y_train: pd.Series, config: dict):
    """
    Train the regression pipeline.

    Args:
        X_train: Training features.
        y_train: Training target values.
        config: Configuration dictionary.

    Returns:
        Trained pipeline object.
    """
    pipeline = build_pipeline(config)

    logger.info(f"Training model: {config['model']['algorithm']}")
    logger.info(f"Hyperparameters: {config['model']['hyperparameters']}")
    logger.info(f"Training samples: {len(X_train)}")

    pipeline.fit(X_train, y_train)

    logger.info("Model training completed successfully")
    return pipeline


def save_model(pipeline, config: dict) -> str:
    """
    Serialize the trained model to disk.

    Args:
        pipeline: Trained sklearn pipeline.
        config: Configuration dictionary.

    Returns:
        Path to the saved model file.
    """
    model_path = config["paths"]["model_file"]

    joblib.dump(pipeline, model_path)

    logger.info(f"Model saved to {model_path}")
    logger.info(f"File size: {os.path.getsize(model_path) / 1024:.2f} KB")

    return model_path


def save_training_metadata(config: dict, X_train, y_train, X_test, y_test) -> None:
    """
    Save metadata about the training run.

    Args:
        config: Configuration dictionary.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
    """
    metadata = {
        "model_version": config["versioning"]["model_version"],
        "experiment_name": config["versioning"]["experiment_name"],
        "algorithm": config["model"]["algorithm"],
        "hyperparameters": config["model"]["hyperparameters"],
        "features": {
            "numeric": config["features"]["numeric"],
            "text": config["features"]["text"],
        },
        "target": config["target"],
        "preprocessing": config["preprocessing"],
        "split": config["split"],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count": len(X_train.columns),
    }

    metadata_path = config["paths"]["metadata_file"]
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Training metadata saved to {metadata_path}")


def main():
    """
    Main training execution.
    """
    global logger
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("Starting ML Training Pipeline - Inflation Forecast")
    logger.info("=" * 60)

    # Load configuration
    config = load_config()
    logger.info(f"Configuration loaded: {config['versioning']['experiment_name']}")

    # Ensure directories exist
    ensure_directories(config)

    # Load features dataset
    df = load_features(config)

    # Temporal split
    X_train, X_test, y_train, y_test, train_df, test_df = temporal_split(df, config)

    # Train model
    pipeline = train_model(X_train, y_train, config)

    # Save model
    save_model(pipeline, config)

    # Save metadata
    save_training_metadata(config, X_train, y_train, X_test, y_test)

    # Save splits for evaluation
    train_df.to_csv(config["paths"]["train_file"], index=False)
    test_df.to_csv(config["paths"]["test_file"], index=False)
    logger.info(f"Train/test splits saved")

    logger.info("=" * 60)
    logger.info("Training Pipeline Completed Successfully")
    logger.info("=" * 60)
    logger.info(f"Model version: {config['versioning']['model_version']}")
    logger.info(f"Artifacts location: {config['paths']['artifacts_dir']}")
    logger.info("Next step: Run model/evaluate.py to assess performance")


if __name__ == "__main__":
    main()
