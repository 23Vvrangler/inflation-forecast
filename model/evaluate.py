"""
Evaluation script for the inflation forecasting regression model.

Computes regression metrics and generates visualizations:
    - RMSE, MAE, MAPE, R2, Explained Variance
    - Predictions vs Actual scatter plot
    - Residuals plot
    - Feature importance (for tree-based models)
    - Time series plot of predictions vs actual

Usage:
    python -m model.evaluate

Output:
    model/artifacts/metrics.json
    model/artifacts/predictions_vs_actual.png
    model/artifacts/residuals.png
    model/artifacts/feature_importance.png
"""
import os
import sys
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.utils import load_config, setup_logging


def load_test_data(config: dict) -> tuple:
    """
    Load test dataset from CSV.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (X_test, y_test, test_df).
    """
    test_path = config["paths"]["test_file"]
    target_col = config["target"]["column"]

    if not Path(test_path).exists():
        raise FileNotFoundError(
            f"Test file not found: {test_path}. "
            "Please run model/train.py first."
        )

    test_df = pd.read_csv(test_path, parse_dates=["date"])

    numeric_features = config["features"]["numeric"]
    text_features = config["features"]["text"]
    feature_cols = numeric_features + text_features
    available_features = [c for c in feature_cols if c in test_df.columns]

    X_test = test_df[available_features]
    y_test = test_df[target_col]

    logger.info(f"Test data loaded: {len(test_df)} samples, {len(available_features)} features")
    return X_test, y_test, test_df


def load_trained_model(config: dict):
    """
    Load the serialized trained model.

    Args:
        config: Configuration dictionary.

    Returns:
        Loaded pipeline object.
    """
    model_path = config["paths"]["model_file"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Please run model/train.py first."
        )

    pipeline = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return pipeline


def compute_metrics(y_true, y_pred) -> dict:
    """
    Compute regression evaluation metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary of metrics.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    evs = float(explained_variance_score(y_true, y_pred))

    # MAPE (handle zero values)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100

    # Directional accuracy (did we predict the direction of change correctly?)
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    if len(y_true_diff) > 0:
        directional_accuracy = float(np.mean(
            (y_true_diff > 0) == (y_pred_diff > 0)
        ))
    else:
        directional_accuracy = None

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "explained_variance": evs,
        "directional_accuracy": directional_accuracy,
        "mean_actual": float(np.mean(y_true)),
        "std_actual": float(np.std(y_true)),
        "mean_predicted": float(np.mean(y_pred)),
        "std_predicted": float(np.std(y_pred)),
    }

    return metrics


def plot_predictions_vs_actual(y_true, y_pred, output_path: str) -> None:
    """
    Generate scatter plot of predictions vs actual values.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        output_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="black", s=50, color="steelblue")

    # Perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")

    ax.set_xlabel("Actual Inflation Rate (%)", fontsize=12)
    ax.set_ylabel("Predicted Inflation Rate (%)", fontsize=12)
    ax.set_title("Predictions vs Actual Values", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add R2 annotation
    r2 = r2_score(y_true, y_pred)
    ax.text(
        0.05, 0.95, f"R2 = {r2:.4f}",
        transform=ax.transAxes, fontsize=12,
        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Predictions vs Actual plot saved to {output_path}")


def plot_residuals(y_true, y_pred, output_path: str) -> None:
    """
    Generate residuals plot.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        output_path: Path to save the plot.
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors="black", s=50, color="coral")
    ax1.axhline(y=0, color="red", linestyle="--", lw=2)
    ax1.set_xlabel("Predicted Inflation Rate (%)", fontsize=12)
    ax1.set_ylabel("Residuals (Actual - Predicted)", fontsize=12)
    ax1.set_title("Residuals vs Predicted", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Histogram of residuals
    ax2 = axes[1]
    ax2.hist(residuals, bins=20, color="steelblue", edgecolor="black", alpha=0.7)
    ax2.axvline(x=0, color="red", linestyle="--", lw=2)
    ax2.set_xlabel("Residuals", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Distribution of Residuals", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Residuals plot saved to {output_path}")


def plot_feature_importance(pipeline, feature_names: list, output_path: str) -> None:
    """
    Generate feature importance bar plot.

    Args:
        pipeline: Trained pipeline.
        feature_names: List of feature names.
        output_path: Path to save the plot.
    """
    regressor = pipeline.named_steps["regressor"]

    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_

        # Sort by importance
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]

        # Take top 20
        top_n = min(20, len(sorted_features))
        sorted_features = sorted_features[:top_n]
        sorted_importances = sorted_importances[:top_n]

        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(sorted_features))
        ax.barh(y_pos, sorted_importances, align="center", color="steelblue")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features)
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance", fontsize=12)
        ax.set_title("Top Feature Importances", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Feature importance plot saved to {output_path}")
    else:
        logger.info("Model does not support feature_importances_, skipping plot")


def plot_time_series(test_df: pd.DataFrame, y_pred: np.ndarray, output_path: str) -> None:
    """
    Generate time series plot of actual vs predicted inflation.

    Args:
        test_df: Test DataFrame with date column.
        y_pred: Predicted values.
        output_path: Path to save the plot.
    """
    if "date" not in test_df.columns:
        logger.warning("No date column in test data, skipping time series plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    dates = test_df["date"]
    y_true = test_df[test_df.columns[test_df.columns.str.contains("inflation")][0]].values

    ax.plot(dates, y_true, "o-", label="Actual", color="steelblue", linewidth=2, markersize=6)
    ax.plot(dates, y_pred, "s--", label="Predicted", color="coral", linewidth=2, markersize=6)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Inflation Rate (%)", fontsize=12)
    ax.set_title("Inflation Forecast: Actual vs Predicted (Test Set)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Time series plot saved to {output_path}")


def save_metrics(metrics: dict, config: dict) -> None:
    """
    Save metrics to JSON file.

    Args:
        metrics: Dictionary of computed metrics.
        config: Configuration dictionary.
    """
    metrics_path = config["paths"]["metrics_file"]

    # Add metadata
    metrics["model_version"] = config["versioning"]["model_version"]
    metrics["dataset"] = config["versioning"]["experiment_name"]
    metrics["algorithm"] = config["model"]["algorithm"]
    metrics["timestamp"] = pd.Timestamp.now().isoformat()

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to {metrics_path}")


def print_metrics_summary(metrics: dict) -> None:
    """
    Print a formatted summary of metrics to console.

    Args:
        metrics: Dictionary of computed metrics.
    """
    logger.info("-" * 50)
    logger.info("EVALUATION RESULTS - INFLATION FORECAST")
    logger.info("-" * 50)
    logger.info(f"RMSE:                    {metrics['rmse']:.4f}")
    logger.info(f"MAE:                     {metrics['mae']:.4f}")
    logger.info(f"MAPE:                    {metrics['mape']:.2f}%")
    logger.info(f"R2 Score:                {metrics['r2']:.4f}")
    logger.info(f"Explained Variance:      {metrics['explained_variance']:.4f}")
    if metrics.get("directional_accuracy") is not None:
        logger.info(f"Directional Accuracy:    {metrics['directional_accuracy']:.4f}")
    logger.info("-" * 50)
    logger.info(f"Mean Actual:             {metrics['mean_actual']:.4f}")
    logger.info(f"Mean Predicted:          {metrics['mean_predicted']:.4f}")
    logger.info(f"Std Actual:              {metrics['std_actual']:.4f}")
    logger.info(f"Std Predicted:           {metrics['std_predicted']:.4f}")
    logger.info("-" * 50)


def main():
    """
    Main evaluation execution.
    """
    global logger
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("Starting Model Evaluation - Inflation Forecast")
    logger.info("=" * 60)

    # Load configuration
    config = load_config()

    # Load test data
    X_test, y_test, test_df = load_test_data(config)
    logger.info(f"Test data loaded: {len(X_test)} samples")

    # Load trained model
    pipeline = load_trained_model(config)

    # Make predictions
    y_pred = pipeline.predict(X_test)
    logger.info(f"Predictions made: {len(y_pred)} samples")

    # Compute metrics
    metrics = compute_metrics(y_test.values, y_pred)

    # Save and print metrics
    save_metrics(metrics, config)
    print_metrics_summary(metrics)

    # Generate plots
    plot_predictions_vs_actual(
        y_test.values, y_pred,
        config["paths"]["predictions_plot"]
    )
    plot_residuals(
        y_test.values, y_pred,
        config["paths"]["residuals_plot"]
    )

    # Feature importance
    numeric_features = config["features"]["numeric"]
    text_features = config["features"]["text"]
    all_features = numeric_features + text_features
    available_features = [c for c in all_features if c in X_test.columns]
    plot_feature_importance(
        pipeline, available_features,
        config["paths"]["feature_importance_plot"]
    )

    # Time series plot
    plot_time_series(test_df, y_pred, config["paths"]["predictions_plot"].replace("predictions_vs_actual", "time_series"))

    logger.info("=" * 60)
    logger.info("Evaluation Completed Successfully")
    logger.info("=" * 60)
    logger.info(f"Metrics file: {config['paths']['metrics_file']}")
    logger.info(f"Plots saved to: {config['paths']['artifacts_dir']}")


if __name__ == "__main__":
    main()
