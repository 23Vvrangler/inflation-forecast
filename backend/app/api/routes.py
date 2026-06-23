"""
API route definitions.
Implements all REST endpoints for the application.
"""
import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse

from app.api.schemas import (
    PredictionRequest, PredictionResult,
    DatasetListResponse, DatasetHistoryResponse,
    UploadResponse, ModelInfo, HealthCheck, ErrorResponse
)
from app.services.prediction import prediction_service
from app.services.dataset_manager import dataset_manager
from app.core.config import config

logger = logging.getLogger(__name__)

# Create router with prefix
router = APIRouter(prefix=config.API_V1_PREFIX)


@router.get(
    "/health",
    response_model=HealthCheck,
    tags=["System"],
    summary="Health Check",
    description="Check if the API and model are operational.",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service unavailable - model not loaded"}
    }
)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    Returns the current status of the API and whether the model is loaded.
    """
    now = datetime.utcnow().isoformat() + "Z"

    status_str = "healthy" if config.is_model_loaded() else "degraded"

    response = {
        "status": status_str,
        "timestamp": now,
        "model_loaded": config.is_model_loaded(),
        "version": config.VERSION
    }

    if not config.is_model_loaded():
        logger.warning("Health check: model not loaded")

    return response


@router.get(
    "/model/info",
    response_model=ModelInfo,
    tags=["Model"],
    summary="Model Information",
    description="Get metadata and performance metrics of the loaded ML model.",
    responses={
        200: {"description": "Model information retrieved successfully"},
        503: {"description": "Model not loaded"}
    }
)
async def get_model_info() -> Dict[str, Any]:
    """
    Retrieve information about the loaded ML model.
    Includes algorithm, version, dataset info, features, and performance metrics.
    """
    info = prediction_service.get_model_info()
    return info


@router.get(
    "/datasets",
    response_model=DatasetListResponse,
    tags=["Datasets"],
    summary="List Available Datasets",
    description="List all pre-loaded country datasets and user-uploaded datasets.",
    responses={
        200: {"description": "Dataset list retrieved successfully"}
    }
)
async def list_datasets() -> Dict[str, Any]:
    """
    List all available datasets.

    Returns pre-loaded country datasets (from World Bank API) and
    any user-uploaded CSV files.
    """
    datasets = dataset_manager.list_datasets()

    # Convert to schema format
    dataset_infos = [
        {
            "id": d["id"],
            "name": d["name"],
            "source": d["source"],
            "records": d["records"],
        }
        for d in datasets
    ]

    return {
        "datasets": dataset_infos,
        "count": len(dataset_infos),
    }


@router.get(
    "/datasets/{dataset_id}/history",
    response_model=DatasetHistoryResponse,
    tags=["Datasets"],
    summary="Get Dataset History",
    description="Retrieve historical inflation data for a specific dataset.",
    responses={
        200: {"description": "Historical data retrieved successfully"},
        404: {"description": "Dataset not found"}
    }
)
async def get_dataset_history(dataset_id: str) -> Dict[str, Any]:
    """
    Get historical data for a specific dataset.

    - **dataset_id**: Country code (PER, ARG, BRA, CHL, COL, MEX) or upload ID.

    Returns the full historical time series for plotting.
    """
    history = dataset_manager.get_dataset_history(dataset_id)

    if history is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found. Use /datasets to list available datasets."
        )

    return history


@router.post(
    "/datasets/upload",
    response_model=UploadResponse,
    tags=["Datasets"],
    summary="Upload Custom CSV Dataset",
    description="Upload a custom CSV file with inflation data for prediction.",
    responses={
        200: {"description": "Upload successful"},
        400: {"description": "Invalid file format or missing columns"},
        422: {"description": "Validation error"}
    }
)
async def upload_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload a custom CSV dataset.

    The CSV must contain at minimum:
    - **date**: Date column (parseable as datetime)
    - **inflation_rate**: Inflation rate values (numeric)

    Optional columns: gdp_growth, interest_rate, and any text-derived features.

    The uploaded dataset will be added to the dataset list and can be used
    for predictions and visualization.
    """
    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are accepted. Please upload a .csv file."
        )

    try:
        content = await file.read()
        result = dataset_manager.save_uploaded_csv(content, file.filename)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.post(
    "/predict",
    response_model=PredictionResult,
    tags=["Prediction"],
    summary="Single Inflation Prediction",
    description="Predict next-month inflation rate (t+1) based on input features.",
    responses={
        200: {"description": "Prediction successful"},
        400: {"description": "Invalid input data"},
        422: {"description": "Validation error"},
        503: {"description": "Model not loaded"}
    }
)
async def predict(request: PredictionRequest) -> Dict[str, Any]:
    """
    Make an inflation prediction for month t+1.

    - **features**: Dictionary of feature names to numeric values.
      Required features depend on the trained model. Common features include:
      - inflation_lag_1, inflation_lag_3, inflation_lag_6, inflation_lag_12
      - rolling_mean_3, rolling_mean_6, rolling_mean_12
      - gdp_growth, interest_rate
      - text_sentiment_score, text_tfidf_mean, text_keyword_inflation

    Returns the predicted inflation rate and model version.
    """
    if not config.is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please ensure the model file is available and the training pipeline has been run."
        )

    try:
        result = prediction_service.predict(request.features)
        return result
    except ValueError as e:
        logger.error(f"Validation error in prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
