"""
Pydantic schemas for request validation and response serialization.
Defines the data contracts for all API endpoints.
"""
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """
    Request schema for single inflation prediction.
    Expects a dictionary of feature names to numeric values.
    """
    features: Dict[str, float] = Field(
        ...,
        description="Feature values for prediction. Example: {"inflation_lag_1": 3.2, "gdp_growth": 2.1}",
        example={
            "inflation_lag_1": 3.2,
            "inflation_lag_3": 3.5,
            "inflation_lag_6": 3.8,
            "inflation_lag_12": 4.1,
            "rolling_mean_3": 3.3,
            "rolling_mean_6": 3.6,
            "rolling_mean_12": 3.8,
            "rolling_std_6": 0.5,
            "gdp_growth": 2.1,
            "interest_rate": 5.5,
            "month_sin": 0.5,
            "month_cos": 0.866,
            "text_sentiment_score": -0.3,
            "text_tfidf_mean": 0.12,
            "text_keyword_inflation": 45,
        }
    )


class PredictionResult(BaseModel):
    """
    Response schema for a single prediction result.
    """
    prediction: float = Field(..., description="Predicted inflation rate for t+1 (%)")
    model_version: str = Field(..., description="Version of the model used")
    input_features: Dict[str, float] = Field(..., description="Input features received")


class DatasetInfo(BaseModel):
    """
    Schema for dataset metadata.
    """
    id: str = Field(..., description="Dataset identifier")
    name: str = Field(..., description="Dataset display name")
    source: str = Field(..., description="Data source: worldbank or user_upload")
    records: int = Field(..., description="Number of records")


class DatasetListResponse(BaseModel):
    """
    Response schema for listing datasets.
    """
    datasets: List[DatasetInfo]
    count: int


class DatasetHistoryResponse(BaseModel):
    """
    Response schema for dataset historical data.
    """
    dataset_id: str
    name: str
    source: str
    records: int
    columns: List[str]
    date_min: Optional[str]
    date_max: Optional[str]
    data: List[Dict[str, Any]]


class UploadResponse(BaseModel):
    """
    Response schema for CSV upload.
    """
    dataset_id: str
    name: str
    records: int
    columns: List[str]
    date_min: str
    date_max: str
    status: str
    message: str


class ModelInfo(BaseModel):
    """
    Response schema for model metadata and performance metrics.
    """
    model_loaded: bool
    project_name: str
    api_version: str
    model: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None


class HealthCheck(BaseModel):
    """
    Response schema for health check endpoint.
    """
    status: str
    timestamp: str
    model_loaded: bool
    version: str


class ErrorResponse(BaseModel):
    """
    Response schema for error responses.
    """
    detail: str
