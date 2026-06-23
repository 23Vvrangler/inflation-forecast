"""
FastAPI application entry point.
Configures the app, loads the model at startup, and registers routes.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import config
from app.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup (model loading) and shutdown events.
    """
    # Startup: Load model
    logger.info("=" * 60)
    logger.info("Starting Inflation Forecast API")
    logger.info("=" * 60)

    success = config.load_model()
    if success:
        logger.info("Model loaded successfully - API ready")
    else:
        logger.warning("Model not loaded - API running in degraded mode")
        logger.warning("Predictions will not be available until model is loaded")
        logger.warning("Run: python -m model.train")

    logger.info(f"API documentation available at: http://localhost:8000/docs")
    logger.info(f"Available datasets: {len(config.discover_datasets())}")

    yield

    # Shutdown
    logger.info("=" * 60)
    logger.info("Shutting down Inflation Forecast API")
    logger.info("=" * 60)


# Create FastAPI application
app = FastAPI(
    title=config.PROJECT_NAME,
    description=config.DESCRIPTION,
    version=config.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, tags=["API v1"])


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint.
    Returns basic API information and available endpoints.
    """
    datasets = config.discover_datasets()
    return {
        "message": "Welcome to the Inflation Forecast API",
        "project": config.PROJECT_NAME,
        "version": config.VERSION,
        "model_loaded": config.is_model_loaded(),
        "available_datasets": len(datasets),
        "documentation": "/docs",
        "endpoints": {
            "health": "/api/v1/health",
            "model_info": "/api/v1/model/info",
            "datasets": "/api/v1/datasets",
            "dataset_history": "/api/v1/datasets/{dataset_id}/history",
            "upload_dataset": "/api/v1/datasets/upload",
            "predict": "/api/v1/predict",
        }
    }
