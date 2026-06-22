# Scraping Inflation Data And Developing A Model With Data From CommonCrawl

A complete Data Mining pipeline that scrapes inflation data from the World Bank API, processes web text data simulating a CommonCrawl pipeline, and builds a regression model to forecast monthly inflation rates. Features a multi-dataset Streamlit frontend and a FastAPI backend, fully containerized with Docker Compose.

---

## Table of Contents

- [Overview](#overview)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Running Locally](#running-locally)
- [API Documentation](#api-documentation)
- [Data Sources](#data-sources)
- [Model Training](#model-training)
- [Endpoints](#endpoints)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [License](#license)

---

## Overview

This project demonstrates a complete data mining pipeline applied to economic forecasting:

1. **Data Acquisition:** Scrapes inflation rates (CPI) from the World Bank API for multiple Latin American countries (Peru, Argentina, Brazil, Chile, Colombia, Mexico).
2. **Web Data Simulation (CommonCrawl):** Simulates a CommonCrawl pipeline by scraping economic news articles via RSS feeds, cleaning HTML, extracting text, and generating NLP features (TF-IDF, keyword frequencies, sentiment scores).
3. **Feature Engineering:** Combines numeric time-series features (lag inflation, rolling means, GDP, interest rates) with text-derived features to create a rich dataset for modeling.
4. **Machine Learning:** Trains a regression model (RandomForest / XGBoost) to predict next-month inflation (t+1).
5. **API & Frontend:** Exposes the model via FastAPI REST endpoints and provides an interactive Streamlit frontend where users can select countries, upload custom CSV datasets, visualize historical inflation series, and run predictions.
6. **Deployment:** Fully containerized with Docker Compose for one-command execution.

**Key Features:**
- Multi-country dataset selector (Peru, Argentina, Brazil, Chile, Colombia, Mexico)
- Custom CSV upload and validation
- Interactive time-series visualizations (Plotly)
- Real-time inflation forecasting via web UI
- Auto-generated Swagger API documentation
- Reproducible ML pipeline with YAML configuration
- Docker Compose orchestration

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data Sources | World Bank API + RSS Feeds | Official inflation data + economic news proxy |
| Scraping | requests + BeautifulSoup4 + feedparser | HTTP requests, HTML parsing, RSS feed reading |
| NLP / Text | NLTK + scikit-learn TF-IDF | Text cleaning, tokenization, vectorization, sentiment |
| ML Framework | scikit-learn + XGBoost + pandas | Regression modeling, feature engineering, preprocessing |
| Backend | FastAPI + Uvicorn | REST API with auto-generated OpenAPI docs |
| Frontend | Streamlit + Plotly | Interactive web UI with charts and multi-dataset support |
| Containerization | Docker + Docker Compose | Service orchestration and portability |
| Configuration | YAML | Centralized hyperparameter and dataset configuration |

---

## Project Structure

```
inflation-forecast/
|
|-- backend/                    # FastAPI server + business logic
|   |-- app/
|   |   |-- api/
|   |   |   |-- __init__.py
|   |   |   |-- routes.py         # REST endpoints (predict, datasets, upload, health)
|   |   |   |-- schemas.py        # Pydantic request/response models
|   |   |-- core/
|   |   |   |-- __init__.py
|   |   |   |-- config.py         # App config + dataset loader + model loader
|   |   |-- services/
|   |   |   |-- __init__.py
|   |   |   |-- prediction.py     # ML inference service
|   |   |   |-- dataset_manager.py # Multi-dataset management (countries + uploads)
|   |   |-- main.py               # FastAPI application entry point
|   |-- Dockerfile
|   |-- requirements.txt
|
|-- frontend/                    # Streamlit application
|   |-- app.py                   # Multi-dataset UI: selector, uploader, charts, prediction
|   |-- Dockerfile
|   |-- requirements.txt
|
|-- scraping/                    # Data acquisition modules
|   |-- __init__.py
|   |-- worldbank_scraper.py     # World Bank API: inflation, GDP, interest rates
|   |-- news_scraper.py          # RSS/API news scraping: proxy for CommonCrawl
|   |-- text_processor.py        # HTML cleaning, tokenization, TF-IDF, sentiment
|
|-- model/                       # ML pipeline and artifacts
|   |-- config.yaml              # Centralized ML configuration
|   |-- pipeline.py              # sklearn pipeline builder (numeric + text features)
|   |-- train.py                 # Training script for regression model
|   |-- evaluate.py              # Evaluation: RMSE, MAE, MAPE, R2, plots
|   |-- utils.py                 # Shared utilities (config loader, logger)
|   |-- artifacts/               # Serialized models and metrics
|   |   |-- .gitkeep
|
|-- data/
|   |-- raw/                     # Original datasets downloaded by scrapers
|   |   |-- .gitkeep
|   |-- processed/               # Cleaned data + engineered features
|   |   |-- .gitkeep
|   |-- external/                # Raw web text (HTML files from news scraper)
|   |   |-- .gitkeep
|   |-- splits/                  # Train / test temporal splits
|   |   |-- .gitkeep
|   |-- uploads/                 # User-uploaded CSV files (frontend)
|   |   |-- .gitkeep
|
|-- notebooks/
|   |-- 01_inflation_eda.ipynb   # EDA of inflation time series by country
|   |-- 02_web_scraping.ipynb    # Demonstration of HTML scraping and text extraction
|   |-- 03_feature_engineering.ipynb # Lag features, rolling stats, TF-IDF merge
|   |-- 04_modeling.ipynb          # Model training, comparison, and selection
|
|-- docs/                          # Academic documentation
|   |-- Arquitectura_Inflation_CommonCrawl_MVP.docx
|   |-- informe.docx               # Final academic report (CRISP-DM)
|   |-- presentacion.pptx          # Presentation slides
|
|-- docker/
|   |-- nginx.conf                 # Optional reverse proxy configuration
|
|-- docker-compose.yml             # Full service orchestration
|-- requirements.txt               # Global dependency reference
|-- .gitignore                     # Git ignore rules
|-- LICENSE                        # MIT License
|-- README.md                      # This file
```

---

## Prerequisites

- Docker Engine 24.0+ and Docker Compose 2.20+
- OR Python 3.11+ with pip (for local development)
- Git (for cloning)

---

## Quick Start

The fastest way to run the entire project is with Docker Compose.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/inflation-forecast.git
cd inflation-forecast
```

### 2. Train the Model (First Time Only)

The model artifacts must be generated before the backend can serve predictions.

```bash
# Option A: Train locally (requires Python 3.11+)
python -m scraping.worldbank_scraper
python -m scraping.news_scraper
python -m scraping.text_processor
python -m model.train
python -m model.evaluate

# Option B: Train using Docker (builds backend image first)
docker compose --profile training run --rm model-trainer
```

After training, verify these files exist:
- `model/artifacts/model_v1.pkl`
- `model/artifacts/metadata.json`
- `model/artifacts/metrics.json`

### 3. Start All Services

```bash
docker compose up --build
```

This command will:
1. Build the backend Docker image (FastAPI + model + scraping code)
2. Build the frontend Docker image (Streamlit)
3. Start the backend service on port 8000
4. Start the frontend service on port 8501 (after backend is healthy)

### 4. Access the Application

| Service | URL | Description |
|---------|-----|-------------|
| Frontend (Streamlit) | http://localhost:8501 | Interactive UI: select country, upload CSV, visualize, predict |
| API Docs (Swagger) | http://localhost:8000/docs | Interactive OpenAPI documentation |
| API Root | http://localhost:8000 | API information and endpoint list |
| Health | http://localhost:8000/api/v1/health | Service health check |

### 5. Stop Services

```bash
docker compose down
```

To remove volumes and images completely:

```bash
docker compose down --volumes --rmi all
```

---

## Running Locally

If you prefer to run without Docker, follow these steps.

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate   # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Data and Train Model

```bash
python -m scraping.worldbank_scraper
python -m scraping.news_scraper
python -m scraping.text_processor
python -m model.train
python -m model.evaluate
```

### 4. Start the Backend

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Start the Frontend (New Terminal)

```bash
cd frontend
streamlit run app.py --server.port 8501
```

---

## API Documentation

FastAPI automatically generates interactive API documentation.

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI Schema:** http://localhost:8000/openapi.json

You can test all endpoints directly from the Swagger UI without any additional tools.

---

## Data Sources

### World Bank API (Structured Data)

The World Bank provides a free REST API for economic indicators. No API key is required.

- **Indicator:** `FP.CPI.TOTL.ZG` (Inflation, consumer prices, annual %)
- **Countries:** Peru (PER), Argentina (ARG), Brazil (BRA), Chile (CHL), Colombia (COL), Mexico (MEX)
- **URL Format:** `https://api.worldbank.org/v2/country/{code}/indicator/FP.CPI.TOTL.ZG?date=2000:2024&format=json`

### News Scraping (CommonCrawl Proxy)

CommonCrawl stores petabytes of raw web data. This project simulates the CommonCrawl pipeline at academic scale by:

1. Scraping RSS feeds from economic news sources (BBC Business, Reuters, etc.)
2. Downloading the raw HTML of each article
3. Cleaning HTML with BeautifulSoup4 (removing scripts, styles, ads)
4. Extracting plain text, tokenizing with NLTK, computing TF-IDF
5. Aggregating monthly keyword frequencies and sentiment scores

This demonstrates understanding of the CommonCrawl workflow (raw HTML -> clean text -> NLP features) without requiring cloud infrastructure to process 100+ TB.

---

## Model Training

The training pipeline is fully configurable via `model/config.yaml`.

### Configuration Parameters

```yaml
dataset:
  countries: ["PER", "ARG", "BRA", "CHL", "COL", "MEX"]
  start_year: 2000
  end_year: 2024
  target_column: "inflation_rate"
  feature_columns:
    - "inflation_lag_1"
    - "inflation_lag_3"
    - "inflation_lag_12"
    - "rolling_mean_3"
    - "rolling_mean_12"
    - "gdp_growth"
    - "interest_rate"
    - "tfidf_mean"
    - "keyword_inflation_count"
    - "sentiment_score"

model:
  algorithm: "xgboost"
  hyperparameters:
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
```

### Training Steps

1. **Scrape World Bank:** Download inflation, GDP, and interest rate data for all countries.
2. **Scrape News:** Download RSS feeds and article HTML for the target period.
3. **Process Text:** Clean HTML, tokenize, compute TF-IDF and keyword frequencies per month.
4. **Feature Engineering:** Create lag features, rolling statistics, merge numeric + text features.
5. **Temporal Split:** Use the last 12 months as test set (no random shuffle to prevent data leakage).
6. **Train Model:** Fit regression pipeline with GridSearchCV for hyperparameter tuning.
7. **Evaluate:** Compute RMSE, MAE, MAPE, R2. Generate prediction vs actual plots.
8. **Serialize:** Save model as `model_v1.pkl` using joblib.

---

## Endpoints

### GET /api/v1/health

Health check endpoint. Returns API status and model load state.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-06-21T14:30:00Z",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### GET /api/v1/datasets

List all available datasets (pre-loaded countries + user uploads).

**Response:**
```json
{
  "datasets": [
    {"id": "PER", "name": "Peru", "source": "worldbank", "records": 288},
    {"id": "ARG", "name": "Argentina", "source": "worldbank", "records": 288},
    {"id": "upload_20260621", "name": "custom_upload.csv", "source": "user", "records": 120}
  ]
}
```

### GET /api/v1/datasets/{dataset_id}/history

Retrieve historical inflation series for a specific dataset.

**Response:**
```json
{
  "dataset_id": "PER",
  "country": "Peru",
  "data": [
    {"date": "2023-01", "inflation_rate": 3.2, "gdp_growth": 2.1},
    {"date": "2023-02", "inflation_rate": 3.1, "gdp_growth": 2.3}
  ]
}
```

### POST /api/v1/datasets/upload

Upload a custom CSV dataset. Validates required columns and stores for prediction.

**Request:** `multipart/form-data` with file field.

**Response:**
```json
{
  "dataset_id": "upload_20260621",
  "name": "my_data.csv",
  "records": 120,
  "columns": ["date", "inflation_rate", "gdp_growth"]
}
```

### POST /api/v1/predict

Make a single inflation prediction for month t+1.

**Request Body:**
```json
{
  "dataset_id": "PER",
  "features": {
    "inflation_lag_1": 3.2,
    "inflation_lag_3": 3.5,
    "inflation_lag_12": 4.1,
    "rolling_mean_3": 3.3,
    "rolling_mean_12": 3.8,
    "gdp_growth": 2.1,
    "interest_rate": 5.5,
    "tfidf_mean": 0.12,
    "keyword_inflation_count": 45,
    "sentiment_score": -0.3
  }
}
```

**Response:**
```json
{
  "prediction": 3.05,
  "confidence_interval": [2.85, 3.25],
  "model_version": "v1.0.0",
  "dataset_id": "PER",
  "target_month": "2024-07"
}
```

### GET /api/v1/model/info

Returns model metadata and performance metrics.

**Response:**
```json
{
  "model_loaded": true,
  "project_name": "Inflation Forecast API",
  "api_version": "1.0.0",
  "model": {
    "version": "v1.0.0",
    "algorithm": "xgboost",
    "dataset": "multi_country_inflation",
    "features": ["inflation_lag_1", "inflation_lag_3", ...],
    "target": "inflation_rate_t_plus_1"
  },
  "performance": {
    "rmse": 0.45,
    "mae": 0.32,
    "mape": 8.5,
    "r2": 0.87
  }
}
```

---

## Screenshots

### Frontend - Dataset Selector and Time Series

The Streamlit frontend provides a dropdown to select pre-loaded countries or uploaded datasets, displaying an interactive Plotly line chart of historical inflation.

### Frontend - Custom CSV Upload

Users can drag and drop their own CSV files with columns `date` and `inflation_rate`. The system validates the format, stores the file, and adds it to the dataset selector.

### Frontend - Prediction Panel

After selecting a dataset, the user adjusts feature sliders (lag inflation, GDP, text features) and clicks "Predict Inflation (t+1)". The result displays the predicted value, confidence interval, and adds an annotation to the time-series chart.

### API Documentation - Swagger UI

The auto-generated Swagger UI at `/docs` allows interactive testing of all endpoints with request/response examples, including file upload.

---

## Architecture

```
User
  |
  | HTTP (Browser)
  v
+------------------+     +------------------+     +------------------+
|    Frontend      | --> |     Backend      | --> |     Model        |
|   (Streamlit)    |     |    (FastAPI)     |     | (sklearn/XGBoost)|
|    Port 8501     |     |    Port 8000     |     |   .pkl file      |
+------------------+     +------------------+     +------------------+
       |                      |                      |
       |                      |                      |
       +----------------------+----------------------+
                              |
                     Docker Compose Network
```

**Data Flow:**
1. User selects a country dataset or uploads a custom CSV in the Streamlit frontend.
2. Frontend requests historical data from `GET /api/v1/datasets/{id}/history` and renders a Plotly chart.
3. User adjusts feature values (inflation lags, GDP, text features) in the prediction form.
4. Frontend sends POST request to `/api/v1/predict` with the feature vector.
5. Backend validates input using Pydantic schemas, loads the serialized regression model.
6. Model performs inference and returns the predicted inflation rate + confidence interval.
7. Backend formats the response and returns JSON.
8. Frontend renders the prediction result, updates the chart with the new forecast point, and displays model metadata.

---

## Troubleshooting

### Backend fails to start: "Model file not found"

The backend requires the model artifact to exist before starting. Run the training pipeline first:

```bash
python -m scraping.worldbank_scraper
python -m scraping.news_scraper
python -m scraping.text_processor
python -m model.train
python -m model.evaluate
```

Verify that `model/artifacts/model_v1.pkl` exists.

### Frontend cannot connect to backend

Ensure the backend is running and accessible. The frontend uses the environment variable `API_URL` (default: `http://backend:8000` in Docker, `http://localhost:8000` locally).

Check backend health:
```bash
curl http://localhost:8000/api/v1/health
```

### Port conflicts

If ports 8000 or 8501 are already in use, modify the port mappings in `docker-compose.yml`:

```yaml
ports:
  - "8001:8000"  # Backend on host port 8001
  - "8502:8501"  # Frontend on host port 8502
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Authors

**Scraping Inflation Data And Developing A Model With Data From CommonCrawl**

- Choquechambi Luque Wrangler
- Ramos Arisapana Frank
- Coaquira Justo Yurins

**Course:** Data Mining
**Teacher:** Mg. Milton Humpiri Flores
**Institution:** Universidad Peruana Union - Facultad de Ingenieria y Arquitectura - Escuela Profesional de Ingenieria de Sistemas
**Location:** Juliaca, Peru
**Year:** 2026

For questions or issues, please open a GitHub issue.
