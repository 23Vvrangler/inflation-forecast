"""
Streamlit Frontend for Inflation Forecast project.

Provides an interactive web interface to:
    - Select datasets (pre-loaded countries or uploaded CSV)
    - Upload custom CSV datasets
    - Visualize historical inflation time series (Plotly)
    - Make predictions using the ML model via API
    - View model metadata and performance metrics

The frontend communicates exclusively with the FastAPI backend.
"""
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Configuration
API_BASE_URL = os.getenv("API_URL", "http://backend:8000")
API_HEALTH = f"{API_BASE_URL}/api/v1/health"
API_MODEL_INFO = f"{API_BASE_URL}/api/v1/model/info"
API_DATASETS = f"{API_BASE_URL}/api/v1/datasets"
API_DATASET_HISTORY = f"{API_BASE_URL}/api/v1/datasets"
API_UPLOAD = f"{API_BASE_URL}/api/v1/datasets/upload"
API_PREDICT = f"{API_BASE_URL}/api/v1/predict"


def check_api_health():
    """Check if the backend API is reachable."""
    try:
        response = requests.get(API_HEALTH, timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except requests.exceptions.ConnectionError:
        return False, None
    except requests.exceptions.Timeout:
        return False, None
    except Exception as e:
        return False, str(e)


def get_model_info():
    """Fetch model metadata and performance metrics from the API."""
    try:
        response = requests.get(API_MODEL_INFO, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def get_datasets():
    """Fetch list of available datasets from the API."""
    try:
        response = requests.get(API_DATASETS, timeout=10)
        if response.status_code == 200:
            return response.json().get("datasets", [])
        return []
    except Exception:
        return []


def get_dataset_history(dataset_id: str):
    """Fetch historical data for a specific dataset."""
    try:
        url = f"{API_DATASET_HISTORY}/{dataset_id}/history"
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def upload_csv_file(file_bytes, filename):
    """Upload a CSV file to the backend API."""
    try:
        files = {"file": (filename, file_bytes, "text/csv")}
        response = requests.post(API_UPLOAD, files=files, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("detail", "Unknown error")}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend API."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out."}
    except Exception as e:
        return {"error": str(e)}


def make_prediction(features: Dict[str, float]):
    """Send prediction request to the backend API."""
    try:
        payload = {"features": features}
        response = requests.post(API_PREDICT, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("detail", "Unknown error")}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend API. Is the service running?"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The backend may be overloaded."}
    except Exception as e:
        return {"error": str(e)}


def render_header():
    """Render the application header and sidebar."""
    st.set_page_config(
        page_title="Inflation Forecast - Data Mining",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Inflation Forecast - Data Mining Project")
    st.subheader("Scraping Inflation Data And Developing A Model With Data From CommonCrawl")
    st.markdown("---")

    with st.sidebar:
        st.header("Project Information")
        st.markdown("""
        **Course:** Data Mining

        **Institution:** Universidad Peruana Union
        **Faculty:** Ingenieria y Arquitectura
        **School:** Ingenieria de Sistemas

        **Authors:**
        - Choquechambi Luque Wrangler
        - Ramos Arisapana Frank
        - Coaquira Justo Yurins

        **Teacher:** Mg. Milton Humpiri Flores

        **Stack:**
        - FastAPI (Backend)
        - XGBoost + scikit-learn (ML)
        - Streamlit + Plotly (Frontend)
        - Docker (Deployment)
        """)

        st.markdown("---")
        st.header("API Status")

        is_healthy, health_data = check_api_health()

        if is_healthy:
            st.success("Backend API Online")
            if health_data:
                st.text(f"Status: {health_data.get('status', 'unknown')}")
                st.text(f"Version: {health_data.get('version', 'unknown')}")
                st.text(f"Model Loaded: {health_data.get('model_loaded', False)}")
        else:
            st.error("Backend API Offline")
            st.warning("Please ensure the backend service is running.")
            st.code("docker compose up backend", language="bash")


def render_dataset_selector():
    """Render dataset selection section."""
    st.header("Dataset Selection")
    st.markdown("Select a pre-loaded country dataset or upload your own CSV file.")

    col1, col2 = st.columns([2, 1])

    with col1:
        datasets = get_datasets()

        if not datasets:
            st.warning("No datasets available. Please ensure the backend is running and data has been scraped.")
            return None

        dataset_options = {f"{d['name']} ({d['source']}, {d['records']} records)": d['id'] for d in datasets}
        selected_label = st.selectbox(
            "Select Dataset",
            options=list(dataset_options.keys()),
            index=0
        )
        selected_id = dataset_options[selected_label]

        st.caption(f"Dataset ID: `{selected_id}`")

        return selected_id

    with col2:
        st.subheader("Upload Custom CSV")
        uploaded_file = st.file_uploader(
            "Upload your inflation dataset",
            type=["csv"],
            help="CSV must contain 'date' and 'inflation_rate' columns. Optional: gdp_growth, interest_rate."
        )

        if uploaded_file is not None:
            with st.spinner("Uploading and validating..."):
                result = upload_csv_file(uploaded_file.getvalue(), uploaded_file.name)

            if "error" in result:
                st.error(f"Upload failed: {result['error']}")
            else:
                st.success(f"Uploaded successfully!")
                st.text(f"Dataset ID: {result['dataset_id']}")
                st.text(f"Records: {result['records']}")
                st.text(f"Columns: {', '.join(result['columns'])}")
                st.balloons()
                # Refresh datasets
                st.rerun()

    return None


def render_time_series_chart(dataset_id: str):
    """Render Plotly time series chart for selected dataset."""
    if not dataset_id:
        return

    history = get_dataset_history(dataset_id)

    if history is None:
        st.warning("Could not load dataset history. Is the backend running?")
        return

    data = history.get("data", [])
    if not data:
        st.info("No historical data available for this dataset.")
        return

    df = pd.DataFrame(data)

    if "date" not in df.columns or "inflation_rate" not in df.columns:
        st.warning("Dataset does not contain expected columns (date, inflation_rate).")
        return

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    st.subheader(f"Historical Inflation: {history.get('name', dataset_id)}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["inflation_rate"],
        mode="lines+markers",
        name="Inflation Rate (%)",
        line=dict(color="steelblue", width=2),
        marker=dict(size=6)
    ))

    # Add GDP growth if available
    if "gdp_growth" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["gdp_growth"],
            mode="lines+markers",
            name="GDP Growth (%)",
            line=dict(color="coral", width=2, dash="dash"),
            marker=dict(size=4),
            yaxis="y2"
        ))

        fig.update_layout(
            yaxis2=dict(
                title="GDP Growth (%)",
                overlaying="y",
                side="right"
            )
        )

    fig.update_layout(
        title=f"Inflation Time Series - {history.get('name', dataset_id)}",
        xaxis_title="Date",
        yaxis_title="Inflation Rate (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Inflation", f"{df['inflation_rate'].mean():.2f}%")
    with col2:
        st.metric("Max Inflation", f"{df['inflation_rate'].max():.2f}%")
    with col3:
        st.metric("Min Inflation", f"{df['inflation_rate'].min():.2f}%")
    with col4:
        st.metric("Std Dev", f"{df['inflation_rate'].std():.2f}%")


def render_prediction_panel():
    """Render the prediction input panel with feature sliders."""
    st.markdown("---")
    st.header("Predict Next-Month Inflation (t+1)")
    st.markdown("Adjust the feature values below and click Predict to forecast the inflation rate for the next month.")

    model_info = get_model_info()

    if model_info is None or not model_info.get("model_loaded", False):
        st.warning("Model information not available. Please ensure the backend is running and the model is loaded.")
        return

    features = model_info.get("model", {}).get("features", {})
    numeric_features = features.get("numeric", [])
    text_features = features.get("text", [])

    feature_values = {}

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Numeric Features")

        # Inflation lag features
        if "inflation_lag_1" in numeric_features:
            feature_values["inflation_lag_1"] = st.slider(
                "Inflation Lag 1 (t-1)",
                min_value=-50.0, max_value=100.0, value=3.0, step=0.1,
                help="Inflation rate from the previous month"
            )
        if "inflation_lag_3" in numeric_features:
            feature_values["inflation_lag_3"] = st.slider(
                "Inflation Lag 3 (t-3)",
                min_value=-50.0, max_value=100.0, value=3.5, step=0.1,
                help="Inflation rate from 3 months ago"
            )
        if "inflation_lag_6" in numeric_features:
            feature_values["inflation_lag_6"] = st.slider(
                "Inflation Lag 6 (t-6)",
                min_value=-50.0, max_value=100.0, value=3.8, step=0.1,
                help="Inflation rate from 6 months ago"
            )
        if "inflation_lag_12" in numeric_features:
            feature_values["inflation_lag_12"] = st.slider(
                "Inflation Lag 12 (t-12)",
                min_value=-50.0, max_value=100.0, value=4.1, step=0.1,
                help="Inflation rate from 12 months ago"
            )

        # Rolling statistics
        if "rolling_mean_3" in numeric_features:
            feature_values["rolling_mean_3"] = st.slider(
                "Rolling Mean 3M",
                min_value=-50.0, max_value=100.0, value=3.3, step=0.1
            )
        if "rolling_mean_6" in numeric_features:
            feature_values["rolling_mean_6"] = st.slider(
                "Rolling Mean 6M",
                min_value=-50.0, max_value=100.0, value=3.6, step=0.1
            )
        if "rolling_mean_12" in numeric_features:
            feature_values["rolling_mean_12"] = st.slider(
                "Rolling Mean 12M",
                min_value=-50.0, max_value=100.0, value=3.8, step=0.1
            )
        if "rolling_std_6" in numeric_features:
            feature_values["rolling_std_6"] = st.slider(
                "Rolling Std Dev 6M",
                min_value=0.0, max_value=50.0, value=0.5, step=0.05
            )

    with col2:
        st.subheader("Macroeconomic & Text Features")

        if "gdp_growth" in numeric_features:
            feature_values["gdp_growth"] = st.slider(
                "GDP Growth (%)",
                min_value=-20.0, max_value=20.0, value=2.1, step=0.1
            )
        if "interest_rate" in numeric_features:
            feature_values["interest_rate"] = st.slider(
                "Interest Rate (%)",
                min_value=0.0, max_value=50.0, value=5.5, step=0.1
            )
        if "month_sin" in numeric_features:
            feature_values["month_sin"] = st.slider(
                "Month Sin (seasonality)",
                min_value=-1.0, max_value=1.0, value=0.5, step=0.01
            )
        if "month_cos" in numeric_features:
            feature_values["month_cos"] = st.slider(
                "Month Cos (seasonality)",
                min_value=-1.0, max_value=1.0, value=0.866, step=0.01
            )

        # Text-derived features
        if "text_sentiment_score" in text_features:
            feature_values["text_sentiment_score"] = st.slider(
                "News Sentiment Score",
                min_value=-1.0, max_value=1.0, value=-0.3, step=0.01,
                help="Average sentiment from economic news articles (-1 = negative, +1 = positive)"
            )
        if "text_tfidf_mean" in text_features:
            feature_values["text_tfidf_mean"] = st.slider(
                "TF-IDF Mean",
                min_value=0.0, max_value=1.0, value=0.12, step=0.01
            )
        if "text_keyword_inflation" in text_features:
            feature_values["text_keyword_inflation"] = st.number_input(
                "Keyword: Inflation (count)",
                min_value=0, max_value=1000, value=45, step=1
            )
        if "text_keyword_crisis" in text_features:
            feature_values["text_keyword_crisis"] = st.number_input(
                "Keyword: Crisis (count)",
                min_value=0, max_value=1000, value=5, step=1
            )
        if "text_keyword_growth" in text_features:
            feature_values["text_keyword_growth"] = st.number_input(
                "Keyword: Growth (count)",
                min_value=0, max_value=1000, value=30, step=1
            )
        if "text_keyword_interest" in text_features:
            feature_values["text_keyword_interest"] = st.number_input(
                "Keyword: Interest (count)",
                min_value=0, max_value=1000, value=20, step=1
            )
        if "text_keyword_gdp" in text_features:
            feature_values["text_keyword_gdp"] = st.number_input(
                "Keyword: GDP (count)",
                min_value=0, max_value=1000, value=15, step=1
            )
        if "text_keyword_currency" in text_features:
            feature_values["text_keyword_currency"] = st.number_input(
                "Keyword: Currency (count)",
                min_value=0, max_value=1000, value=10, step=1
            )
        if "text_keyword_market" in text_features:
            feature_values["text_keyword_market"] = st.number_input(
                "Keyword: Market (count)",
                min_value=0, max_value=1000, value=25, step=1
            )
        if "text_article_count" in text_features:
            feature_values["text_article_count"] = st.number_input(
                "Articles Count",
                min_value=0, max_value=1000, value=50, step=1
            )

    st.markdown("---")

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button("Predict Inflation (t+1)", type="primary", use_container_width=True)

    if predict_clicked:
        is_healthy, _ = check_api_health()

        if not is_healthy:
            st.error("Backend API is not available. Please start the backend service.")
            return

        with st.spinner("Sending request to ML model..."):
            result = make_prediction(feature_values)

        if "error" in result:
            st.error(f"Prediction Error: {result['error']}")
            return

        prediction = result.get("prediction", 0.0)
        model_version = result.get("model_version", "unknown")
        input_features = result.get("input_features", {})

        st.markdown("---")
        st.header("Prediction Result")

        res_col, info_col = st.columns([1, 2])

        with res_col:
            st.metric(
                "Predicted Inflation (t+1)",
                f"{prediction:.2f}%",
                delta=None
            )
            st.caption(f"Model Version: {model_version}")

        with info_col:
            st.subheader("Input Features Used")
            feat_df = pd.DataFrame([
                {"Feature": k, "Value": round(v, 4)}
                for k, v in input_features.items()
            ])
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

        # Interpretation
        st.markdown("---")
        st.subheader("Interpretation")

        if prediction < 0:
            st.info("""
            **Deflation predicted.** The model forecasts a decrease in the general price level.
            This may indicate weak demand, economic contraction, or monetary policy tightening.
            """)
        elif prediction < 3:
            st.success("""
            **Low/Moderate inflation predicted.** Within typical central bank target ranges (2-3%).
            Indicates stable economic conditions.
            """)
        elif prediction < 10:
            st.warning("""
            **High inflation predicted.** Above typical central bank targets.
            May indicate overheating economy, supply shocks, or loose monetary policy.
            """)
        else:
            st.error("""
            **Very high inflation / Hyperinflation predicted.** Critical economic alert.
            Suggests severe monetary instability, currency crisis, or structural economic problems.
            """)


def render_model_info():
    """Render model information section."""
    st.markdown("---")
    st.header("Model Information")

    model_info = get_model_info()

    if model_info is None:
        st.warning("Unable to fetch model information. Is the backend running?")
        return

    if not model_info.get("model_loaded", False):
        st.warning("Model is not loaded on the backend.")
        return

    model_data = model_info.get("model", {})
    performance = model_info.get("performance", {})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Metadata")
        metadata_items = {
            "Algorithm": model_data.get("algorithm", "N/A"),
            "Version": model_data.get("version", "N/A"),
            "Experiment": model_data.get("experiment_name", "N/A"),
            "Train Samples": model_data.get("train_samples", "N/A"),
            "Test Samples": model_data.get("test_samples", "N/A"),
            "Feature Count": model_data.get("feature_count", "N/A"),
        }

        for key, value in metadata_items.items():
            st.text(f"{key}: {value}")

        st.subheader("Features Used")
        features = model_data.get("features", {})
        numeric = features.get("numeric", [])
        text = features.get("text", [])

        st.markdown("**Numeric Features:**")
        for feat in numeric:
            st.markdown(f"- {feat}")

        st.markdown("**Text-Derived Features:**")
        for feat in text:
            st.markdown(f"- {feat}")

    with col2:
        st.subheader("Performance Metrics")

        if performance:
            metrics = {
                "RMSE": performance.get("rmse", 0),
                "MAE": performance.get("mae", 0),
                "MAPE": performance.get("mape", 0),
                "R2 Score": performance.get("r2", 0),
                "Explained Variance": performance.get("explained_variance", 0),
                "Directional Accuracy": performance.get("directional_accuracy", 0),
            }

            for metric_name, value in metrics.items():
                if value is not None:
                    if metric_name == "MAPE":
                        st.metric(metric_name, f"{value:.2f}%")
                    elif metric_name in ["R2 Score", "Explained Variance", "Directional Accuracy"]:
                        st.metric(metric_name, f"{value:.4f}")
                    else:
                        st.metric(metric_name, f"{value:.4f}")
                else:
                    st.metric(metric_name, "N/A")
        else:
            st.info("No performance metrics available.")


def main():
    """Main application entry point."""
    render_header()

    # Dataset selection
    selected_dataset = render_dataset_selector()

    # Time series visualization
    if selected_dataset:
        render_time_series_chart(selected_dataset)

    # Prediction panel
    render_prediction_panel()

    # Model information
    render_model_info()

    # Footer
    st.markdown("---")
    st.caption(f"Inflation Forecast - Data Mining Project | Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
