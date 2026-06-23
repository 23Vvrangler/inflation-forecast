"""
ML Pipeline definition for inflation forecasting regression.

Constructs a complete preprocessing + modeling pipeline that handles:
    - Numeric features: imputation + scaling
    - Text-derived features: imputation with zeros
    - Regression model: XGBoost or RandomForest

Uses sklearn ColumnTransformer to apply different preprocessing
strategies to different feature groups.
"""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from typing import Dict, Any, List


def build_pipeline(config: Dict[str, Any]) -> Pipeline:
    """
    Build a scikit-learn pipeline for inflation regression.

    The pipeline includes:
    1. ColumnTransformer for mixed feature types:
       - Numeric features: Imputation + Scaling
       - Text-derived features: Imputation (fill with 0) + pass-through
    2. Regression model (XGBoost or RandomForest)

    Args:
        config: Configuration dictionary with preprocessing and model settings.

    Returns:
        Configured sklearn Pipeline object.
    """
    preprocessing_config = config.get("preprocessing", {})
    model_config = config.get("model", {})
    features_config = config.get("features", {})

    numeric_features = features_config.get("numeric", [])
    text_features = features_config.get("text", [])

    # --- Numeric preprocessing ---
    numeric_prep_config = preprocessing_config.get("numeric", {})
    numeric_imputer_strategy = numeric_prep_config.get("imputer_strategy", "median")
    numeric_scaler_type = numeric_prep_config.get("scaler", "standard")

    numeric_steps = [("imputer", SimpleImputer(strategy=numeric_imputer_strategy))]

    if numeric_scaler_type == "standard":
        numeric_steps.append(("scaler", StandardScaler()))
    elif numeric_scaler_type == "minmax":
        numeric_steps.append(("scaler", MinMaxScaler()))
    elif numeric_scaler_type == "robust":
        numeric_steps.append(("scaler", RobustScaler()))
    # else: no scaler

    numeric_transformer = Pipeline(numeric_steps)

    # --- Text-derived preprocessing ---
    text_prep_config = preprocessing_config.get("text", {})
    text_imputer_strategy = text_prep_config.get("imputer_strategy", "constant")
    text_fill_value = text_prep_config.get("imputer_fill_value", 0)

    text_transformer = SimpleImputer(
        strategy=text_imputer_strategy,
        fill_value=text_fill_value,
    )

    # --- Combine with ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("text", text_transformer, text_features),
        ],
        remainder="drop",  # Drop any columns not explicitly listed
    )

    # --- Model ---
    algorithm = model_config.get("algorithm", "random_forest")
    hyperparameters = model_config.get("hyperparameters", {})
    random_state = model_config.get("random_state", 42)

    if algorithm == "xgboost":
        regressor = xgb.XGBRegressor(
            n_estimators=hyperparameters.get("n_estimators", 200),
            max_depth=hyperparameters.get("max_depth", 6),
            learning_rate=hyperparameters.get("learning_rate", 0.1),
            subsample=hyperparameters.get("subsample", 0.8),
            colsample_bytree=hyperparameters.get("colsample_bytree", 0.8),
            reg_alpha=hyperparameters.get("reg_alpha", 0.1),
            reg_lambda=hyperparameters.get("reg_lambda", 1.0),
            objective=hyperparameters.get("objective", "reg:squarederror"),
            random_state=random_state,
            n_jobs=-1,
        )
    elif algorithm == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=hyperparameters.get("n_estimators", 200),
            max_depth=hyperparameters.get("max_depth", 10),
            min_samples_split=hyperparameters.get("min_samples_split", 2),
            min_samples_leaf=hyperparameters.get("min_samples_leaf", 1),
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        regressor = RandomForestRegressor(random_state=random_state, n_jobs=-1)

    # Build final pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])

    return pipeline
