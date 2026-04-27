"""
utils/model_training.py
------------------------
Trains three ML models, evaluates them, and saves the best one.
Aligns with CO3 (Machine Learning) and CO4 (Performance Analysis).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── Where to save trained models ─────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_all_models(X: pd.DataFrame, y: pd.Series):
    """
    Trains Linear Regression, Decision Tree, and Random Forest.
    Returns:
        results  – dict of {model_name: metrics_dict}
        best_name – name of best model by R²
    """
    # ── Train / test split (80/20) ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # ── Scale features (benefits Linear Regression) ──────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Save scaler with compression
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"), compress=3)

    # ── Model definitions ─────────────────────────────────────────────────────
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree":     DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=60, max_depth=12, random_state=42, n_jobs=-1),
    }

    results = {}

    for name, model in models.items():
        # Linear Regression uses scaled data; tree models use raw
        if name == "Linear Regression":
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)

        results[name] = {
            "MAE":  round(mae,  4),
            "MSE":  round(mse,  4),
            "RMSE": round(rmse, 4),
            "R²":   round(r2,   4),
            "model": model,
            "y_test":  y_test.values,
            "y_pred":  y_pred,
        }

        # Save each model with compression
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl"), compress=3)

    # ── Pick best model by R² ─────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["R²"])
    joblib.dump(results[best_name]["model"], os.path.join(MODELS_DIR, "best_model.pkl"), compress=3)
    joblib.dump(X.columns.tolist(),          os.path.join(MODELS_DIR, "feature_cols.pkl"), compress=3)

    best_r2 = results[best_name]["R²"]
    print(f"\nBest model: {best_name}  |  R2 = {best_r2}")
    return results, best_name


def load_best_model():
    """Load saved best model and scaler."""
    model   = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler  = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feat    = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
    return model, scaler, feat


def predict_demand(model, scaler, feature_cols, input_dict: dict) -> float:
    """
    Run inference for a single input row.
    input_dict must contain all feature_cols keys.
    """
    row = pd.DataFrame([input_dict])[feature_cols]
    row_sc = scaler.transform(row)
    return float(model.predict(row_sc)[0])
