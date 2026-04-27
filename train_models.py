"""
utils/model_training.py
------------------------
DemandLens — Train & save ML models.

Models (CO3):
  Linear Regression, Decision Tree, Random Forest

Libraries: scikit-learn, NumPy, joblib
"""

import os
import numpy as np
import joblib

from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing   import StandardScaler

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def train_all_models(X, y):
    """
    Train all syllabus-specified models, save to disk, return results dict.

    Models (CO3):
        Linear Regression, Decision Tree, Random Forest

    Returns
    -------
    results   : dict  — {model_name: {R², MAE, MSE, RMSE, y_test, y_pred, model}}
    best_name : str   — name of best model by R²
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standard scaling — required for Linear Regression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Model definitions (CO3) ──────────────────────────────────────────
    model_defs = {
        "Linear Regression": (
            LinearRegression(),
            True,   # needs scaling
        ),
        "Decision Tree": (
            DecisionTreeRegressor(max_depth=8, random_state=42),
            False,
        ),
        "Random Forest": (
            RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            False,
        ),
    }

    results   = {}
    best_r2   = -np.inf
    best_name = None

    for name, (model, scaled) in model_defs.items():
        Xtr = X_train_sc if scaled else X_train
        Xte = X_test_sc  if scaled else X_test

        model.fit(Xtr, y_train)
        preds = model.predict(Xte)

        r2   = r2_score(y_test, preds)
        mae  = mean_absolute_error(y_test, preds)
        mse  = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)

        results[name] = {
            "R²":    r2,
            "MAE":   mae,
            "MSE":   mse,
            "RMSE":  rmse,
            "y_test": y_test,
            "y_pred": preds,
            "model":  model,
            "scaled": scaled,
        }

        # Save model with compression
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, os.path.join(MODELS_DIR, f"{safe_name}.pkl"), compress=3)

        if r2 > best_r2:
            best_r2   = r2
            best_name = name

    # Save shared artefacts
    joblib.dump(scaler,    os.path.join(MODELS_DIR, "scaler.pkl"), compress=3)
    joblib.dump(results[best_name]["model"],
                os.path.join(MODELS_DIR, "best_model.pkl"), compress=3)

    return results, best_name