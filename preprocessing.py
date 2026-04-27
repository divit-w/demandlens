"""
utils/preprocessing.py
-----------------------
Handles all data loading, cleaning, encoding, and scaling.
This is the first step in the ML pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# ── Path to the dataset ──────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "retail_store_inventory.csv")


def load_raw_data() -> pd.DataFrame:
    """Load the raw CSV without any transformation."""
    df = pd.read_csv(DATA_PATH)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      1. Parse dates
      2. Handle missing values
      3. Remove outliers (IQR method on Units Sold)
      4. Encode categoricals
      5. Derive time features
    Returns a clean DataFrame ready for feature engineering.
    """

    df = df.copy()

    # ── 1. Parse dates ────────────────────────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"]  = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"] = df["Date"].dt.dayofweek          # 0=Mon … 6=Sun
    df["Quarter"]   = df["Date"].dt.quarter

    # ── 2. Handle missing values ──────────────────────────────────────────────
    # Numeric columns: fill with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Categorical columns: fill with mode
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # ── 3. Outlier removal on Units Sold (IQR) ────────────────────────────────
    Q1 = df["Units Sold"].quantile(0.01)
    Q3 = df["Units Sold"].quantile(0.99)
    df = df[(df["Units Sold"] >= Q1) & (df["Units Sold"] <= Q3)]

    # ── 4. Encode categoricals ────────────────────────────────────────────────
    le = LabelEncoder()
    encode_cols = ["Category", "Region", "Weather Condition", "Seasonality"]
    for col in encode_cols:
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    # ── 5. Discount as fraction ───────────────────────────────────────────────
    df["Discount_frac"] = df["Discount"] / 100.0

    return df


def get_feature_matrix(df: pd.DataFrame):
    """
    Returns X (feature matrix) and y (target: Units Sold).
    Features used for ML models.
    """
    feature_cols = [
        "Price",
        "Discount_frac",
        "Competitor Pricing",
        "Inventory Level",
        "Demand Forecast",
        "Holiday/Promotion",
        "Month",
        "Week",
        "Quarter",
        "DayOfWeek",
        "Category_enc",
        "Region_enc",
        "Weather Condition_enc",
        "Seasonality_enc",
    ]
    X = df[feature_cols].copy()
    y = df["Units Sold"].copy()
    return X, y, feature_cols
