"""
utils/feature_engineering.py
------------------------------
Creates derived business features on top of the preprocessed DataFrame.
These features align with CO1 (Python/Pandas/NumPy) requirements.
"""

import pandas as pd
import numpy as np


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds domain-driven derived columns:
      - Revenue, Cost, Profit, Profit Margin
      - Effective Price (after discount)
      - Competitor Difference
      - Demand Category (High/Medium/Low)
      - Price Sensitivity tag
      - Stock Turnover Rate
    """
    df = df.copy()

    # ── Revenue ───────────────────────────────────────────────────────────────
    # Revenue = Effective Price × Units Sold
    df["Effective Price"] = df["Price"] * (1 - df["Discount"] / 100)
    df["Revenue"]         = df["Effective Price"] * df["Units Sold"]

    # ── Cost & Profit (assume cost = 70 % of selling price) ──────────────────
    df["Cost Price"]      = df["Price"] * 0.70
    df["Profit"]          = (df["Effective Price"] - df["Cost Price"]) * df["Units Sold"]
    df["Profit Margin %"] = ((df["Effective Price"] - df["Cost Price"]) / df["Price"] * 100).round(2)

    # ── Competitor Difference ─────────────────────────────────────────────────
    df["Competitor Diff"] = (df["Price"] - df["Competitor Pricing"]).round(2)
    df["Price vs Competitor"] = df["Competitor Diff"].apply(
        lambda x: "Above Market" if x > 1 else ("Below Market" if x < -1 else "At Market")
    )

    # ── Demand Category ───────────────────────────────────────────────────────
    low_thresh  = df["Units Sold"].quantile(0.33)
    high_thresh = df["Units Sold"].quantile(0.66)
    df["Demand Category"] = pd.cut(
        df["Units Sold"],
        bins=[-np.inf, low_thresh, high_thresh, np.inf],
        labels=["Low", "Medium", "High"]
    )

    # ── Stock Turnover Rate ───────────────────────────────────────────────────
    df["Stock Turnover"] = (df["Units Sold"] / df["Inventory Level"].replace(0, np.nan)).round(4)

    # ── Discount Effectiveness ────────────────────────────────────────────────
    df["Discount Effectiveness"] = (df["Units Sold"] / (df["Discount"] + 1)).round(2)

    return df


def compute_price_elasticity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Price Elasticity of Demand at the Product-Category level.

    Formula:
        Elasticity = (% ΔDemand) / (% ΔPrice)

    We calculate this by fitting a log-log regression per category:
        ln(Units Sold) ~ β × ln(Price)  →  β is the elasticity coefficient.

    Returns a DataFrame with columns:
        Category | Elasticity | Sensitivity
    """
    from scipy.stats import linregress

    results = []
    for cat, grp in df.groupby("Category"):
        grp = grp[grp["Price"] > 0].copy()
        if len(grp) < 10:
            continue
        log_price  = np.log(grp["Price"])
        log_demand = np.log(grp["Units Sold"].replace(0, 0.1))
        slope, intercept, r, p, se = linregress(log_price, log_demand)
        sensitivity = "Elastic" if abs(slope) > 1 else "Inelastic"
        results.append({
            "Category": cat,
            "Elasticity": round(slope, 4),
            "R²": round(r**2, 4),
            "Sensitivity": sensitivity,
        })

    return pd.DataFrame(results).sort_values("Elasticity")


def compute_product_elasticity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same as above but at Product ID level (top 20 products by volume).
    """
    from scipy.stats import linregress

    top_products = df.groupby("Product ID")["Units Sold"].sum().nlargest(20).index
    sub = df[df["Product ID"].isin(top_products)]

    results = []
    for pid, grp in sub.groupby("Product ID"):
        grp = grp[grp["Price"] > 0].copy()
        if len(grp) < 5:
            continue
        log_price  = np.log(grp["Price"])
        log_demand = np.log(grp["Units Sold"].replace(0, 0.1))
        slope, intercept, r, p, se = linregress(log_price, log_demand)
        cat = grp["Category"].mode()[0]
        results.append({
            "Product ID": pid,
            "Category": cat,
            "Avg Price": round(grp["Price"].mean(), 2),
            "Avg Units Sold": round(grp["Units Sold"].mean(), 1),
            "Elasticity": round(slope, 4),
            "Sensitivity": "Elastic" if abs(slope) > 1 else "Inelastic",
        })

    return pd.DataFrame(results).sort_values("Elasticity")
