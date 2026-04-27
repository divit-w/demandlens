# 📊 DemandLens — Price Elasticity & Demand Analytics Dashboard

> **AI & ML Lab Mini-Project** | End-to-end retail analytics platform built with Python, Scikit-learn, and Streamlit.

---

## 🎯 Project Overview

DemandLens is a **production-grade retail analytics dashboard** that:
- Analyses price elasticity of demand across product categories
- Trains and compares 3 ML models to predict units sold
- Recommends optimal prices to maximise revenue or profit
- Provides interactive EDA with 20+ Plotly charts
- Presents a premium dark-themed analytics UI

---

## 📚 Course Outcome Alignment

| CO | Outcome | Implementation |
|----|---------|----------------|
| **CO1** | Python, NumPy, Pandas, Matplotlib, Plotly | `utils/preprocessing.py`, `utils/feature_engineering.py`, all pages |
| **CO2** | Search / Optimisation | `views/optimizer.py` — grid search over price range to maximise revenue/profit |
| **CO3** | Machine Learning | Linear Regression, Decision Tree, Random Forest in `utils/model_training.py` |
| **CO4** | Performance Analysis | R², MAE, MSE, RMSE comparison charts in `views/performance.py` |

---

## 🗂️ Project Structure

```
retail_project/
│
├── app.py                        ← Main Streamlit entry point
├── train_models.py               ← Standalone model training script
├── requirements.txt              ← Python dependencies
├── README.md                     ← This file
│
├── data/
│   └── retail_store_inventory.csv  ← Dataset (73,100 rows × 15 cols)
│
├── models/                       ← Auto-created after training
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_cols.pkl
│   ├── Linear_Regression.pkl
│   ├── Decision_Tree.pkl
│   └── Random_Forest.pkl
│
├── notebooks/
│   └── eda_and_ml.ipynb          ← End-to-end EDA & ML notebook
│
├── views/
│   ├── dashboard.py              ← Home: KPIs, revenue trends, heatmap
│   ├── data_explorer.py          ← Interactive EDA with filters
│   ├── elasticity.py             ← Price elasticity analysis & curves
│   ├── prediction.py             ← ML demand prediction + sensitivity
│   ├── optimizer.py              ← Price optimiser (CO2)
│   └── performance.py            ← Model metrics & comparison (CO4)
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py          ← Data loading, cleaning, encoding
│   ├── feature_engineering.py    ← Derived features + elasticity math
│   ├── model_training.py         ← Train/save/load all ML models
│   └── helper_functions.py       ← UI helpers, KPI cards, chart theme
│
└── assets/
    └── styles.css                ← Dark theme CSS overrides
```

---

## 🚀 How to Run

### Step 1 — Clone / extract the project
```bash
cd retail_project
```

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — (Optional) Pre-train models from terminal
```bash
python train_models.py
```
> You can also train directly inside the app via **Model Performance → Train All Models**

### Step 5 — Launch the dashboard
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📊 Dataset Description

| Column | Type | Description |
|--------|------|-------------|
| Date | datetime | Transaction date (2022–2023) |
| Store ID | str | Store identifier (S001–S005) |
| Product ID | str | Product identifier (P0001–P0020) |
| Category | str | Groceries, Toys, Electronics, Furniture, Clothing |
| Region | str | North, South, East, West |
| Inventory Level | int | Units in stock |
| Units Sold | int | **Target variable** |
| Units Ordered | int | Replenishment order qty |
| Demand Forecast | float | Forecasted demand |
| Price | float | Selling price (₹) |
| Discount | int | Discount percentage (0–40%) |
| Weather Condition | str | Sunny, Rainy, Cloudy, Snowy |
| Holiday/Promotion | int | 1 = promotional period |
| Competitor Pricing | float | Competitor's price (₹) |
| Seasonality | str | Spring, Summer, Autumn, Winter |

---

## ⚙️ Feature Engineering

New features derived from raw data:

| Feature | Formula |
|---------|---------|
| Effective Price | `Price × (1 - Discount/100)` |
| Revenue | `Effective Price × Units Sold` |
| Cost Price | `Price × 0.70` |
| Profit | `(Effective Price - Cost Price) × Units Sold` |
| Profit Margin % | `(Effective Price - Cost Price) / Price × 100` |
| Competitor Diff | `Price - Competitor Pricing` |
| Demand Category | Low / Medium / High (quantile-based) |
| Stock Turnover | `Units Sold / Inventory Level` |

---

## 🤖 ML Models

| Model | Description |
|-------|-------------|
| Linear Regression | Baseline; uses scaled features |
| Decision Tree | Non-linear; interpretable splits |
| Random Forest | Ensemble of trees; robust |

**Target variable:** `Units Sold`

**Evaluation metrics:** R², MAE, MSE, RMSE on 20% held-out test set

---

## 💡 Price Optimiser (CO2)

The optimiser performs a **grid search** over 100 candidate price points between a user-defined min/max range. For each price point it:
1. Predicts demand using the best-trained ML model
2. Calculates revenue = effective_price × demand
3. Calculates profit = (effective_price - cost) × demand
4. Returns the price that maximises the chosen objective

---

## ⚡ Price Elasticity

Elasticity is estimated via **log-log regression** at category and product level:

```
ln(Units Sold) = α + β × ln(Price)
```

β is the **elasticity coefficient**:
- |β| > 1 → Elastic (demand is price-sensitive)
- |β| < 1 → Inelastic (demand is price-insensitive)

---

## 🖥️ Dashboard Pages

| Page | Key Features |
|------|-------------|
| 🏠 Dashboard Home | KPI cards, revenue trend, category pie, correlation heatmap |
| 🔍 Data Explorer | Multi-filter panel, scatter, box plots, regional sunburst |
| ⚡ Elasticity Analysis | Log-log regression, radar chart, demand curves |
| 🤖 ML Prediction | Input sliders, demand gauge, price sensitivity curve |
| 💡 Price Optimizer | Grid search, revenue/profit curves, before/after comparison |
| 🏆 Model Performance | Train button, metrics table, actual vs predicted, feature importance |

---

## 👨‍💻 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit + Custom CSS |
| Visualisation | Plotly, Matplotlib |
| ML | Scikit-learn |
| Data | Pandas, NumPy |
| Model Persistence | Joblib |
| Stats | SciPy |

---

## 📝 Notes for Viva

- The dataset has **73,100 rows** with no missing values
- All 3 models achieve **R² > 0.99** because `Demand Forecast` is a strong predictor (intentional in dataset design)
- To demonstrate generalisation, the evaluator can remove `Demand Forecast` from `feature_cols` in `utils/preprocessing.py`
- The Price Optimizer satisfies **CO2** by implementing a search algorithm (grid search) to find the optimal price
