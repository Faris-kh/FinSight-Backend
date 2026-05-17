import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime, timezone
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------------------------
# 1. Synthetic data generation
# ---------------------------------------------------------------------------

INDUSTRIES = {
    "retail": {
        "revenue_range": (500_000, 15_000_000),
        "expense_ratio_range": (0.72, 0.88),
        "current_asset_ratio": (0.25, 0.50),
        "current_liability_ratio": (0.15, 0.35),
        "total_asset_ratio": (0.40, 0.80),
        "debt_asset_ratio": (0.20, 0.45),
        "annual_growth_mean": 0.06,
        "annual_growth_std": 0.08,
        "seasonality": [0.90, 0.85, 0.92, 0.95, 1.00, 1.05, 1.10, 1.05, 1.00, 1.05, 1.15, 1.40],
    },
    "restaurant": {
        "revenue_range": (300_000, 8_000_000),
        "expense_ratio_range": (0.80, 0.93),
        "current_asset_ratio": (0.08, 0.20),
        "current_liability_ratio": (0.10, 0.25),
        "total_asset_ratio": (0.30, 0.60),
        "debt_asset_ratio": (0.30, 0.55),
        "annual_growth_mean": 0.04,
        "annual_growth_std": 0.10,
        "seasonality": [0.88, 0.82, 0.90, 1.00, 1.05, 1.10, 1.20, 1.15, 1.00, 0.95, 0.90, 1.05],
    },
    "manufacturing": {
        "revenue_range": (2_000_000, 50_000_000),
        "expense_ratio_range": (0.65, 0.82),
        "current_asset_ratio": (0.35, 0.65),
        "current_liability_ratio": (0.20, 0.40),
        "total_asset_ratio": (0.80, 1.50),
        "debt_asset_ratio": (0.30, 0.60),
        "annual_growth_mean": 0.05,
        "annual_growth_std": 0.07,
        "seasonality": [0.92, 0.90, 0.95, 1.00, 1.02, 1.05, 0.98, 0.95, 1.02, 1.05, 1.08, 1.08],
    },
    "it_services": {
        "revenue_range": (500_000, 20_000_000),
        "expense_ratio_range": (0.55, 0.78),
        "current_asset_ratio": (0.30, 0.60),
        "current_liability_ratio": (0.10, 0.25),
        "total_asset_ratio": (0.25, 0.55),
        "debt_asset_ratio": (0.10, 0.30),
        "annual_growth_mean": 0.12,
        "annual_growth_std": 0.12,
        "seasonality": [0.95, 0.90, 0.95, 1.00, 1.02, 1.05, 0.98, 0.90, 1.05, 1.10, 1.05, 1.05],
    },
    "construction": {
        "revenue_range": (1_000_000, 40_000_000),
        "expense_ratio_range": (0.75, 0.90),
        "current_asset_ratio": (0.40, 0.80),
        "current_liability_ratio": (0.30, 0.55),
        "total_asset_ratio": (0.60, 1.20),
        "debt_asset_ratio": (0.35, 0.65),
        "annual_growth_mean": 0.05,
        "annual_growth_std": 0.15,
        "seasonality": [0.85, 0.88, 0.95, 1.05, 1.10, 1.10, 1.05, 0.90, 1.00, 1.05, 1.05, 0.97],
    },
    "healthcare": {
        "revenue_range": (800_000, 25_000_000),
        "expense_ratio_range": (0.65, 0.80),
        "current_asset_ratio": (0.20, 0.45),
        "current_liability_ratio": (0.15, 0.30),
        "total_asset_ratio": (0.50, 1.00),
        "debt_asset_ratio": (0.20, 0.40),
        "annual_growth_mean": 0.08,
        "annual_growth_std": 0.06,
        "seasonality": [1.02, 0.98, 1.00, 0.98, 0.95, 0.92, 0.90, 0.88, 1.02, 1.05, 1.10, 1.10],
    },
}

N_COMPANIES = 2500
rng = np.random.default_rng(42)

print("[*] Generating synthetic SME data...")

rows = []
industry_names = list(INDUSTRIES.keys())

for _ in range(N_COMPANIES):
    ind_name = rng.choice(industry_names)
    ind = INDUSTRIES[ind_name]

    revenue = rng.uniform(*ind["revenue_range"])
    expense_ratio = rng.uniform(*ind["expense_ratio_range"])
    expenses = revenue * expense_ratio

    currentAssets = revenue * rng.uniform(*ind["current_asset_ratio"])
    currentLiabilities = revenue * rng.uniform(*ind["current_liability_ratio"])
    totalAssets = revenue * rng.uniform(*ind["total_asset_ratio"])
    totalDebt = totalAssets * rng.uniform(*ind["debt_asset_ratio"])
    equity = totalAssets - totalDebt

    base_cf = revenue - expenses
    cashFlow = base_cf + rng.normal(0, abs(base_cf) * 0.05)

    annual_growth = rng.normal(ind["annual_growth_mean"], ind["annual_growth_std"])
    monthly_growth = (1 + annual_growth) ** (1 / 12) - 1

    current_month_idx = int(rng.integers(0, 12))
    monthly_base = cashFlow / 12

    for month_ahead in range(1, 7):
        future_month_idx = (current_month_idx + month_ahead) % 12
        seasonal = ind["seasonality"][future_month_idx]
        growth = (1 + monthly_growth) ** month_ahead
        noise = rng.normal(0, abs(monthly_base) * 0.08)
        future_cashFlow = monthly_base * seasonal * growth + noise

        rows.append({
            "revenue": revenue,
            "expenses": expenses,
            "currentAssets": currentAssets,
            "currentLiabilities": currentLiabilities,
            "totalAssets": totalAssets,
            "totalDebt": totalDebt,
            "equity": equity,
            "cashFlow": cashFlow,
            "month_ahead": month_ahead,
            "future_cashFlow": future_cashFlow,
        })

df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
df.to_csv("data/sme_data.csv", index=False)
print(f"   Saved {len(df):,} rows -> data/sme_data.csv")

# ---------------------------------------------------------------------------
# 2. Train
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "revenue", "expenses", "currentAssets", "currentLiabilities",
    "totalAssets", "totalDebt", "equity", "cashFlow", "month_ahead",
]
TARGET_COLUMN = "future_cashFlow"

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"[*] Training on {len(X_train):,} samples, testing on {len(X_test):,} samples.")

model = LGBMRegressor(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=127,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

print("[*] Training model...")
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 3. Evaluate and save stats
# ---------------------------------------------------------------------------

y_pred = model.predict(X_test)
mae = float(mean_absolute_error(y_test, y_pred))
r2 = float(r2_score(y_test, y_pred))

print("\nModel Performance:")
print(f"   Mean Absolute Error : {mae:,.0f} SAR")
print(f"   R² Score            : {r2:.4f}  (1.0 = perfect, 0.0 = random)")

stats = {
    "mae": round(mae, 2),
    "r2": round(r2, 4),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "trained_at": datetime.now(timezone.utc).isoformat(),
}
with open("model_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

joblib.dump(model, "model.pkl")
print("\n[OK] model.pkl and model_stats.json saved.")
