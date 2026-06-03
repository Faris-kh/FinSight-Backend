"""
PoD Model Training Pipeline
----------------------------
Trains a LightGBM Probability of Default classifier on the Polish Companies
Bankruptcy Dataset (UCI ML Repository ID 365, 43,405 observations, 4.8%
bankruptcy rate across 1–5 year prediction windows).

Feature definitions (source of truth for inference alignment):
    roa           — A7:  EBIT / total assets
    current_ratio — A4:  current assets / short-term liabilities
    quick_ratio   — A46: (current assets − inventory) / short-term liabilities
    ebit_margin   — A42: profit on operating activities / sales
    debt_to_assets— A2:  total liabilities / total assets
    icr           — A27: profit on operating activities / financial expenses
    dscr_proxy    — A26: (net profit + depreciation) / total liabilities

Usage:
    python train_pod_model.py [--data path/to/polish_bankruptcy.csv]

Outputs:
    artifacts/lightgbm_pod_model.pkl  — serialised model for FastAPI inference
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ── Config ───────────────────────────────────────────────────────────────────

DATA_FILE    = "polish_bankruptcy.csv"
ARTIFACT_DIR = "artifacts"
MODEL_FILE   = os.path.join(ARTIFACT_DIR, "lightgbm_pod_model.pkl")
TARGET_COL   = "class"
TEST_SIZE    = 0.20
RANDOM_SEED  = 42

# Polish dataset column → canonical inference name.
# Definitions match standard financial ratio conventions; inference formulas
# must replicate these exactly (see services/forecasting.py).
FEATURE_MAP: dict[str, str] = {
    "A7":  "roa",            # EBIT / total assets
    "A4":  "current_ratio",  # current assets / short-term liabilities
    "A46": "quick_ratio",    # (current assets − inventory) / short-term liabilities
    "A42": "ebit_margin",    # profit on operating activities / sales
    "A2":  "debt_to_assets", # total liabilities / total assets
    "A27": "icr",            # profit on operating activities / financial expenses
    "A26": "dscr_proxy",     # (net profit + depreciation) / total liabilities
}

# Training column order — the inference service MUST present features in this
# exact sequence when calling predict_proba().
FEATURE_COLS: list[str] = [
    "roa", "current_ratio", "quick_ratio", "ebit_margin",
    "debt_to_assets", "icr", "dscr_proxy",
]


# ── Data ─────────────────────────────────────────────────────────────────────

def load_and_prepare(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {path}.")

    missing_feats = [c for c in FEATURE_MAP if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Dataset is missing expected columns:\n  {missing_feats}")

    y = df[TARGET_COL].astype(int)
    # Select only the 7 feature columns; 'year' and all other metadata columns
    # are intentionally excluded — they are dataset artefacts, not financial ratios.
    X = (
        df[list(FEATURE_MAP.keys())]
        .rename(columns=FEATURE_MAP)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return X[FEATURE_COLS], y


# ── Training ─────────────────────────────────────────────────────────────────

def train(X_train: pd.DataFrame, y_train: pd.Series) -> LGBMClassifier:
    model = LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=100,
        max_depth=6,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model: LGBMClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Solvent", "Bankrupt"]))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR-AUC  : {average_precision_score(y_test, y_proba):.4f}"
          "  (preferred for imbalanced sets)")
    print("---\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(data_path: str = DATA_FILE) -> None:
    print(f"Loading data from '{data_path}' ...")
    X, y = load_and_prepare(data_path)
    print(f"  {len(X):,} samples  |  bankruptcy rate: {y.mean():.2%}")
    print(f"  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    # --- Step 1: Stratified train / test split BEFORE any resampling ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"\nSplit  ->  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # --- Step 2: Apply SMOTE ONLY to the training set (prevents data leakage) ---
    print("Applying SMOTE to training set only ...")
    X_train_res, y_train_res = SMOTE(random_state=RANDOM_SEED).fit_resample(
        X_train, y_train
    )
    print(
        f"  Post-SMOTE train: {len(X_train_res):,} samples"
        f"  (bankrupt share: {y_train_res.mean():.2%})"
    )

    # --- Step 3: Train on balanced data, evaluate on original test distribution ---
    print("\nTraining LGBMClassifier ...")
    model = train(X_train_res, y_train_res)
    print("  Training complete.")
    evaluate(model, X_test, y_test)

    # --- Step 4: Persist artifact ---
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(MODEL_FILE, "wb") as fh:
        pickle.dump(model, fh)
    print(f"Artifact saved  ->  {MODEL_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the FinSight PoD model.")
    parser.add_argument("--data", default=DATA_FILE,
                        help=f"Path to the Polish bankruptcy CSV (default: {DATA_FILE})")
    args = parser.parse_args()
    main(data_path=args.data)
