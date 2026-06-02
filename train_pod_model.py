"""
PoD Model Training Pipeline
----------------------------
Trains a LightGBM Probability of Default classifier on the Taiwanese Bankruptcy
Prediction Dataset.

Usage:
    python train_pod_model.py [--data path/to/data.csv]

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

DATA_FILE    = "data.csv"
ARTIFACT_DIR = "artifacts"
MODEL_FILE   = os.path.join(ARTIFACT_DIR, "lightgbm_pod_model.pkl")
TARGET_COL   = "Bankrupt?"
TEST_SIZE    = 0.20
RANDOM_SEED  = 42

# Exact column names as they appear in the Taiwanese dataset (after strip).
# The dict value is the canonical name used in the inference service.
FEATURE_MAP: dict[str, str] = {
    "ROA(C) before interest and depreciation before interest": "roa",
    "Current Ratio":                                           "current_ratio",
    "Quick Ratio":                                             "quick_ratio",
    "Operating Profit Rate":                                   "ebitda_margin",
    "Debt ratio %":                                            "debt_to_assets",
    "Interest Coverage Ratio (Interest expense to EBIT)":     "icr",
    "Cash Flow to Liability":                                  "dscr_proxy",
}

# Training column order — the inference service MUST present features in this
# exact sequence when calling predict_proba().
FEATURE_COLS: list[str] = [
    "roa", "current_ratio", "quick_ratio", "ebitda_margin",
    "debt_to_assets", "icr", "dscr_proxy",
]


# ── Data ─────────────────────────────────────────────────────────────────────

def load_and_prepare(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    missing_target = TARGET_COL not in df.columns
    if missing_target:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {path}.")

    missing_feats = [c for c in FEATURE_MAP if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Dataset is missing expected columns:\n  {missing_feats}")

    y = df[TARGET_COL].astype(int)
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
                        help=f"Path to the Taiwanese dataset CSV (default: {DATA_FILE})")
    args = parser.parse_args()
    main(data_path=args.data)
