"""
LightGBM Behavioral Risk & Default Classifier — Training Pipeline

Ingests synthetic_sme_data.csv, trains an LGBMClassifier, evaluates it on a
held-out test set, and writes two artifacts for FastAPI inference:

  lgbm_risk_classifier.joblib   serialised LightGBM model
  industry_categories.json      fixed category order for live inference encoding
"""

import json
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ── Config ──────────────────────────────────────────────────────────────────
DATA_FILE   = "synthetic_sme_data.csv"
MODEL_FILE  = "lgbm_risk_classifier.joblib"
CATS_FILE   = "industry_categories.json"
TARGET      = "defaulted_in_12_months"
TEST_SIZE   = 0.20
RANDOM_SEED = 42

# Alphabetically sorted so the integer codes are deterministic across runs.
# The FastAPI predictor MUST load this file rather than hard-coding it.
INDUSTRY_CATS = sorted(["Construction", "Logistics", "Retail", "SaaS"])


# ── Data ─────────────────────────────────────────────────────────────────────
def load_and_preprocess(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in {path}.")

    unknown = set(df["industry"].unique()) - set(INDUSTRY_CATS)
    if unknown:
        raise ValueError(f"Unseen industry values in data: {unknown}")

    # Fixed-order Categorical → LightGBM reads .cat.codes internally.
    # The same ordered list is saved to CATS_FILE for inference parity.
    df["industry"] = pd.Categorical(df["industry"], categories=INDUSTRY_CATS)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


# ── Model ────────────────────────────────────────────────────────────────────
def build_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        # Penalises missed defaults on imbalanced training data.
        class_weight="balanced",
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        verbose=-1,
    )


# ── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(model: LGBMClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n── Classification Report ──────────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc  = average_precision_score(y_test, y_proba)

    print(f"ROC-AUC : {roc_auc:.4f}")
    # PR-AUC is more informative than ROC-AUC on class-imbalanced sets because
    # it is not inflated by the large true-negative pool.
    print(f"PR-AUC  : {pr_auc:.4f}   (preferred metric for imbalanced default detection)")
    print("───────────────────────────────────────────────────────────────\n")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    # 1. Load & preprocess
    print(f"Loading data from '{DATA_FILE}' ...")
    X, y = load_and_preprocess(DATA_FILE)
    print(f"  {len(X):,} samples  |  overall default rate: {y.mean():.2%}")

    # 2. Stratified split preserves the default rate in both partitions.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # 3. Train
    print("\nTraining LGBMClassifier ...")
    model = build_model()
    model.fit(
        X_train,
        y_train,
        categorical_feature=["industry"],
    )
    print("  Training complete.")

    # 4. Evaluate on held-out test set
    evaluate(model, X_test, y_test)

    # 5. Persist artifacts
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved        → {MODEL_FILE}")

    with open(CATS_FILE, "w") as f:
        json.dump(INDUSTRY_CATS, f, indent=2)
    print(f"Category map saved → {CATS_FILE}")

    print("\nBoth artifacts are required by the FastAPI predictor.")
    print("Load them with:")
    print("  model = joblib.load('lgbm_risk_classifier.joblib')")
    print("  cats  = json.load(open('industry_categories.json'))")
    print("  df['industry'] = pd.Categorical(df['industry'], categories=cats)")


if __name__ == "__main__":
    main()
