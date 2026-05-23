"""
Synthetic SME training data generator for the LightGBM Behavioral Risk & Default Classifier.

Outputs a CSV of 5,000 SME profiles with engineered default probability based on
Altman Z-Score, DSCR, and supplementary financial health metrics.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_SAMPLES = 5_000
OUTPUT_FILE = "synthetic_sme_data.csv"
NOISE_STD = 0.05  # Gaussian noise to prevent overfitting to hard rules

np.random.seed(RNG_SEED)


def generate_features(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "current_ratio":   np.random.uniform(0.2, 4.0, n),
        "debt_to_equity":  np.random.uniform(0.0, 5.0, n),
        "ebitda_margin":   np.random.uniform(-20.0, 50.0, n),
        "icr":             np.random.uniform(-2.0, 15.0, n),
        "dscr":            np.random.uniform(-2.0, 5.0, n),
        "altman_z_score":  np.random.uniform(-2.0, 8.0, n),
        "industry":        np.random.choice(
            ["SaaS", "Retail", "Construction", "Logistics"], n
        ),
    })


def compute_default_probability(df: pd.DataFrame) -> np.ndarray:
    n = len(df)

    # --- Continuous base score ---
    # Start at a neutral 25% and drift based on supplementary risk signals.
    p = np.full(n, 0.25)

    # Rule 3a: High leverage is a distress signal.
    risky_dte = df["debt_to_equity"] > 3.0
    p += risky_dte * 0.15

    # Rule 3b: Low liquidity is a distress signal.
    risky_cr = df["current_ratio"] < 0.8
    p += risky_cr * 0.15

    # Compound penalty when both conditions hold simultaneously.
    p += (risky_dte & risky_cr) * 0.10

    # Weak profitability and interest coverage nudge risk upward.
    # Each is normalized to its feature range before weighting.
    p -= (df["ebitda_margin"].clip(-20, 50) / 50.0) * 0.08
    p -= (df["icr"].clip(-2, 15) / 15.0) * 0.08

    p = np.clip(p, 0.05, 0.90)

    # --- Hard rule overrides (applied after the continuous base) ---

    # Rule 1: Altman Z'' distress zone + poor debt service → high default probability.
    high_distress = (df["altman_z_score"] < 1.1) & (df["dscr"] < 1.0)
    p[high_distress] = 0.85

    # Rule 2: Altman Z'' safe zone + healthy debt service → low default probability.
    healthy = (df["altman_z_score"] > 2.6) & (df["dscr"] > 1.25)
    p[healthy] = 0.05

    # --- Gaussian noise to prevent deterministic overfitting ---
    noise = np.random.normal(0.0, NOISE_STD, n)
    p = np.clip(p + noise, 0.0, 1.0)

    return p


def main() -> None:
    df = generate_features(N_SAMPLES)
    p_default = compute_default_probability(df)

    df["defaulted_in_12_months"] = (
        np.random.uniform(size=N_SAMPLES) < p_default
    ).astype(int)

    float_cols = ["current_ratio", "debt_to_equity", "ebitda_margin",
                  "icr", "dscr", "altman_z_score"]
    df[float_cols] = df[float_cols].round(4)

    df.to_csv(OUTPUT_FILE, index=False)

    # Sanity-check summary
    high_distress_mask = (df["altman_z_score"] < 1.1) & (df["dscr"] < 1.0)
    healthy_mask = (df["altman_z_score"] > 2.6) & (df["dscr"] > 1.25)

    print(f"Saved {len(df):,} records  →  {OUTPUT_FILE}")
    print(f"Overall default rate    : {df['defaulted_in_12_months'].mean():.2%}")
    print(f"High-distress zone rate : {df.loc[high_distress_mask, 'defaulted_in_12_months'].mean():.2%}  "
          f"(n={high_distress_mask.sum()})")
    print(f"Healthy zone rate       : {df.loc[healthy_mask, 'defaulted_in_12_months'].mean():.2%}  "
          f"(n={healthy_mask.sum()})")
    print(f"Middle zone rate        : {df.loc[~high_distress_mask & ~healthy_mask, 'defaulted_in_12_months'].mean():.2%}  "
          f"(n={(~high_distress_mask & ~healthy_mask).sum()})")


if __name__ == "__main__":
    main()
