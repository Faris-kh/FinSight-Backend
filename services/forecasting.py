"""
PoD Inference Service
---------------------
Loads the trained LightGBM artifact and exposes a single method for per-month
Probability of Default inference during the DES forecast loop.

Feature contract (must match train_pod_model.py FEATURE_COLS exactly):
    roa, current_ratio, quick_ratio, ebitda_margin,
    debt_to_assets, icr, dscr_proxy
"""

import os
import pickle

import pandas as pd

# Canonical feature order — must be identical to FEATURE_COLS in train_pod_model.py.
_FEATURE_COLS: list[str] = [
    "roa",
    "current_ratio",
    "quick_ratio",
    "ebitda_margin",
    "debt_to_assets",
    "icr",
    "dscr_proxy",
]

_DEFAULT_ARTIFACT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "artifacts", "lightgbm_pod_model.pkl")
)


class PodPredictor:
    """
    Wraps the serialised LightGBM PoD model for single-row monthly inference.
    Loaded once at API startup; shared across all requests.
    """

    def __init__(self, artifact_path: str = _DEFAULT_ARTIFACT) -> None:
        with open(artifact_path, "rb") as fh:
            self._model = pickle.load(fh)

    def predict_monthly_pod(self, reconstructed_balance_sheet: dict) -> float:
        """
        Parameters
        ----------
        reconstructed_balance_sheet : dict
            Must contain:
              Pre-computed ratios (extracted directly):
                roa, current_ratio, quick_ratio, ebitda_margin, icr
              Raw balance-sheet values (used to derive remaining features):
                total_debt, total_assets      → debt_to_assets
                operating_cash_flow, total_liabilities  → dscr_proxy

        Returns
        -------
        float
            Probability of default for this month, rounded to 4 decimal places.
        """
        bs = reconstructed_balance_sheet

        # Derived features — computed here so the caller passes raw financials
        # and does not need to know the exact ratio conventions the model expects.
        debt_to_assets = bs["total_debt"] / max(bs["total_assets"], 1.0)
        dscr_proxy     = bs["operating_cash_flow"] / max(bs["total_liabilities"], 1.0)

        row = {
            "roa":            bs["roa"],
            "current_ratio":  bs["current_ratio"],
            "quick_ratio":    bs["quick_ratio"],
            "ebitda_margin":  bs["ebitda_margin"],
            "debt_to_assets": debt_to_assets,
            "icr":            bs["icr"],
            "dscr_proxy":     dscr_proxy,
        }

        df = pd.DataFrame([row], columns=_FEATURE_COLS)
        return float(round(self._model.predict_proba(df)[0][1], 4))
