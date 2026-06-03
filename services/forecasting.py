"""
PoD Inference Service
---------------------
Loads the trained LightGBM artifact and exposes a single method for per-month
Probability of Default inference during the DES forecast loop.

Feature contract (must match train_pod_model.py FEATURE_COLS exactly):
    roa, current_ratio, quick_ratio, ebit_margin,
    debt_to_assets, icr, dscr_proxy

Training definitions (Polish Companies Bankruptcy Dataset, UCI ID 365):
    roa           = EBIT / total_assets                          (A7)
    current_ratio = current_assets / short_term_liabilities      (A4)
    quick_ratio   = (current_assets - inventory) / STL           (A46)
    ebit_margin   = operating_profit / sales                     (A42)
    debt_to_assets= total_liabilities / total_assets             (A2)
    icr           = operating_profit / financial_expenses        (A27)
    dscr_proxy    = (net_profit + depreciation) / total_liab.    (A26)

Inference approximations (documented, not hidden):
    debt_to_assets: uses caller-supplied total_liabilities (not total_debt).
    dscr_proxy:     numerator uses operating_cash_flow directly; training used
                    (net_profit + depreciation) — both are cash-generation proxies
                    relative to total liabilities.
    roa:            uses projected rolling total_assets; training used
                    period-end reported total_assets.
"""

import os
import pickle

import numpy as np
import pandas as pd

# Canonical feature order — must be identical to FEATURE_COLS in train_pod_model.py.
_FEATURE_COLS: list[str] = [
    "roa",
    "current_ratio",
    "quick_ratio",
    "ebit_margin",
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
              Pre-computed ratios (passed through directly):
                roa, current_ratio, quick_ratio, ebit_margin, icr
              Raw balance-sheet values (used to derive remaining ratios):
                total_assets, total_liabilities  → debt_to_assets
                operating_cash_flow, total_liabilities  → dscr_proxy

            Any value may be float('nan') when the underlying denominator is
            zero; the model's trained missing-value branches handle these.

        Returns
        -------
        float
            Probability of default for this month, rounded to 4 decimal places.
        """
        bs = reconstructed_balance_sheet

        total_assets      = bs["total_assets"]
        total_liabilities = bs["total_liabilities"]

        # debt_to_assets: total_liabilities / total_assets (matches Polish A2).
        # total_debt is NOT used here — the two leverage inputs serve different
        # models (scoring-engine D/E uses total_debt; PoD model uses total_liab.).
        debt_to_assets = (
            total_liabilities / total_assets
            if total_assets != 0
            else float("nan")
        )

        # dscr_proxy: operating_cash_flow / total_liabilities (matches Polish A26
        # denominator; numerator uses OCF directly instead of net_profit +
        # depreciation — documented approximation, not a silent substitution).
        dscr_proxy = (
            bs["operating_cash_flow"] / total_liabilities
            if total_liabilities != 0
            else float("nan")
        )

        row = {
            "roa":            bs["roa"],
            "current_ratio":  bs["current_ratio"],
            "quick_ratio":    bs["quick_ratio"],
            "ebit_margin":    bs["ebit_margin"],
            "debt_to_assets": debt_to_assets,
            "icr":            bs["icr"],
            "dscr_proxy":     dscr_proxy,
        }

        df = pd.DataFrame([row], columns=_FEATURE_COLS)
        return float(round(self._model.predict_proba(df)[0][1], 4))
