from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
import numpy as np

from services.forecasting import PodPredictor

# --- 1. App setup ---
app = FastAPI(
    title="FinSight Forecast API",
    description="Cash flow forecasting backend for FinSight SME funding assessment",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://finsight-gp.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. ML Artifacts (loaded once at startup, not per-request) ---
_pod_predictor = PodPredictor()


# --- 3. Schemas ---

class AssessRequest(BaseModel):
    currentAssets: float
    currentLiabilities: float
    totalAssets: float
    totalDebt: float
    equity: float
    revenue: float
    expenses: float
    retainedEarnings: float | None = None

class AltmanZScore(BaseModel):
    score: float
    zone: str                           # "Safe" | "Grey" | "Distress"
    variant: str = "Altman Z'' (Private Non-Manufacturing, 1995)"

class AssessResponse(BaseModel):
    altmanZScore: AltmanZScore
    flags: list[str] = []


class LoanParams(BaseModel):
    profit_rate: float = Field(default=0.08, gt=0, le=1.0)
    tenor_months: int = Field(default=36, gt=0)


class ForecastRequest(BaseModel):
    historicalCashFlows: list[float]    # time-ordered monthly series (min 2 values)
    currentAssets: float
    currentLiabilities: float
    totalAssets: float
    totalDebt: float                    # financial debt — used for D/E covenant ceiling
    totalLiabilities: float             # all obligations — used for PoD ML features
    equity: float
    revenue: float
    expenses: float
    industry: Literal["Construction", "Logistics", "Retail", "SaaS", "Manufacturing", "Tourism", "Healthcare"] | None = None
    retainedEarnings: float | None = None
    inventory: float | None = None
    debtService: float | None = None    # annual figure
    interest_expense: float | None = None  # annual interest expense; used to compute real ICR
    confidenceTier: Literal["narrow", "standard", "wide"] = "standard"
    loan_params: LoanParams | None = None

class MonthForecast(BaseModel):
    month: str
    forecastedCashFlow: float
    upperBound: float
    lowerBound: float
    dscr: float | None
    quickRatio: float | None
    currentRatio: float | None
    probabilityOfDefault: float
    flags: list[str] = []

class LoanCeilings(BaseModel):
    dscr_ceiling: float | None = None
    icr_ceiling: float | None = None
    debt_ebitda_ceiling: float | None = None
    de_ceiling: float | None = None


class LoanRecommendation(BaseModel):
    base_max_capacity: float
    stressed_max_capacity: float
    binding_constraint: str
    stressed_binding_constraint: str | None = None
    status: str
    ceilings: LoanCeilings
    stressed_ceilings: LoanCeilings | None = None
    inputs_used: dict
    flags: list[str]


class ForecastResponse(BaseModel):
    forecastedCashflow: list[MonthForecast]
    confidenceTier: str
    confidence_band_note: str
    forecast_method: str            # "ARIMA" or "DES" — the method that produced the numbers
    forecast_flags: list[str] = []  # e.g. ["ARIMA_FAILED_FELL_BACK_TO_DES"]
    loan_recommendation: LoanRecommendation
    pod_model_notes: list[str]


# --- 4. Helpers ---

MONTH_LABELS = ["Month 1", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6"]

_TIER_MULTIPLIERS: dict[str, float] = {
    "narrow":   1.0,
    "standard": 1.5,
    "wide":     2.0,
}

# Methodology disclosures returned with every forecast response so consumers
# know exactly where training and inference definitions diverge.
_POD_MODEL_NOTES: list[str] = [
    "Model: LightGBM trained on Polish Companies Bankruptcy Dataset (UCI ID 365, "
    "43,405 obs., 4.8% bankruptcy rate, 1–5 year prediction windows).",
    "roa: inference uses projected rolling total_assets (balance sheet drifts "
    "forward each month); training (Polish A7) used period-end reported total_assets.",
    "ebit_margin: computed as EBIT / revenue; training feature (Polish A42) is "
    "operating_profit / sales — equivalent when EBIT ≈ operating_profit.",
    "dscr_proxy: inference numerator is operating_cash_flow; training (Polish A26) "
    "used (net_profit + depreciation). Both measure cash generation relative to "
    "total_liabilities. Denominator is identical.",
    "debt_to_assets: uses total_liabilities (all obligations), NOT total_debt. "
    "total_debt is reserved for the scoring-engine D/E covenant ceiling only.",
    "icr: computed as EBIT / interest_expense — same direction as training "
    "(Polish A27: operating_profit / financial_expenses; higher = safer). "
    "Zero-debt firms are capped at 6,961 (Polish A27 p99) rather than a sentinel, "
    "keeping the value in-distribution.",
]


def _compute_altman_z(
    current_assets: float,
    current_liabilities: float,
    total_assets: float,
    total_debt: float,
    equity: float,
    ebit: float,
    retained_earnings: float | None,
) -> tuple[AltmanZScore, list[str]]:
    if total_assets == 0:
        raise ValueError("totalAssets cannot be zero.")

    flags: list[str] = []

    x1 = (current_assets - current_liabilities) / total_assets

    if retained_earnings is None:
        x2 = equity / total_assets
        flags.append("X2_RETAINED_EARNINGS_MISSING_EQUITY_SUBSTITUTED")
    else:
        x2 = retained_earnings / total_assets

    x3 = ebit / total_assets

    if total_debt != 0:
        x4 = equity / total_debt
    else:
        # Zero-debt firms have an undefined (infinite) equity/debt ratio.
        # 10.0 is a documented display sentinel; places the firm firmly in Safe
        # zone without producing an astronomically large score.
        x4 = 10.0
        flags.append("X4_ZERO_DEBT_SENTINEL_APPLIED")

    score = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4

    if score > 2.6:
        zone = "Safe"
    elif score >= 1.1:
        zone = "Grey"
    else:
        zone = "Distress"

    return AltmanZScore(
        score=round(score, 4),
        zone=zone,
        variant="Altman Z'' (Private Non-Manufacturing, 1995)",
    ), flags


# Minimum real months required to route to ARIMA instead of DES.
_ARIMA_MIN_HISTORY: int = 24


def _run_des(historical_cfs: list[float], n: int = 6) -> list[float]:
    if len(historical_cfs) < 4:
        raise ValueError(
            "At least 4 historical cash flow values are required. "
            "Damped-trend ExponentialSmoothing with estimated initialisation "
            "is unstable on shorter series."
        )
    series = np.array(historical_cfs, dtype=float)
    fit = ExponentialSmoothing(
        series,
        trend="add",
        damped_trend=True,
        initialization_method="estimated",
    ).fit(optimized=True)
    return fit.forecast(n).tolist()


def _run_arima(historical_cfs: list[float], n: int = 6) -> list[float]:
    """
    Fit ARIMA on the full historical series and forecast n steps.
    Search space is bounded (max p/d/q = 3/2/3) to keep per-request latency
    under ~600 ms even on lumpy SME series. auto_arima selects order via AIC.
    Raises ValueError when the fit produces non-finite or explosive forecasts.
    """
    series = np.array(historical_cfs, dtype=float)
    model = auto_arima(
        series,
        seasonal=False,
        information_criterion="aic",
        max_p=3, max_d=2, max_q=3,
        start_p=0, start_q=0, d=None,
        error_action="ignore",
        suppress_warnings=True,
    )
    forecasts = np.asarray(model.predict(n), dtype=float)
    if not all(np.isfinite(v) for v in forecasts):
        raise ValueError("ARIMA produced non-finite forecast values.")
    hist_mean = float(np.abs(series).mean()) or 1.0
    if any(abs(v) > hist_mean * 100 for v in forecasts):
        # Explosive divergence — treat as degenerate fit.
        raise ValueError("ARIMA forecast diverges beyond 100× historical mean.")
    return forecasts.tolist()


def _route_forecast(
    historical_cfs: list[float],
) -> tuple[list[float], str, list[str]]:
    """
    Length-adaptive router.
    >=24 real months → ARIMA; <24 → DES.
    DES is the fallback when ARIMA fails; a flag is emitted when fallback fires.
    Returns (forecasted_cfs, method_used, forecast_flags).
    """
    flags: list[str] = []
    if len(historical_cfs) >= _ARIMA_MIN_HISTORY:
        try:
            return _run_arima(historical_cfs), "ARIMA", flags
        except Exception:
            flags.append("ARIMA_FAILED_FELL_BACK_TO_DES")
    return _run_des(historical_cfs), "DES", flags


def _project_covenants(
    data: ForecastRequest,
    forecasted_cfs: list[float],
    baseline_avg: float,
) -> list[MonthForecast]:
    inventory        = data.inventory or 0.0
    cl               = data.currentLiabilities
    std_dev          = float(np.std(data.historicalCashFlows, ddof=1)) if len(data.historicalCashFlows) >= 2 else 0.0
    ebit             = data.revenue - data.expenses
    band_multiplier  = _TIER_MULTIPLIERS[data.confidenceTier]

    # ebit_margin: EBIT / revenue — matches Polish A42 (operating profit / sales).
    # NaN when revenue is zero; flagged per-month below.
    ebit_margin_ratio: float = ebit / data.revenue if data.revenue != 0 else float("nan")

    icr_flags: list[str] = []
    if data.interest_expense is not None and data.interest_expense > 0:
        icr = ebit / data.interest_expense
    elif data.totalDebt == 0:
        # Polish A27 p99 ≈ 6,961 — in-distribution upper bound for zero-debt firms.
        # 999.0 was a sentinel with no basis in the training distribution.
        icr = 6_961.0
    else:
        icr = 0.0
        if data.interest_expense is None and data.totalDebt > 0:
            icr_flags.append("ICR_ASSUMED_ZERO_INTEREST_EXPENSE_NOT_PROVIDED")

    projections: list[MonthForecast] = []
    cum_cash_delta = 0.0  # running total of monthly CF deviations from historical baseline

    for i, cf in enumerate(forecasted_cfs):
        margin = std_dev * (1 + 0.15 * i) * band_multiplier

        # Accumulate each month's deviation so the balance sheet drifts forward in
        # time rather than resetting to the starting position every iteration.
        cum_cash_delta += cf - baseline_avg

        adj_current_assets = data.currentAssets + cum_cash_delta
        adj_total_assets   = data.totalAssets   + cum_cash_delta

        month_flags: list[str] = list(icr_flags)

        # DSCR (display only; single-month CF annualised — not a trailing-12-months figure).
        dscr_val: float | None = None
        if data.debtService and data.debtService > 0:
            dscr_val = round((cf * 12) / data.debtService, 4)
            month_flags.append("DSCR_IS_SINGLE_MONTH_ANNUALISED_NOT_TTM")

        # Current and quick ratios: undefined when current_liabilities is zero.
        # NaN propagates into the ML feature row; LightGBM routes it via its
        # trained missing-value branch rather than receiving a fake denominator.
        if cl > 0:
            quick_ratio_ml   = (adj_current_assets - inventory) / cl
            current_ratio_ml =  adj_current_assets              / cl
            quick_ratio_disp:   float | None = round(quick_ratio_ml,   4)
            current_ratio_disp: float | None = round(current_ratio_ml, 4)
            if data.inventory is None:
                month_flags.append("QUICK_RATIO_INVENTORY_ASSUMED_ZERO")
        else:
            quick_ratio_ml   = float("nan")
            current_ratio_ml = float("nan")
            quick_ratio_disp   = None
            current_ratio_disp = None
            month_flags.append("QUICK_CURRENT_RATIO_UNDEFINED_ZERO_CURRENT_LIABILITIES")

        # ROA: undefined when projected total_assets is zero (extreme insolvency).
        if adj_total_assets != 0:
            roa_feature: float = ebit / adj_total_assets
        else:
            roa_feature = float("nan")
            month_flags.append("ROA_UNDEFINED_ZERO_PROJECTED_ASSETS")

        if np.isnan(ebit_margin_ratio):
            month_flags.append("EBIT_MARGIN_UNDEFINED_ZERO_REVENUE")

        # Balance sheet passed to PoD inference.
        # total_liabilities is the real input field — total_debt is NOT used here.
        balance_sheet = {
            "roa":                 roa_feature,
            "current_ratio":       current_ratio_ml,
            "quick_ratio":         quick_ratio_ml,
            "ebit_margin":         ebit_margin_ratio,
            "icr":                 icr,
            "total_assets":        adj_total_assets,
            "operating_cash_flow": cf,
            "total_liabilities":   data.totalLiabilities,
        }
        pod = _pod_predictor.predict_monthly_pod(balance_sheet)

        projections.append(MonthForecast(
            month=MONTH_LABELS[i],
            forecastedCashFlow=round(cf, 2),
            upperBound=round(cf + margin, 2),
            lowerBound=round(max(0.0, cf - margin), 2),
            dscr=dscr_val,
            quickRatio=quick_ratio_disp,
            currentRatio=current_ratio_disp,
            probabilityOfDefault=pod,
            flags=month_flags,
        ))

    return projections


def _run_capacity_calc(
    ebit_val: float,
    existing_debt_service: float,
    total_existing_debt: float,
    equity: float,
    ebit_proxy: float,
    profit_rate: float,
    tenor_months: int,
) -> tuple[float, str, LoanCeilings, list[str]]:
    """
    Runs the 4-covenant ceiling calculation for the supplied EBIT figure.
    Returns (raw_capacity, binding_constraint_name, ceilings, flags).
    Reused for both the base (LTM EBIT) and stressed (worst-month) scenarios.
    """
    flags: list[str] = []

    # Ceiling 1: DSCR — back-solve via PV of annuity
    max_annual_ds = (ebit_val - existing_debt_service) / 1.25
    pmt = max_annual_ds / 12.0
    r   = profit_rate / 12.0
    dscr_cap = pmt * ((1.0 - (1.0 + r) ** -tenor_months) / r) if r > 0 else pmt * tenor_months

    # Ceiling 2: ICR — max debt s.t. ICR >= 2.0x
    icr_cap = (ebit_val / 2.0) / profit_rate - total_existing_debt

    # Ceiling 3: Debt / EBIT <= 3.5x (EBIT proxy; D&A unavailable; constant across both scenarios)
    debt_ebitda_cap = (3.5 * ebit_proxy) - total_existing_debt

    # Ceiling 4: D/E <= 2.0x (skipped when equity <= 0)
    de_cap: float | None = None
    if equity > 0:
        de_cap = (2.0 * equity) - total_existing_debt
    else:
        flags.append("NEGATIVE_OR_ZERO_EQUITY_DE_CONSTRAINT_SKIPPED")

    ceilings = LoanCeilings(
        dscr_ceiling=round(dscr_cap, 2),
        icr_ceiling=round(icr_cap, 2),
        debt_ebitda_ceiling=round(debt_ebitda_cap, 2),
        de_ceiling=round(de_cap, 2) if de_cap is not None else None,
    )

    candidates: dict[str, float] = {
        "dscr_ceiling":        dscr_cap,
        "icr_ceiling":         icr_cap,
        "debt_ebitda_ceiling": debt_ebitda_cap,
    }
    if de_cap is not None:
        candidates["de_ceiling"] = de_cap

    positive = {k: v for k, v in candidates.items() if v > 0}
    if not positive:
        return 0.0, "NONE", ceilings, flags + ["ALL_CONSTRAINTS_NON_POSITIVE"]

    binding_name = min(positive, key=lambda k: positive[k])
    return positive[binding_name], binding_name, ceilings, flags


def _compute_loan_recommendation(
    data: ForecastRequest,
    forecasted_cfs: list[float],
) -> LoanRecommendation:
    params = data.loan_params or LoanParams()
    profit_rate  = params.profit_rate
    tenor_months = params.tenor_months

    ltm_ebit               = data.revenue - data.expenses
    min_forecast_monthly_cf = float(min(forecasted_cfs))
    stressed_ebit           = 12.0 * min_forecast_monthly_cf
    debt_service_absent     = data.debtService is None
    existing_debt_service   = data.debtService if not debt_service_absent else 0.0
    total_existing_debt     = data.totalDebt
    equity                  = data.equity
    ebit_proxy              = ltm_ebit

    inputs_used: dict = {
        "ltm_ebit":                round(ltm_ebit, 2),
        "stressed_ebit":           round(stressed_ebit, 2),
        "stressed_ebit_note":      "12 × worst forecast monthly CF; monthly CF used as EBIT proxy — may diverge from period EBIT",
        "min_forecast_monthly_cf": round(min_forecast_monthly_cf, 2),
        "existing_debt_service":   round(existing_debt_service, 2),
        "debt_service_absent":     debt_service_absent,
        "total_existing_debt":     round(total_existing_debt, 2),
        "equity":                  round(equity, 2),
        "ebit_proxy":              round(ebit_proxy, 2),
        "profit_rate":             profit_rate,
        "tenor_months":            tenor_months,
        "icr_ceiling_assumption":  "profit_rate used as proxy for borrower's blended cost of debt in ICR ceiling",
    }

    pre_flags: list[str] = []
    if debt_service_absent:
        pre_flags.append("DEBT_SERVICE_ASSUMED_ZERO_NOT_PROVIDED")

    # --- Early exit: LTM earnings are non-positive — both capacities are 0 ---
    if ltm_ebit <= 0:
        return LoanRecommendation(
            base_max_capacity=0.0,
            stressed_max_capacity=0.0,
            binding_constraint="NONE",
            status="NOT_RECOMMENDED_INSUFFICIENT_EARNINGS",
            ceilings=LoanCeilings(),
            inputs_used=inputs_used,
            flags=pre_flags + ["LTM_EBIT_NON_POSITIVE"],
        )

    # --- Base scenario: 4 ceilings driven by LTM EBIT ---
    base_cap, binding_name, ceilings, flags = _run_capacity_calc(
        ltm_ebit, existing_debt_service, total_existing_debt,
        equity, ebit_proxy, profit_rate, tenor_months,
    )
    flags = pre_flags + flags

    if base_cap == 0.0:
        return LoanRecommendation(
            base_max_capacity=0.0,
            stressed_max_capacity=0.0,
            binding_constraint=binding_name,
            status="NOT_RECOMMENDED_ALL_CONSTRAINTS_BINDING",
            ceilings=ceilings,
            inputs_used=inputs_used,
            flags=flags,
        )

    # --- Stressed scenario: same 4 ceilings driven by 12 × worst monthly CF ---
    if stressed_ebit <= 0:
        stressed_cap = 0.0
        stressed_binding = "NONE"
        stressed_ceilings_val = LoanCeilings()
    else:
        stressed_cap, stressed_binding, stressed_ceilings_val, _ = _run_capacity_calc(
            stressed_ebit, existing_debt_service, total_existing_debt,
            equity, ebit_proxy, profit_rate, tenor_months,
        )

    # --- Kafalah hard caps applied to both scenarios ---
    kafalah_cap_value: int | None = None
    if data.revenue <= 3_000_000:
        kafalah_cap_value = 2_500_000
    elif data.revenue <= 40_000_000:
        kafalah_cap_value = 5_000_000
    elif data.revenue <= 200_000_000:
        kafalah_cap_value = 15_000_000
    else:
        flags.append("KAFALAH_CAP_NOT_APPLICABLE_REVENUE_EXCEEDS_200M_SAR")

    kafalah_applied = False
    if kafalah_cap_value is not None:
        if base_cap > kafalah_cap_value:
            base_cap = float(kafalah_cap_value)
            kafalah_applied = True
        if stressed_cap > kafalah_cap_value:
            stressed_cap = float(kafalah_cap_value)
            kafalah_applied = True
    if kafalah_applied:
        flags.append(f"KAFALAH_CAP_APPLIED_{kafalah_cap_value:,}_SAR")

    return LoanRecommendation(
        base_max_capacity=round(base_cap, 2),
        stressed_max_capacity=round(stressed_cap, 2),
        binding_constraint=binding_name,
        stressed_binding_constraint=stressed_binding,
        status="RECOMMENDED",
        ceilings=ceilings,
        stressed_ceilings=stressed_ceilings_val,
        inputs_used={
            **inputs_used,
            "kafalah_cap_applied": kafalah_applied,
            "kafalah_cap_value":   kafalah_cap_value,
        },
        flags=flags,
    )


# --- 5. Endpoints ---

@app.post("/api/computeAssessment", response_model=AssessResponse)
def assess(data: AssessRequest):
    try:
        ebit = data.revenue - data.expenses
        z_score, z_flags = _compute_altman_z(
            current_assets=data.currentAssets,
            current_liabilities=data.currentLiabilities,
            total_assets=data.totalAssets,
            total_debt=data.totalDebt,
            equity=data.equity,
            ebit=ebit,
            retained_earnings=data.retainedEarnings,
        )
        return AssessResponse(altmanZScore=z_score, flags=z_flags)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/forecast", response_model=ForecastResponse)
def forecast_cash_flow(data: ForecastRequest):
    try:
        forecasted_cfs, method, forecast_flags = _route_forecast(data.historicalCashFlows)
        baseline_avg = float(np.mean(data.historicalCashFlows))
        projections = _project_covenants(data, forecasted_cfs, baseline_avg)
        loan_rec = _compute_loan_recommendation(data, forecasted_cfs)
        band_mult = _TIER_MULTIPLIERS[data.confidenceTier]
        if method == "ARIMA":
            band_note = (
                f"{data.confidenceTier} (±{band_mult:.1f}σ) — "
                "tier-based σ-bands applied to ARIMA forecast; "
                "ARIMA native prediction intervals not used — "
                "tier-based bands kept for display consistency across methods; "
                "does not affect point forecast or model inputs"
            )
        else:
            band_note = (
                f"{data.confidenceTier} (±{band_mult:.1f}σ) — "
                "tier-based σ-bands applied to DES forecast; "
                "does not affect point forecast or model inputs"
            )
        return ForecastResponse(
            forecastedCashflow=projections,
            confidenceTier=data.confidenceTier,
            confidence_band_note=band_note,
            forecast_method=method,
            forecast_flags=forecast_flags,
            loan_recommendation=loan_rec,
            pod_model_notes=_POD_MODEL_NOTES,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- 6. Health check ---

@app.get("/")
def health_check():
    return {
        "status": "FinSight API is running",
        "forecast": f"ARIMA (≥{_ARIMA_MIN_HISTORY} months) / DES fallback + LightGBM Risk Classifier",
    }
