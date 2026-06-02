from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
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

class AssessResponse(BaseModel):
    altmanZScore: AltmanZScore


class LoanParams(BaseModel):
    profit_rate: float = 0.08
    tenor_months: int = 36


class ForecastRequest(BaseModel):
    historicalCashFlows: list[float]    # time-ordered monthly series (min 2 values)
    currentAssets: float
    currentLiabilities: float
    totalAssets: float
    totalDebt: float
    equity: float
    revenue: float
    expenses: float
    industry: Literal["Construction", "Logistics", "Retail", "SaaS", "Manufacturing", "Tourism", "Healthcare"] | None = None
    retainedEarnings: float | None = None
    inventory: float | None = None
    debtService: float | None = None    # annual figure
    interest_expense: float = 0.0       # annual interest expense; used to compute real ICR
    confidenceTier: str = "standard"
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

class LoanCeilings(BaseModel):
    dscr_ceiling: float | None = None
    icr_ceiling: float | None = None
    debt_ebitda_ceiling: float | None = None
    de_ceiling: float | None = None


class LoanRecommendation(BaseModel):
    base_max_capacity: float
    stressed_max_capacity: float
    binding_constraint: str
    status: str
    ceilings: LoanCeilings
    inputs_used: dict
    flags: list[str]


class ForecastResponse(BaseModel):
    forecastedCashflow: list[MonthForecast]
    confidenceTier: str
    loan_recommendation: LoanRecommendation


# --- 4. Helpers ---

MONTH_LABELS = ["Month 1", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6"]


def _compute_altman_z(
    current_assets: float,
    current_liabilities: float,
    total_assets: float,
    total_debt: float,
    equity: float,
    ebit: float,
    retained_earnings: float | None,
) -> AltmanZScore:
    if total_assets == 0:
        raise ValueError("totalAssets cannot be zero.")

    x1 = (current_assets - current_liabilities) / total_assets
    x2 = (retained_earnings if retained_earnings is not None else equity) / total_assets
    x3 = ebit / total_assets
    x4 = equity / total_debt if total_debt != 0 else 0.0

    score = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4

    if score > 2.6:
        zone = "Safe"
    elif score >= 1.1:
        zone = "Grey"
    else:
        zone = "Distress"

    return AltmanZScore(score=round(score, 4), zone=zone)


def _run_des(historical_cfs: list[float], n: int = 6) -> list[float]:
    if len(historical_cfs) < 2:
        raise ValueError("At least 2 historical cash flow values are required.")
    series = np.array(historical_cfs, dtype=float)
    fit = ExponentialSmoothing(
        series,
        trend="add",
        damped_trend=True,
        initialization_method="estimated",
    ).fit(optimized=True)
    return fit.forecast(n).tolist()


def _project_covenants(
    data: ForecastRequest,
    forecasted_cfs: list[float],
    baseline_avg: float,
) -> list[MonthForecast]:
    inventory = data.inventory or 0.0
    cl        = data.currentLiabilities
    std_dev   = float(np.std(data.historicalCashFlows, ddof=1)) if len(data.historicalCashFlows) >= 2 else 0.0
    ebit      = data.revenue - data.expenses

    # Scalars constant across all 6 projected months.
    ebitda_margin_ratio = ebit / max(data.revenue, 1.0)   # 0-1, matches Taiwanese dataset scale

    if data.interest_expense > 0:
        icr = ebit / data.interest_expense
    elif data.totalDebt == 0:
        icr = 999.0  # zero-debt firm: no interest burden, set high ceiling
    else:
        icr = 0.0   # debt exists but interest_expense not provided; conservative fallback

    projections: list[MonthForecast] = []
    cum_cash_delta = 0.0  # running total of monthly CF deviations from historical baseline

    for i, cf in enumerate(forecasted_cfs):
        margin = std_dev * (1 + 0.15 * i)

        # Accumulate each month's deviation so the balance sheet drifts forward in
        # time rather than resetting to the starting position every iteration.
        cum_cash_delta += cf - baseline_avg

        adj_current_assets = data.currentAssets + cum_cash_delta
        adj_total_assets   = data.totalAssets   + cum_cash_delta

        # DSCR (display only; derived from the annual debt-service figure in the request).
        dscr_val: float | None = None
        if data.debtService and data.debtService > 0:
            dscr_val = round((cf * 12) / data.debtService, 4)

        # Quick and current ratios — used both for display and as ML inputs.
        quick_ratio_ml   = (adj_current_assets - inventory) / max(cl, 1.0)
        current_ratio_ml =  adj_current_assets              / max(cl, 1.0)
        quick_ratio_disp:   float | None = round(quick_ratio_ml,   4) if cl > 0 else None
        current_ratio_disp: float | None = round(current_ratio_ml, 4) if cl > 0 else None

        # Reconstruct the month-specific balance sheet and run PoD inference.
        balance_sheet = {
            "roa":                 ebit / max(adj_total_assets, 1.0),
            "current_ratio":       current_ratio_ml,
            "quick_ratio":         quick_ratio_ml,
            "ebitda_margin":       ebitda_margin_ratio,
            "icr":                 icr,
            "total_debt":          data.totalDebt,
            "total_assets":        max(adj_total_assets, 1.0),
            "operating_cash_flow": cf,
            "total_liabilities":   max(data.totalDebt, 1.0),
        }
        pod = _pod_predictor.predict_monthly_pod(balance_sheet)

        projections.append(MonthForecast(
            month=MONTH_LABELS[i],
            forecastedCashFlow=round(cf, 2),
            upperBound=round(cf + margin, 2),
            lowerBound=round(cf - margin, 2),
            dscr=dscr_val,
            quickRatio=quick_ratio_disp,
            currentRatio=current_ratio_disp,
            probabilityOfDefault=pod,
        ))

    return projections


def _run_capacity_calc(
    ebit_val: float,
    existing_debt_service: float,
    total_existing_debt: float,
    equity: float,
    ebitda_proxy: float,
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

    # Ceiling 3: Debt / EBITDA <= 3.5x (trailing EBITDA constant across both scenarios)
    debt_ebitda_cap = (3.5 * ebitda_proxy) - total_existing_debt

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
    stressed_ebit           = 12.0 * min_forecast_monthly_cf   # worst-month annualised
    existing_debt_service   = data.debtService or 0.0
    total_existing_debt     = data.totalDebt
    equity                  = data.equity
    ebitda_proxy            = ltm_ebit   # D&A unavailable; LTM EBIT used as conservative proxy

    inputs_used: dict = {
        "ltm_ebit":                round(ltm_ebit, 2),
        "stressed_ebit":           round(stressed_ebit, 2),
        "min_forecast_monthly_cf": round(min_forecast_monthly_cf, 2),
        "existing_debt_service":   round(existing_debt_service, 2),
        "total_existing_debt":     round(total_existing_debt, 2),
        "equity":                  round(equity, 2),
        "ebitda_proxy":            round(ebitda_proxy, 2),
        "profit_rate":             profit_rate,
        "tenor_months":            tenor_months,
    }

    # --- Early exit: LTM earnings are non-positive — both capacities are 0 ---
    if ltm_ebit <= 0:
        return LoanRecommendation(
            base_max_capacity=0.0,
            stressed_max_capacity=0.0,
            binding_constraint="NONE",
            status="NOT_RECOMMENDED_INSUFFICIENT_EARNINGS",
            ceilings=LoanCeilings(),
            inputs_used=inputs_used,
            flags=["LTM_EBIT_NON_POSITIVE"],
        )

    # --- Base scenario: 4 ceilings driven by LTM EBIT ---
    base_cap, binding_name, ceilings, flags = _run_capacity_calc(
        ltm_ebit, existing_debt_service, total_existing_debt,
        equity, ebitda_proxy, profit_rate, tenor_months,
    )

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
    else:
        stressed_cap, _, _, _ = _run_capacity_calc(
            stressed_ebit, existing_debt_service, total_existing_debt,
            equity, ebitda_proxy, profit_rate, tenor_months,
        )

    # --- Kafalah hard caps applied to both scenarios ---
    kafalah_cap_value: int | None = None
    if data.revenue <= 3_000_000:
        kafalah_cap_value = 2_500_000
    elif data.revenue <= 40_000_000:
        kafalah_cap_value = 5_000_000
    elif data.revenue <= 200_000_000:
        kafalah_cap_value = 15_000_000

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
        status="RECOMMENDED",
        ceilings=ceilings,
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
        z_score = _compute_altman_z(
            current_assets=data.currentAssets,
            current_liabilities=data.currentLiabilities,
            total_assets=data.totalAssets,
            total_debt=data.totalDebt,
            equity=data.equity,
            ebit=ebit,
            retained_earnings=data.retainedEarnings,
        )
        return AssessResponse(altmanZScore=z_score)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/forecast", response_model=ForecastResponse)
def forecast_cash_flow(data: ForecastRequest):
    try:
        forecasted_cfs = _run_des(data.historicalCashFlows)
        baseline_avg = float(np.mean(data.historicalCashFlows))
        projections = _project_covenants(data, forecasted_cfs, baseline_avg)
        loan_rec = _compute_loan_recommendation(data, forecasted_cfs)
        return ForecastResponse(
            forecastedCashflow=projections,
            confidenceTier=data.confidenceTier,
            loan_recommendation=loan_rec,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- 6. Health check ---

@app.get("/")
def health_check():
    return {"status": "FinSight API is running", "model": "DES + LightGBM Risk Classifier"}
