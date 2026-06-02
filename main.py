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
    industry: str | None = None
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
    else:
        # Zero interest expense: no debt burden (matches zero-debt firms in Taiwanese dataset).
        icr = 0.0

    projections: list[MonthForecast] = []

    for i, cf in enumerate(forecasted_cfs):
        margin         = std_dev * (1 + 0.15 * i)
        cash_shortfall = baseline_avg - cf

        # Adjust current assets and total assets by the cumulative cash shortfall.
        adj_current_assets = data.currentAssets - cash_shortfall
        adj_total_assets   = data.totalAssets   - cash_shortfall

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


def _compute_loan_recommendation(
    data: ForecastRequest,
    forecasted_cfs: list[float],
) -> LoanRecommendation:
    params = data.loan_params or LoanParams()
    profit_rate = params.profit_rate
    tenor_months = params.tenor_months

    ltm_ebit = data.revenue - data.expenses
    # Annualise the 6-month monthly forecast to compare against LTM figures.
    avg_forecast_ebit = float(np.mean(forecasted_cfs)) * 12.0
    min_forecast_monthly_cf = float(min(forecasted_cfs))
    existing_debt_service = data.debtService or 0.0
    total_existing_debt = data.totalDebt
    equity = data.equity
    # D&A not available; use LTM EBIT as a conservative EBITDA proxy.
    ebitda_proxy = ltm_ebit

    ebit_stress = min(ltm_ebit, 0.8 * avg_forecast_ebit, 12.0 * min_forecast_monthly_cf)

    flags: list[str] = []

    base_inputs: dict = {
        "ltm_ebit": round(ltm_ebit, 2),
        "avg_forecast_ebit_annualized": round(avg_forecast_ebit, 2),
        "min_forecast_monthly_cf": round(min_forecast_monthly_cf, 2),
        "ebit_stress": round(ebit_stress, 2),
        "existing_debt_service": round(existing_debt_service, 2),
        "total_existing_debt": round(total_existing_debt, 2),
        "equity": round(equity, 2),
        "profit_rate": profit_rate,
        "tenor_months": tenor_months,
    }

    if ebit_stress <= 0:
        return LoanRecommendation(
            base_max_capacity=0.0,
            binding_constraint="NONE",
            status="NOT_RECOMMENDED_INSUFFICIENT_EARNINGS",
            ceilings=LoanCeilings(),
            inputs_used=base_inputs,
            flags=["EBIT_STRESS_NON_POSITIVE"],
        )

    # --- Ceiling 1: DSCR (back-solved via PV of annuity) ---
    max_annual_debt_service = (ebit_stress - existing_debt_service) / 1.25
    pmt = max_annual_debt_service / 12.0
    r_monthly = profit_rate / 12.0
    if r_monthly > 0:
        dscr_cap = pmt * ((1.0 - (1.0 + r_monthly) ** -tenor_months) / r_monthly)
    else:
        dscr_cap = pmt * tenor_months  # zero-rate edge case

    # --- Ceiling 2: ICR (max debt s.t. ICR >= 2.0) ---
    icr_cap = (ebit_stress / 2.0) / profit_rate - total_existing_debt

    # --- Ceiling 3: Debt / EBITDA <= 3.5x ---
    debt_ebitda_cap = (3.5 * ebitda_proxy) - total_existing_debt

    # --- Ceiling 4: D/E <= 2.0x (skipped when equity <= 0) ---
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

    # --- Binding constraint: smallest positive ceiling ---
    candidates: dict[str, float] = {
        "dscr_ceiling": dscr_cap,
        "icr_ceiling": icr_cap,
        "debt_ebitda_ceiling": debt_ebitda_cap,
    }
    if de_cap is not None:
        candidates["de_ceiling"] = de_cap

    positive_candidates = {k: v for k, v in candidates.items() if v > 0}

    if not positive_candidates:
        return LoanRecommendation(
            base_max_capacity=0.0,
            binding_constraint="NONE",
            status="NOT_RECOMMENDED_ALL_CONSTRAINTS_BINDING",
            ceilings=ceilings,
            inputs_used={**base_inputs, "ebitda_proxy": round(ebitda_proxy, 2)},
            flags=flags + ["ALL_CONSTRAINTS_NON_POSITIVE"],
        )

    binding_name = min(positive_candidates, key=lambda k: positive_candidates[k])
    base_max_capacity = positive_candidates[binding_name]

    # --- Kafalah hard caps (segment guardrails by trailing revenue) ---
    kafalah_cap_value: int | None = None
    if data.revenue <= 3_000_000:
        kafalah_cap_value = 2_500_000
    elif data.revenue <= 40_000_000:
        kafalah_cap_value = 5_000_000
    elif data.revenue <= 200_000_000:
        kafalah_cap_value = 15_000_000

    kafalah_applied = False
    if kafalah_cap_value is not None and base_max_capacity > kafalah_cap_value:
        base_max_capacity = float(kafalah_cap_value)
        kafalah_applied = True
        flags.append(f"KAFALAH_CAP_APPLIED_{kafalah_cap_value:,}_SAR")

    return LoanRecommendation(
        base_max_capacity=round(base_max_capacity, 2),
        binding_constraint=binding_name,
        status="RECOMMENDED",
        ceilings=ceilings,
        inputs_used={
            **base_inputs,
            "ebitda_proxy": round(ebitda_proxy, 2),
            "kafalah_cap_applied": kafalah_applied,
            "kafalah_cap_value": kafalah_cap_value,
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
