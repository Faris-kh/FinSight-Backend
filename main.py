import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

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
_model = joblib.load("lgbm_risk_classifier.joblib")
_industry_cats: list[str] = json.load(open("industry_categories.json"))

# Must match the column order used in train_lgbm_model.py exactly.
_FEATURE_COLS = [
    "current_ratio", "debt_to_equity", "ebitda_margin",
    "icr", "dscr", "altman_z_score", "industry",
]


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
    industry: str                       # Must be in industry_categories.json
    icr: float | None = None            # If absent, proxied from EBIT / estimated interest
    dscr: float | None = None           # If absent, defaults to neutral 1.0

class AltmanZScore(BaseModel):
    score: float
    zone: str                           # "Safe" | "Grey" | "Distress"

class AssessResponse(BaseModel):
    altmanZScore: AltmanZScore
    probabilityOfDefault: float         # LightGBM PoD [0.0 – 1.0]


class ForecastRequest(BaseModel):
    historicalCashFlows: list[float]
    currentAssets: float
    currentLiabilities: float
    totalAssets: float
    totalDebt: float
    equity: float
    inventory: float | None = None
    debtService: float | None = None
    confidenceTier: str = "standard"

class MonthForecast(BaseModel):
    month: str
    forecastedCashFlow: float
    upperBound: float
    lowerBound: float
    dscr: float | None
    quickRatio: float | None
    currentRatio: float | None

class ForecastResponse(BaseModel):
    forecastedCashflow: list[MonthForecast]
    confidenceTier: str


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


def _build_feature_df(data: AssessRequest, z_score: AltmanZScore) -> pd.DataFrame:
    ebit = data.revenue - data.expenses

    current_ratio = (
        data.currentAssets / data.currentLiabilities
        if data.currentLiabilities != 0 else 0.0
    )
    debt_to_equity = (
        data.totalDebt / data.equity
        if data.equity != 0 else 0.0
    )
    ebitda_margin = (
        (ebit / data.revenue) * 100.0
        if data.revenue != 0 else 0.0
    )
    # Proxy ICR via a 6% assumed interest rate on outstanding debt when not supplied.
    if data.icr is not None:
        icr = data.icr
    else:
        estimated_interest = data.totalDebt * 0.06
        icr = ebit / estimated_interest if estimated_interest > 0 else 0.0

    # Neutral break-even fallback when DSCR is not supplied.
    dscr = data.dscr if data.dscr is not None else 1.0

    row = {
        "current_ratio":  current_ratio,
        "debt_to_equity": debt_to_equity,
        "ebitda_margin":  ebitda_margin,
        "icr":            icr,
        "dscr":           dscr,
        "altman_z_score": z_score.score,
        "industry":       data.industry,
    }
    df = pd.DataFrame([row], columns=_FEATURE_COLS)
    df["industry"] = pd.Categorical(df["industry"], categories=_industry_cats)
    return df


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
    cl = data.currentLiabilities
    std_dev = float(np.std(data.historicalCashFlows, ddof=1)) if len(data.historicalCashFlows) >= 2 else 0.0
    projections: list[MonthForecast] = []

    for i, cf in enumerate(forecasted_cfs):
        margin = std_dev * (1 + 0.15 * i)
        cash_shortfall = baseline_avg - cf

        dscr: float | None = None
        if data.debtService and data.debtService > 0:
            dscr = round((cf * 12) / data.debtService, 4)

        quick_ratio: float | None = None
        if cl > 0:
            quick_ratio = round((data.currentAssets - cash_shortfall - inventory) / cl, 4)

        current_ratio: float | None = None
        if cl > 0:
            current_ratio = round((data.currentAssets - cash_shortfall) / cl, 4)

        projections.append(MonthForecast(
            month=MONTH_LABELS[i],
            forecastedCashFlow=round(cf, 2),
            upperBound=round(cf + margin, 2),
            lowerBound=round(cf - margin, 2),
            dscr=dscr,
            quickRatio=quick_ratio,
            currentRatio=current_ratio,
        ))

    return projections


# --- 5. Endpoints ---

@app.post("/api/computeAssessment", response_model=AssessResponse)
def assess(data: AssessRequest):
    if data.industry not in _industry_cats:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown industry '{data.industry}'. Accepted values: {_industry_cats}",
        )
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
        feature_df = _build_feature_df(data, z_score)
        pod = float(round(_model.predict_proba(feature_df)[0][1], 4))
        return AssessResponse(altmanZScore=z_score, probabilityOfDefault=pod)
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
        return ForecastResponse(
            forecastedCashflow=projections,
            confidenceTier=data.confidenceTier,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- 6. Health check ---

@app.get("/")
def health_check():
    return {"status": "FinSight API is running", "model": "DES + LightGBM Risk Classifier"}
