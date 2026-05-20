from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import json
import os

# --- 1. App setup ---
app = FastAPI(
    title="FinSight Forecast API",
    description="Cash flow forecasting backend for FinSight SME funding assessment",
    version="1.0.0"
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

# --- 2. Load model and stats ---
MODEL_PATH = "model.pkl"
STATS_PATH = "model_stats.json"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("model.pkl not found. Please run: python train_model.py")

model = joblib.load(MODEL_PATH)
print("[OK] Model loaded.")

_model_stats: dict = {}
if os.path.exists(STATS_PATH):
    with open(STATS_PATH) as f:
        _model_stats = json.load(f)
    print(f"[OK] Model stats loaded (MAE={_model_stats.get('mae')}, R2={_model_stats.get('r2')})")

# --- 3. Schemas ---

class ForecastRequest(BaseModel):
    # Core model features (required)
    revenue: float
    expenses: float
    currentAssets: float
    currentLiabilities: float
    totalAssets: float
    totalDebt: float
    equity: float
    cashFlow: float
    # Optional balance sheet items for ratio projections
    inventory: float | None = None
    interestExpense: float | None = None
    debtService: float | None = None
    # Echoed back in the response
    confidenceTier: str = "standard"

class MonthForecast(BaseModel):
    month: str
    forecastedCashFlow: float
    upperBound: float
    lowerBound: float
    confidence: float

class BreachProjection(BaseModel):
    month: int
    ratio: str
    projectedValue: float

class ForecastResponse(BaseModel):
    forecastedCashflow: list[MonthForecast]
    confidenceTier: str
    projectedBreaches: list[BreachProjection]

# --- 4. Helpers ---

MONTH_LABELS = ["Month 1", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6"]

MODEL_FEATURES = [
    "revenue", "expenses", "currentAssets", "currentLiabilities",
    "totalAssets", "totalDebt", "equity", "cashFlow", "month_ahead",
]

def _build_input_row(data: ForecastRequest, month_ahead: int) -> pd.DataFrame:
    row = {
        "revenue": data.revenue,
        "expenses": data.expenses,
        "currentAssets": data.currentAssets,
        "currentLiabilities": data.currentLiabilities,
        "totalAssets": data.totalAssets,
        "totalDebt": data.totalDebt,
        "equity": data.equity,
        "cashFlow": data.cashFlow,
        "month_ahead": month_ahead,
    }
    # Explicitly convert any None to np.nan so LightGBM routes them
    # down its learned missing-value branches rather than treating them as 0.
    row = {k: (np.nan if v is None else v) for k, v in row.items()}
    return pd.DataFrame([row])[MODEL_FEATURES]

def _project_breaches(
    data: ForecastRequest,
    monthly_predictions: list[float],
) -> list[BreachProjection]:
    breaches: list[BreachProjection] = []

    monthly_interest = (data.interestExpense / 12) if data.interestExpense else None
    monthly_debt_svc = (data.debtService / 12) if data.debtService else None
    inventory = data.inventory or 0.0
    current_liabilities = data.currentLiabilities

    # Cumulative cash position change: each month's cash flow adds to current assets.
    cumulative_cf = 0.0

    for i, predicted_cf in enumerate(monthly_predictions):
        month_num = i + 1
        cumulative_cf += predicted_cf

        # DSCR — requires debtService
        if monthly_debt_svc and monthly_debt_svc > 0:
            dscr = predicted_cf / monthly_debt_svc
            if dscr < 1.0:
                breaches.append(BreachProjection(
                    month=month_num,
                    ratio="DSCR",
                    projectedValue=round(dscr, 4),
                ))

        # ICR — requires interestExpense; uses monthly CF as EBIT proxy
        if monthly_interest and monthly_interest > 0:
            icr = predicted_cf / monthly_interest
            if icr < 1.0:
                breaches.append(BreachProjection(
                    month=month_num,
                    ratio="ICR",
                    projectedValue=round(icr, 4),
                ))

        # Quick Ratio — cash position shifts current assets each month
        if current_liabilities > 0:
            projected_current_assets = data.currentAssets + cumulative_cf
            quick_ratio = (projected_current_assets - inventory) / current_liabilities
            if quick_ratio < 0.8:
                breaches.append(BreachProjection(
                    month=month_num,
                    ratio="QuickRatio",
                    projectedValue=round(quick_ratio, 4),
                ))

    return breaches

# --- 5. Forecast endpoint ---

@app.post("/api/forecast", response_model=ForecastResponse)
def forecast_cash_flow(data: ForecastRequest):
    try:
        monthly_forecasts: list[MonthForecast] = []
        monthly_predictions: list[float] = []

        for month_ahead in range(1, 7):
            input_row = _build_input_row(data, month_ahead)
            predicted = float(model.predict(input_row)[0])
            monthly_predictions.append(predicted)

            uncertainty_pct = 0.08 + (month_ahead * 0.01)
            band = abs(predicted) * uncertainty_pct
            upper = predicted + band
            lower = max(0.0, predicted - band)
            confidence = round(max(0.70, 0.95 - (month_ahead * 0.03)), 2)

            monthly_forecasts.append(MonthForecast(
                month=MONTH_LABELS[month_ahead - 1],
                forecastedCashFlow=round(predicted, 2),
                upperBound=round(upper, 2),
                lowerBound=round(lower, 2),
                confidence=confidence,
            ))

        breaches = _project_breaches(data, monthly_predictions)

        return ForecastResponse(
            forecastedCashflow=monthly_forecasts,
            confidenceTier=data.confidenceTier,
            projectedBreaches=breaches,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 6. Model stats ---

@app.get("/api/model/stats")
def model_stats():
    if not _model_stats:
        raise HTTPException(status_code=404, detail="model_stats.json not found. Run train_model.py first.")
    return _model_stats

# --- 7. Health check ---

@app.get("/")
def health_check():
    return {"status": "FinSight API is running"}
