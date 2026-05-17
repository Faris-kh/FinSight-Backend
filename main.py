from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

# --- 1. Create the FastAPI app ---
app = FastAPI(
    title="FinSight Forecast API",
    description="Cash flow forecasting backend for FinSight SME funding assessment",
    version="1.0.0"
)

# --- 2. Allow frontend to talk to this backend (CORS) ---
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

# --- 3. Load the trained model once when the server starts ---
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        "model.pkl not found. Please run: python train_model.py"
    )

model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully.")

# --- 4. Define what the frontend must send (input shape) ---
class ForecastRequest(BaseModel):
    revenue: float
    expenses: float
    currentAssets: float
    currentLiabilities: float
    totalAssets: float
    totalDebt: float
    equity: float
    cashFlow: float

# --- 5. Define what the backend sends back (output shape) ---
class MonthForecast(BaseModel):
    month: str
    forecastedCashFlow: float
    upperBound: float
    lowerBound: float
    confidence: float

# --- 6. Month labels for the response ---
MONTH_LABELS = ["Month 1", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6"]

# --- 7. The forecast endpoint ---
@app.post("/api/forecast", response_model=list[MonthForecast])
def forecast_cash_flow(data: ForecastRequest):
    try:
        results = []

        for month_ahead in range(1, 7):
            input_row = pd.DataFrame([{
                "revenue": data.revenue,
                "expenses": data.expenses,
                "currentAssets": data.currentAssets,
                "currentLiabilities": data.currentLiabilities,
                "totalAssets": data.totalAssets,
                "totalDebt": data.totalDebt,
                "equity": data.equity,
                "cashFlow": data.cashFlow,
                "month_ahead": month_ahead
            }])

            predicted = float(model.predict(input_row)[0])

            uncertainty_pct = 0.08 + (month_ahead * 0.01)
            upper = predicted * (1 + uncertainty_pct)
            lower = predicted * (1 - uncertainty_pct)

            confidence = round(max(0.70, 0.95 - (month_ahead * 0.03)), 2)

            results.append(MonthForecast(
                month=MONTH_LABELS[month_ahead - 1],
                forecastedCashFlow=round(predicted, 2),
                upperBound=round(upper, 2),
                lowerBound=round(lower, 2),
                confidence=confidence
            ))

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 8. Health check ---
@app.get("/")
def health_check():
    return {"status": "FinSight API is running ✅"}