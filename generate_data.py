import pandas as pd
import numpy as np

np.random.seed(42)

NUM_COMPANIES = 3000
MONTHS_AHEAD = 6

rows = []

for i in range(NUM_COMPANIES):

    # --- Assign company health profile ---
    # 60% healthy, 25% struggling, 15% distressed (like your test company)
    profile = np.random.choice(['healthy', 'struggling', 'distressed'],
                                p=[0.60, 0.25, 0.15])

    if profile == 'healthy':
        revenue         = np.random.uniform(500_000, 10_000_000)
        expense_ratio   = np.random.uniform(0.50, 0.80)
        equity_ratio    = np.random.uniform(0.40, 0.75)   # equity is positive
        cf_ratio        = np.random.uniform(0.08, 0.20)   # strong cash flow

    elif profile == 'struggling':
        revenue         = np.random.uniform(80_000, 2_000_000)
        expense_ratio   = np.random.uniform(0.85, 1.05)   # near break-even or slight loss
        equity_ratio    = np.random.uniform(0.10, 0.35)   # thin equity
        cf_ratio        = np.random.uniform(-0.05, 0.05)  # near-zero cash flow

    else:  # distressed — like Company X
        revenue         = np.random.uniform(50_000, 500_000)
        expense_ratio   = np.random.uniform(1.10, 1.60)   # spending more than earning
        equity_ratio    = np.random.uniform(-0.30, 0.05)  # negative or near-zero equity
        cf_ratio        = np.random.uniform(-0.25, -0.05) # negative cash flow

    expenses            = revenue * expense_ratio
    current_assets      = np.random.uniform(30_000, revenue * 0.8)
    current_liabilities = current_assets * np.random.uniform(0.5, 2.5)
    total_assets        = current_assets * np.random.uniform(1.2, 4.0)
    equity              = total_assets * equity_ratio
    total_debt          = total_assets - equity
    base_cash_flow      = revenue * cf_ratio

    profitability = (revenue - expenses) / revenue
    liquidity     = current_assets / max(current_liabilities, 1)
    leverage      = total_debt / max(abs(equity), 1) * (1 if equity > 0 else -1)

    for month in range(1, MONTHS_AHEAD + 1):

        if profile == 'healthy':
            growth    = 1 + (profitability * 0.02 * month)
            liq_boost = min(liquidity * 0.01, 0.05)
            lev_drag  = min(max(leverage, 0) * 0.015, 0.08)
            forecast  = base_cash_flow * growth * (1 + liq_boost - lev_drag)

        elif profile == 'struggling':
            # Slight decline each month
            decay    = 1 - (0.015 * month)
            forecast = base_cash_flow * decay

        else:  # distressed
            # Accelerating decline — gets worse each month
            decay    = 1 - (0.05 * month)
            forecast = base_cash_flow * decay

        noise    = np.random.uniform(-0.08, 0.08)
        forecast = forecast * (1 + noise)

        rows.append({
            "revenue":            revenue,
            "expenses":           expenses,
            "currentAssets":      current_assets,
            "currentLiabilities": current_liabilities,
            "totalAssets":        total_assets,
            "totalDebt":          total_debt,
            "equity":             equity,
            "cashFlow":           base_cash_flow * 12,
            "month_ahead":        month,
            "future_cashFlow":    forecast
        })

df = pd.DataFrame(rows)
df.to_csv("sme_data.csv", index=False)

print(f"✅ Generated {len(df)} rows across {NUM_COMPANIES} companies.")

# Show profile breakdown
profiles = ['healthy'] * int(NUM_COMPANIES * 0.60) + \
           ['struggling'] * int(NUM_COMPANIES * 0.25) + \
           ['distressed'] * int(NUM_COMPANIES * 0.15)
print(f"   Healthy:    {int(NUM_COMPANIES * 0.60)}")
print(f"   Struggling: {int(NUM_COMPANIES * 0.25)}")
print(f"   Distressed: {int(NUM_COMPANIES * 0.15)}")