import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. Load the data ---
print("📂 Loading data...")
df = pd.read_csv("sme_data.csv")

# These are the inputs the model will receive
FEATURE_COLUMNS = [
    "revenue",
    "expenses",
    "currentAssets",
    "currentLiabilities",
    "totalAssets",
    "totalDebt",
    "equity",
    "cashFlow",
    "month_ahead"   # month 1, 2, 3, 4, 5, or 6
]

TARGET_COLUMN = "future_cashFlow"

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# --- 2. Split into training set (80%) and test set (20%) ---
# Training set = what the model learns from
# Test set = data the model has never seen, used to check accuracy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"🏋️  Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

# --- 3. Train the LightGBM model ---
# n_estimators = number of decision trees to build (more = better, but slower)
# learning_rate = how big each step of learning is
# num_leaves = complexity of each tree
model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    random_state=42,
    verbose=-1  # suppress training logs
)

print("🚀 Training model...")
model.fit(X_train, y_train)

# --- 4. Evaluate accuracy on the test set ---
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"   Mean Absolute Error : {mae:,.0f} SAR")
print(f"   R² Score            : {r2:.4f}  (1.0 = perfect, 0.0 = random)")

# --- 5. Save the trained model to disk ---
joblib.dump(model, "model.pkl")
print("\n✅ Model saved as model.pkl")