import pandas as pd
import pickle
import json
import os
from sklearn.metrics import mean_squared_error, r2_score

# Define paths
PROCESSED_PATH = os.path.join("data", "processed")
MODELS_PATH = os.path.join("models", "models")
DATA_OUT_PATH = os.path.join("models", "data")
METRICS_PATH = os.path.join("metrics")
os.makedirs(METRICS_PATH, exist_ok=True)
os.makedirs(DATA_OUT_PATH, exist_ok=True)

# Load test data
X_test = pd.read_csv(os.path.join(PROCESSED_PATH, "X_test_scaled.csv"))
y_test = pd.read_csv(os.path.join(PROCESSED_PATH, "y_test.csv")).squeeze()

# Load model
with open(os.path.join(MODELS_PATH, "model.pkl"), "rb") as f:
    model = pickle.load(f)

# Predict
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
scores = {"MSE": round(mse, 6), "R2": round(r2, 6)}
print(f"Finished evaluation")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# Save metrics
scores_path = os.path.join(METRICS_PATH, "scores.json")
with open(scores_path, "w") as f:
    json.dump(scores, f, indent=2)

# Save predictions
predictions_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
pred_path = os.path.join(DATA_OUT_PATH, "predictions.csv")
predictions_df.to_csv(pred_path, index=False)
