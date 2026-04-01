import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor

# Define paths 
PROCESSED_PATH = os.path.join("data", "processed")
MODELS_PATH = os.path.join("models", "models")
os.makedirs(MODELS_PATH, exist_ok=True)

# Load data
X_train = pd.read_csv(os.path.join(PROCESSED_PATH, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_PATH, "y_train.csv")).squeeze()

# Load best params
with open(os.path.join(MODELS_PATH, "best_params.pkl"), "rb") as f:
    best_params = pickle.load(f)

print(f"Using params: {best_params}")

# Train final model
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

# Save trained model
model_path = os.path.join(MODELS_PATH, "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Finished training model")
