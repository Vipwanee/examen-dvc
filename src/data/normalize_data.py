import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler

# Define paths
PROCESSED_PATH = os.path.join("data", "processed")
MODELS_PATH = os.path.join("models", "data")
os.makedirs(MODELS_PATH, exist_ok=True)

# Load data
X_train = pd.read_csv(os.path.join(PROCESSED_PATH, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(PROCESSED_PATH, "X_test.csv"))

# Fit scaler on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Convert to dataframe
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X_test.columns)

# Save scaled datasets to CSV
X_train_scaled.to_csv(os.path.join(PROCESSED_PATH, "X_train_scaled.csv"), index=False)
X_test_scaled.to_csv(os.path.join(PROCESSED_PATH,  "X_test_scaled.csv"),  index=False)

# Save scaler pickle file
with open(os.path.join(MODELS_PATH, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print("Finished data normalization")
