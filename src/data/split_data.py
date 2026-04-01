import pandas as pd
from sklearn.model_selection import train_test_split
import os

RAW_PATH = os.path.join("data", "raw", "raw.csv")
PROCESSED_PATH = os.path.join("data", "processed")
os.makedirs(PROCESSED_PATH, exist_ok=True)

df = pd.read_csv(RAW_PATH)

# Drop the date column
df = df.drop(columns=["date"], errors="ignore")

# Split features and target
X = df.drop(columns=["silica_concentrate"])
y = df["silica_concentrate"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save data
X_train.to_csv(os.path.join(PROCESSED_PATH, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(PROCESSED_PATH, "X_test.csv"),  index=False)
y_train.to_csv(os.path.join(PROCESSED_PATH, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(PROCESSED_PATH, "y_test.csv"),  index=False)

print("Finished splitting data")
