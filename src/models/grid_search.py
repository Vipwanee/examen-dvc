import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Define paths
PROCESSED_PATH = os.path.join("data", "processed")
MODELS_PATH = os.path.join("models", "models")
os.makedirs(MODELS_PATH, exist_ok=True)

# Load training data
X_train = pd.read_csv(os.path.join(PROCESSED_PATH, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_PATH, "y_train.csv")).squeeze()

# Grid search
param_grid = {"n_estimators": [50, 100, 200],
              "max_depth":    [None, 5, 10],
              "min_samples_split": [2, 5],}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           cv=3,
                           scoring="r2",
                           n_jobs=-1,
                           verbose=1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score  = grid_search.best_score_

print(f"Finished Grid Search")
print(f"Best CV R2: {best_score:.4f}")
print(f"Best params: {best_params}")

# Save best params
with open(os.path.join(MODELS_PATH, "best_params.pkl"), "wb") as f:
    pickle.dump(best_params, f)
