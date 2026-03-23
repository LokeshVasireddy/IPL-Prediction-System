import math
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

print("Started training baselines...")

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("../data/data1.csv", index_col="Unnamed: 0")

input_features = ["batting_team", "bowling_team", "venue"]
xfeats = ["over", "ball"]
yfeats = ["runs", "wickets"]

# -----------------------------
# PREPROCESS (SELF-CONTAINED)
# -----------------------------
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
scalerx = StandardScaler()
scalery = StandardScaler()

encoded_categorical = encoder.fit_transform(data[input_features])
scaled_x = scalerx.fit_transform(data[xfeats])
scaled_y = scalery.fit_transform(data[yfeats])

X = np.hstack((encoded_categorical, scaled_x))
y = scaled_y

# -----------------------------
# SPLIT
# -----------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def evaluate_model(name, model):

    model.fit(X_train, y_train)

    # latency
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()

    total_latency_ms = (end_time - start_time) * 1000
    avg_latency = total_latency_ms / X_test.shape[0]

    # metrics (REAL SCALE)
    y_pred_real = scalery.inverse_transform(y_pred)
    y_test_real = scalery.inverse_transform(y_test)

    mse = mean_squared_error(y_test_real, y_pred_real)
    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test_real, y_pred_real)

    n = X_test.shape[0]
    p = X_test.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    return {
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Adj_R2": adjusted_r2,
        "Latency(ms)": total_latency_ms,
        "Per_Sample(ms)": avg_latency
    }

# -----------------------------
# MODELS
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=20, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    ),
}

# -----------------------------
# RUN
# -----------------------------
results = []

for name, model in models.items():
    results.append(evaluate_model(name, model))

results_df = pd.DataFrame(results).sort_values(by="RMSE")

print("\nMODEL LEADERBOARD:\n")
print(results_df.to_string(index=False))