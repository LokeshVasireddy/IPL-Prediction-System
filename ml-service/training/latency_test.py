import time

import pandas as pd
from core.model_loader import load_latest_model

# -----------------------------
# Load model (PRODUCTION BUNDLE)
# -----------------------------
model = load_latest_model(stage="production")

# -----------------------------
# INPUT
# -----------------------------
input_data = {
    "batting_team": "Mumbai Indians",
    "bowling_team": "Chennai Super Kings",
    "venue": "Wankhede Stadium",
    "over": 10,
    "ball": 3,
}

df = pd.DataFrame([input_data])

# -----------------------------
# LATENCY MEASURE
# -----------------------------
start_time = time.time()

pred = model.predict(df)

end_time = time.time()

latency_ms = (end_time - start_time) * 1000

# -----------------------------
# OUTPUT
# -----------------------------
runs, wickets = pred[0]

print(f"Predicted Runs: {runs:.2f}")
print(f"Predicted Wickets: {wickets:.2f}")
print(f"Latency: {latency_ms:.3f} ms")
