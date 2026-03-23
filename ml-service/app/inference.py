import pandas as pd
from core.model_loader import load_latest_model

model = load_latest_model(stage="production")

def run_prediction(data):
    df = pd.DataFrame([data])
    preds = model.predict(df)

    return {
        "predicted_runs": float(preds[0][0]),
        "predicted_wickets": float(preds[0][1])
    }