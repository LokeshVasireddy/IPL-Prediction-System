import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = BASE_DIR / "models" / "production"


def test_prediction_runs():

    model_files = list(PRODUCTION_DIR.glob("*.pkl"))
    assert len(model_files) > 0, "No model found"

    with open(model_files[0], "rb") as f:
        bundle = pickle.load(f)

    # create dummy feature matrix
    n_features = bundle.model.input_shape[2]

    sample = np.zeros((1, n_features))

    pred = bundle.predict(sample)

    assert pred is not None
    assert pred.shape[0] == 1