import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = BASE_DIR / "models" / "production"


def test_model_bundle_loads():

    model_files = list(PRODUCTION_DIR.glob("*.pkl"))
    assert len(model_files) > 0, "No model bundle found"

    with open(model_files[0], "rb") as f:
        bundle = pickle.load(f)

    # check class structure
    assert hasattr(bundle, "model")
    assert hasattr(bundle, "dataset_version")
    assert hasattr(bundle, "feature_version")

    # check methods
    assert hasattr(bundle, "predict")
    assert hasattr(bundle, "preprocess")
    assert hasattr(bundle, "info")

    # check info output
    info = bundle.info()

    assert "dataset_version" in info
    assert "feature_version" in info
