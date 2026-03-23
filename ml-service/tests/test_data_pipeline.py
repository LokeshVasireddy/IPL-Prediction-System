import os
import pandas as pd

ML_SERVICE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(
    ML_SERVICE_DIR,
    "data",
    "processed",
    "v2_alpha",
    "dataset.parquet"
)

def test_dataset_loads():
    assert os.path.exists(DATA_PATH)
    df = pd.read_parquet(DATA_PATH)
    assert len(df) > 0

def test_target_columns_exist():
    df = pd.read_parquet(DATA_PATH)
    assert "y_runs" in df.columns
    assert "y_wickets" in df.columns

def test_no_nulls():
    df = pd.read_parquet(DATA_PATH)
    assert df.isnull().sum().sum() == 0