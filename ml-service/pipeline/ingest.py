import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="Unnamed: 0")
    return df
