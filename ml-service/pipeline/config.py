DATASET_VERSION = "v2_alpha"

RAW_PATH = "../ml-service/data/data1.csv"

PROCESSED_DIR = "../ml-service/data/processed/"
OUTPUT_PATH = f"{PROCESSED_DIR}{DATASET_VERSION}/dataset.parquet"

METADATA_PATH = f"../ml-service/data/metadata/{DATASET_VERSION}.json"

INPUT_FEATURES = ["batting_team", "bowling_team", "venue"]
X_FEATS = ["over", "ball"]
Y_FEATS = ["runs", "wickets"]
