import hashlib
import json
from datetime import datetime
from pathlib import Path

from core.config import METADATA_PATH


def generate_sample_hash(df, n=1000):
    sample = df.head(n).to_csv(index=False)
    return hashlib.md5(sample.encode()).hexdigest()


def save_metadata(
    dataset_name,
    dataset_path,
    raw_sources,
    preprocessing,
    df=None,  # ← NEW
):

    metadata = {
        "dataset_name": dataset_name,
        "created_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_path": str(dataset_path),
        "raw_sources": raw_sources,
        "preprocessing": preprocessing,
    }

    # ✅ Add data-level metadata ONLY if df is passed
    if df is not None:
        metadata.update(
            {
                "row_count": int(len(df)),
                "column_count": int(len(df.columns)),
                "columns": list(df.columns),
                "null_counts": df.isnull().sum().to_dict(),
                "sample_hash": generate_sample_hash(df),
            }
        )

    METADATA_PATH.mkdir(parents=True, exist_ok=True)

    metadata_file = METADATA_PATH / f"{dataset_name}.json"

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    print("Metadata saved:", metadata_file)
