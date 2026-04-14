import json
from datetime import datetime
from pathlib import Path

from core.config import METADATA_PATH


def save_metadata(dataset_name, dataset_path, raw_sources, preprocessing):

    metadata = {
        "dataset_name": dataset_name,
        "created_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_path": str(dataset_path),
        "raw_sources": raw_sources,
        "preprocessing": preprocessing,
    }

    METADATA_PATH.mkdir(parents=True, exist_ok=True)

    metadata_file = METADATA_PATH / f"{dataset_name}.json"

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    print("Metadata saved:", metadata_file)
