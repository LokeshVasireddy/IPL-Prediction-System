import json
import os
from datetime import datetime

def save_metadata(df, config):
    metadata = {
        "dataset_version": config.DATASET_VERSION,
        "created_on": datetime.now().strftime("%Y-%m-%d"),
        "rows": len(df),
        "features": config.INPUT_FEATURES + config.X_FEATS,
        "target": config.Y_FEATS,
        "raw_source": config.RAW_PATH,
        "preprocessing": [
            "onehot_encoding",
            "standard_scaling"
        ]
    }

    os.makedirs(os.path.dirname(config.METADATA_PATH), exist_ok=True)

    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print("Metadata saved:", config.METADATA_PATH)