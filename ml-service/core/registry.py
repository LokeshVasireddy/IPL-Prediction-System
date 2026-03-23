from core.logger import setup_logger
import os
import shutil
import json
from datetime import datetime

logger = setup_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STAGING_DIR = os.path.join(BASE_DIR, "models", "staging")
PRODUCTION_DIR = os.path.join(BASE_DIR, "models", "production")
HISTORY_DIR = os.path.join(BASE_DIR, "models", "history")


def list_staging_models():
    logger.info("Listing staging models")

    if not os.path.exists(STAGING_DIR):
        raise Exception("Staging directory missing")

    return sorted(os.listdir(STAGING_DIR))


def get_latest_staging_model():

    models = list_staging_models()

    if not models:
        logger.error("No models in staging")
        raise Exception("No models in staging")

    latest = models[-1]

    logger.info(f"Latest staging model: {latest}")
    return latest


def promote_model(model_name, model_type, dataset_version, feature_version, run_id):

    logger.info("Starting model promotion")
    logger.info(f"Model: {model_name}")

    if not os.path.exists(STAGING_DIR):
        raise Exception("Staging directory does not exist")

    src = os.path.join(STAGING_DIR, model_name)

    if not os.path.exists(src):
        raise Exception("Model not found in staging")

    os.makedirs(PRODUCTION_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    current_model_file = os.path.join(PRODUCTION_DIR, "current_model.txt")
    metadata_file = os.path.join(PRODUCTION_DIR, "metadata.json")

    # ----------------------------
    # move old production to history
    # ----------------------------

    if os.path.exists(current_model_file):

        with open(current_model_file, "r") as f:
            old_model = f.read().strip()

        old_model_path = os.path.join(PRODUCTION_DIR, old_model)

        if os.path.exists(old_model_path):
            logger.info(f"Moving old model to history: {old_model}")

            shutil.move(old_model_path, os.path.join(HISTORY_DIR, old_model))

    # ----------------------------
    # move new model to production
    # ----------------------------

    dst = os.path.join(PRODUCTION_DIR, model_name)

    shutil.copy(src, dst)

    logger.info("Model copied to production")

    # ----------------------------
    # update current model
    # ----------------------------

    with open(current_model_file, "w") as f:
        f.write(model_name)

    metadata = {
        "model_name": model_name,
        "model_type": model_type,
        "run_id": run_id,
        "dataset_version": dataset_version,
        "feature_version": feature_version,
        "stage": "production",
        "source": "staging",
        "promoted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info("Metadata written")
    logger.info(f"Production now contains: {model_name}")
