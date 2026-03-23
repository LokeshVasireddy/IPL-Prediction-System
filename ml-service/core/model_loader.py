from core.logger import setup_logger
import os
import pickle
import json

logger = setup_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTION_DIR = os.path.join(BASE_DIR, "models", "production")


def load_production_model():

    logger.info("Loading production model")

    current_model_file = os.path.join(PRODUCTION_DIR, "current_model.txt")
    metadata_file = os.path.join(PRODUCTION_DIR, "metadata.json")

    # check production folder
    if not os.path.exists(PRODUCTION_DIR):
        logger.error("Production directory missing")
        raise Exception("Production directory missing")

    # check current model
    if not os.path.exists(current_model_file):
        logger.error("No current_model.txt found")
        raise Exception("No production model found")

    with open(current_model_file, "r") as f:
        model_name = f.read().strip()

    if not model_name:
        logger.error("current_model.txt is empty")
        raise Exception("Production registry corrupted")

    model_path = os.path.join(PRODUCTION_DIR, model_name)

    if not os.path.exists(model_path):
        logger.error("Production model file missing")
        raise Exception("Production model file missing")

    # check metadata
    if not os.path.exists(metadata_file):
        logger.error("metadata.json missing")
        raise Exception("Production metadata missing")

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    logger.info(f"Model: {metadata.get('model_name')}")
    logger.info(f"Dataset: {metadata.get('dataset_version')}")
    logger.info(f"Run ID: {metadata.get('run_id')}")

    # load model
    try:
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
    except Exception as e:
        logger.error("Model loading failed")
        raise Exception("Production model corrupted") from e

    logger.info("Production model loaded successfully")

    return bundle, metadata