# MLOps Foundations — IPL Prediction System

A comprehensive record of all MLOps practices implemented across the IPL Prediction ML service.

## Project Structure

```
IPL-Prediction-System/
├── analytics-service/
├── api-gateway/
├── data-service/
├── docs/
├── frontend/
├── ml-service/                        ← Primary focus of these notes
│   ├── app/
│   ├── configs/
│   │   └── lstm_v1.yaml               ← YAML config (single source of truth)
│   ├── core/
│   │   ├── config_loader.py
│   │   ├── logger.py
│   │   ├── model_bundle.py
│   │   ├── model_loader.py
│   │   └── registry.py
│   ├── data/
│   │   ├── processed/                 ← Versioned parquet datasets
│   │   └── metadata/                  ← Per-version metadata JSON files
│   ├── experiments/
│   │   └── mlruns/                    ← MLflow tracking DB and artifacts
│   ├── logs/                          ← Rotated log files (git-ignored)
│   ├── models/
│   │   ├── staging/
│   │   ├── production/
│   │   └── history/
│   ├── tests/
│   │   ├── test_data_pipeline.py
│   │   ├── test_model_bundle.py
│   │   ├── test_inference.py
│   │   └── test_registry.py
│   ├── training/
│   │   ├── simple_train_test.py       ← Main training entrypoint
│   │   ├── model.py
│   │   ├── baselines.py
│   │   └── latency_test.py
│   ├── Dockerfile
│   ├── pytest.ini
│   └── requirements.txt
├── docker-compose.yml
└── README.md
```

**Run training:**

```bash
python -m training.simple_train_test --config configs/lstm_v1.yaml
```

## Table of Contents

1. [Experiment Tracking](#1-experiment-tracking)
2. [Model Versioning](#2-model-versioning)
3. [Reproducible Training Pipeline](#3-reproducible-training-pipeline)
4. [Data Version Awareness](#4-data-version-awareness)
5. [Minimal Testing](#5-minimal-testing)
6. [Logging & Log Rotation](#6-logging--log-rotation)
7. [Model Registry](#7-model-registry)

## 1. Experiment Tracking

### Overview

The goal was to move from ad-hoc model training to a **reproducible, trackable, and comparable experimentation system**.

### Objectives

- Track all ML experiments systematically
- Ensure reproducibility of training runs
- Enable comparison between different models and configurations
- Store models and preprocessing pipelines as artifacts
- Build a foundation for production deployment

### Tools Used

- MLflow (Experiment Tracking)
- TensorFlow / Keras (Modeling)
- Scikit-learn (Preprocessing, metrics)

### Implementation Details

#### MLflow Setup

MLflow is configured with a SQLite backend and the experiment is set from YAML config:

```python
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment(EXPERIMENT_NAME)
```

MLflow run artifacts and experiment data are stored under `experiments/mlruns/`. A cleanup block at the end of training automatically relocates any stray `mlruns/` folder that TensorFlow or MLflow creates at the service root into the correct `experiments/` directory.

#### Training Pipeline Integration

The entire pipeline runs inside a single MLflow run with full exception handling:

```python
with mlflow.start_run():
    try:
        ...
    except Exception as e:
        mlflow.set_tag("status", "failed")
        logger.exception("Training failed")
        raise
```

#### Reproducibility

Seeds are set from config for Python `random`, NumPy, and TensorFlow:

```python
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

#### Feature Pipeline

The pipeline now loads a **pre-processed parquet file** (produced by the data-service) instead of raw CSVs. Features are already encoded and scaled — the training script only reshapes for LSTM:

```python
X = df.drop(columns=["y_runs", "y_wickets"]).values
y = df[["y_runs", "y_wickets"]].values
X = X.reshape(X.shape[0], 1, X.shape[1])
```

#### Parameter Logging

All parameters are logged from YAML config — nothing is hardcoded:

```python
mlflow.log_param("model_type", MODEL_TYPE)
mlflow.log_param("dataset_version", DATASET_VERSION)
mlflow.log_param("feature_version", FEATURE_VERSION)
mlflow.log_param("dataset_path", DATA_PATH)
mlflow.log_param("epochs", EPOCHS)
mlflow.log_param("batch_size", BATCH_SIZE)
mlflow.log_param("lstm1_units", LSTM1_UNITS)
mlflow.log_param("lstm2_units", LSTM2_UNITS)
mlflow.log_param("learning_rate", LEARNING_RATE)
mlflow.log_param("config_file", args.config)
mlflow.log_artifact(metadata_path)   # dataset metadata JSON
```

#### Model Architecture

Built dynamically from config values:

```python
model = keras.Sequential([
    keras.layers.Input(shape=(1, X.shape[2])),
    keras.layers.LSTM(LSTM1_UNITS, return_sequences=True),
    keras.layers.LSTM(LSTM2_UNITS),
    keras.layers.Dense(DENSE1, activation='relu'),
    keras.layers.Dense(DENSE2, activation='relu'),
    keras.layers.Dense(2)
])

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
```

#### Metrics Tracking

| Split      | Metrics                              |
|------------|--------------------------------------|
| Training   | Loss, MAE                            |
| Validation | Val Loss, Val MAE                    |
| Test       | Loss, MAE, MSE, RMSE, R²             |
| Custom     | Real-scale R², Adjusted R²           |

#### Artifact Logging

```python
mlflow.tensorflow.log_model(model, name="lstm_model")
mlflow.log_artifact(save_path)    # model bundle .pkl
mlflow.log_artifact(args.config)  # YAML config
```

Preprocessing artifacts (`encoder.pkl`, `scalerx.pkl`, `scalery.pkl`) are not logged here — they do not exist in ml-service. The data-service owns all preprocessing and delivers a ready-to-train parquet file.

#### Experiment Variations

Conducted multiple runs with variations in hyperparameters, model configurations, and training behavior. All variations are managed through different YAML config files, enabling comparison without touching the training code.

### Production Training Script

`ml-service/training/simple_train_test.py` — run via:

```bash
python -m training.simple_train_test --config configs/lstm_v1.yaml
```

```python
from core.model_bundle import IPLModelBundle
from core.registry import promote_model
from core.config_loader import load_config
from core.logger import setup_logger

import argparse
import pickle
import os
import shutil
import random

import numpy as np
import pandas as pd
import mlflow
import mlflow.tensorflow

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

# -----------------------------
# CLI
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to config file")
args = parser.parse_args()
config = load_config(args.config)

logger = setup_logger(__name__)
logger.info("Training script started")
logger.info(f"Using config: {args.config}")

# -----------------------------
# PATHS
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_SERVICE_DIR = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(ML_SERVICE_DIR, config["data"]["path"])
STAGING_DIR = os.path.join(ML_SERVICE_DIR, config["paths"]["staging_dir"])
PRODUCTION_DIR = os.path.join(ML_SERVICE_DIR, config["paths"]["production_dir"])
MLFLOW_DB = os.path.join(ML_SERVICE_DIR, config["paths"]["mlflow_db"])
HISTORY_DIR = os.path.join(ML_SERVICE_DIR, "models", "history")

os.makedirs(STAGING_DIR, exist_ok=True)
os.makedirs(PRODUCTION_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# -----------------------------
# CONFIG VALUES
# -----------------------------
EXPERIMENT_NAME   = config["experiment"]["name"]
DATASET_VERSION   = config["data"]["dataset_version"]
FEATURE_VERSION   = config["data"]["feature_version"]
TEST_SIZE         = config["data"]["test_size"]
VAL_SIZE          = config["data"]["val_size"]
SEED              = config["data"]["random_seed"]
EPOCHS            = config["model"]["epochs"]
BATCH_SIZE        = config["model"]["batch_size"]
GATE_MAE          = config["validation_gate"]["test_mae"]
GATE_R2           = config["validation_gate"]["test_r2"]
LSTM1_UNITS       = config["model"]["lstm1_units"]
LSTM2_UNITS       = config["model"]["lstm2_units"]
DENSE1            = config["model"]["dense1"]
DENSE2            = config["model"]["dense2"]
LEARNING_RATE     = config["model"]["learning_rate"]
MODEL_TYPE        = config["model"]["type"]

# -----------------------------
# REPRODUCIBILITY
# -----------------------------
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# MLFLOW
# -----------------------------
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment(EXPERIMENT_NAME)

# -----------------------------
# LOAD DATA
# -----------------------------
logger.info("Loading dataset")
logger.info(f"Dataset path: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    logger.error(f"Dataset not found: {DATA_PATH}")
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

metadata_path = os.path.join(ML_SERVICE_DIR, "data", "metadata", f"{DATASET_VERSION}.json")

if not os.path.exists(metadata_path):
    logger.error(f"Metadata missing for dataset {DATASET_VERSION}")
    raise FileNotFoundError(f"Metadata not found for dataset {DATASET_VERSION}")

if DATA_PATH.endswith(".parquet"):
    df = pd.read_parquet(DATA_PATH)
else:
    raise ValueError("Unsupported data format")

# -----------------------------
# FEATURES
# -----------------------------
X = df.drop(columns=["y_runs", "y_wickets"]).values
y = df[["y_runs", "y_wickets"]].values
X = X.reshape(X.shape[0], 1, X.shape[1])

# -----------------------------
# TRAINING
# -----------------------------
with mlflow.start_run():
    try:
        logger.info("MLflow run started")

        mlflow.log_param("model_type", MODEL_TYPE)
        mlflow.log_param("dataset_version", DATASET_VERSION)
        mlflow.log_param("feature_version", FEATURE_VERSION)
        mlflow.log_param("dataset_path", DATA_PATH)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("val_size", VAL_SIZE)
        mlflow.log_param("seed", SEED)
        mlflow.log_param("lstm1_units", LSTM1_UNITS)
        mlflow.log_param("lstm2_units", LSTM2_UNITS)
        mlflow.log_param("dense1", DENSE1)
        mlflow.log_param("dense2", DENSE2)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("config_file", args.config)
        mlflow.log_artifact(metadata_path)

        # SPLIT
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=VAL_SIZE, random_state=SEED)

        logger.info(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

        # MODEL
        model = keras.Sequential([
            keras.layers.Input(shape=(1, X.shape[2])),
            keras.layers.LSTM(LSTM1_UNITS, return_sequences=True),
            keras.layers.LSTM(LSTM2_UNITS),
            keras.layers.Dense(DENSE1, activation='relu'),
            keras.layers.Dense(DENSE2, activation='relu'),
            keras.layers.Dense(2)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='mean_squared_error',
            metrics=['mae', 'mse',
                     tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                     tf.keras.metrics.R2Score(name='r2')]
        )

        # TRAIN
        logger.info("Training started")
        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                            validation_data=(X_val, y_val), verbose=1)
        logger.info("Training completed")

        # SAVE BUNDLE
        bundle = IPLModelBundle(model=model, dataset_version=DATASET_VERSION,
                                feature_version=FEATURE_VERSION)
        run_id = mlflow.active_run().info.run_id[:6]
        model_name = (f"{EXPERIMENT_NAME}_dataset_{DATASET_VERSION}_"
                      f"features_{FEATURE_VERSION}_run_{run_id}.pkl")
        save_path = os.path.join(STAGING_DIR, model_name)

        with open(save_path, "wb") as f:
            pickle.dump(bundle, f)

        logger.info(f"Model saved to staging: {save_path}")
        mlflow.set_tag("model_file", model_name)
        mlflow.set_tag("model_stage", "staging")

        # METRICS
        mlflow.log_metric("train_loss", history.history['loss'][-1])
        mlflow.log_metric("val_loss",   history.history['val_loss'][-1])
        mlflow.log_metric("train_mae",  history.history['mae'][-1])
        mlflow.log_metric("val_mae",    history.history['val_mae'][-1])

        test_results = model.evaluate(X_test, y_test, verbose=0)
        mlflow.log_metric("test_loss", test_results[0])
        mlflow.log_metric("test_mae",  test_results[1])
        mlflow.log_metric("test_mse",  test_results[2])
        mlflow.log_metric("test_rmse", test_results[3])
        mlflow.log_metric("test_r2",   test_results[4])

        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        n, p = X_val.shape[0], X_val.shape[2]
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        mlflow.log_metric("r2_real",     r2)
        mlflow.log_metric("adjusted_r2", adjusted_r2)

        mlflow.tensorflow.log_model(model, name="lstm_model")
        mlflow.log_artifact(save_path)
        mlflow.log_artifact(args.config)

        logger.info("Training pipeline completed")

        # FIX MLRUNS LOCATION
        stray_mlruns  = os.path.join(ML_SERVICE_DIR, "mlruns")
        target_mlruns = os.path.join(ML_SERVICE_DIR, "experiments", "mlruns")

        if os.path.exists(stray_mlruns):
            if not os.path.exists(target_mlruns):
                shutil.move(stray_mlruns, target_mlruns)
            else:
                for item in os.listdir(stray_mlruns):
                    src = os.path.join(stray_mlruns, item)
                    dst = os.path.join(target_mlruns, item)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                shutil.rmtree(stray_mlruns)

        # VALIDATION GATE
        test_mae = test_results[1]
        test_r2  = test_results[4]

        logger.info(f"Validation gate — MAE: {test_mae:.4f} (threshold: {GATE_MAE}) | R2: {test_r2:.4f} (threshold: {GATE_R2})")

        if test_mae < GATE_MAE and test_r2 > GATE_R2:
            logger.info("Model passed validation gate — promoting to production")
            mlflow.set_tag("model_stage", "production")
            mlflow.set_tag("registry_status", "production")
            mlflow.set_tag("production_model_file", model_name)
            mlflow.set_tag("promoted_run_id", run_id)
            mlflow.log_param("promoted_model", model_name)
            mlflow.log_param("production_dataset", DATASET_VERSION)
            mlflow.log_param("production_feature_version", FEATURE_VERSION)

            promote_model(
                model_name=model_name,
                model_type=MODEL_TYPE,
                dataset_version=DATASET_VERSION,
                feature_version=FEATURE_VERSION,
                run_id=run_id
            )
            logger.info("Model promotion completed")
        else:
            logger.warning("Model did not pass validation gate — remains in staging")

    except Exception as e:
        mlflow.set_tag("status", "failed")
        logger.exception("Training failed")
        raise
```

### Key Learnings

- **Experiment tracking is not just logging** — multiple runs with variation are required for tracking to be meaningful.
- **Reproducibility is critical** — without dataset + feature version tracking, results are unreliable.
- **Model alone is not enough** — the preprocessing pipeline must also be saved for production inference.
- **Metrics must be structured** — consistent naming with Train/Validation/Test separation enables proper comparison.
- **MLflow UI is essential** — helps identify the best model, enables debugging and analysis.
- **Custom metrics matter** — real-world evaluation via inverse scaling gives better insights.

## 2. Model Versioning

### Overview

This phase transformed a basic training script into a structured and production-aware **model versioning and lifecycle system**.

Initially, the project only trained a TensorFlow LSTM model and saved it as a file with no control over model versions, preprocessing consistency, deployment safety, traceability, or production stability.

### Problems in the Old System

| Problem | Impact |
|---------|--------|
| Model and preprocessing were separated | Encoder/scaler mismatch, unreliable inference |
| No model registry | Overwriting risk, no lifecycle control |
| No version naming | Difficult tracing, weak MLflow linkage |
| Backend could load any model | Staging model could accidentally serve production |
| MLflow and model files not linked | Broken traceability |
| No validation gate | Bad models could reach production |
| Raw encoder files saved separately in ml-service | Coupled preprocessing to training, fragile and unportable |

### Solution — 7 Key Changes

#### 1. Bundle Model + Version Metadata

Created a model bundle:

```
IPLModelBundle
    model
    dataset_version
    feature_version
```

Saved as a single `.pkl` file. Encoders and scalers are **not** part of the bundle — preprocessing is handled entirely by the data-service, which outputs a fully processed `dataset.parquet` ready for training. The bundle only needs to carry the model and the version metadata that links it back to the dataset and feature pipeline it was trained on.

#### 2. Model Registry Structure

```
models/
    staging/
    production/
```

Promotion system (`promote_model()`) separates experimental models from the production model and enforces lifecycle.

#### 3. Version Naming

Enforced naming convention:

```
ipl_lstm_v2_dataset_v1_features_basic_v1_run_4f2a9b.pkl
```

Using the MLflow run ID guarantees uniqueness and traceability.

#### 4. Production Loader

Created `core/model_loader.py` which loads only from:

```
models/production/current_model.txt
```

FastAPI loads the model once at startup, ensuring stable and fast inference.

#### 5. MLflow Run → Model File Link

Linked MLflow with the model file via `run_id`, `model_version`, `model_file`, `model_stage`, and `metadata.json`.

#### 6. Validation Gate

Added performance thresholds — promotion only happens if:

```
MAE < 0.35
R2 > 0.70
```

#### 7. Preprocessing Moved to Data-Service

Encoding, scaling, and feature engineering are the data-service's responsibility. It outputs a fully processed `data/processed/v2_alpha/dataset.parquet` that ml-service consumes directly. This means the ml-service has no encoder or scaler files at all — the split is clean and each service owns its domain.

### Final Architecture

```
Training → MLflow Run → Bundle Creation → Staging Model
    → Validation Gate → Promotion → Production Model
    → FastAPI Loader → Prediction
```

### Key Learnings

- **Model versioning is not file naming** — it is lifecycle control.
- **MLflow alone is not MLOps** — registry and production logic are required.
- **The data-service owns preprocessing, the ml-service owns the model** — clean separation means no encoders or scalers live in ml-service at all.
- **Simplicity is powerful** — a lightweight registry can still be industry-grade.
- **Deployment should be performance-driven** — never automatic.

## 3. Reproducible Training Pipeline

### Objective

Convert the IPL training script into a **config-driven and reproducible training pipeline** with MLflow tracking and deterministic behavior.

The goal: any experiment can be reproduced later with the same configuration, dataset, and model parameters without modifying training code.

### What Was Implemented

#### Config-Driven Training

Introduced YAML-based configuration covering:

- Experiment name
- Dataset path and version
- Feature version
- Model hyperparameters
- Validation thresholds
- MLflow DB path
- Staging and production paths

Training now runs via:

```bash
python -m training.simple_train_test --config configs/lstm_v1.yaml
```

This eliminates all hardcoded parameters from the script.

#### Environment Reproducibility

Dependencies are pinned in `requirements.txt` with exact versions, serving as both the dependency list and the environment lockfile:

```
# requirements.txt acts as the pinned lockfile
# All versions are fixed — no floating ranges
```

This ensures the same environment can be reconstructed exactly 6 months later. The `Dockerfile` installs from this file directly, making training reproducible across local machines, Docker containers, and CI environments.

There is also a `requirements-prod.txt` for the production API image — a leaner subset containing only inference-time dependencies, not training ones.

#### Centralized Configuration Loader

Created `core/config_loader.py` to ensure:

- Correct path resolution
- Consistent config loading
- Portability across environments
- Compatibility with Docker and CI

#### Fully Reproducible Training

Added deterministic seeds for Python `random`, NumPy, and TensorFlow — ensuring consistent model behavior and repeatable experiments.

#### Model Architecture from Config

Moved all model parameters to YAML (LSTM units, Dense layers, learning rate, batch size, epochs). The training script now builds the model dynamically from config, allowing multiple experiments without touching code.

#### MLflow Integration Improvements

Now logs: model type, dataset version, feature version, hyperparameters, learning rate, config file, metrics, TensorFlow model, and pickle bundle.

#### Validation Gate & Automated Promotion

Model promotion is controlled by config thresholds and is **fully automated inside the training script** — no manual step is required:

```python
# From simple_train_test.py
test_mae = test_results[1]
test_r2  = test_results[4]

if test_mae < GATE_MAE and test_r2 > GATE_R2:
    # Auto-promote to production
    promote_model(
        model_name=model_name,
        model_type=MODEL_TYPE,
        dataset_version=DATASET_VERSION,
        feature_version=FEATURE_VERSION,
        run_id=run_id
    )
    logger.info("Model promotion completed")
else:
    logger.warning("Model did not pass validation gate — remains in staging")
```

Thresholds come from YAML config:

```yaml
validation_gate:
  test_mae: 0.35
  test_r2: 0.70
```

If the model passes → automatically promoted to production via `registry.py`. If it fails → stays in staging, logged as a warning. This makes the system a **controlled release pipeline**, not a manual one.

### Key Learnings

- **Config-driven systems are essential** — hardcoded scripts are not scalable; YAML separates code and config, enabling automation and reproducibility.
- **Reproducibility requires more than seeds** — dataset version, hyperparameters, model architecture, config file, and MLflow tracking all must be in sync.
- **MLflow becomes more useful with structured configs** — without configs, MLflow logs lose context.
- **Validation gate enables safe model promotion** — training and deployment should never be automatic.

## 4. Data Version Awareness

### Objective

Ensure every trained model can be traced back to the exact dataset and feature pipeline used during training, without adding heavy tools like DVC or Airflow.

### What Was Implemented

#### Versioned Dataset Usage

The training pipeline loads processed datasets, not raw CSV files.

**Flow:**

```
data-service → processed dataset → ml-service → training
```

Dataset path controlled through YAML config:

```yaml
data:
  path: data/processed/v2_alpha/dataset.parquet
  dataset_version: v2_alpha
  feature_version: lstm_features_v1
```

#### Dataset Metadata Validation

Before training begins, the system checks:

- Dataset exists
- Metadata file exists (`data/metadata/{dataset_version}.json`)
- Dataset version is valid

```python
if not os.path.exists(metadata_path):
    raise FileNotFoundError(...)
```

This prevents training on untracked datasets.

The actual metadata file (`data/metadata/v2_alpha.json`) produced by the data-service:

```json
{
    "dataset_version": "v2_alpha",
    "created_on": "2026-03-22",
    "rows": 260430,
    "features": [
        "batting_team",
        "bowling_team",
        "venue",
        "over",
        "ball"
    ],
    "target": [
        "runs",
        "wickets"
    ],
    "raw_source": "../ml-service/data/data1.csv",
    "preprocessing": [
        "onehot_encoding",
        "standard_scaling"
    ]
}
```

This metadata documents the full lineage: what raw source was used, what preprocessing was applied, what features and targets were produced, how many rows, and when it was created. The ml-service validates this file exists before any training run begins.

#### Dataset Traceability in MLflow

Each MLflow run logs the metadata file as an artifact alongside the version params:

```python
mlflow.log_param("dataset_version", DATASET_VERSION)
mlflow.log_param("feature_version", FEATURE_VERSION)
mlflow.log_param("dataset_path", DATA_PATH)
mlflow.log_artifact(metadata_path)
```

#### Model–Dataset Link

Each saved model bundle includes dataset and feature version, creating a strong link:

```
model → dataset → features → experiment
```

### Key Learnings

- **Raw data should never be used directly in training.**
- **Dataset versioning is critical for reproducibility.**
- **Metadata validation prevents silent errors** — training cannot start without a valid, tracked dataset.
- **MLflow should track dataset and feature versions**, not just model parameters.
- **The data-service owns full lineage** — preprocessing steps, feature list, row count, and source are all recorded at dataset creation time, not at training time.

## 5. Minimal Testing

### Goal

Set up a lightweight but meaningful testing structure to ensure:

- Dataset integrity
- Model bundle validity
- Inference pipeline works
- Project is CI-ready
- Reproducibility across environments

### Tools Used

- pytest

### Project Structure

```
ml-service/
├── core/
├── data/
├── models/
├── tests/
│     ├── test_data_pipeline.py
│     ├── test_model_bundle.py
│     ├── test_inference.py
│     └── test_registry.py
└── pytest.ini
```

### Tests Implemented

#### 1. Data Pipeline Test (`test_data_pipeline.py`)

Checks: dataset exists, parquet loads, target column exists, no null values.

*Purpose:* Ensure training data is valid and reproducible.

#### 2. Model Bundle Test (`test_model_bundle.py`)

Checks: model bundle exists, bundle loads correctly, required attributes present, metadata accessible.

*Purpose:* Ensure production model artifact is valid and structured.

#### 3. Inference Test (`test_inference.py`)

Checks: bundle loads, dummy input runs, prediction works.

*Purpose:* Ensure model can run inference end-to-end.

#### 4. Registry Test (`test_registry.py`)

Checks: registry structure is valid, production directory exists, `current_model.txt` and `metadata.json` are present and well-formed.

*Purpose:* Ensure the model registry is intact and the production model is correctly registered.

### Configuration

**`pytest.ini`:**

```ini
[pytest]
testpaths = tests
pythonpath = .
```

`pythonpath = .` adds the project root to the Python path directly — no `conftest.py` needed. This ensures `core.*` modules are importable when pickle loads the model bundle, and tests run consistently from any environment.

### Key Engineering Learnings

- **Tests must match architecture** — model bundle is a class, so tests should use `bundle.predict()` and `bundle.info()`, not dictionary access.
- **`pythonpath = .` in `pytest.ini` is sufficient** — no `conftest.py` is needed for path setup; pytest handles it natively from pytest 7+.
- **Avoid loading wrong files** — use `glob("*.pkl")` instead of `os.listdir()` to prevent accidentally loading `.gitignore` or other non-model files.
- **Minimal tests are enough** — only test pipeline integrity, artifact validity, and inference execution. No need to test model accuracy or training logic.

### Final Result

```
pytest
collected 6 items

tests/test_data_pipeline.py  ...   [ 50%]
tests/test_inference.py      .     [ 66%]
tests/test_model_bundle.py   .     [ 83%]
tests/test_registry.py       .     [100%]

6 passed in 4.43s
```

Testing is stable, reproducible, CI-ready, and MLOps compliant.

## 6. Logging & Log Rotation

### Overview

The goal was to introduce **production-ready logging** to the ML training pipeline and model management system.

**Before:** `print()` statements — no structured output, no traceability, no production visibility, no debugging support.

**After:** Structured logs with centralized logger, file + console logging, environment-based output, error tracking, and model lifecycle traceability.

### Central Logger

Created `core/logger.py` with a `setup_logger(name)` function.

**Usage:**

```python
from core.logger import setup_logger
logger = setup_logger(__name__)
```

**Log format:**

```
timestamp | level | module | message
# Example:
2026-03-22 16:10:02 | INFO | training.simple_train_test | Training started
```

### Environment-Based Logging

| Mode | Console Output | File Output |
|------|---------------|-------------|
| `ENV=dev` | INFO, WARNING, ERROR | All levels |
| `ENV=prod` | WARNING, ERROR only | INFO, WARNING, ERROR |

This prevents console clutter in production.

### Log Rotation

Implemented using `TimedRotatingFileHandler`:

- Rotates daily at midnight: `ml_service.log` → `ml_service.log.2026-03-22.log`
- Retention policy: 7 days (`LOG_RETENTION_DAYS = 7`)
- On logger startup: `cleanup_old_logs()` removes logs older than 7 days

### Files Updated

**`simple_train_test.py`** — Added logging at training start, dataset load, data split, model build, training completion, test results, validation gate, promotion, and exception handling. All `print()` replaced with `logger.info()`, `logger.warning()`, `logger.error()`, `logger.exception()`.

**`model_loader.py`** — Added logging for loading production model, missing model/file, and successful load.

**`registry.py`** — Added logging for staging model listing, model promotion, directory checks, production cleanup, metadata creation, and promotion success.

### Environment Variable Setup

| Platform | Command |
|----------|---------|
| Linux/Mac | `ENV=dev python -m training.simple_train_test` |
| Windows PowerShell | `$env:ENV="dev"; python -m training.simple_train_test` |
| Windows CMD | `set ENV=dev && python -m training.simple_train_test` |

### Notes on External Logs

- **TensorFlow logs** (oneDNN, CPU instructions, GPU warnings) come from the TensorFlow backend and are independent of our logger. Production systems typically silence them.
- **MLflow warning** ("TensorFlow model saved without signature") indicates inference schema is missing — to be addressed in the model serving stage.

### Key Learnings

- **Logging is mandatory in ML systems** — structured and traceable logs are required, not optional.
- **Dev and prod logging must differ** — more logs in development, less console noise in production.
- **Log rotation prevents system failure** — without it, logs grow forever, disk fills, and systems crash.
- **Timed rotation is better than size-only rotation** — daily logs provide clean, predictable history.
- **Central logger improves architecture** — single logger, shared configuration, consistent format is industry practice.

### Git Ignore

```
logs/
*.log
```

Logs are runtime artifacts and should never be committed to Git.

## 7. Model Registry

### Overview

A lightweight model registry manages the lifecycle of trained IPL prediction models, ensuring only validated models reach production while maintaining traceability and rollback capability.

This registry connects: **Training → Staging → Validation Gate → Production → History**

### Folder Structure

```
models/
├── staging/
├── production/
│     ├── current_model.txt
│     ├── metadata.json
│     └── model.pkl
└── history/
```

| Folder | Purpose |
|--------|---------|
| `staging/` | Stores newly trained models before validation |
| `production/` | Contains the active model used by the API |
| `history/` | Stores previous production models for rollback |

### Promotion Flow

```
1. Training
   Model trained via simple_train_test.py → saved to models/staging/
   MLflow logs: dataset version, feature version, parameters, metrics, artifact

2. Validation Gate
   test_mae < gate_mae AND test_r2 > gate_r2
   → Passed: promoted to production
   → Failed: remains in staging

3. Promotion to Production (registry.py)
   a. Move old production model → history
   b. Copy new staging model → production
   c. Update current_model.txt
   d. Update metadata.json
```

### Metadata Tracking

Two metadata files exist in the system:

**Dataset metadata** (`data/metadata/v2_alpha.json`) — owned by data-service, describes the dataset (see Section 4).

**Production model metadata** (`models/production/metadata.json`) — owned by ml-service registry, describes the deployed model:

```json
{
    "model_name": "ipl_lstm_v2_dataset_v2_alpha_features_lstm_features_v1_run_27bb30.pkl",
    "model_type": "LSTM",
    "run_id": "27bb30",
    "dataset_version": "v2_alpha",
    "feature_version": "lstm_features_v1",
    "stage": "production",
    "source": "staging",
    "promoted_at": "2026-03-22 18:19:44"
}
```

This links the production model back to its MLflow run, dataset version, and feature version — enabling full traceability from deployment back to the training experiment.

### Registry Design — Custom vs MLflow Registry

This system uses a **custom folder-based registry** (`staging/production/history/`) rather than the MLflow Model Registry (`mlflow.register_model()`). This is a deliberate Week 1 choice — the custom registry is simpler, has no external dependencies, and is sufficient for a single-service setup. MLflow Model Registry integration is a planned improvement for later stages when multi-service deployment and API-driven model management become relevant.

### Production Model Loader

`model_loader.py` loads the active model using `current_model.txt` and `metadata.json`.

Validates:
- Production directory exists
- Model file exists
- Metadata exists
- Model is loadable

Returns: `model_bundle` + `metadata`

### MLflow Integration

MLflow tracks model stage, dataset version, feature version, run ID, and promoted model — linking experiments to production deployment.

### Final Flow

```
Training
   ↓
Staging
   ↓
Validation Gate
   ↓
Production
   ↓
History (rollback available)
   ↓
FastAPI loads production model
```

## Summary — MLOps Foundation Progress

| # | Component | Status |
|---|-----------|--------|
| 1 | Experiment Tracking (MLflow) | ✅ Complete |
| 2 | Model Versioning | ✅ Complete |
| 3 | Reproducible Training Pipeline | ✅ Complete |
| 4 | Data Version Awareness | ✅ Complete |
| 5 | Minimal Testing (pytest) | ✅ Complete |
| 6 | Logging & Log Rotation | ✅ Complete |
| 7 | Model Registry (custom, lightweight) | ✅ Complete |
| — | Inference contract (input/output schema) | 🔜 Planned — Week 2/3 |
| — | Inference logging (prediction/latency) | 🔜 Planned — Week 2/3 |
| — | MLflow Model Registry (`register_model`) | 🔜 Optional — later stage |

The IPL Prediction ML service now supports config-driven training, structured experiment tracking, automated model promotion, controlled model lifecycle, full dataset traceability, production-safe deployment, and a clean logging system — forming a solid MLOps foundation for the next stages of development.

Inference contract and prediction-time logging are intentionally deferred to Week 2/3 when the FastAPI layer and serving pipeline are built out properly.
