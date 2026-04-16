# IPL Prediction System — Week 1 Milestone Report

> **Project:** IPL Match Prediction System  
> **Week:** 1  
> **Roles covered:** ML Engineer · Web Developer  
> **Purpose:** Reviewer milestone submission — end of Week 1

## Table of Contents

1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [Web Development — Microservices & Infrastructure](#2-web-development--microservices--infrastructure)
3. [ML Engineering — Data Pipeline](#3-ml-engineering--data-pipeline)
4. [ML Engineering — MLOps Foundations](#4-ml-engineering--mlops-foundations)
5. [ML Engineering — Baseline Model Comparison & Model Decision](#5-ml-engineering--baseline-model-comparison--model-decision)
6. [Dataset v2 Strategy — Finalized Direction](#6-dataset-v2-strategy--finalized-direction)
7. [Week 1 Evaluation (Web Dev)](#7-week-1-evaluation-web-dev)
8. [Known Gaps — To Be Resolved Before Week 2/3](#8-known-gaps--to-be-resolved-before-week-2/3)
9. [Overall Progress Summary](#9-overall-progress-summary)

## 1. Project Overview & Architecture

### System Architecture

```
Frontend (React)
     ↓
API Gateway (FastAPI / Express)
     ↓
ML Service (FastAPI + LSTM Model)
     ↓
Data Service (Pipeline → Parquet)
```

### Project Root Structure

```bash
IPL-Prediction-System/
├── analytics-service/         # (planned)
├── api-gateway/               # Gateway — routing + logging
├── data-service/              # Data pipeline (parquet output)
├── docs/
├── frontend/                  # React app
├── ml-service/                # FastAPI + LSTM + MLOps
│   ├── app/
│   ├── configs/
│   │   └── lstm_v1.yaml       # YAML config — single source of truth
│   ├── core/
│   │   ├── config_loader.py
│   │   ├── logger.py
│   │   ├── model_bundle.py
│   │   ├── model_loader.py
│   │   └── registry.py
│   ├── data/
│   │   ├── processed/         # Versioned parquet datasets
│   │   └── metadata/          # Per-version metadata JSON files
│   ├── experiments/
│   │   └── mlruns/            # MLflow tracking DB and artifacts
│   ├── logs/                  # Rotated log files (git-ignored)
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
│   │   ├── simple_train_test.py   # Main training entrypoint
│   │   ├── model.py
│   │   ├── baselines.py
│   │   └── latency_test.py
│   ├── Dockerfile
│   ├── pytest.ini
│   └── requirements.txt
├── docker-compose.yml
└── README.md
```

## 2. Web Development — Microservices & Infrastructure

### 2.1 Microservices Overview

#### ML Service (FastAPI)

- Built using **FastAPI**
- Endpoints:
  - `POST /predict` → returns dummy prediction
  - `GET /health` → health check
- Uses **Pydantic** models for request/response validation
- Auto-generated OpenAPI docs at `/docs`

#### API Gateway

- Built using FastAPI / Express
- Responsibilities: request routing, logging middleware, JWT auth placeholder

**Routing Table:**

| Gateway Endpoint | Forwards To |
|---|---|
| `POST /predict` | `ml-service:8000/predict` |
| `GET /health` | `ml-service:8000/health` |

#### Frontend

- **React**-based UI
- Connects to API Gateway

#### Future Services (planned)

- `data-service` → data handling
- `analytics-service` → insights & stats

### 2.2 Docker Setup

Each service has its own `Dockerfile`, orchestrated via `docker-compose`.

```bash
docker compose up --build
```

**Services included:** `ml-service`, `api-gateway`, `mongo` (for future use)

**Docker Compose output:**

```bash
[+] Running 3/3
 ✔ Container mongo          Started
 ✔ Container ml-service     Started
 ✔ Container api-gateway    Started
```

**Sample API calls:**

```bash
# Health check via Gateway
curl http://localhost:8080/health
# Response: { "status": "ok" }

# Prediction via Gateway
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"team1": "CSK", "team2": "MI", "venue": "Wankhede"}'
# Response: { "winner": "CSK", "confidence": 0.72 }
```

### 2.3 Dev Tooling

| Layer | Tools |
|---|---|
| Frontend (React) | ESLint + Prettier |
| Backend (ML Service) | Black + isort |
| Git Hooks | pre-commit |
| CI/CD | GitHub Actions |

**Frontend setup:**

```bash
cd frontend
npm install --save-dev eslint prettier eslint-config-prettier eslint-plugin-react eslint-plugin-react-hooks

npm run lint      # Run ESLint
npm run format    # Run Prettier
```

**ML Service setup:**

```bash
cd ml-service
python -m venv venv
source venv/bin/activate
pip install black isort

black --check .        # Check formatting
isort --check-only .   # Check import order
```

**Pre-commit hooks:**

```bash
pip install pre-commit
pre-commit install
```

Enforces on every commit: ESLint on frontend files, Black & isort on Python files.

**GitHub Actions CI** (`.github/workflows/ci.yml`) runs on push to `main`:
- ESLint checks
- Black & isort validation
- Test execution (if available)

### 2.4 Web Dev Deliverables

- [x] All services start via `docker compose up --build`
- [x] ML service responds with dummy predictions
- [x] API Gateway routes requests correctly
- [x] Linting and formatting checks pass
- [x] Pre-commit hooks enforce code quality
- [x] GitHub Actions CI passes on `main`

## 3. ML Engineering — Data Pipeline

### What Was Built

A modular data pipeline to transform raw IPL match data into model-ready format.

### Pipeline Steps

1. **Ingestion** — Load CSV data and remove invalid rows with missing key fields.
2. **Encoding** — Applied One-Hot Encoding to categorical features (teams, venue) for model compatibility.
3. **Scaling** — Standardized numerical inputs (`over`, `ball`) and targets (`runs`, `wickets`).
4. **Feature Assembly** — Combined encoded categorical and scaled numerical features into a single feature matrix (`X`).
5. **Target Structuring** — Structured as a multi-output regression problem (predicting runs and wickets jointly).
6. **Dataset Splitting** — Train / validation / test split using consistent random seeds for reproducibility.
7. **Output** — Saved processed dataset in efficient Parquet format for downstream ML usage.

### Key Properties

- Fully reproducible via a single execution script (`pipeline.py`)
- Clean foundation for training, evaluation, and MLOps integration
- Data-service owns all preprocessing; ml-service consumes the ready parquet file

## 4. ML Engineering — MLOps Foundations

### 4.1 Experiment Tracking (MLflow)

MLflow is configured with a SQLite backend. All experiments are tracked under `experiments/mlruns/`.

**Setup:**

```python
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment(EXPERIMENT_NAME)
```

**What is tracked per run:**

| Category | Details |
|---|---|
| Parameters | model_type, dataset_version, feature_version, epochs, batch_size, LSTM units, learning rate, config file |
| Metrics (train) | Loss, MAE |
| Metrics (val) | Val Loss, Val MAE |
| Metrics (test) | Loss, MAE, MSE, RMSE, R², Real-scale R², Adjusted R² |
| Artifacts | Model bundle (.pkl), YAML config, dataset metadata JSON |

**Reproducibility:** Seeds set from config for Python `random`, NumPy, and TensorFlow.

**Run entrypoint:**

```bash
python -m training.simple_train_test --config configs/lstm_v1.yaml
```

### 4.2 Model Versioning

All model parameters are driven from `configs/lstm_v1.yaml` — nothing is hardcoded. Different experiments are managed via separate YAML config files, enabling comparison without modifying training code.

**Model architecture (config-driven):**

```python
model = keras.Sequential([
    keras.layers.Input(shape=(1, X.shape[2])),
    keras.layers.LSTM(LSTM1_UNITS, return_sequences=True),
    keras.layers.LSTM(LSTM2_UNITS),
    keras.layers.Dense(DENSE1, activation='relu'),
    keras.layers.Dense(DENSE2, activation='relu'),
    keras.layers.Dense(2)   # outputs: runs, wickets
])
```

### 4.3 Reproducible Training Pipeline

- Config loaded via CLI (`--config`), no hardcoded values anywhere
- Seeds applied consistently across Python, NumPy, TensorFlow
- Full pipeline runs inside a single MLflow run with exception handling
- Dataset loaded from versioned Parquet (not raw CSV)
- LSTM reshape handled in training script (`reshape(N, 1, features)`)

### 4.4 Data Version Awareness

Each dataset version has a corresponding metadata JSON (`data/metadata/v2_alpha.json`):

```json
{
    "dataset_version": "v2_alpha",
    "feature_version": "lstm_features_v1",
    "created_at": "...",
    "rows": ...,
    "features": [...]
}
```

This metadata is logged as an MLflow artifact, linking every experiment run back to its exact dataset and feature version.

### 4.5 Minimal Testing (pytest)

**Test coverage:**

| Test File | What It Tests |
|---|---|
| `test_data_pipeline.py` | Pipeline integrity — correct output shape and format |
| `test_model_bundle.py` | Bundle validity — loadable, correct attributes |
| `test_inference.py` | Inference execution — `bundle.predict()` runs without error |
| `test_registry.py` | Registry integrity — staging/production/history structure |

**Test result:**

```
pytest
collected 6 items

tests/test_data_pipeline.py  ...   [ 50%]
tests/test_inference.py      .     [ 66%]
tests/test_model_bundle.py   .     [ 83%]
tests/test_registry.py       .     [100%]

6 passed in 4.43s
```

**`pytest.ini`:**

```ini
[pytest]
testpaths = tests
pythonpath = .
```

### 4.6 Logging & Log Rotation

**Before:** `print()` statements — no structured output, no traceability.  
**After:** Structured logs with centralized logger, file + console output, environment-based configuration.

**Log format:**

```
2026-03-22 16:10:02 | INFO | training.simple_train_test | Training started
```

**Environment-based output:**

| Mode | Console | File |
|---|---|---|
| `ENV=dev` | INFO, WARNING, ERROR | All levels |
| `ENV=prod` | WARNING, ERROR only | INFO, WARNING, ERROR |

**Log rotation:** Daily via `TimedRotatingFileHandler`, 7-day retention. Old logs auto-cleaned on logger startup.

**Files updated:** `simple_train_test.py`, `model_loader.py`, `registry.py` — all `print()` replaced with structured logger calls.

### 4.7 Model Registry

A lightweight, custom folder-based registry manages the full model lifecycle.

```
models/
├── staging/          # Newly trained models, awaiting validation
├── production/       # Active model served by the API
│   ├── current_model.txt
│   ├── metadata.json
│   └── model.pkl
└── history/          # Previous production models (for rollback)
```

**Promotion flow:**

```
Training → Staging → Validation Gate → Production → History
```

**Validation gate:**

```
test_mae < gate_mae AND test_r2 > gate_r2
  → Passed: model promoted to production
  → Failed: model stays in staging
```

**Production metadata (`models/production/metadata.json`):**

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

This links the deployed model back to its MLflow run, dataset version, and feature version — full traceability from production to experiment.

> **Note:** A custom folder-based registry is used intentionally for Week 1 simplicity. MLflow Model Registry (`register_model`) is planned for later stages when multi-service deployment becomes relevant.

## 5. ML Engineering — Baseline Model Comparison & Model Decision

### Overview

To avoid committing to LSTM without justification, a full model family comparison was conducted across 7 models on the same dataset and evaluation protocol. All models were tracked via MLflow.

### Leaderboard

| Model | RMSE | MAE | R² | Adj. R² | Latency (ms) | Per Sample (ms) |
|---|---|---|---|---|---|---|
| **GRU** | **10.887** | 5.993 | **0.7901** | **0.7895** | 1024.4 | 0.0393 |
| XGBoost | 10.925 | **5.966** | 0.7863 | 0.7857 | 38.0 | 0.0015 |
| **LSTM** | 10.982 | 6.042 | 0.7874 | 0.7868 | 1131.9 | 0.0435 |
| RNN | 11.401 | 6.307 | 0.7823 | 0.7817 | 784.2 | 0.0301 |
| Linear Regression | 12.141 | 6.716 | 0.7352 | 0.7345 | 4.0 | 0.0002 |
| Random Forest | 12.209 | 6.669 | 0.7464 | 0.7457 | 160.4 | 0.0062 |
| Decision Tree | 12.314 | 6.593 | 0.7388 | 0.7381 | 6.0 | 0.0002 |

### Key Findings

**Top 3 performers by R²:** GRU (0.7901) → LSTM (0.7874) → XGBoost (0.7863)

The separation between the top 3 is narrow (~0.004 R²), but the pattern is clear:

- **Sequential models (GRU, LSTM) lead on accuracy** — GRU edges LSTM on every metric while being ~10% faster. This validates the hypothesis that over-by-over IPL data has temporal structure that tree models cannot fully exploit.
- **XGBoost is a strong challenger** — nearly matches LSTM on accuracy at ~30x lower latency. Retained as the primary non-sequential baseline.
- **Simpler models (Linear Regression, Decision Tree) lag significantly** — ~5–6 point RMSE gap confirms the problem is non-linear and feature interactions matter.

### Model Decision

> **Primary model: LSTM** (retained for production pipeline)  
> **Reason:** Consistent top-tier accuracy, existing MLOps integration, and direct upgrade path to GRU if latency becomes a constraint in serving.  
> **GRU noted as a drop-in upgrade** — same architecture family, better metrics, lower latency. Planned switch in Week 2/3 if inference SLA tightens.  
> **XGBoost retained as ensemble candidate** — will be evaluated for stacking with LSTM/GRU in later weeks.

This comparison satisfies the reviewer requirement of justifying the model family choice before freezing the architecture.

## 6. Dataset v2 Strategy — Finalized Direction

### Decision

After a joint review between the ML Engineer and Web Developer, the Dataset v2 feature strategy has been **finalized and frozen**.

### Core Decision: Embeddings over One-Hot Encoding

| Component | v1 (Current) | v2 (Finalized) |
|---|---|---|
| Team representation | One-Hot Encoding | Learned embeddings |
| Venue representation | One-Hot Encoding | Learned embeddings |
| Player representation | Not included | Learned embeddings |
| Features | over, ball | over, ball + match phase + context |

**Why embeddings:**
- One-hot encoding for 10 teams and 50+ venues creates a sparse, high-dimensional input that does not capture relationships between entities.
- Learned embeddings allow the model to represent team strength, venue character, and player quality as dense vectors trained end-to-end with the prediction objective.
- This is the same approach used in production sports ML systems and raises the feature engineering bar significantly above typical resume projects.

### Planned Feature Groups for v2

| Group | Features | Status |
|---|---|---|
| Team embeddings | Per-team dense vector (trainable) | 🔜 Week 2/3 |
| Venue embeddings | Per-venue dense vector (trainable) | 🔜 Week 2/3 |
| Player embeddings | Per-player dense vector (batters, bowlers) | 🔜 Week 2/3 |
| Match context | over, ball, innings, powerplay/death flag | 🔜 Week 2/3 |
| Historical context | Rolling team form, venue run bias | 🔜 Week 2/3 |

### Formal Specification

A formal feature specification document (`docs/feature_spec.md`) is a **tracked gap** — see Section 8.

## 7. Week 1 Evaluation (Web Dev)

> Evaluation done strictly against Week 1 scope — not full project expectations.

**Final Score: 8 / 10**

| Category | Score | Notes |
|---|---|---|
| Microservice architecture | 9/10 | Correct structure, clean separation, future services defined |
| ML service skeleton | 8.5/10 | FastAPI, Pydantic, dummy prediction, OpenAPI — all correct. Minor gaps: internal folder structure, logging |
| API Gateway | 7.5/10 | Routing concept solid, but routing example/structure could be more explicit |
| Docker setup | 8/10 | docker-compose and services correct. Would improve with port documentation and run proof |
| Dev tooling | 9/10 | ESLint, Prettier, Black, isort, pre-commit, GitHub Actions — strong and often skipped in Week 1 |
| Documentation | 8.5/10 | Structured and clear. Missing: architecture diagram |

**What was done well:**
- Correct microservice structure without overengineering
- Tooling and CI set up from day one — uncommon and valuable
- Stayed within Week 1 scope (no Kubernetes, Redis, full frontend, etc.)
- Clean, structured documentation

**Areas to improve for Week 2/3:**
- Add a simple architecture flow diagram (`Frontend → Gateway → ML Service`)
- Include docker compose run proof (screenshot or terminal output)
- Make API Gateway routing more explicit in documentation

> Real evaluation depth begins in **Week 2/3** when API integration, ML integration, and real frontend work begin.

## 8. Known Gaps — To Be Resolved Before Week 2/3

These are acknowledged gaps from the reviewer's evaluation. All are tracked with owner and target.

| Gap | Owner | Status | Target |
|---|---|---|---|
| `docs/feature_spec.md` — formal feature schema with source, type, description per feature | ML Engineer | ❌ Not started | Before Week 2/3 start |
| Dataset v2 implementation — embeddings pipeline replacing one-hot encoding | ML Engineer | ❌ Design frozen, implementation pending | Week 2/3 |
| Architecture diagram — visual system flow (React → Gateway → ML → Data) | Web Developer | ❌ Not started | Before Week 2/3 start |
| API Gateway routing — more explicit route documentation | Web Developer | ⚠️ Partially done | Week 2/3 |

**Note:** Baseline model comparison and model family decision (previously flagged as gaps by the reviewer) are now **resolved** — see Section 5.

**Note:** Dataset v2 feature strategy (embeddings) is now **finalized** — see Section 6. Implementation begins in Week 2/3.

## 9. Overall Progress Summary

### ML Engineering

| Component | Status |
|---|---|
| Modular data pipeline (CSV → Parquet) | ✅ Complete |
| Experiment Tracking (MLflow) | ✅ Complete |
| Model Versioning (config-driven YAML) | ✅ Complete |
| Reproducible Training Pipeline | ✅ Complete |
| Data Version Awareness (metadata JSON) | ✅ Complete |
| Minimal Testing (pytest, 6 tests passing) | ✅ Complete |
| Logging & Log Rotation | ✅ Complete |
| Model Registry (staging → production → history) | ✅ Complete |
| Baseline model comparison (7 models, full leaderboard) | ✅ Complete |
| Model family decision (LSTM primary, GRU upgrade path, XGBoost candidate) | ✅ Complete |
| Dataset v2 strategy — embeddings direction frozen | ✅ Complete |
| Inference contract (input/output schema) | 🔜 Planned — Week 2/3 |
| Inference logging (prediction/latency) | 🔜 Planned — Week 2/3 |
| Dataset v2 implementation (embeddings pipeline) | 🔜 Week 2/3 |
| Feature specification document (`docs/feature_spec.md`) | ❌ Gap — before Week 2/3 |
| MLflow Model Registry (`register_model`) | 🔜 Optional — later stage |

### Web Development

| Component | Status |
|---|---|
| Microservice folder structure | ✅ Complete |
| FastAPI ML service skeleton | ✅ Complete |
| API Gateway skeleton | ✅ Complete |
| Docker Compose setup | ✅ Complete |
| Dev tooling (ESLint, Black, isort, pre-commit) | ✅ Complete |
| GitHub Actions CI | ✅ Complete |
| Architecture documentation | ✅ Complete |
| Architecture diagram | ❌ Gap — before Week 2/3 |
| Real frontend | 🔜 Week 2/3 |
| JWT auth & database integration | 🔜 Week 2/3 |

### Reviewer Gap Resolution Status

| Reviewer-flagged Gap | Status |
|---|---|
| Model family comparison and decision | ✅ Resolved — see Section 5 |
| Dataset v2 strategy and feature direction | ✅ Resolved — see Section 6 |
| Feature specification document | ❌ Pending — before Week 2/3 |
| Architecture diagram | ❌ Pending — before Week 2/3 |

*Week 1 establishes the full foundation: microservice architecture, MLOps pipeline, experiment tracking, model registry, automated testing, structured logging, CI/CD, a full 7-model comparison, and a finalized Dataset v2 embeddings strategy. The 2 remaining gaps (feature spec doc, architecture diagram) are tracked and will be closed before Week 2/3 begins.*
