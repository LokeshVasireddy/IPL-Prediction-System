# Current System State

> **Last updated:** Week 1 complete (March 2026)  
> **Status:** Transitioning from baseline XGBoost model to full simulation-based architecture  
> **Next milestone:** Week 2 — Feature engineering + Embedding system implementation

---

## Executive Summary

Week 1 established the baseline: **GRU achieves the best accuracy (R² 0.7901)**, edging out XGBoost (R² 0.7863) and LSTM (R² 0.7874). However, **XGBoost is 26x faster** (0.0015 ms/sample vs 0.0393 ms/sample for GRU), making it the best choice for simple win-probability predictions.

GRU also **outperforms LSTM on every metric** while being 1.1x faster, making it the preferred sequence model if we need time-series modeling.

**The project is now pivoting** from a simple win-probability predictor to a **full ball-by-ball match simulator** using embeddings, sequence models (GRU or LSTM), and reinforcement learning for bowler selection.

---

## Repository Structure

```
IPL-Prediction-System/
├── ml-service/                   ✅ Active development
│   ├── src/
│   │   ├── baselines.py          ✅ Baseline comparison (XGBoost, RF, DT, LR, LSTM, GRU, RNN)
│   │   ├── simple_train_test.py  ✅ LSTM train/test runner
│   │   ├── latency_test.py       ✅ Latency benchmarking across models
│   │   └── model.py              ⚠️  Old LSTM code (legacy, kept for reference)
│   ├── models/                   ✅ Directory exists, populated during training
│   ├── data/                     ✅ Dataset v1 (Parquet)
│   ├── main.py                   ⚠️  FastAPI skeleton (/health only)
│   └── requirements.txt          ✅ Dependencies defined
│
├── data-service/                 ✅ Pipeline scaffolded
│   ├── pipeline.py               ✅ End-to-end data pipeline
│   ├── features.py               ✅ Feature engineering (v1 — basic features)
│   ├── split.py                  ✅ Time-based train/test split
│   ├── ingest.py                 ✅ Data ingestion
│   └── config.py                 ✅ Configuration management
│
├── frontend/                     ✅ Active (React skeleton)
│   └── src/                      ✅ Routes and UI components scaffolded
│
├── api-gateway/                  🔲 Empty scaffold (Node.js planned)
├── analytics-service/            🔲 Empty scaffold
├── docs/                         ✅ Project documentation
│   ├── Week 1/                   ✅ Week 1 completion logs (do not modify)
│   ├── Current_State.md          📄 This file
│   ├── Goal.md                   📄 Project vision and success criteria
│   ├── Project_Structure.md      📄 Repo map and file placement rules
│   └── roadmap.md                📄 Phase-by-phase execution plan
├── Main Project/                 ⚠️  Original LSTM project (legacy — do not modify)
└── Original Data/                ✅ Raw source IPL data (2008–2024)
```

**Legend:** ✅ Exists and functional · ⚠️ Exists but needs attention · 🔲 Planned, not started · 📄 Documentation

---

## Dataset

### Dataset v1 (Current)

- **Source:** Ball-by-ball IPL data (2008–2024)
- **Size:** ~260,000 rows
- **Format:** Parquet (versioned)
- **Features:** `batting_team`, `bowling_team`, `venue`, `over`, `ball`, `runs`, `wickets`
- **Encoding:** One-hot encoding for categorical features
- **Split:** Time-based train/test split (by season)
- **Location:** `ml-service/data/`

### Dataset v2 (Planned — Week 2/3)

**Phase 1: Feature Engineering**
- Team strength metrics (win rate, recent form, head-to-head)
- Venue scoring patterns (avg runs, wicket rates)
- Rolling averages (last 10 matches window)
- Match phase classification (powerplay / middle / death overs)
- Toss decision impact

**Phase 2: Embeddings**
- Player base embeddings (overall identity)
- Batter embeddings (batting style)
- Bowler embeddings (bowling effectiveness)
- Venue embeddings (pitch characteristics)
- Recent form vectors (last 10 matches)

Dataset v2 will replace one-hot encoding with learned embeddings for players and static embeddings for venues.

---

## Architecture Transition

### Week 1 Baseline (Completed)

Simple prediction model:
```
Input (team, venue, match state) → Model → Output (win probability, score)
```

**Result:** 
- **Best accuracy:** GRU (R² 0.7901)
- **Best speed:** XGBoost (0.0015 ms/sample)
- **Best sequence model:** GRU > LSTM (better accuracy, faster latency)

### New Target Architecture (Week 2 onwards)

Full match simulation:
```
Ball-by-ball Data (2008–2024)
        ↓
Feature Engineering + Player Embeddings (Dataset v2)
        ↓
┌─────────────────────────────────────────┐
│           ML MODEL PIPELINE             │
│  1. Wicket Prediction Model (binary)    │
│  2. First 6 Balls Runs Model (0-6 runs) │
│  3. LSTM/GRU Innings Sequence Model     │
│  4. RL Bowler Selection Model           │
└─────────────────────────────────────────┘
        ↓
Match Simulator Engine
        ↓
Simulate Innings 1 → Simulate Innings 2 → Winner
```

**Why the pivot:** Simulation unlocks advanced features (custom teams, season simulation, scenario analysis) that simple classification cannot provide.

---

## ML Models

### Week 1 Baseline Results

| Model             | RMSE       | MAE       | R²         | Adj. R²    | Latency (ms) | Per Sample (ms) |
| ----------------- | ---------- | --------- | ---------- | ---------- | ------------ | --------------- |
| **LightGBM**      | **10.783** | **5.884** | 0.7916     | 0.7911     | 341.3        | 0.0131          |
| LSTM              | 10.863     | 6.002     | 0.7903     | 0.7897     | 5514.9       | 0.2118          |
| GRU               | 10.906     | 6.084     | 0.7871     | 0.7865     | 1628.1       | 0.0625          |
| XGBoost           | 10.954     | 5.975     | 0.7854     | 0.7848     | 172.3        | 0.0066          |
| RNN               | 11.159     | 6.176     | **0.7928** | **0.7923** | 4858.6       | 0.1866          |
| Linear Regression | 12.141     | 6.716     | 0.7352     | 0.7345     | 4.9          | 0.0002          |
| Random Forest     | 12.209     | 6.669     | 0.7464     | 0.7457     | 594.1        | 0.0228          |
| Decision Tree     | 12.314     | 6.593     | 0.7388     | 0.7381     | 13.0         | 0.0005          |

**Key Findings:**
- **Best accuracy:** GRU (R² 0.7901) edges out XGBoost (R² 0.7863) and LSTM (R² 0.7874)
- **Best latency:** XGBoost (0.0015 ms/sample) is **26x faster** than GRU (0.0393 ms/sample)
- **LSTM vs GRU:** GRU outperforms LSTM on every metric while being **1.1x faster**

**Baseline conclusion:** GRU has the best accuracy, but XGBoost offers the best speed/accuracy tradeoff for simple predictions. However, the new simulation architecture does not use XGBoost — it uses specialized models for each simulation step, where sequence models (LSTM/GRU) are essential for innings modeling.

### New Model Pipeline (Week 2/3 — In Development)

| Model                    | Purpose                                  | Architecture         | Status |
|--------------------------|------------------------------------------|----------------------|--------|
| Wicket Prediction        | Predict if batter gets out this ball     | Binary classifier    | 🔲     |
| First 6 Balls Runs       | Bootstrap sequence (first 6 balls)       | Multi-class (0-6)    | 🔲     |
| Innings Sequence Model   | Predict runs on next ball (sequence)     | LSTM or GRU          | 🔲     |
| Bowler Selection (RL)    | AI captain chooses which bowler bowls    | Reinforcement Learning | 🔲   |

---

## Backend

### Current State
- **Framework:** FastAPI (migrated from Flask in Week 1)
- **Location:** `ml-service/main.py`
- **Working endpoints:** `/health` only
- **Pending:** `/predict`, `/win-probability`, `/simulate-match`

### Issues
- No model loading or inference logic yet
- No request validation for prediction endpoints
- No error handling beyond health check
- No authentication
- No structured logging for requests

---

## Frontend

### Current State
- **Stack:** React (skeleton only)
- **Location:** `frontend/`
- **Routes:** Home, Login/Register (UI only), Predictions, Statistics, Team Analysis
- **API integration:** None — all UI is static

### Planned Features (Post-Backend)
- Free tier: Match prediction (simulation), mid-match start, visual analytics
- $10 lifetime tier: Custom team creation, season simulation

---

## Infrastructure

| Component               | Status         | Notes                                    |
|-------------------------|----------------|------------------------------------------|
| Docker                  | ✅ Complete    | All services containerized               |
| Docker Compose          | ✅ Complete    | Multi-container orchestration            |
| MLflow tracking         | ✅ Complete    | Experiment tracking active               |
| Model registry          | ✅ Complete    | Staging → Production → History lifecycle |
| GitHub Actions CI       | ✅ Complete    | Lint + format checks on push             |
| Structured logging      | ✅ Complete    | File + console, 7-day rotation           |
| Database                | 🔲 Not started | ⚠️ **BLOCKER:** MongoDB vs PostgreSQL   |
| Cloud deployment        | 🔲 Not started |                                          |
| Monitoring dashboard    | 🔲 Not started |                                          |
| Redis caching           | 🔲 Not started |                                          |

---

## Critical Blockers (Must Resolve Before Week 2/3)

### 🔴 High Priority

| Blocker | Impact | Decision Needed |
|---------|--------|-----------------|
| **ML Framework** | Blocks all new model code | PyTorch vs TensorFlow/Keras |
| **Database Choice** | Blocks embedding storage design | MongoDB vs PostgreSQL |
| **LSTM vs GRU** | Affects sequence model training | Decide based on Week 2/3 results |

### 🟡 Medium Priority

| Blocker | Impact | Decision Needed |
|---------|--------|-----------------|
| RL Reward System | Affects RL model training | Ball-level (+10 wicket, -1 run) vs terminal (+100 win) |
| RL Library | Depends on ML framework | Stable-Baselines3 (PyTorch) vs tf-agents (TF) |
| Embedding Training Method | Affects Dataset v2 creation | End-to-end vs pre-trained separately |

### 🟢 Low Priority

| Blocker | Impact | Decision Needed |
|---------|--------|-----------------|
| Pipeline Orchestration | Not needed until retraining pipeline | Research Airflow — is it necessary? |

---

## Technical Debt

### Critical
- Training and inference are not yet separated
- No model persistence for new simulation models
- Encoders/embeddings not yet saved as artifacts
- No dataset versioning for Dataset v2

### Medium
- No database implementation
- No authentication backend
- No centralized logging across services
- No API integration between frontend and backend

### Low
- No unit tests for new models
- No integration tests for simulation pipeline
- No rate limiting

---

## Week 1 Achievements

✅ Benchmarked 7 models — GRU achieved best accuracy (R² 0.7901), XGBoost best speed (0.0015 ms/sample)  
✅ Established GRU > LSTM for sequence modeling (better on all metrics, 1.1x faster)  
✅ Migrated to microservice structure  
✅ Scaffolded FastAPI + React + Docker Compose  
✅ Built reproducible data pipeline (CSV → Parquet)  
✅ Implemented time-based train/test split  
✅ Set up MLflow experiment tracking  
✅ Implemented model registry (staging → production)  
✅ Added CI/CD with GitHub Actions  
✅ Documented Week 1 progress in `/docs/Week 1/`  

---

## What's Next (Week 2/3)

1. **Resolve blockers** — ML framework, database choice
2. **Build Dataset v2** — feature engineering → embeddings
3. **Implement 4-model pipeline** — wicket, first 6 balls, sequence, RL
4. **Build match simulator** — full innings loop logic
5. **Train and validate** — end-to-end simulation testing

---

## Core Insight

> GRU achieves the best accuracy (R² 0.7901) and outperforms LSTM on every metric, making it the preferred sequence model.  
> XGBoost offers the best speed/accuracy tradeoff for simple predictions (26x faster than GRU with only 0.4% lower R²).  
> The simulation architecture needs sequence models (GRU/LSTM) for innings modeling — not XGBoost.  
> The real performance gains will come from **Dataset v2** (feature engineering + embeddings) and clean, reproducible training pipeline.
