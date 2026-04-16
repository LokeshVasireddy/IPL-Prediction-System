# Project Structure Reference

> This file exists to give an AI assistant (or new contributor) immediate orientation.  
> Always consult this before creating or modifying files.

## Root

```
IPL-Prediction-System/
├── ml-service/           ML models, training pipeline, simulation engine, inference API
├── data-service/         Data ingestion, feature engineering, embedding creation
├── frontend/             React user interface (Free + $10 tier features)
├── api-gateway/          Request routing between services (Node.js — not yet implemented)
├── analytics-service/    Team/venue/match analytics endpoints (not yet implemented)
├── docs/                 Project documentation
│   └── Week 1/           Week 1 completion logs (historical — do not modify)
├── Main Project/         ⚠️ LEGACY — original LSTM project. Do not modify.
├── Original Data/        Raw source IPL data files (2008–2024)
└── .gitignore
```

## ml-service/

**Purpose:** Everything ML — training, model artifacts, simulation engine, and inference API.

```
ml-service/
├── src/
│   ├── baselines.py              Week 1 baseline comparison (XGBoost, RF, DT, LR, LSTM, GRU, RNN)
│   ├── simple_train_test.py      Standalone XGBoost train + evaluate script (Week 1)
│   ├── latency_test.py           Benchmarks prediction latency per model (Week 1)
│   ├── model.py                  ⚠️ Legacy LSTM model code — do not use for new work
│   │
│   ├── train/                    🔜 Week 2/3 — New training scripts
│   │   ├── train_wicket.py       Train wicket prediction model
│   │   ├── train_first6.py       Train first 6 balls runs model
│   │   ├── train_sequence.py     Train LSTM/GRU innings sequence model
│   │   └── train_rl_bowler.py    Train RL bowler selection model
│   │
│   ├── inference/                🔜 Week 2/3 — Inference logic
│   │   ├── simulator.py          Match simulation engine (innings loop)
│   │   └── predict.py            Load models + run simulation
│   │
│   └── embeddings/               🔜 Week 2/3 — Embedding management
│       ├── create.py             Generate player/batter/bowler/venue embeddings
│       └── storage.py            Save/load embeddings from database
│
├── models/
│   ├── (xgboost_v1.pkl)          Week 1 baseline model artifact
│   ├── wicket_v1.pkl             🔜 Wicket prediction model
│   ├── first6_v1.pkl             🔜 First 6 balls runs model
│   ├── sequence_v1.pkl           🔜 LSTM/GRU sequence model
│   ├── rl_bowler_v1.pkl          🔜 RL bowler selection model
│   └── encoders/                 🔜 Scalers, label encoders (if needed)
│
├── data/
│   ├── dataset_v1.parquet        Week 1 dataset (basic features, one-hot encoding)
│   └── dataset_v2.parquet        🔜 Week 2/3 dataset (engineered features + embeddings)
│
├── main.py                       FastAPI app — currently /health only
├── models.py                     Pydantic request/response schemas
└── requirements.txt
```

**What exists:** Week 1 baseline scripts, FastAPI skeleton, Dataset v1, model artifacts from baseline.  
**What's missing:** 4-model pipeline (wicket, first 6, sequence, RL), simulation engine, `/simulate-match` endpoint, Dataset v2.

**When adding new files here:**
- Training code → `src/train/`
- Inference logic → `src/inference/`
- Embedding creation → `src/embeddings/`
- New endpoints → `main.py`
- Saved models → `models/`
- New datasets → `data/`

## data-service/

**Purpose:** Data ingestion, cleaning, feature engineering, embedding creation, and dataset versioning.

```
data-service/
├── pipeline.py     Orchestrates full pipeline (ingest → features → embeddings → split)
├── features.py     Feature engineering functions
│                   - v1: basic team/venue/over features (one-hot encoding)
│                   - v2: team strength, venue patterns, rolling averages (engineered features)
├── embeddings.py   🔜 Create player/batter/bowler/venue/form embeddings
├── split.py        Train/test split (time-based by season)
├── ingest.py       Loads raw data from Original Data/
└── config.py       File paths and configuration constants
```

**What exists:** Full v1 pipeline with basic features and one-hot encoding.  
**What's missing:** Dataset v2 feature engineering, embedding creation logic.

**Dataset v2 Roadmap:**
1. **Phase 1 — Feature Engineering** (add to `features.py`):
   - Team strength metrics (win rate, head-to-head, recent form)
   - Venue scoring patterns (avg runs, wicket rates by venue)
   - Rolling averages (last 10 matches window)
   - Match phase classification (powerplay / middle / death)
   - Toss decision impact

2. **Phase 2 — Embeddings** (add to `embeddings.py`):
   - Player base embeddings (overall identity)
   - Batter embeddings (batting style)
   - Bowler embeddings (bowling effectiveness)
   - Venue embeddings (pitch characteristics — static)
   - Recent form vectors (last 10 matches — computed)

**When modifying Dataset v2:**
- Add feature engineering functions to `features.py` (keep v1 functions for comparison)
- Add embedding creation to `embeddings.py`
- Version output as `dataset_v2.parquet`

## frontend/

**Purpose:** React user interface for Free tier and $10 lifetime tier features.

```
frontend/
├── src/
│   ├── pages/
│   │   ├── Home.jsx              Landing page
│   │   ├── Login.jsx             Authentication (UI only — no backend yet)
│   │   ├── Register.jsx          Registration (UI only — no backend yet)
│   │   ├── Predictions.jsx       Free tier — Match simulation UI
│   │   ├── CustomTeam.jsx        🔜 $10 tier — Custom team creation
│   │   ├── SeasonSim.jsx         🔜 $10 tier — Season simulation
│   │   ├── Statistics.jsx        Analytics dashboard
│   │   └── TeamAnalysis.jsx      Team performance page
│   │
│   ├── components/               Reusable UI components
│   └── api/                      🔜 API integration layer (not yet wired)
│
├── index.html
├── vite.config.ts
├── tailwind.config.js
├── tsconfig.json
└── package.json
```

**Stack:** React + Vite + Tailwind CSS (TypeScript migration planned).  
**What exists:** Routes for Home, Login/Register (UI only), Predictions, Statistics, Team Analysis.  
**What's missing:** Real auth backend, `/simulate-match` API integration, custom team creation UI, season simulation UI.

**Free Tier Features:**
- Match simulation (any real IPL teams)
- Mid-match start (input current match state)
- Visual analytics (win probability, run rate, phase breakdown)

**$10 Lifetime Tier Features:**
- Custom team creation (pick real IPL players)
- Season simulation (real IPL season or custom league)

**Note:** Do not add new pages until the corresponding backend endpoint exists.

## api-gateway/

**Purpose:** Single entry point that routes requests to ml-service, data-service, analytics-service.  
**Current state:** Empty scaffold. Do not add logic here until Phase 3.  
**Planned stack:** Node.js + Express

## analytics-service/

**Purpose:** Serve team stats, venue stats, historical match data.  
**Current state:** Empty scaffold. Do not add logic here until Phase 3.

## docs/

**Purpose:** Project documentation for AI assistants, contributors, and recruiters.

```
docs/
├── Week 1/                   Week 1 completion logs (historical — do not modify)
│   ├── day1.md
│   ├── final_decision.md     Architectural decisions from Week 1
│   ├── mlops_foundations.md
│   ├── to_do.md              Full to-do list (basis for roadmap)
│   ├── walkthrough.md
│   └── week1.md              Week 1 summary
│
├── Current_State.md          What exists right now, repo structure, blockers
├── Goal.md                   Target architecture, success criteria, product features
├── roadmap.md                Phase-by-phase execution plan (updated from to_do.md)
└── Project_Structure.md      This file — repo map and file placement rules
```

## Legacy Warning

`Main Project/` contains the original LSTM-based project built before the architectural redesign. It is kept for reference only.

**Do not:**
- Copy patterns from it (training/inference coupling, Flask setup, LSTM model)
- Modify files in it
- Use it as a basis for new work

The new architecture uses:
- FastAPI (not Flask)
- Microservices (not monolith)
- Simulation (not classification)
- Embeddings (not one-hot encoding)

## Key Conventions

### File Placement
- All new ML training scripts → `ml-service/src/train/`
- All inference logic → `ml-service/src/inference/`
- Model artifacts → `ml-service/models/` (never committed raw to git)
- Dataset files → `ml-service/data/` or `data-service/output/`
- Raw source files → `Original Data/` (do not modify)
- Frontend pages → `frontend/src/pages/`
- Reusable components → `frontend/src/components/`

### Dataset Versioning
- Dataset v1 → one-hot encoding, basic features
- Dataset v2 → engineered features + embeddings
- Always version output files: `dataset_v1.parquet`, `dataset_v2.parquet`

### Model Versioning
- Model artifacts saved with version suffix: `wicket_v1.pkl`, `sequence_v2.pkl`
- MLflow tracks experiments and stores artifacts
- Model registry: staging → production → history

### API Design
- Frontend never calls ml-service directly
- All calls go through api-gateway (once built)
- Endpoints follow REST conventions: `/simulate-match`, `/analytics/team-stats`, etc.

### Embeddings
- Stored in database (MongoDB or PostgreSQL — TBD)
- Retrieved at inference time (not recomputed)
- Updated: recent form vectors after each match, base embeddings after retraining

## Navigation Guide

**Need to understand the project?** → Read `Goal.md`  
**Need to know what exists?** → Read `Current_State.md`  
**Need to know what to build next?** → Read `roadmap.md`  
**Need to know where to put a file?** → Read this file (`Project_Structure.md`)  
**Need to understand Week 1 work?** → Read `/docs/Week 1/final_decision.md`
