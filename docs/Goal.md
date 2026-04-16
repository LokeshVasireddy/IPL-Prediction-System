# Project Goal

## What This Project Is

A production-grade IPL match simulation and analytics platform — demonstrating end-to-end ML engineering from raw data pipelines to a subscription-based, API-driven product with real-time ball-by-ball match simulation.

This is not a research notebook or a simple classifier. The goal is a **deployable simulation system** with clean separation between training, inference, and serving layers.

## Primary Objective

Build a subscription-tier IPL simulation platform that:
- **Simulates full IPL matches** ball-by-ball using ML models
- Predicts match outcomes by simulating both innings completely
- Supports mid-match start (user provides current match state)
- Enables custom team creation and season simulation (paid tier)
- Serves predictions via a low-latency REST API
- Presents insights through an interactive analytics dashboard
- Is reproducible, versioned, and deployable

## Core Product Features

### Free Tier

| Feature | Details |
|---------|---------|
| **IPL Match Simulation** | Full ball-by-ball simulation for any real IPL teams |
| **Mid-Match Start** | User inputs: venue, innings, current score (runs/wickets), over + ball. System infers batting order position and current bowler based on team and over. |
| **Visual Analytics** | Charts and graphs — win probability, run rate, phase-wise breakdown, match progression |

### $10 Lifetime Tier

| Feature | Details |
|---------|---------|
| **Custom Team Creation** | Pick real IPL players from existing rosters to build a custom team, then simulate matches with it |
| **Season Simulation** | Two modes: (1) Simulate all matches in a real IPL season → generate points table + winner; (2) Simulate a full custom league season using user-created teams |

## System Architecture

### Match Simulation Pipeline

```
Ball-by-ball Data (2008–2024)
        ↓
Feature Engineering + Player Embeddings (Dataset v2)
        ↓
┌─────────────────────────────────────────┐
│           ML MODEL PIPELINE             │
│  1. Wicket Prediction Model             │
│     → Binary: out or not out            │
│                                         │
│  2. First 6 Balls Runs Model            │
│     → Bootstrap sequence (0-6 runs)     │
│                                         │
│  3. LSTM/GRU Innings Sequence Model     │
│     → Predict runs on next ball         │
│                                         │
│  4. RL Bowler Selection Model           │
│     → AI captain chooses bowler         │
└─────────────────────────────────────────┘
        ↓
Match Simulator Engine
        ↓
Simulate Innings 1 → Simulate Innings 2 → Declare Winner
```

**Why simulation over classification:**
- Enables advanced features: custom teams, season simulation, scenario analysis
- Produces interpretable ball-by-ball output, not just a final prediction
- Allows "what-if" scenarios (change bowler, change batting order, etc.)

## Target Tech Stack

|     Layer     |               Technology                      |
|---------------|-----------------------------------------------|
| ML            | PyTorch or TensorFlow (TBD), scikit-learn, MLflow |
| Embeddings    | Learned player/batter/bowler embeddings       |
| Reinforcement Learning | Stable-Baselines3 (PyTorch) or tf-agents (TF) |
| Data pipeline | Python, pandas, custom feature engineering    |
| Backend       | FastAPI (ML service), Node.js (API Gateway)   |
| Frontend      | React, TypeScript (planned), Tailwind CSS     |
| Database      | MongoDB or PostgreSQL (TBD)                   |
| Infra         | Docker, Docker Compose, GitHub Actions CI     |
| Deployment    | Cloud-hosted (TBD: Render / Railway / GCP)    |

## Core Capabilities (Must-Have)

### Simulation Engine
- **Ball-by-ball match simulation** — full innings 1 + innings 2
- **Wicket prediction** — binary classifier per ball
- **Runs prediction** — sequence model (LSTM/GRU) + first 6 balls model
- **Bowler selection** — RL agent acting as AI captain
- **Match state tracking** — runs, wickets, overs, balls, strike rotation, batting order

### Analytics & Visualization
- Win probability over time (ball-by-ball)
- Phase-wise analysis (powerplay / middle / death overs)
- Team performance dashboards
- Venue scoring patterns
- Match progression charts

### API & Product
- REST API with `/simulate-match` endpoint (<1s latency target)
- Subscription tier enforcement (Free vs $10 features)
- Mid-match simulation support
- Custom team management (paid tier)

## Secondary Capabilities (Future)

- Scenario simulation engine (what-if: change bowler, change team composition)
- Real-time match tracking (pseudo-live simulation mode)
- Player performance predictions
- Advanced analytics (batting order optimization, bowling strategy)

**Explicitly out of scope:** IPL Chatbot (cancelled)

## Why This Is Non-Trivial

### Data Engineering
- **Data leakage risk** — match joins must avoid future information bleeding into training
- **Time-aware validation required** — random splits overestimate performance; must validate on future seasons
- **Feature engineering dominates model performance** — team strength, rolling form, venue bias require careful construction from raw ball-by-ball data

### ML Engineering
- **4 separate models** — wicket, first 6 balls, sequence, RL bowler selection
- **Embedding system** — player base, batter, bowler, venue, recent form (5 types)
- **Reinforcement learning** — bowler selection trained in simulation environment
- **Sequence modeling** — LSTM/GRU with strike rotation and match state tracking

### System Design
- **Inference and training fully decoupled** — embeddings, scalers, and model artifacts must be saved and versioned separately
- **Latency constraints** — end-to-end simulation must complete in <1s
- **Subscription enforcement** — feature gating across Free vs $10 tiers
- **Database design** — embedding storage and retrieval at inference time

## Success Criteria

### ML Performance
- End-to-end simulation produces **realistic IPL scores** (120–200 range, wickets 0–10)
- Wicket prediction model: **precision/recall > 0.75**
- Sequence model: **RMSE < 2 runs per ball**
- RL bowler selection: **beats rule-based selection** in win rate

### Engineering Quality
- Clean training / inference separation
- MLflow experiment tracking and model versioning
- API latency **< 1 second** end-to-end for full match simulation
- Reproducible pipeline: data → features → embeddings → models → artifacts
- All 6 ML unit tests passing (pipeline, bundle, inference, registry)

### Product Completeness
- Fully integrated frontend + backend
- Live hosted demo
- No dummy data — all UI elements backed by real simulation outputs
- Subscription tier enforcement working (Free vs $10 features)
- Payment integration (for $10 tier)

## Non-Goals

- No deep learning unless benchmarks justify it
- No Kubernetes or overengineered infra
- No betting-style or gambling-adjacent predictions
- No focus on real-time data ingestion (static dataset 2008–2024 is sufficient for v1)
- No IPL Chatbot (cancelled)

## Strategic Principles

### 1. Feature Quality > Model Complexity > UI Aesthetics

A well-engineered embedding system on rich features will outperform a poorly-featured neural network — and be faster, more interpretable, and easier to deploy.

### 2. Simulation > Classification

Ball-by-ball simulation unlocks product features that simple win-probability classification cannot provide (custom teams, season simulation, scenario analysis).

### 3. Reproducibility = Professional Signal

The difference between a "project" and a "product" is:
- Versioned datasets
- Saved artifacts (models + encoders + embeddings)
- MLflow tracking
- Docker containerization
- Clean separation of training and inference

## Target Audience

### Recruiters & Hiring Managers
Demonstrates full-stack ML engineering:
- Data pipeline design
- Feature engineering at scale
- Multi-model training orchestration
- Reinforcement learning
- FastAPI microservices
- Docker + CI/CD
- Product thinking (subscription tiers, UX)

### Cricket Fans & Users
- Free tier: Simulate any IPL match, mid-match predictions, analytics
- Paid tier: Build custom teams, simulate full seasons
