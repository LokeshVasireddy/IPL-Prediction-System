# Development Roadmap

> **Status:** Week 1 complete. Starting Week 2.  
> **Approach:** Phased execution with clear blockers flagged at each stage.  
> **Timeline:** 8-9 weeks total (2 people: ML Engineer + Web Developer)

---

## ✅ Week 1 — System Audit & Baseline (COMPLETE)

### Completed Deliverables

**ML & Baselines**
- ✅ Benchmarked 7 models (XGBoost, RF, DT, LR, LSTM, GRU, RNN)
- ✅ Established XGBoost as baseline (R² 0.786, latency 0.002ms)
- ✅ Audited original LSTM project (identified sequence_length=1 flaw)
- ✅ Deprecated LSTM in favor of XGBoost for baseline

**Infrastructure**
- ✅ Migrated to microservice folder structure
- ✅ Scaffolded: `ml-service`, `data-service`, `frontend`, `api-gateway`, `analytics-service`
- ✅ Basic FastAPI setup in `ml-service/main.py` (`/health` endpoint)
- ✅ Docker + Docker Compose for all services
- ✅ GitHub Actions CI (lint + format checks)

**Data Pipeline**
- ✅ Data pipeline scaffolded in `data-service/` (pipeline, features, split, ingest)
- ✅ Dataset v1 created (Parquet, basic features, one-hot encoding)
- ✅ Time-based train/test split (by season)
- ✅ Data validation and missing value handling

**MLOps**
- ✅ MLflow experiment tracking
- ✅ Model registry (staging → production → history)
- ✅ Model serialization (.pkl bundles)
- ✅ Structured logging (file + console, 7-day rotation)
- ✅ 6 ML unit tests passing (pipeline, bundle, inference, registry)

**Documentation**
- ✅ Week 1 completion logs (`/docs/Week 1/`)
- ✅ Architecture decisions documented (`final_decision.md`)

---

## 🔴 Critical Blockers (Must Resolve Before Week 2/3)

| Blocker | Decision Needed | Impact | Owner |
|---------|-----------------|--------|-------|
| **ML Framework** | PyTorch vs TensorFlow/Keras | Blocks all new model code (wicket, first 6, sequence, RL) | ML Engineer |
| **Database Choice** | MongoDB vs PostgreSQL | Blocks embedding storage design | ML Engineer |
| **LSTM vs GRU** | Decide based on Week 2/3 training results | Affects sequence model architecture | ML Engineer |
| **RL Reward System** | Ball-level vs terminal reward | Affects RL training strategy | ML Engineer |
| **Embedding Training Method** | End-to-end vs pre-trained | Affects Dataset v2 creation | ML Engineer |

**Action Required:** Schedule decision meeting before Week 2 coding starts.

---

## Week 2–3 — Dataset v2 + Core ML Pipeline

### Phase 1: Feature Engineering (Week 2)

**Owner:** ML Engineer  
**Goal:** Create Dataset v2 with engineered features (not embeddings yet)

**Data Engineering** (`data-service/features.py`)
- [ ] Add team strength metrics (win rate, recent form, head-to-head records)
- [ ] Add venue scoring patterns (avg runs, wicket rates by venue)
- [ ] Add rolling averages (last 10 matches window)
- [ ] Add match phase classification (powerplay / middle / death overs)
- [ ] Add toss decision impact features
- [ ] Version output as `dataset_v2_features.parquet`

**Validation**
- [ ] Feature schema documentation (`docs/feature_spec.md` — **gap from Week 1**)
- [ ] Data quality checks on new features
- [ ] Verify no data leakage (time-aware validation)

### Phase 2: Embedding System (Week 2–3)

**Owner:** ML Engineer  
**Goal:** Replace one-hot encoding with learned embeddings

**Embedding Creation** (`data-service/embeddings.py`)
- [ ] Implement player base embeddings (overall identity)
- [ ] Implement batter embeddings (batting style)
- [ ] Implement bowler embeddings (bowling effectiveness)
- [ ] Implement venue embeddings (pitch characteristics — static)
- [ ] Implement recent form vectors (last 10 matches — computed)
- [ ] Version final output as `dataset_v2.parquet`

**Database Integration**
- [ ] Design embedding storage schema (depends on database choice blocker)
- [ ] Implement embedding save/load logic (`ml-service/src/embeddings/storage.py`)
- [ ] Build embedding retrieval API (fetch by player_id at inference time)

### Phase 3: Core ML Models (Week 3)

**Owner:** ML Engineer  
**Prerequisites:** Dataset v2 complete, ML framework decided

**Model Training** (`ml-service/src/train/`)
- [ ] Train wicket prediction model (binary classifier)
- [ ] Train first 6 balls runs model (multi-class: 0-6 runs)
- [ ] Train LSTM/GRU innings sequence model (decide architecture by results)
- [ ] Save all model artifacts to `ml-service/models/`
- [ ] Save embeddings to database
- [ ] Log experiments with MLflow

**Validation**
- [ ] Wicket model: precision/recall > 0.75
- [ ] First 6 balls model: accuracy > 0.70
- [ ] Sequence model: RMSE < 2 runs per ball
- [ ] Cross-validation on time-based splits

---

## Week 4 — Match Simulator + RL Bowler Selection

### Phase 1: Simulation Engine

**Owner:** ML Engineer  
**Goal:** Build full innings loop that uses trained models

**Simulator** (`ml-service/src/inference/simulator.py`)
- [ ] Implement match state tracking (runs, wickets, overs, balls, strike)
- [ ] Implement batting order logic
- [ ] Implement bowling order logic (rule-based initially)
- [ ] Implement ball loop:
  - [ ] Choose bowler (rule-based for now)
  - [ ] Predict wicket
  - [ ] If wicket → next batter, wickets += 1
  - [ ] If not out → predict runs (first 6 balls model or LSTM/GRU)
  - [ ] Update match state
  - [ ] Rotate strike (odd runs or end of over)
  - [ ] Check innings end conditions
- [ ] Simulate Innings 1 → Simulate Innings 2 → Declare Winner

**Validation**
- [ ] End-to-end simulation produces realistic IPL scores (120–200 range)
- [ ] Wickets fall in realistic distribution (0–10)
- [ ] Match duration = 40 overs (2 innings × 20 overs)
- [ ] Strike rotation works correctly

### Phase 2: RL Bowler Selection

**Owner:** ML Engineer  
**Prerequisites:** Simulator working with rule-based bowler selection

**RL Training** (`ml-service/src/train/train_rl_bowler.py`)
- [ ] Define RL state space (over, runs, wickets, batters, available bowlers)
- [ ] Define action space (select bowler)
- [ ] Implement reward system (decided by Week 2 blocker resolution)
- [ ] Train RL agent using simulator as environment
- [ ] Compare RL agent vs rule-based selection (win rate, runs conceded)
- [ ] Replace rule-based selection with trained RL agent in simulator

**Validation**
- [ ] RL agent beats rule-based selection in win rate
- [ ] Bowler selection choices are interpretable (not random)

---

## Week 5 — Backend API + Inference Pipeline

### Backend Engineering

**Owner:** ML Engineer  
**Goal:** Production-grade inference API

**ML Service** (`ml-service/main.py`)
- [ ] Implement `/simulate-match` endpoint
  - [ ] Load all 4 models (wicket, first 6, sequence, RL)
  - [ ] Load embeddings from database
  - [ ] Run full match simulation
  - [ ] Return winner + ball-by-ball output
- [ ] Implement `/win-probability` endpoint (ball-by-ball probabilities)
- [ ] Add request validation (Pydantic schemas in `models.py`)
- [ ] Add error handling middleware
- [ ] Add structured logging for requests (log input + output + latency)
- [ ] Add latency benchmarking (target: <1s end-to-end)

**Inference Optimization**
- [ ] Model loading (load once at startup, not per request)
- [ ] Embedding caching (Redis — if database decision supports it)
- [ ] Response caching for repeated requests

**Testing**
- [ ] Unit tests for `/simulate-match` endpoint
- [ ] Integration tests (end-to-end simulation)
- [ ] Load testing (concurrent requests)

---

## Week 6 — Frontend Integration + Product Features

### Frontend Engineering

**Owner:** Web Developer  
**Prerequisites:** `/simulate-match` API working

**Free Tier Features** (`frontend/src/pages/`)
- [ ] Connect Predictions page to `/simulate-match`
- [ ] Build match simulation UI:
  - [ ] Team selection dropdowns
  - [ ] Venue selection
  - [ ] Mid-match start inputs (innings, current score, over + ball)
  - [ ] "Simulate Match" button
- [ ] Display simulation output:
  - [ ] Winner announcement
  - [ ] Ball-by-ball progression (runs, wickets, overs)
  - [ ] Win probability chart (over time)
  - [ ] Phase-wise breakdown (powerplay / middle / death)
- [ ] Connect Statistics page to analytics endpoints (when available)
- [ ] Ensure no dummy data remains in any connected page

**$10 Lifetime Tier Features** (`frontend/src/pages/`)
- [ ] Build Custom Team creation page:
  - [ ] Player selection UI (pick from real IPL rosters)
  - [ ] Team composition validation (11 players, roles balanced)
  - [ ] Save custom team to database
- [ ] Build Season Simulation page:
  - [ ] Mode 1: Simulate real IPL season (all matches → points table)
  - [ ] Mode 2: Simulate custom league (user-created teams)
  - [ ] Display points table + winner

**Authentication & Subscription**
- [ ] JWT authentication backend (`api-gateway/auth.js`)
- [ ] Login/Register pages (real backend integration)
- [ ] Subscription tier enforcement (Free vs $10 feature gating)
- [ ] Payment integration UI (Stripe / PayPal)

**UI Polish**
- [ ] Responsive design (mobile + desktop)
- [ ] Loading states for async operations
- [ ] Error boundaries
- [ ] Dark / light theme (optional)

---

## Week 7 — Analytics Service + API Gateway

### Analytics Service

**Owner:** Web Developer  
**Goal:** Team/venue/match analytics endpoints

**Endpoints** (`analytics-service/`)
- [ ] `/analytics/team-stats` (win rate, avg score, recent form)
- [ ] `/analytics/venue-stats` (avg runs, wicket rates by venue)
- [ ] `/analytics/match-history` (past match results, ball-by-ball data)
- [ ] `/analytics/player-stats` (batter/bowler performance)

**Caching**
- [ ] Redis setup for analytics responses
- [ ] Cache invalidation strategy (update after each match)

### API Gateway

**Owner:** Web Developer  
**Goal:** Single entry point for all frontend requests

**Gateway** (`api-gateway/`)
- [ ] Route `/simulate-match` → `ml-service`
- [ ] Route `/analytics/*` → `analytics-service`
- [ ] Route `/auth/*` → auth endpoints
- [ ] Add rate limiting (per user, per IP)
- [ ] Add request logging (trace ID across services)
- [ ] Add CORS policy
- [ ] Add API versioning (`/v1/simulate-match`)

---

## Week 8 — Deployment + Production Hardening

### DevOps & Infrastructure

**Deployment**
- [ ] Deploy backend services (Render / Railway / GCP)
- [ ] Deploy frontend (static hosting or same platform)
- [ ] Domain + HTTPS setup
- [ ] Environment-specific configs (dev / staging / prod)

**Monitoring**
- [ ] Centralized logging (all services → single log aggregator)
- [ ] Distributed tracing (trace ID per request across Gateway → ML Service → DB)
- [ ] API latency monitoring (track `/simulate-match` latency)
- [ ] System health monitoring (per service)
- [ ] Error tracking (Sentry or similar)

**Security**
- [ ] Secrets management (no hardcoded keys, env-based)
- [ ] Secure headers (CSP, HSTS, etc.)
- [ ] Input validation on all endpoints
- [ ] Rate limiting enforcement

**Testing**
- [ ] End-to-end testing (full user flow: login → simulate → view results)
- [ ] Load testing (concurrent simulations)
- [ ] Latency validation (<1s target for `/simulate-match`)

### Documentation & Polish

**Architecture Documentation**
- [ ] Architecture diagram (React → Gateway → ML → Data → DB)
- [ ] Data flow diagram (ball-by-ball simulation flow)
- [ ] Deployment architecture diagram (services + infrastructure)

**Recruiter Signal**
- [ ] Updated README with setup instructions
- [ ] Live hosted URL
- [ ] Demo video (2-3 min walkthrough)
- [ ] Screenshots (simulation output, analytics dashboard, custom teams)
- [ ] Performance metrics writeup (latency, accuracy, win rate)
- [ ] Blog article / write-up for portfolio

---

## Parallel Tracks (Ongoing)

### MLOps (ML Engineer — Weeks 2–8)
- [ ] Retraining pipeline (automated on new data)
- [ ] Model versioning via MLflow model registry
- [ ] Drift detection (monitor prediction distribution)
- [ ] Automated evaluation on new seasons

### Testing (Both — Weeks 5–8)
- [ ] Backend unit tests (FastAPI endpoints)
- [ ] Frontend unit tests (React components)
- [ ] Integration tests (API → Database)
- [ ] Coverage reports (aim for >80%)

### Security (Web Developer — Weeks 6–8)
- [ ] Password hashing (bcrypt)
- [ ] JWT token expiry and refresh
- [ ] Role-based access control (Free vs $10 tier)
- [ ] HTTPS enforcement

---

## Open Research Questions (Low Priority)

| Question | Owner | Timeline |
|----------|-------|----------|
| Pipeline orchestration (Airflow) — is it needed? | ML Engineer | Week 4–5 |
| LightGBM model — should we benchmark it? | ML Engineer | Week 3 |
| Real-time match tracking (pseudo-live mode) | ML Engineer | Post-launch |
| Advanced analytics (batting order optimization) | ML Engineer | Post-launch |

---

## Execution Rules

### Do NOT
- Add LLM chatbot (explicitly cancelled)
- Build new UI pages until the corresponding backend endpoint exists
- Optimize models before fixing data (Dataset v2 first)
- Skip MLflow setup — reproducibility is a core deliverable
- Use Kubernetes or overengineered infra

### DO
- Resolve critical blockers before Week 2 starts
- Version all datasets (v1, v2_features, v2)
- Save all model artifacts (models + encoders + embeddings)
- Log all experiments with MLflow
- Write unit tests for new code
- Document all architectural decisions

---

## Success Metrics (Final Deliverable)

### Technical
- ✅ End-to-end simulation latency < 1 second
- ✅ Wicket model precision/recall > 0.75
- ✅ Sequence model RMSE < 2 runs per ball
- ✅ RL agent beats rule-based selection in win rate
- ✅ Simulation produces realistic IPL scores (120–200 range)
- ✅ All 6 ML unit tests passing + new tests for new models

### Product
- ✅ Live hosted platform (frontend + backend)
- ✅ Free tier working (match simulation, mid-match start, analytics)
- ✅ $10 tier working (custom teams, season simulation)
- ✅ Payment integration functional
- ✅ No dummy data — all UI backed by real simulation outputs

### Engineering
- ✅ MLflow experiment tracking active
- ✅ Model registry with versioning (staging → production)
- ✅ Reproducible pipeline (data → features → embeddings → models → API)
- ✅ Docker + CI/CD working
- ✅ Clean separation of training and inference
- ✅ Structured logging across all services

### Recruiter Signal
- ✅ Polished README with setup instructions
- ✅ Architecture diagrams (3: system, data flow, deployment)
- ✅ Demo video (2-3 min)
- ✅ Performance metrics writeup
- ✅ Blog post / portfolio writeup
