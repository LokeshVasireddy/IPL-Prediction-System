# IPL Prediction System — Final Decision Document

> **Purpose:** Single source of truth for all architectural and ML decisions going into Week 2/3.
> **Owner:** ML Engineer
> **Last updated:** Week 1 → Week 2/3 transition
> **Legend:** ✅ Decided | ⚠️ Open — must decide before implementation | 🔜 Planned

## 1. System Goal

✅ **Decided.**

Simulate a full IPL match ball-by-ball using ML models. The output is a **match winner**, derived from simulating both innings completely.

```
Simulate Innings 1 → Simulate Innings 2 → Declare Winner
```

This is not a simple win-probability classifier. The system simulates the match and the result emerges from that simulation.

**Explicitly out of scope:** IPL Chatbot — cancelled.

## 2. Subscription Tiers

✅ **Decided.**

### Free Tier

| Feature | Details |
|---|---|
| IPL Match Prediction (Simulation) | Full match simulation, any real IPL teams |
| Start from mid-match | User inputs: venue, innings number, current score (runs/wickets), over + ball number. System infers batting order position and current bowler based on team and over. |
| Visual Analytics | Charts and graphs — win probability, run rate, and related stats |

### $10 Lifetime Tier

| Feature | Details |
|---|---|
| Own Team Creation & Simulation | User picks real IPL players from existing rosters to build a custom team, then simulates a match with it |
| Season Simulation | Two modes: (1) simulate all matches in a real IPL season → generate points table + winner; (2) simulate a full custom league season using user-created teams |

## 3. High-Level Architecture

✅ **Decided.**

```
Ball-by-ball Data (2008–2024)
        ↓
Feature Engineering + Player Embeddings
        ↓
┌─────────────────────────────────────┐
│           ML MODELS                 │
│  1. Wicket Prediction Model         │
│  2. First 6 Balls Runs Model        │
│  3. LSTM or GRU Innings Model       │
│  4. RL Bowler Selection Model       │
└─────────────────────────────────────┘
        ↓
Match Simulator Engine
        ↓
Winner Prediction
```

## 4. ML Framework

⚠️ **Open — must decide before Week 2/3 implementation starts.**

Week 1 used TensorFlow/Keras. The same architecture could be built in PyTorch. These cannot both be used for the same models without a clear boundary.

**Options:**
- Switch fully to PyTorch
- Stay on TensorFlow/Keras

**Impact:** Affects LSTM/GRU, wicket model, and runs model. Decide this first before writing any new model code.

## 4. Embedding System

### 4.1 Embedding Types

✅ **Decided.**

| Embedding | Purpose | Type |
|---|---|---|
| Player base embedding | Overall player identity | Learned |
| Batter embedding | Batting style and behavior | Learned |
| Bowler embedding | Bowling style and effectiveness | Learned |
| Venue embedding | Pitch and ground characteristics | Static |
| Recent form vector | Current player form (last 10 matches) | Computed |

### 4.2 Form Window

✅ **Decided.**

```
last 10 matches + current season stats
```

Rationale: last 5 is too noisy, last 15 is too slow to react, whole season becomes outdated.

### 4.3 Embedding Dimensions

🔜 **Experimental — to be determined during Week 2/3 training.**

Dimensions are not fixed upfront. They will be treated as a hyperparameter and tuned based on model performance. Start small to avoid overfitting and increase if results justify it.

### 4.4 How Embeddings Are Trained

🔜 **To be decided and implemented in Week 2/3.**

Two valid approaches exist — end-to-end (embedding layers trained with the model) or pre-trained separately (trained from player stats first, then used as input). The choice will be made during Week 2/3 as part of the embedding implementation work.

### 4.5 Venue Embedding

✅ **Decided.** Static — venue characteristics are stable enough that a fixed embedding is acceptable.

## 5. Embedding Storage & Retrieval

✅ **Decided.**

Player embeddings are stored in the database and retrieved at inference time — they are not recomputed on every prediction.

**What is stored:**

| Entity | Stored Data |
|---|---|
| Batter | batter embedding, recent form vector |
| Bowler | bowler embedding, recent form vector |
| Player (shared) | player base embedding |
| Venue | venue embedding (static) |

**Retrieval flow:**

```
Simulation request
      ↓
Look up player IDs from team squads
      ↓
Fetch embeddings from DB
      ↓
Feed into models
```

**Update strategy:** Recent form vectors are recomputed and updated in the database after each match (last 10 matches window slides forward). Base embeddings are updated on a less frequent schedule (e.g., end of season or after retraining).

**Note:** Database choice (MongoDB vs PostgreSQL) is still open — see Section 10. This storage design works with either.

## 6. Model Components

### 6.1 Wicket Prediction Model

✅ **Decided.**

- **Type:** Binary classifier
- **Output:** `0 = not out`, `1 = wicket`
- **Inputs:** bowler embedding, batter embedding, venue embedding, recent form, over, ball, runs, wickets fallen, match phase, innings

**If wicket:** next batter enters, wickets += 1, runs = 0 for that ball.
**If not out:** pass to runs model.

### 6.2 First 6 Balls Runs Model

✅ **Decided.**

- **Purpose:** Bootstrap the sequence model — LSTM/GRU needs a sequence history to start. First 6 balls of a batter's innings provide that.
- **Output:** runs per ball (`0, 1, 2, 3, 4, 6`)
- **Inputs:** batter embedding, bowler embedding, match state, phase, over, ball

Result feeds directly into the LSTM as its initial sequence.

### 6.3 Sequence Innings Model

⚠️ **Architecture open — LSTM or GRU, decided by results.**

Both are valid sequence models for this task. Week 1 comparison showed GRU edges LSTM slightly on every metric at lower latency. The final choice will be made based on performance during Week 2/3 training with the new embeddings-based dataset.

- **Purpose:** Model innings as a time series and predict runs on the next ball.
- **Input sequence:** last 6 balls + match state + player embeddings + wickets + overs
- **Output:** runs on next ball

**Strike rotation:**
- Odd number of runs → strike rotates
- End of over → strike rotates

### 6.4 RL Bowler Selection Model

⚠️ **In scope, but approach and reward system not yet decided.**

- **Purpose:** Choose which bowler bowls the next over (Captain AI).
- **Agent state:** over number, runs, wickets, batter embeddings, available bowlers, overs left, match phase
- **Action:** select bowler
- **Reward system:** ⚠️ Open — two options under consideration: (1) `+10 wicket, -1 per run` per ball; (2) `+100 win / -100 loss` as terminal reward. To be decided during RL implementation.
- **RL library/framework:** ⚠️ Open — depends on ML framework decision (PyTorch vs Keras).

**Training approach (order is decided):**
1. Train wicket model, runs model, sequence model first.
2. Build simulator with rule-based bowler selection.
3. Train RL agent using the simulator as the environment.
4. Replace rule-based with trained RL agent.

## 7. Match Simulator Engine

✅ **Decided.**

Controls the full innings loop. Tracks: runs, wickets, overs, balls, strike batter, batting order, bowling order, target.

**Ball loop logic:**
```
choose bowler (RL)
predict wicket
if wicket → next batter
else → predict runs (first 6 balls model or LSTM)
update match state
rotate strike if needed
check innings end
```

**Innings end conditions:**
- 10 wickets → innings over
- 20 overs completed → innings over
- Target chased (2nd innings) → batting team wins
- Target not chased → bowling team wins

## 8. Training Order

✅ **Decided.**

| Step | What |
|---|---|
| 1 | Train player embeddings, wicket model, first 6 balls model, sequence model (LSTM or GRU) |
| 2 | Build match simulator with rule-based bowler selection |
| 3 | Train RL bowler selection model using simulator as environment |
| 4 | Replace rule-based selection with trained RL model |
| 5 | Full system integration and end-to-end testing |

## 9. Data

✅ **Decided.**

- **Source:** Ball-by-ball IPL data, 2008–2024
- **Format:** Parquet (versioned, from Week 1 pipeline)
- **Dataset version moving to:** v2 (embeddings replacing one-hot encoding)

## 10. Database

⚠️ **Open.**

Week 1 docker-compose has MongoDB. Architecture doc specifies PostgreSQL. These serve different purposes (document store vs relational), and the right choice depends on how embeddings and player data are stored and queried.

**Must decide:** Which database stores player embeddings, team squads, batting orders, and bowling pools.

## 11. Infrastructure & MLOps

### Confirmed from Week 1

✅ MLflow — experiment tracking (stays)
✅ Custom model registry (staging → production → history) (stays for now)
✅ Docker + Docker Compose — all services containerized
✅ GitHub Actions CI — linting, formatting, tests on push to main
✅ Structured logging with rotation

### Pipeline Orchestration

⚠️ **Open — requires research.**

The architecture doc mentions Airflow for orchestrating the data pipeline. This is not yet understood or decided. Research what Airflow is and whether it's needed at this stage before committing.

## 12. Tech Stack (Confirmed Portions)

| Layer | Tool | Status |
|---|---|---|
| ML (framework) | TensorFlow/Keras or PyTorch | ⚠️ Open |
| Sequence model | LSTM or GRU | ⚠️ Decided by results in Week 2/3 |
| RL library | To be decided with framework | ⚠️ Open |
| Data processing | Pandas, NumPy | ✅ Decided |
| Experiment tracking | MLflow | ✅ Decided |
| API | FastAPI | ✅ Decided |
| Containerisation | Docker + Docker Compose | ✅ Decided |
| Database | MongoDB or PostgreSQL | ⚠️ Open |
| Frontend | React | ✅ Decided |

## 13. Open Decisions — Summary

| Decision | Priority | Notes |
|---|---|---|
| ML framework (PyTorch vs Keras) | 🔴 High — blocks all model code | Decide before any new model code |
| LSTM vs GRU | 🟡 Medium | Decided by Week 2/3 training results |
| RL approach & reward system | 🟡 Medium | Decided during RL implementation (Step 3 of training order) |
| RL library | 🟡 Medium | Depends on framework decision |
| Database (MongoDB vs PostgreSQL) | 🟡 Medium | Needed before storing embeddings and squads |
| Pipeline orchestration (Airflow) | 🟢 Low — not urgent | Research what it is first; not needed for Week 2/3 |

## 14. Week 1 Gaps Still Open

| Gap | Owner | Target |
|---|---|---|
| `docs/feature_spec.md` — formal feature schema | ML Engineer | Before Week 2/3 start |
| Architecture diagram (React → Gateway → ML → Data) | Web Developer | Before Week 2/3 start |

*This document reflects only finalized decisions and clearly flagged open questions. It replaces the AI-generated design docs which contained unfinalized content.*
