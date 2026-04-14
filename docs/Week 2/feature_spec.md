# IPL Cricket Intelligence Platform — Feature Specification

**Scope**: IPL Only | **Format**: T20 (20 overs per inning)

## Executive Summary

ML-powered cricket intelligence platform for IPL T20 matches with:
- **Pre-computed embeddings** (player & venue, seasonal, chronologically safe)
- **Wicket prediction** (binary classifier)
- **Run prediction** (Unified zero-padded LSTM for all balls)
- **RL agent** (bowling strategy optimization)

**Core Principles**: Simplicity (no wicket types, fielding, toss) | Legal deliveries only | Single unified model | No future data leakage

## 1. Feature Specification & Normalization

### 1.1 Match Context Features

| Feature | Type | Raw Range | Normalization | Notes |
|---------|------|-----------|---------------|-------|
| `inning` | int | 0 - 1 | None | Categorical identifier |
| `over` | int | 1 - 20 | `over / 20` | Normalized to [0.05, 1.0] |
| `sin_ball` | float | -1 to 1 | `sin(2π × ball / 6)` | Cyclic encoding of ball position |
| `cos_ball` | float | -1 to 1 | `cos(2π × ball / 6)` | Cyclic encoding of ball position |
| `balls_remaining` | int | 0 - 120 | `balls_remaining / 120` | Normalized to [0, 1]|
| `current_score` | int | 0 - 300 | **TBD**: min-max / ÷100 / ÷200 | Test during training |
| `wickets_fallen` | int | 0 - 10 | `wickets_fallen / 10` | Normalized to [0, 1] |
| `target` | int | 0 - 300 | Same as `current_score` | 0 for inning 1 |
| `phase` | categorical | 3 classes | One-hot encoding | powerplay/middle/death |

**Notes**
- **Formulae**: balls_remaining = (20 - current_over) * 6 - current_ball
- **For Phase**: 0: powerplay (1 - 6 overs) / 1: middle (7-15 overs)/ 2: death (16 - 20 overs)
- **inning Switch**: Acts as a binary switch to zero out chase weights.
- **Ball Encoding Rationale**: Using `sin_ball` and `cos_ball` instead of raw `ball` captures the cyclic nature (ball 6 → ball 1 transition) and ensures smooth gradients.

### 1.2 Extended Context Features

| Feature | Type | Raw Range | Normalization | Notes |
|---------|------|-----------|---------------|-------|
| `last_over_runs` | int | 0 - 37 | Same as `current_score` | Recent momentum |
| `balls_since_boundary` | int | 0 - 120 | `balls_since_boundary / 120` | Pressure indicator |
| `percentage_target_achieved` | float | 0 - 1 | None | 0.0 for inning 1; `current_score / target` for inning 2 |

### 1.3 Player Features

| Feature | Type | Raw Range | Normalization | Notes |
|---------|------|-----------|---------------|-------|
| `macro_batter_embedding` | vector | [-1, 1] | None | Pre-normalized |
| `micro_batter_embedding` | vector | [-1, 1] | None | Pre-normalized |
| `micro_non_striker_embedding` | vector | [-1, 1] | None | Pre-normalized |
| `macro_bowler_embedding` | vector | [-1, 1] | None | Pre-normalized |
| `micro_bowler_embedding` | vector | [-1, 1] | None | Pre-normalized |

### 1.4 Venue & Metadata

| Feature | Type | Raw Range | Normalization | Notes |
|---------|------|-----------|---------------|-------|
| `macro_venue_embedding` | vector | [-1, 1] | None | Pre-normalized |
| `season_year` | int | 2008 - 2030 | `(year - μ) / σ` | Standardization (mean & std) |

## 2. Normalization Strategy for Run-Related Features

**Features requiring run normalization**: `current_score`, `target`, `last_over_runs`

**Approach**: To be finalized during model training phase. Test these options:

1. **Min-Max Scaling**: `(score - min) / (max - min)` where max ≈ 300
2. **Division by 100**: `score / 100` (simple, interpretable)
3. **Division by 200**: `score / 200` (ensures most values < 1)
4. **Standardization**: `(score - μ) / σ` using training set statistics

**Recommendation**: Start with **÷200** for simplicity, then experiment with min-max if model struggles.

## 3. Player & Venue Embeddings

**Pre-computed offline** and placed in dataset as fixed vectors (NOT trainable during model training).

### 3.1 Embedding Strategy

**Initial embedding dimensions**:

- batter embedding → 20
- bowler embedding → 20
- venue embedding → 10

These values are starting points and may be tuned based on validation performance.

**Generation**:
- Computed externally using historical delivery data
- Micro embeddings by TabTransformer and Macro by PCA / Standardization of Stats
- Season N embeddings: Use data from seasons 2008 to N-1 only
- Dimension: 20-70 (players), 10-15 (venues)

**Cold Start**:
- New players (That Season): Use role-based league average
- New venues (That Season): Use league-average venue embedding

**Chronological Safety**: Embeddings for 2024 use 2008-2023 data; 2025 uses 2008-2024 data.

## 4. Extras Handling

**Extras NOT predicted by ML models** — handled probabilistically in simulator.

### Bowler Extras Stats

| Stat | Description |
|------|-------------|
| `powerplay_wide_rate` | P(wide per legal delivery in powerplay) |
| `powerplay_no_ball_rate` | P(no-ball per legal delivery in powerplay) |
| `middle_wide_rate` | P(wide per legal delivery in middle) |
| `middle_no_ball_rate` | P(no-ball per legal delivery in middle) |
| `death_wide_rate` | P(wide per legal delivery in death) |
| `death_no_ball_rate` | P(no-ball per legal delivery in death) |

>Note: if overs by bowler < 5 for particular phase → league average

### Simulator Logic

The system uses multiple specialized models (wicket, runs, sequence, RL). There is no single unified model.<br>
Match outcome emerges from the interaction of these models inside the simulator.

1. **Sample extras**: If `random() < wide_rate` → add run, repeat ball; if `random() < no_ball_rate` → add run, repeat ball, next is free hit
2. **Get ML prediction**: Wicket model → LSTM Run model
3. **Update state**: Scores, strike rotation

**Edge Cases**: Free hit disables wicket model | Multiple wides possible | No-ball + wide = 2 runs

## 5. Model Architecture

### 5.1 Wicket Model
- **Type**: Binary classifier (wicket / no wicket)
- **Input**: All normalized features from Sections 1.1-1.4
- **Output**: P(wicket) ∈ [0, 1]
- **Architecture**: During Building Phase

### 5.2 Unified LSTM Run Model (All Balls)
- **Type**: Sequence-based multi-class classifier
- **Input**: Last 30 balls (30 × feature_dim)
- **Output**: P(runs) for {0, 1, 2, 3, 4, 5, 6}
- **Architecture**: During Building Phase
- **Padding**: Left-zero-padding used for dummy historical deliveries when predicting balls 1 through 29

**Model Transition**:
```
Sequence predicting Ball 1:  [Pad, Pad, ..., Pad (x29), Ball_1_Features]
Sequence predicting Ball 2:  [Pad, Pad, ..., Pad (x28), Ball_1_Features, Ball_2_Features]
...
Sequence predicting Ball 30: [Ball_1, Ball_2, Ball_3, ..., Ball_30_Features]
Sequence predicting Ball 31: [Ball_2, Ball_3, Ball_4, ..., Ball_31_Features]
```
LSTM sequence resets at start of each inning

## 6. Match Simulation Flow

### Target Variables Definition

Each ML component has a clearly defined target variable:

1. Wicket Model → target: wicket (0/1)
2. LSTM Run Model → target: runs per ball (0,1,2,3,4,5,6)
3. RL Bowler Selection Model → target: reward signal based on match outcome

### 6.1 Strike Rotation Logic

| Event | Action |
|-------|--------|
| Odd runs (1, 3, 5) | Swap striker ↔ non-striker |
| End of over | Swap striker ↔ non-striker |
| Wicket | New batter takes striker's end |
| Even runs (0, 2, 4, 6) | No swap |

### 6.2 Bowling Constraints

| Constraint | Rule |
|------------|------|
| Max overs | 4 per bowler per inning |
| Consecutive | Cannot bowl overs N and N+1 in same inning |
| Validity | Must be in playing XI |
| Minimum | < 5 bowlers allowed (flexibility) |

### 6.3 Match Structure

- **Format**: 2 inning, 20 overs each, 120 legal balls per inning
- **Termination**: 20 overs complete OR 10 wickets fall OR target reached (inning 2)
- **Target**: `inning_1_score + 1` for inning 2; `0` for inning 1
- **RL Control**: Selects bowler at start of each over for both teams

## 7. Reinforcement Learning Environment

### 7.1 RL Agent Role

**Objective**: Select optimal bowler at the start of each over to maximize win probability.

- **Decision frequency**: Once per over (~20 per inning)
- **Scope**: Controls bowling for both teams

### 7.2 RL State Space

**To be finalized during RL training phase.** Initial proposal includes:

**Core Features**: All match context, player, venue features from Section 1

**Bowler-Specific Features**:
- `available_bowlers`: List of valid bowler IDs
- `overs_bowled_per_bowler`: Dict of overs bowled
- `overs_left_per_bowler`: Dict of overs remaining

### 7.3 RL Action & Reward

- **Action**: Select `bowler_id` from `available_bowlers`
- **Validation**: Simulator enforces constraints; invalid action → negative penalty
- **Reward**: To be decided during training

## 8. Team Composition

### 8.1 Playing XI & Batting Order

| Mode | Playing XI | Batting Order |
|------|------------|---------------|
| **Developer** | Fixed from database | Historical order |
| **User** | Custom selection | Custom order (1-11) |

**Validation**: All 11 players must have embeddings for the season.

### 8.2 Bowling Order

**Not predefined** — RL agent selects bowler each over subject to constraints in Section 6.2.

## 9. Dataset Requirements

### 9.1 Data Scope

- **League**: IPL only (2008-present)
- **Granularity**: Ball-by-ball legal deliveries (wides/no-balls removed)
- **Structure**: Data processed in overlapping sliding windows of exactly 30 balls.

### 9.2 Chronological Split

| Set | Description |
|-----|-------------|
| **Training** | Seasons 2010 to 2024 |
| **Validation** | Season 2025 (First Half) |
| **Test** | Season 2025 (Second Half) |

**Critical**: No future data leakage in embeddings or features.<br>
**Note**: 2023 and 2024 are critical training years to ensure the model adapts to the modern ~185+ par score meta introduced by the Impact Player rule. Validation and Test are split mid-season 2025 to evaluate the model's performance on pitch degradation.

### 9.3 Data Quality

**Required Fields**: Player IDs, venue ID, match outcome, ball outcome (runs/wicket), over/ball counts
**Action if Missing**: Exclude ball/match from dataset

## 10. Implementation Pipeline

### Phase 1: Data Pipeline
1. Extract ball-by-ball data from IPL sources
2. Clean data, remove extras, validate fields
3. Engineer features (`phase`, `last_over_runs`, etc.)
4. Compute bowler phase wise extras stats (`powerplay_wide_rate`, `death_no_ball_rate`, etc.)
5. Create chronological train/val/test splits

### Phase 2: Embedding Generation (External)
1. Compute macro and micro player embeddings (batters, bowlers) from historical data
2. Compute macro venue embeddings
3. Generate league-average embeddings for cold-start
4. Place embeddings in dataset as fixed vectors
5. Ensure chronological constraint (season N uses data up to N-1)

### Phase 3: ML Models
1. Train wicket model (binary classifier with class weights)
2. Train unified LSTM using overlapping sliding window sequences to prevent data starvation (using zero-padding for early balls)
3. Validate on validation set, tune hyperparameters
4. Evaluate on test set

### Phase 4: Simulator
1. Integrate wicket and LSTM run models
2. Implement strike rotation and bowling constraints
3. Build full 2-inning match loop
4. Test with historical match replays

### Phase 5: RL Environment
1. Wrap simulator as RL environment
2. Define state encoding and action space
3. Train RL agent (methodology TBD)
4. Evaluate vs historical bowling orders

### Phase 6: Deployment
1. Build microservices for models (wicket, LSTM, RL)
2. Create API endpoints for simulation and prediction
3. Deploy embedding lookup service
4. Integrate with frontend

## 11. Model Outputs Summary

| Model | Output |
|-------|--------|
| **Wicket** | P(wicket) ∈ [0, 1] |
| **Unified LSTM** | P(runs) for {0,1,2,3,4,5,6} |
| **RL Agent** | bowler_id from available_bowlers |

**Excluded**: Wicket types (bowled/caught/LBW), fielding positions, shot types

## 12. Key Design Decisions

### Finalized
- **LSTM Sequence Strateg**y: Overlapping sliding windows of 30 balls to massively increase training data volume and capture multi-over momentum.
- **Player Representation**: Concatenation of TabTransformer micro-interactions and normalized macro-season stats at the sequence input.
- **inning Handling**: 0 for inning 1
- **Main Model**: Multi-model cricket simulation system with specialized models interacting through a simulator
- **Ball Encoding**: Cyclic (sin/cos) to capture position
- **Embeddings**: Pre-computed externally, not trainable
- **Bowling Flexibility**: < 5 bowlers allowed if needed
- **Consecutive Overs**: Constraint applies within inning only

### To Be Decided During Training
- **Run normalization**: Test min-max / ÷100 / ÷200 / standardization
- **RL reward structure**: Win/loss only vs intermediate rewards
- **RL training methodology**: Algorithm, exploration, opponent generation

## 13. Quick Reference

### Normalization Cheat Sheet
```
over               → over / 20
ball               → sin(2π × ball/6), cos(2π × ball/6)
balls_remaining    → balls_remaining / 120
wickets_fallen     → wickets_fallen / 10
balls_since_boundary → balls_since_boundary / 120
season_year        → (year - μ) / σ
runs (all)         → TBD (min-max / ÷100 / ÷200)
embeddings         → None (already normalized)
phase              → One-hot encoding
inning            → None (0 or 1)
percentage_target_achieved  → 0.0 (1st inning) or current/target (2nd inning)
```

### Model Flow
```
For each ball:
    1. Check phase wide
      → add run
      → repeat ball

  2. Check phase no-ball
      → add run
      → free hit = True

  3. If free hit:
          wicket = 0
          Run LSTM model
    Else:
          predict wicket
          if wicket:
              scores = 0, new batter
          else:
              Run LSTM model
  4. If over complete: RL selects next bowler

  "Assumption: Wicket deliveries yield 0 runs in simulation."
```

### Constraints Validation
```
✓ Max 4 overs per bowler per inning
✓ No overs N and N+1 by same bowler (same inning)
✓ Bowler must be in playing XI
✓ < 5 bowlers permitted (flexibility)
```