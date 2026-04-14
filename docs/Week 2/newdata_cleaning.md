# New Data Cleaning Pipeline (Dataset v3_alpha)

## Overview

This document describes the data cleaning and preprocessing pipeline built for the IPL Cricket Intelligence Platform.

The goal of this stage is to transform raw IPL datasets into clean, standardized, and versioned parquet datasets that can be used for feature engineering, player stats, embeddings, and ML pipelines.

This stage ensures:

* consistent schema
* removal of invalid matches
* standardized column naming
* chronological ordering
* versioned datasets
* metadata tracking
* reproducible data pipeline

---

# Raw Datasets

## Matches Dataset

```
New Data/matches_updated_ipl_upto_2025.csv
```

Contains:

* matchId
* season
* venue
* team1
* team2
* winner
* date
* city
* toss details
* umpire details
* other metadata

---

## Deliveries Dataset

```
New Data/deliveries_updated_ipl_upto_2025.csv
```

Contains:

* matchId
* inning
* over
* ball
* batsman
* bowler
* non_striker
* batsman_runs
* extras
* isWide
* isNoBall
* Byes
* LegByes
* Penalty
* dismissal_kind
* player_dismissed
* date

---

# Pipeline Structure

```
data-service/

core/
    config.py
    metadata.py

pipeline/
    clean_matches.py
    clean_deliveries.py
    build_dataset.py
    player_stats_and_embeddings.py
    venue_embeddings.py

validate_raw_data.py
run_pipeline.py
newdata_cleaning.md
```

---

# Dataset Versioning

Dataset version:

```
v3_alpha
```

Output location:

```
ml-service/data/processed/v3_alpha/
```

Metadata location:

```
ml-service/data/metadata/v3_alpha/
```

---

# Pipeline Execution

Command:

```
py run_pipeline.py
```

---

# Pipeline Steps

## STEP 1: Clean Matches

### File

```
pipeline/clean_matches.py
```

### Operations Performed

1. Load raw matches dataset
2. Fix season format

Examples:

```
2020/21 → 2020
2016 → 2016
```

3. Handle missing values

```
winner_runs → 0
winner_wickets → 0
winner filled using eliminator
```

4. Drop unnecessary columns

Removed:

* event
* umpires
* toss decision
* city
* gender
* method
* outcome
* match_number
* extra metadata

5. Remove matches without winner

Reason:

* RL reward requires winner
* invalid matches removed

6. Convert types

```
date → datetime
winner_runs → int
winner_wickets → int
season → int
```

7. Sort matches by date

Ensures chronological order.

8. Create version folder if not present

```
ml-service/data/processed/v3_alpha/
```

9. Save parquet

```
clean_matches.parquet
```

10. Save metadata

```
clean_matches.json
```

---

# STEP 2: Clean Deliveries

### File

```
pipeline/clean_deliveries.py
```

### Operations Performed

1. Load deliveries dataset

2. Convert date to datetime

3. Fill missing values

```
isWide → 0
isNoBall → 0
player_dismissed → 0
Byes → 0
LegByes → 0
Penalty → 0
```

4. Create batsman runs

Byes and LegByes adjusted properly.

5. Drop unnecessary columns

Removed:

```
Byes
LegByes
over_ball
dismissal_kind
extras
```

6. Keep only innings 1 and 2

Removes super overs.

7. Handle penalty

```
isWide += Penalty
Penalty removed
```

8. Map innings

```
1 → 0
2 → 1
```

9. Convert types

```
batsman_runs → int
isWide → int
isNoBall → int
```

10. Remove washed-out matches

Using:

```
clean_matches.parquet
```

Only valid match IDs retained.

11. Sort chronologically

```
date
matchId
inning
over
ball
```

12. Remove duplicate rows

Ensures data integrity.

13. Save parquet

```
clean_deliveries.parquet
```

14. Save metadata

```
clean_deliveries.json
```

---

# Metadata Tracking

Each cleaned dataset produces metadata.

Example:

```
clean_matches.json
clean_deliveries.json
```

Contains:

* dataset version
* rows
* columns
* creation time
* source file
* output file

This enables:

* dataset tracking
* reproducibility
* debugging
* version control

---

# Pipeline Output

## Processed Data

```
ml-service/data/processed/v3_alpha/

clean_matches.parquet
clean_deliveries.parquet
```

---

## Metadata

```
ml-service/data/metadata/v3_alpha/

clean_matches.json
clean_deliveries.json
```

---

# Terminal Execution

Example run:

```
py run_pipeline.py
```

Output:

STEP 1: CLEAN MATCHES
Loading matches dataset...
Fixing season format...
Handling missing values...
Dropping unnecessary columns...
Removing matches without winner...
Converting types...
Sorting by date...
Checking if Folder is present...
Saving parquet...
Clean matches saved successfully.
Metadata saved

STEP 2: CLEAN DELIVERIES
Loading deliveries dataset...
Converting date...
Filling missing values...
Creating Batsman runs...
Dropping unnecessary columns...
Keeping only innings 1 and 2...
Properly Calculating..
Mapping innings...
Converting types...
Removing washed-out matches from deliveries...
Sorting by date...
Removing duplicate rows...
Saving parquet...
Clean deliveries saved successfully.
Metadata saved

Pipeline completed successfully

---

# Outcome

The new IPL datasets are now:

* cleaned
* standardized
* versioned
* chronologically sorted
* metadata tracked
* reproducible

Datasets are ready for:

```
Step 3: Player Stats Pipeline
```

---

# Next Step

```
pipeline/player_stats_and_embeddings.py
```

This will generate:

* player stats
* batting metrics
* bowling metrics
* embeddings-ready dataset

from:

```
clean_deliveries.parquet
clean_matches.parquet
```

---

# Status

Task 1: Feature specification — Completed
Task 2: Dataset identification — Completed
Task 3 (Cleaning pipeline) — Completed

Ready to proceed to Player Stats Pipeline.
