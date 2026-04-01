# Task 2 — Dataset Identification and Verification

## Objective

Identify and verify the required raw IPL datasets needed for building Dataset v2 and ML pipelines.

## Work Completed

### 1. Ball-by-Ball Dataset Identified

**File**

```
New Data/deliveries_updated_ipl_upto_2025.csv
```

**Contains**

* matchId
* inning
* over
* ball
* batting_team
* bowling_team
* batsman
* non_striker
* bowler
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

**Status:** Valid and usable

### 2. Match Metadata Dataset Identified

**File**

```
New Data/matches_updated_ipl_upto_2025.csv
```

**Contains**

* matchId
* season
* venue
* team1
* team2
* winner
* toss_winner
* date
* city
* match_number
* outcome
* other metadata

**Status:** Valid and usable

### 3. Dataset Validation

A simple test script was executed to verify:

* files load correctly
* columns exist
* structure is consistent
* data is readable
* no missing core fields

**Result:**

* Ball-by-ball dataset verified
* Matches dataset verified
* Required columns present
* No blocking issues found

**Minor issues found**:

* 3 duplicate rows in deliveries (to be inspected later)
* 23 matches missing winner (to be inspected later)

No blocking issues found.

## Required Raw Data Confirmed

```
deliveries_updated_ipl_upto_2025.csv
matches_updated_ipl_upto_2025.csv
```

These datasets are sufficient to proceed with data standardization and feature engineering.

## Outcome

Task 2 is completed.

The required IPL datasets have been:

* identified
* inspected
* validated
* confirmed usable for pipeline development