## Data Integrity

✅ Missing `matchId`.
✅ Missing `inning`.
✅ Missing `over`.
✅ Missing `batsman_runs`.
✅ Missing `isWide`.
✅ Missing `isNoBall`.
✅ Missing `batsman`.
✅ Missing `non_striker`.
✅ Missing `bowler`.
✅ Missing `player_dismissed`.
✅ Missing `venue` after merge.
✅ Missing `season` after map.
✅ Duplicate rows where same delivery repeated more than twice.
✅ `inning` values outside {0,1}.
✅ `over` negative.
✅ `over` greater than 19.
✅ Non-integer `over`.
✅ `ball` negative.
✅ `ball` equal to 0.
✅ `ball` fractional non-scorebook values.
✅ `ball` greater than 6 before filtering.
✅ `ball` stored as string.
✅ `batsman_runs` negative.
✅ `batsman_runs` > 6 before clipping.
✅ `isWide` negative.
✅ `isNoBall` negative.
✅ `isNoBall` > 1 but not true duplicate no-balls.
✅ `Byes` negative.
✅ `LegByes` negative.
✅ `Penalty` negative.
✅ `Penalty` values other than 0 or 5.
✅ `batting_team` equals `bowling_team`.
✅ `player_dismissed` null instead of `"Not Out"`.
✅ `player_dismissed` set to dismissed player on non-wicket rows.
✅ `date` malformed or unparsable.
✅ `matchId` present in balls but absent in matches table.
✅ Same `matchId` mapped to multiple venues.
✅ Same `matchId` mapped to multiple seasons.
✅ Rows unsorted within innings before cumulative calculations.
✅ Mixed datatypes in numeric columns.
✅ Infinite values in engineered numeric columns.
✅ NaN introduced by division during normalization.
✅ CSV export truncating float precision. --> End Product is parquet, csv is only for testing
✅ Hidden duplicate index values after repeats/resetting.
✅ Multiple batters names for same striker row after duplicate aggregation.
✅ Multiple bowlers for same ball after duplicate aggregation.
✅ Negative `total_runs` after recomputation.
✅ `total_runs` inconsistent with component extras.
✅ Same innings containing rows from two matches due to bad merge.

## Friend Special Request

✅ SRH 287
✅ RCB 49

## Cricket Logic

✅ More than 10 wickets in an innings.
✅ Wicket on wide/no-ball incorrectly treated as impossible type. --> Wide for runout and stumped and nothing for noball
✅ Legal ball count increases on wides.
✅ Legal ball count increases on no-balls.
✅ Over contains more than 6 legal balls after processing.
✅ Innings contains more than 120 legal balls in normal IPL innings.
✅ Innings contains fewer than 120 legal balls without all-out/chase complete.
✅ Chase innings continues after target already reached.
✅ Chase innings target set to first innings score instead of +1.
✅ First innings target non-zero.
✅ Second innings target zero.
✅ `balls_remaining` negative.
✅ `balls_remaining` increases within innings.
✅ Over numbers skip backwards within innings.
✅ Ball sequence decreases within over.
✅ Same striker/non-striker never rotate despite singles/odd runs.
✅ Bowler changes mid-over without explicit interruption.
✅ Same bowler bowls consecutive overs beyond allowed spells not reflected.
✅ Innings starts at over > 0.
✅ Innings starts at ball > 1.
✅ First innings wickets start above 0.
✅ Second innings current score starts above 0.
✅ Two wickets credited on one row without run-out context.
✅ Byes and leg-byes simultaneously positive on same delivery.
✅ All-out innings continues with further deliveries.
✅ Final wicket row still has non-dismissed striker continuing.
✅ Powerplay phase assigned beyond over 6.
✅ Death phase starts before over 16.
✅ Last over runs for over 1 referencing previous innings.
✅ Wickets fall on `"Not Out"` rows.
✅ Match with only one innings retained.

## Feature Engineering

✅ Duplicate aggregation sums `batsman_runs` across truly separate events incorrectly.
✅ Duplicate aggregation keeps first batter/bowler when duplicates disagree.
✅ `isNoBall > 1` converted into batsman runs incorrectly.
✅ Wide repeats create extra rows but no-ball repeats do not.
✅ `ball` recomputation based only on legal balls loses original event numbering.
✅ Filtering `ball <= 6` removes valid extra deliveries instead of preserving event rows.
✅ `balls_bowled` based on legal balls but `total_balls` based on all events.
✅ `balls_remaining = 120 - balls_bowled` wrong for innings ending early.
✅ `over_number = over + 1` wrong if `over` not zero-indexed.
✅ `phase` mislabels over 15 boundary if condition order wrong.
✅ `current_score` computed before later edits to `batsman_runs`.
✅ `current_score` excludes later clipped batsman runs inconsistently.
✅ Adding `Byes` and `LegByes` into `batsman_runs` distorts batting stats.
✅ Clipping `batsman_runs > 6` hides legitimate overthrow totals.
✅ `is_wicket = player_dismissed != "Not Out"` fails for nulls/blanks.
✅ `wickets_fallen` cumulative count includes retired hurt if encoded as player name.
✅ `target` forward fill across innings if merge logic leaks.
✅ `last_over_runs` for first over null/incorrectly zero-filled.
✅ `is_boundary` based on modified `batsman_runs` after adding byes/leg-byes.
✅ Byes of 4 misclassified as boundary.
✅ `balls_since_boundary` uses transformed boundary definition, not true batsman boundary.
✅ `balls_since_boundary` starts at -1 then later normalized.
✅ Negative normalized `balls_since_boundary` values fed to LSTM.
✅ `percentage_target_achieved` division by zero when target = 0.
✅ `percentage_target_achieved` > 1 after chase complete.
✅ `percentage_target_achieved` negative when score corrections occur.
✅ `overs_bowled = balls_bowled/6` ignores partial over display semantics.
✅ `current_run_rate` infinite when overs_bowled = 0.
✅ `required_run_rate` infinite when overs_remaining = 0 and runs_required > 0.
✅ `required_run_rate` negative when target surpassed.
✅ Scaling `over = over/20` compresses zero-indexed over to max 0.95 not 1.0.
✅ `sin_ball`,`cos_ball` built from recomputed ball values only 1..6, losing extra-ball context.
✅ `sin_ball`,`cos_ball` undefined for filtered/missing balls.
✅ Dropping `ball` removes interpretable sequencing column before modeling.
✅ Dropping team/date columns loses potentially predictive context.
✅ `season` unnormalized while other numerics normalized.
✅ `venue` categorical left unencoded for LSTM.
✅ Player names retained unencoded if not later removed externally.
✅ `target/200` saturates values >200.
✅ `current_score/200` >1 for scores above 200.
✅ `last_over_runs/200` >1 for overs above 200 impossible but coding allows.
✅ `balls_remaining/120` negative if balls_bowled >120.
✅ `wickets_fallen/10` >1 if wicket count bug >10.
✅ `batsman_runs/6` after clipping masks data issues.
✅ `total_runs/6` can exceed 1 on 7+ run balls.
✅ Export to CSV may coerce booleans to text.

## Statistical Boundaries

✅ First innings score = 0.
✅ First innings score = 1.
✅ First innings score at known IPL extreme high (287 by Sunrisers Hyderabad, 2024).
✅ Chase score = 0.
✅ Chase exactly equals target.
✅ Chase exceeds target by large margin.
✅ Team all out for single digits.
✅ Innings with 10 wickets.
✅ Innings with 0 wickets.
✅ Over with 0 runs.
✅ Over with 36 runs (six sixes legal balls).
✅ Over exceeding 36 via no-balls/wides.
✅ Consecutive maiden overs.
✅ Consecutive boundary balls entire over.
✅ `balls_since_boundary = 0`.
✅ `balls_since_boundary = 120` theoretical no-boundary innings.
✅ `balls_remaining = 120`.
✅ `balls_remaining = 0`.
✅ `balls_remaining` near zero with many runs required.
✅ `required_run_rate = 0`.
✅ Extremely high `required_run_rate` late chase.
✅ `current_run_rate = 0`.
✅ Extremely high `current_run_rate` after one scoring ball.
✅ `percentage_target_achieved = 0`.
✅ `percentage_target_achieved = 1`.
✅ `percentage_target_achieved > 1`.
✅ `batsman_runs = 0`.
✅ `batsman_runs = 6`.
✅ `total_runs = 0`.
✅ `total_runs = 7+` on extra-filled delivery.
✅ `last_over_runs = 0`.
✅ `last_over_runs = 36+`.
✅ `wickets_fallen = 0`.
✅ `wickets_fallen = 10`.

## Temporal / Sequence

✅ Rows not sorted by (`matchId`,`inning`,`over`,`ball`) before cumulative ops.
✅ Duplicate rows inserted after sorting but before later cumulative ops.
✅ Innings 2 rows appearing before innings 1 rows.
✅ Over 10 appearing before over 9.
✅ Ball 3 appearing before ball 2 within over.
✅ Reset index omitted causing stale positional assumptions.
✅ `current_score` decreases between consecutive rows.
✅ `wickets_fallen` decreases between rows.
✅ `balls_bowled` decreases between rows.
✅ `balls_remaining` increases between rows.
✅ `balls_since_boundary` decreases without a boundary.
✅ `last_over_runs` changes within same over when intended constant carryover.
✅ First row of innings inherits previous innings cumulative state.
✅ First row of match innings 2 target missing until later rows.
✅ Last row of innings not terminal state due truncated data.
✅ Over transition from 5.6 to 7.1 skipping over 6.
✅ Venue/season changes mid-match after merge.
✅ Consecutive rows with same legal ball number after wides not handled consistently.
✅ Boundary row duplicated causing double reset of `balls_since_boundary`.
✅ Chase target reached mid-over but later rows retained.
✅ Wicket on final legal ball followed by same innings continuation without next batter.
✅ End-of-innings row still shows positive balls_remaining after successful chase.
✅ First over last_over_runs not initialized deterministically.
✅ `total_balls_in_over` not resetting to 1 on new over.
✅ `total_balls_in_over` skipping values within over.
✅ `sin_ball`,`cos_ball` sequence not aligned with true event order after extras.