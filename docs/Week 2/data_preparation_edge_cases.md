## Data Integrity ✅

 Missing `matchId`.
 Missing `inning`.
 Missing `over`.
 Missing `batsman_runs`.
 Missing `isWide`.
 Missing `isNoBall`.
 Missing `batsman`.
 Missing `non_striker`.
 Missing `bowler`.
 Missing `player_dismissed`.
 Missing `venue` after merge.
 Missing `season` after map.
 Duplicate rows where same delivery repeated more than twice.
 `inning` values outside {0,1}.
 `over` negative.
 `over` greater than 19
 Non-integer `over`.
 `ball` negative.
 `ball` equal to 0.
 `ball` fractional non-scorebook values.
30. `ball` greater than 6 before filtering.
 `ball` stored as string.
 `batsman_runs` negative.
 `batsman_runs` > 6 before clipping.
 `isWide` negative.
 `isNoBall` negative.
37. `isNoBall` > 1 but not true duplicate no-balls.
 `Byes` negative.
 `LegByes` negative.
 `Penalty` negative.
 `Penalty` values other than 0 or 5.
 `batting_team` equals `bowling_team`.
43. Team names inconsistent across rows of same innings.
 `player_dismissed` null instead of `"Not Out"`.
 `player_dismissed` set to dismissed player on non-wicket rows.
 `date` malformed or unparsable.
47. `matchId` present in balls but absent in matches table.
 Same `matchId` mapped to multiple venues.
 Same `matchId` mapped to multiple seasons.
 Rows unsorted within innings before cumulative calculations.
 Mixed datatypes in numeric columns.
52. Infinite values in engineered numeric columns.
53. NaN introduced by division during normalization.
54. CSV export truncating float precision.
55. Hidden duplicate index values after repeats/resetting.
56. Multiple batters names for same striker row after duplicate aggregation.
57. Multiple bowlers for same ball after duplicate aggregation.
58. Negative `total_runs` after recomputation.
59. `total_runs` inconsistent with component extras.
60. Same innings containing rows from two matches due to bad merge.

## Friend Special Request

 SRH 287
 RCB 49

## Cricket Logic

 More than 10 wickets in an innings.
2. Wicket count increases on retired hurt rows treated as dismissal.
3. Wicket on wide/no-ball incorrectly treated as impossible type.
4. Legal ball count increases on wides.
5. Legal ball count increases on no-balls.
6. Over ends with fewer than 6 legal balls in completed innings.
7. Over contains more than 6 legal balls after processing.
8. Innings contains more than 120 legal balls in normal IPL innings.
9. Innings contains fewer than 120 legal balls without all-out/chase complete.
10. Chase innings continues after target already reached.
11. Chase innings target set to first innings score instead of +1.
12. First innings target non-zero.
13. Second innings target zero.
14. `runs_required` negative before innings termination.
15. `balls_remaining` negative.
16. `balls_remaining` increases within innings.
17. Over numbers skip backwards within innings.
18. Ball sequence decreases within over.
19. Same striker/non-striker never rotate despite singles/odd runs.
20. Bowler changes mid-over without explicit interruption.
21. Same bowler bowls consecutive overs beyond allowed spells not reflected.
22. Both teams batting in same innings.
23. Innings starts at over > 0.
24. Innings starts at ball > 1.
25. First innings wickets start above 0.
26. Second innings current score starts above 0.
27. Two wickets credited on one row without run-out context.
28. Five penalty runs assigned as batsman boundary.
29. Byes and leg-byes simultaneously positive on same delivery.
30. Wide and byes combination mishandled.
31. No-ball and byes combination mishandled.
32. Wide with batsman_runs > 0.
33. Impossible batsman_runs = 5 without overthrow context.
34. All-out innings continues with further deliveries.
35. Final wicket row still has non-dismissed striker continuing.
36. Powerplay phase assigned beyond over 6.
37. Death phase starts before over 16.
38. Last over runs for over 1 referencing previous innings.
39. Wickets fall on `"Not Out"` rows.
40. Match with only one innings retained.

## Feature Engineering

1. Duplicate aggregation sums `batsman_runs` across truly separate events incorrectly.
2. Duplicate aggregation keeps first batter/bowler when duplicates disagree.
3. `isNoBall > 1` converted into batsman runs incorrectly.
4. Wide repeats create extra rows but no-ball repeats do not.
5. `ball` recomputation based only on legal balls loses original event numbering.
6. Filtering `ball <= 6` removes valid extra deliveries instead of preserving event rows.
7. `legal_ball` recomputed twice with inconsistent dtype.
8. `balls_bowled` based on legal balls but `total_balls_in_over` based on all events.
9. `balls_remaining = 120 - balls_bowled` wrong for innings ending early.
10. `over_number = over + 1` wrong if `over` not zero-indexed.
11. `phase` mislabels over 15 boundary if condition order wrong.
12. `current_score` computed before later edits to `batsman_runs`.
13. `current_score` excludes later clipped batsman runs inconsistently.
14. `Penalty==5` reassigned into `batsman_runs`.
15. Adding `Byes` and `LegByes` into `batsman_runs` distorts batting stats.
16. Clipping `batsman_runs > 6` hides legitimate overthrow totals.
17. `is_wicket = player_dismissed != "Not Out"` fails for nulls/blanks.
18. `wickets_fallen` cumulative count includes retired hurt if encoded as player name.
19. `target` forward fill across innings if merge logic leaks.
20. `last_over_runs` for first over null/incorrectly zero-filled.
21. `is_boundary` based on modified `batsman_runs` after adding byes/leg-byes.
22. Byes of 4 misclassified as boundary.
23. `balls_since_boundary` uses transformed boundary definition, not true batsman boundary.
24. `balls_since_boundary` resets incorrectly on consecutive boundaries.
25. `balls_since_boundary` starts at -1 then later normalized.
26. Negative normalized `balls_since_boundary` values fed to LSTM.
27. `percentage_target_achieved` division by zero when target = 0.
28. `percentage_target_achieved` > 1 after chase complete.
29. `percentage_target_achieved` negative when score corrections occur.
30. `overs_bowled = balls_bowled/6` ignores partial over display semantics.
31. `current_run_rate` infinite when overs_bowled = 0.
32. `required_run_rate` infinite when overs_remaining = 0 and runs_required > 0.
33. `required_run_rate` negative when target surpassed.
34. Scaling `over = over/20` compresses zero-indexed over to max 0.95 not 1.0.
35. `sin_ball`,`cos_ball` built from recomputed ball values only 1..6, losing extra-ball context.
36. `sin_ball`,`cos_ball` undefined for filtered/missing balls.
37. Dropping `ball` removes interpretable sequencing column before modeling.
38. Dropping team/date columns loses potentially predictive context.
39. `season` unnormalized while other numerics normalized.
40. `venue` categorical left unencoded for LSTM.
41. Player names retained unencoded if not later removed externally.
42. `target/200` saturates values >200.
43. `current_score/200` >1 for scores above 200.
44. `last_over_runs/200` >1 for overs above 200 impossible but coding allows.
45. `balls_remaining/120` negative if balls_bowled >120.
46. `wickets_fallen/10` >1 if wicket count bug >10.
47. `batsman_runs/6` after clipping masks data issues.
48. `total_runs/6` can exceed 1 on 7+ run balls.
49. Export to CSV may coerce booleans to text.

## Statistical Boundaries

1. First innings score = 0.
2. First innings score = 1.
 First innings score at known IPL extreme high (287 by Sunrisers Hyderabad, 2024).
4. Chase score = 0.
5. Chase exactly equals target.
6. Chase exceeds target by large margin.
7. Team all out for single digits.
8. Innings with 10 wickets.
9. Innings with 0 wickets.
10. Over with 0 runs.
11. Over with 36 runs (six sixes legal balls).
12. Over exceeding 36 via no-balls/wides.
13. Consecutive maiden overs.
14. Consecutive boundary balls entire over.
15. No boundary entire innings.
16. Boundary on first ball of innings.
17. Boundary on last ball of innings.
18. `balls_since_boundary = 0`.
19. `balls_since_boundary = 120` theoretical no-boundary innings.
20. `balls_remaining = 120`.
21. `balls_remaining = 0`.
22. `balls_remaining` near zero with many runs required.
23. `required_run_rate = 0`.
24. Extremely high `required_run_rate` late chase.
25. `current_run_rate = 0`.
26. Extremely high `current_run_rate` after one scoring ball.
27. `percentage_target_achieved = 0`.
28. `percentage_target_achieved = 1`.
29. `percentage_target_achieved > 1`.
30. `batsman_runs = 0`.
31. `batsman_runs = 6`.
32. `total_runs = 0`.
33. `total_runs = 7+` on extra-filled delivery.
34. `last_over_runs = 0`.
35. `last_over_runs = 36+`.
36. `wickets_fallen = 0`.
37. `wickets_fallen = 10`.

## Temporal / Sequence

1. Rows not sorted by (`matchId`,`inning`,`over`,`ball`) before cumulative ops.
2. Duplicate rows inserted after sorting but before later cumulative ops.
3. Innings 2 rows appearing before innings 1 rows.
4. Over 10 appearing before over 9.
5. Ball 3 appearing before ball 2 within over.
6. Reset index omitted causing stale positional assumptions.
7. `current_score` decreases between consecutive rows.
8. `wickets_fallen` decreases between rows.
9. `balls_bowled` decreases between rows.
10. `balls_remaining` increases between rows.
11. `balls_since_boundary` decreases without a boundary.
12. `last_over_runs` changes within same over when intended constant carryover.
13. First row of innings inherits previous innings cumulative state.
14. First row of match innings 2 target missing until later rows.
15. Last row of innings not terminal state due truncated data.
16. Over transition from 5.6 to 7.1 skipping over 6.
17. Over transition repeats same over after next over started.
18. Match rows interleaved with another matchId.
19. Venue/season changes mid-match after merge.
20. Consecutive rows with same legal ball number after wides not handled consistently.
21. Filtering `ball<=6` after recompute removes sequence continuity checks.
22. Boundary row duplicated causing double reset of `balls_since_boundary`.
23. Chase target reached mid-over but later rows retained.
24. Wicket on final legal ball followed by same innings continuation without next batter.
25. End-of-innings row still shows positive balls_remaining after successful chase.
26. First over last_over_runs not initialized deterministically.
27. `total_balls_in_over` not resetting to 1 on new over.
28. `total_balls_in_over` skipping values within over.
29. `sin_ball`,`cos_ball` sequence not aligned with true event order after extras.
30. Exported CSV row order differing from in-memory sorted order.