# Decisions log

A running record of the choices made in this project, why, and their status.
(This file did not exist before Stage 5a. It was created then and seeded from the
decisions of Stages 1 to 4, so earlier entries are a summary.)

Status: **settled** = decided and used; **open** = waiting for the team's decision;
**post-hoc** = decided after seeing validation results (not a clean test of that choice).

---

## Stages 1 to 4 (summary)

| Stage | Decision | Why | Status |
|---|---|---|---|
| 1 | Match data from football-data.co.uk, one CSV per season, 2015-16 to 2025-26 | Free, downloadable, complete (380 matches per season) | settled |
| 1 | Player data: nothing downloaded; recommended option is the vaastav FPL archive on GitHub | Live FPL API only shows the current season. Decision postponed to Stage 8 | open (Stage 8) |
| 2 | `TEAM_NAME_FIXES` is empty | Stage 1 found no club spelled two ways | settled |
| 2 | Odds columns go to `data/processed/odds.csv`, never used as model features | Course plan; odds used only as a benchmark | settled |
| 2 | Train = 2015-16 to 2024-25, test = 2025-26; tuning scheme = fit on 2015-16 to 2022-23, validate on 2023-24 and 2024-25 | Time-based split, no leakage | settled |
| 3 | 2020-21 (no fans) is kept in the data | EDA showed a real outlier in home advantage; Stage 5a tested what to do with it (see below) | settled in 5a |
| 4 | Elo constants K=30, H=50, keep=0.9, promoted offset=-100, tuned on training seasons only | Best on 2015-16 to 2022-23, also beats the baseline on validation | settled |
| 4 | `keep=0.9` is the edge of the tuning grid; the EDA-based 0.75 is only 0.0003 worse | Data cannot tell them apart | open (low priority) |
| 4 | Teams with no match in the previous season start with Elo 1400 and an empty form history | Avoid stale history for promoted/returning teams | settled |
| 4 | Matches on the same date are treated as simultaneous | Stricter leakage protection | settled |
| 4 | Head-to-head features kept in the table | Plan said to test them on validation data | settled in 5a; excluded from the primary models by the amendment below (post-hoc) |

---

## Stage 5a: baselines and Logistic Regression (original v1 analysis)

All choices in this section were made on the validation seasons (2023-24, 2024-25, 760 matches)
with models fitted on 2015-16 to 2022-23, using the frozen v1 feature sets
(`feature_sets_v1.json`). The 2025-26 test season was not touched.

**Noise rule used:** adopt a change only if the 95% paired-bootstrap interval of the
validation log-loss improvement excludes zero. If it does not, keep the simpler option and
say the data cannot tell the options apart. One standard error of accuracy on 760 matches is
about 1.8 points, so accuracy alone never decides anything. With about 9 comparisons, roughly
one "clearly better" result is expected by chance at the 95% level, so a borderline hit is weak
evidence.

| Question | Result (improvement in validation log loss vs the default; positive = better) | v1 decision |
|---|---|---|
| Feature set (tuned C each) | FULL (34, default) 0.9608; REDUCED (27) +0.0016 [-0.0007, +0.0040]; SMALL (3) -0.0055 [-0.0163, +0.0045] | The data cannot tell them apart; by the rule the simplest, SMALL, was picked. **Superseded by Amendment 1 below** |
| Head-to-head features | Without h2h: +0.0025 [+0.0004, +0.0048] | Interval just excludes zero, but it is borderline |
| `season_no_fans` | Exclude the feature: -0.0033 [-0.0069, -0.0002] (clearly worse). Keep it but drop 2020-21 rows: +0.0017 [-0.0013, +0.0047] (cannot tell) | **Keep the feature as it is** (the default). Dropping 2020-21 rows is not needed |
| Training window (from 2022-23 back) | Last 3: -0.0254 [-0.0375, -0.0140] (clearly worse); last 5: -0.0018 [-0.0096, +0.0054] (cannot tell); all 8 with recency weights: -0.0153 [-0.0241, -0.0071] (clearly worse) | **Use all seasons, no weights** |
| Warm-up rows in training | Exclude: -0.0140 [-0.0211, -0.0075] (clearly worse) | **Keep warm-up rows.** The diagnostic showed the effect comes from `league_avg_goals_prev380`, not from the warm-up rows themselves (without that feature: -0.0003 [-0.0021, +0.0015]) |
| Regularization C (FULL v1) | C = 0.001, 0.01, 0.1, 1, 10 gave validation log loss 0.9615, 0.9608, 0.9669, 0.9690, 0.9699 | C = 0.01 for FULL |
| `class_weight="balanced"` | Predicts 104 draws instead of 1, accuracy 53.3% vs 55.4%, log loss 0.9673 vs 0.9608 | Keep the unweighted model (honest probabilities) |

The full v1 table is `reports/ablations_logreg.csv` (kept unchanged as the record). The v1 rule
result (SMALL, C = 10, validation log loss 0.9663) is **no longer the saved model**.

---

## Amendment 1, 2026-11-14: after the drift diagnostic

**These three decisions are POST-HOC and DIAGNOSTIC-DRIVEN.** They were made after the validation
results and the `league_avg_goals_prev380` diagnostic had been seen. The validation scores are
therefore not a clean test of these choices. To limit the damage: the primary feature set was fixed
before any new numbers were looked at; the justification does not rest only on validation scores;
`feature_sets_v1.json` is kept byte-for-byte unchanged (its hash is checked) and the change is
written as `feature_sets_v2.json` with its date and reason.

| # | Decision | Justification | Status |
|---|---|---|---|
| 1 | **Primary Logistic Regression feature set = ELO2** (`elo_diff` + `abs_elo_diff`). FULL (v2, 33 features) stays as a reference row for Stage 6 | The noise rule (the feature sets were indistinguishable, so keep the simpler one) and the extrapolation risk of the league-average feature. **Post-hoc, diagnostic-driven** | settled (post-hoc) |
| 2 | **`league_avg_goals_prev380` is removed from every classifier feature set** (Logistic Regression, Random Forest, XGBoost). It stays in `features.csv` | It is far outside its training range in the validation seasons (mean +4.5 training standard deviations, 73% of validation rows above 3). A level-neutral form (team goals relative to the league average) will be tested in Stage 5d. **Post-hoc, diagnostic-driven** | settled (post-hoc) |
| 3 | **Head-to-head features are kept only in FULL**, excluded from the primary models | Weak evidence: a borderline result (interval lower end +0.0004) among about 9 comparisons. **Post-hoc, diagnostic-driven** | settled (post-hoc) |

**Feature sets in `feature_sets_v2.json`:** FULL = 33 features (everything except the league average;
keeps head-to-head; reference row), REDUCED = 24 (v1 REDUCED without the league average and
head-to-head), ELO2 = 2 (primary). The v1 set SMALL is retired.

**Diagnostics added (not used to choose anything):**
- **Drift table** (`reports/drift_validation.csv`): all 34 features, validation mean z and maximum |z|
  in training standard deviations. By the rule "flag max |z| above 3", 22 of 34 are flagged, including
  both ELO2 features, but only `league_avg_goals_prev380` drifts systematically (mean z +4.51, 73% of
  validation rows above 3). Every other feature has a mean z between -0.38 and +0.45 and at most 4% of
  rows above 3. `elo_diff` (max 3.17) and `abs_elo_diff` (max 3.82) have about 2 and 6 unusual rows out of
  760. **To be repeated for the test features in Stage 6, labelled a diagnostic, not a tuning step.**
- **Home-win check per validation season:** 2023-24 home teams won 46.1%, ELO2 predicted 45.2% (-0.9
  points), bookmaker 44.6%. 2024-25 home teams won only 40.8% (EDA 02), ELO2 predicted 45.2% (+4.4),
  bookmaker 44.6% (+3.8), class frequencies 44.8% (+4.0). Every predictor over-predicted home wins in
  2024-25; one standard error for 380 matches is about 2.5 points, so the gap is under two standard errors.
  The home-win over-optimism in the calibration plot is mostly that season, not clear evidence of a model flaw.

**Results after the amendment (validation seasons, fit on 2015-16 to 2022-23):**

| Model | Features | C | Validation log loss | Accuracy |
|---|---|---|---|---|
| Class frequencies | none | | 1.0675 | 43.4% |
| Bookmaker | | | 0.9400 | 56.8% |
| FULL reference (v2) | 33 | 0.01 | 0.9577 | 55.5% |
| **ELO2 primary** | 2 | 10 | **0.9582** | 55.4% |

ELO2 vs FULL: -0.0006 [-0.0116, +0.0102] (cannot tell apart). ELO2 vs class frequencies: +0.1093
[+0.0784, +0.1402]. ELO2 vs bookmaker: the bookmaker is clearly better by 0.0182 [0.0066, 0.0292].
Both models were refit on 2015-16 to 2024-25 and saved (`models/logreg.joblib` = ELO2 primary,
`models/logreg_full_reference.joblib` = FULL reference); neither was scored on the test season.

---

## Stage 5b: Random Forest (2026-11-14)

Same protocol as 5a: all choices on the validation seasons (2023-24, 2024-25) with models fitted on 2015-16 to
2022-23; the 2025-26 test season was not touched; the same noise rule (adopt a change only if the 95%
paired-bootstrap interval of the log-loss improvement excludes zero, otherwise keep the simpler option).

**Frozen before any forest result was seen:** `data/processed/feature_sets_v3.json`, SHA-256
`9ca210478c2e9fa25c254e6db557ac7c578a014e366cf4a686eb72fc86b4e946` (also stored in `feature_sets_v3.sha256` and
checked by `src/check_rf.py`). v1 and v2 are unchanged (hashes checked). Sets: FULL_v2 (33, keeps head-to-head),
REDUCED (24), ELO2 (2), DIFF (16 level-neutral features: home_X minus away_X for 12 statistics, plus `elo_diff`,
`abs_elo_diff`, `season_no_fans`, `home_matches_played`). DIFF is same-row arithmetic computed in code, so
`features.csv` is unchanged.

**Hyperparameters** (grid of 20 on the default FULL_v2 features, 500 trees): `min_samples_leaf` = 80,
`max_features` = 0.5, `max_depth` = 8. **`min_samples_leaf` = 80 is the edge of the grid**, so a larger value might
be even better. The whole grid is flat (validation log loss 0.9649 to 0.9759); training log loss falls from 0.93 to
0.75 as leaves get smaller while validation does not improve (overfitting).

**Validation results (760 matches):**

| Predictor | Accuracy | Log loss |
|---|---|---|
| Class frequencies | 43.4% | 1.0675 |
| Bookmaker | 56.8% | 0.9400 |
| ELO2 Logistic Regression (primary LR) | 55.4% | 0.9582 |
| FULL reference Logistic Regression | 55.5% | 0.9577 |
| Random Forest, default (FULL_v2) | 55.9% | 0.9649 |
| Random Forest, chosen by the rule (ELO2 features) | 54.6% | 0.9733 |

Random Forest (default) vs ELO2 LR: -0.0067 [-0.0135, +0.0006] (cannot tell apart, leaning worse); vs bookmaker:
the bookmaker is clearly better by 0.0249 [0.0131, 0.0359]. The forest predicted 0 draws.

| Experiment (vs the default forest) | Result (improvement; positive = better) | Verdict |
|---|---|---|
| a. Feature set | REDUCED +0.0002 [-0.0017, +0.0022]; ELO2 -0.0084 [-0.0169, +0.0005]; DIFF +0.0063 [-0.0006, +0.0134] | None clearly better or worse. **By the rule the simplest, ELO2, is chosen** (flagged below) |
| b. Head-to-head | Without: -0.0005 [-0.0018, +0.0009] | Cannot tell; simpler "without" (moot for the chosen ELO2 set) |
| c. `season_no_fans` | Exclude: 0.0000 [-0.0012, +0.0013] | Cannot tell; simpler "exclude" (moot for ELO2) |
| d. Training window | Last 3 (extra row, added because CLAUDE.md asks for it): -0.0047 [-0.0125, +0.0032]; last 5: -0.0012 [-0.0058, +0.0034]; recency weights: -0.0008 [-0.0057, +0.0039] | **Keep all 8 seasons, no weights** (none clearly better) |
| e. Warm-up rows | Exclude: -0.0000 [-0.0018, +0.0018] | **Keep warm-up rows.** Unlike Logistic Regression, the forest was not affected by `league_avg_goals_prev380` (it is not in the sets) |

**Rule outcome and the problem with it.** The combined configuration is a forest on `elo_diff` + `abs_elo_diff`
only (validation log loss 0.9733). On identical inputs it is **clearly worse than the ELO2 Logistic Regression**:
-0.0151 [-0.0239, -0.0056]. "Keep the simpler option" therefore picks a model that a simpler model dominates.
(The ELO2 interval against the default only just contains zero: upper end +0.0005.) DIFF has the lowest loss
(0.9586) and is the best set in both validation seasons (0.9227 and 0.9946), but not clearly better.

**Saved models** (not scored on the test set): `models/random_forest.joblib` (the rule's choice, ELO2 features) and
`models/random_forest_full_reference.joblib` (the default FULL_v2 forest, a reference row for Stage 6, as in 5a),
0.8 MB each (limit 50 MB). Validation probabilities of all predictors: `reports/val_predictions_5b.csv`.

**Understanding results (diagnostics, not used to choose):** permutation importance on validation puts `elo_diff`
first by a wide margin (+0.077 log loss when shuffled), then `home_elo` (+0.015); shuffling 12 of 33 features does not hurt
at all. Impurity importance over-rates noisy, finely-valued shot features (for example `away_shots_against_10`: 0.036
impurity share, +0.0013 permutation). The partial-dependence curve of the ELO2 forest follows the Logistic Regression
curve with some zigzags (noise); no clear non-linear shape was found.

---

### Stage 5b: decisions approved after review (2026-11-14)

| # | Decision | Status |
|---|---|---|
| 1 | **Primary Random Forest = ELO2** (`elo_diff` + `abs_elo_diff`, `min_samples_leaf` 80, `max_features` 0.5, `max_depth` 8), as chosen by the noise rule. `models/random_forest.joblib` | settled |
| 2 | **The FULL_v2 forest is kept as the reference model** saved alongside it, matching the Stage 5a pair (`models/random_forest_full_reference.joblib`). Both are scored on the test set together in Stage 6 | settled |
| 3 | **The `min_samples_leaf` grid is NOT extended** (80 stays the edge of the grid; the time is better spent on Stage 5c) | settled |

**DIFF diagnostic (recorded now, before any test result exists).** The DIFF feature set had the best validation score
among the forests (log loss 0.9586, against 0.9649 for FULL_v2 and 0.9733 for ELO2; best in both validation seasons:
0.9227 in 2023-24 and 0.9946 in 2024-25), but the difference against the default is **within noise** (+0.0063
[-0.0006, +0.0134]). It will be scored on the test set in Stage 6 **as a diagnostic, labelled as such, and not treated as
a competing primary model.** To keep it a diagnostic and not a tuning step, its configuration is fixed now and must not be
changed after the test result is seen: feature set `DIFF` from `feature_sets_v3.json` (16 features), 500 trees,
`min_samples_leaf` 80, `max_features` 0.5, `max_depth` 8, `random_state` 42, refit on all ten training seasons
(2015-16 to 2024-25), warm-up rows included, no sample weights, no re-tuning.

**The 2024-25 pattern (recorded as an observation, not a decision).** The ELO2 Logistic Regression beat all four forests in
2024-25 (log loss 0.9852, against 0.9946 for the DIFF forest, 0.9990 for the FULL_v2 forest, 0.9998 for the REDUCED forest and
1.0084 for the ELO2 forest). This is the season in which home teams won only **40.8%** of matches (EDA 02; 46.1% in 2023-24).
Every predictor was worse in 2024-25 than in 2023-24 (bookmaker 0.9092 to 0.9708, ELO2 Logistic Regression 0.9313 to 0.9852,
FULL_v2 forest 0.9308 to 0.9990, ELO2 forest 0.9382 to 1.0084, even the class-frequency baseline 1.0540 to 1.0809), and every one
over-predicted home wins in 2024-25 (average predicted home-win probability 44.6% to 45.4%, against 40.8% real). The forests lost more than the
Logistic Regression (+0.068 and +0.070 against +0.054), so the Logistic Regression's edge is clear only in that season: ELO2
Logistic Regression vs ELO2 forest +0.0233 [+0.0102, +0.0362] in 2024-25 and +0.0070 [-0.0059, +0.0201] in 2023-24; vs the
FULL_v2 forest +0.0139 [+0.0035, +0.0238] in 2024-25 and -0.0005 [-0.0101, +0.0096] in 2023-24. **Caveats:** the link to the low
home-win rate is plausible but not proven (it is consistent with a low-home-win season hurting all predictors, and the more
variable forest probabilities most); splitting the validation set by season was done after seeing the results, so this is a
post-hoc observation based on 380 matches per season.

---

## Stage 5c: XGBoost (2026-11-14)

Same protocol as 5a and 5b: all choices on the validation seasons (2023-24, 2024-25) with models fitted on 2015-16 to
2022-23; the 2025-26 test season was not touched; feature sets reused from `feature_sets_v3.json` (no v4; hashes checked).

**Corrections and readings made at the start of the stage (the prompt and the record differ):**
- The stage prompt said head-to-head "helped LR slightly". The Stage 5a record says the opposite: removing it was better for the
  Logistic Regression, +0.0025 [+0.0004, +0.0048] (borderline). We follow the recorded result (head-to-head stays in FULL only).
- "All 8 seasons beat the last 3, recency hurt" holds for the Logistic Regression; for the forests no window was clearly different.
- **Early stopping** is done in two steps, always on a time-based split inside the data a model may use: fit on the earlier seasons
  of the window, stop on its last two seasons (for the default: fit on 2015-16 to 2020-21, stop on 2021-22 and 2022-23) to find the
  number of trees; the scored model is then refit on the WHOLE window with that number of trees. So "all 8 fit seasons" are used for
  fitting and the validation seasons are never used for stopping. The inner-only model's validation score is kept in the grid for
  information (it is slightly worse: 0.9758 against 0.9699 for the default). The two final models choose their tree count by
  stopping on 2023-24 and 2024-25 and are then refit on all ten seasons; they are never scored on validation or test again.
  (`reports/xgb_early_stopping_audit.json` lists every early-stopping run; `src/check_xgb.py` checks them.)
- "Same noise rule" was read as including the tie-break of 5a and 5b: adopt a change only if the 95% interval of the improvement
  excludes zero, otherwise keep the simpler option.
- **SHAP** was computed with XGBoost's own exact TreeSHAP (`pred_contribs`), with our own beeswarm plot, so no new package was
  needed (`requirements.txt` unchanged; the `shap` package is not installed).

**Hyperparameters** (grid of 12 on FULL_v2, up to 2000 trees allowed, `subsample` = `colsample_bytree` = 0.8, `reg_lambda` = 1,
`reg_alpha` = 0): `max_depth` 3, `learning_rate` 0.1, `min_child_weight` 20; early stopping chose **25 trees**
(here a "tree" means a boosting round; XGBoost builds one small tree per class in every round, so 25 rounds are 75 small trees). All three are at the
**edge of the grid** (smallest depth, largest learning rate, largest `min_child_weight`), so a simpler or more regularised model might
do even better. The grid is flat (validation log loss 0.9699 to 0.9843); deeper trees and small `min_child_weight` overfit (training
log loss down to 0.735 with no validation gain).

**Validation results (760 matches):**

| Predictor | Accuracy | Log loss | Draws predicted |
|---|---|---|---|
| Class frequencies | 43.4% | 1.0675 | 0 |
| Bookmaker | 56.8% | 0.9400 | 0 |
| ELO2 Logistic Regression | 55.4% | 0.9582 | 0 |
| FULL reference Logistic Regression | 55.5% | 0.9577 | 1 |
| Random Forest ELO2 | 54.6% | 0.9733 | 0 |
| Random Forest FULL_v2 | 55.9% | 0.9649 | 0 |
| XGBoost default (FULL_v2) | 55.3% | 0.9699 | 0 |
| XGBoost chosen by the rule (ELO2 features) | 54.2% | 0.9714 | 8 |

Paired bootstrap (improvement in log loss; positive = XGBoost better): default vs ELO2 LR -0.0117 [-0.0208, -0.0019] (clearly
worse); vs bookmaker -0.0299 [-0.0420, -0.0169] (bookmaker clearly better); vs Random Forest ELO2 +0.0035 [-0.0070, +0.0134]; vs
Random Forest FULL_v2 -0.0050 [-0.0103, +0.0004]. Chosen vs ELO2 LR -0.0132 [-0.0224, -0.0038]; vs bookmaker -0.0315 [-0.0449, -0.0173];
vs Random Forest ELO2 +0.0019 [-0.0030, +0.0069]; vs Random Forest FULL_v2 -0.0065 [-0.0161, +0.0025].

| Experiment (vs the default XGBoost) | Result (improvement; positive = better) | Verdict |
|---|---|---|
| a. Feature set (max_depth re-tuned) | REDUCED -0.0017 [-0.0064, +0.0028]; ELO2 -0.0016 [-0.0113, +0.0083]; DIFF +0.0048 [-0.0034, +0.0135] | None clearly different. **By the rule the simplest, ELO2, is chosen** (flagged below) |
| b. Head-to-head | Without: +0.0001 [-0.0028, +0.0032] | Cannot tell; simpler "without" (moot for ELO2) |
| c. `season_no_fans` | Exclude: -0.0003 [-0.0031, +0.0026] | Cannot tell; simpler "exclude" (moot for ELO2) |
| d. Training window | Last 3 (extra row, CLAUDE.md pitfall): -0.0120 [-0.0231, -0.0008] (clearly worse); last 5: -0.0037 [-0.0103, +0.0027]; recency weights: -0.0072 [-0.0144, 0.0000] (the interval ends at zero) | **Keep all 8 seasons, no weights** |
| e. Warm-up rows | Exclude: +0.0015 [-0.0038, +0.0067] | **Keep warm-up rows** (cannot tell) |
| f. Monotone constraint on `elo_diff` | -0.0004 [-0.0047, +0.0041] | **Unconstrained** (cannot tell) |

**Monotone constraint, what we found.** In XGBoost's multi-class mode the constraint is applied to every class's score, so it does
not guarantee that the Home-win probability rises with `elo_diff` (it depends on the three scores relative to each other). Checked
empirically: even with `elo_diff` as the only Elo feature (ELO2), 9 of 24 steps of the Home-win curve still go down (11 of 24
unconstrained); in FULL_v2 the unconstrained `home_elo` and `away_elo` can bypass it (5 of 24 down). The constraint did not change the
validation score on any set (FULL_v2 -0.0004, ELO2 +0.0008 [-0.0046, +0.0059], DIFF +0.0007 [-0.0045, +0.0058]; the last two are
diagnostics, not used for selection). A true monotone Home-win probability would need a different construction (for example a
separate binary home-win model with a +1 constraint); not tried.

**Rule outcome.** The combined configuration is a boosted model on `elo_diff` + `abs_elo_diff` only (validation log loss 0.9714,
57 trees), clearly worse than the ELO2 Logistic Regression (-0.0132 [-0.0224, -0.0038]), as the ELO2 forest was in 5b. The default
FULL_v2 model is also clearly worse than it (-0.0117 [-0.0208, -0.0019]). Boosting did not beat the simple model or the bookmaker.

**Saved models** (not scored on the test set): `models/xgboost.joblib`, `models/xgboost.ubj` and `models/xgboost_config.json` (the rule's
choice, ELO2 features, 62 trees after the final-period early stopping) and, as in 5a and 5b, `models/xgboost_full_reference.*` (the default
FULL_v2 model, 30 trees). All files are under 0.2 MB. Validation probabilities: `reports/val_predictions_5c.csv`.

**Understanding results (diagnostics, not used to choose):** three importances of the default model disagree on the minor features.
`elo_diff` is first on all three (gain 17.4%, mean |SHAP| 34.4%, permutation +0.0795 log loss). `season_no_fans` has a high gain (3.0%,
rank 9: the trees used it to isolate 2020-21 while training) but permutation importance 0.0000 (it is 0 in every validation match);
`away_ppg_5` is never used (gain 0) and `h2h_n` has a gain rank of 32. 13 of 33 features do not hurt when shuffled. The boosting curve
for stumps reaches its lowest validation log loss after about 107 trees (0.9699) and gets worse by 500 (0.9916) while the training loss
keeps falling to 0.896. Probabilities are not over-sharpened: the largest predicted Home-win probability is 0.81 (default), against
0.93 for the ELO2 Logistic Regression and 0.88 for the bookmaker.

---

### Stage 5c: decisions approved after review (2026-11-14)

| # | Decision | Status |
|---|---|---|
| 1 | **Primary XGBoost = ELO2** (`elo_diff` + `abs_elo_diff`, `max_depth` 3, `learning_rate` 0.1, `min_child_weight` 20), as chosen by the noise rule: `models/xgboost.joblib` / `.ubj`. **The FULL_v2 XGBoost is kept as the reference for Stage 6**, matching the 5a and 5b pair pattern (`models/xgboost_full_reference.*`). Both are scored on the test set together in Stage 6 | settled |
| 2 | **The DIFF XGBoost is a Stage 6 diagnostic, together with the DIFF Random Forest.** It is labelled as a diagnostic, is not a competing primary model, and its configuration is fixed now so that it cannot become a tuning step: DIFF set (16 features, `feature_sets_v3.json`), `max_depth` 3, `learning_rate` 0.1, `min_child_weight` 20, `subsample` 0.8, `colsample_bytree` 0.8, `reg_lambda` 1, `reg_alpha` 0, number of rounds chosen by early stopping (fit on the earlier seasons, stop on 2023-24 and 2024-25), then refit on all ten training seasons (2015-16 to 2024-25), warm-up rows included, no sample weights, no re-tuning. (Why it is a diagnostic: it scored best of the four sets, 0.9651, but within noise: +0.0048 [-0.0034, +0.0135]; best in both validation seasons.) | settled |
| 3 | **The hyperparameter grid is NOT extended** (depth 3, `learning_rate` 0.1 and `min_child_weight` 20 stay at the edge of the grid) | settled |
| 4 | **No two-model monotone construction.** Softprob objective doesn't allow a truly monotone constraint on Home-win probability; a binary two-model workaround exists but was out of scope for this project. (The monotone constraint on `elo_diff` stays out of the model: it changed nothing on the validation score and did not make the Home-win curve monotone.) | settled |
| 5 | **Both unrequested extras are kept:** the FULL_v2 reference model and the "last 3 seasons" row in the training-window experiment | settled |

---

## Stage 5d: Poisson regression for goals (2026-11-14)

Same protocol: all choices on the validation seasons (2023-24, 2024-25), models fitted on 2015-16 to 2022-23, the 2025-26 test
season not used for fitting or scoring. The one place the 2025-26 file is opened is `load_fixtures_2025_26()` in `src/poisson.py`:
it reads the FIXTURE LIST only (match id, date, the two team names, via `usecols`); no score or result column is ever loaded
(checked in `src/check_poisson.py`). Stage 7 needs the fixtures.

**Wording corrected at the start of the stage:**
- EDA 03 showed an *excess* of 0-goal and 4+ goal results and variance 1.13 to 1.15 times the mean (over-dispersion), for each side's
  goals separately. It did not measure the joint scores 0-0 and 1-1, so "under-dispersion at 0-0 and 1-1" was not an EDA finding;
  the Dixon-Coles premise was tested in this stage instead (see below).
- "Cross-validated deviance" is a single held-out validation split, not cross-validation.
- `league_avg_goals_prev380` "exactly the signal a goal model should track" is a hypothesis, tested in experiment b.
- "League average over the same window": the league average of the same quantity over the same date range as the team's own window;
  for venue features goals conceded are divided by the OTHER side's scoring average (what a home team concedes is what away teams
  score). Every `_rel` feature is built from the same prior matches as its parent (all 8 parents recomputed and compared with
  `features.csv`; perturbation tests pass). `features.csv` is unchanged; `features_goals.csv` has the same 4,180 rows.
- P(3+ goals) at lambda 1.55 is 0.204 (not 0.21). The "bivariate Poisson grid" is the product of two independent Poissons
  (plus Dixon-Coles), not the Karlis-Ntzoufras bivariate Poisson with a shared component. `is_home` only exists in the stacked table
  (built in code), not in the one-row-per-match CSV.
- "Simpler" = fewer parameters (stacked B before independent A); primary metric = Poisson deviance per match for changes to the
  rates and match-outcome log loss for Dixon-Coles (it changes only the joint, not the rates).

**Three models on the validation seasons (760 matches).** Deviance is per team-goal count (lower is better):

| Model | MAE home / away / total | Deviance | Outcome log loss | Accuracy | Draws predicted |
|---|---|---|---|---|---|
| A independent Poisson | 0.984 / 0.896 / 1.360 | 1.1132 | 0.9565 | 55.7% | 0 |
| B stacked Poisson | 0.982 / 0.897 / 1.356 | 1.1140 | 0.9566 | 55.4% | 0 |
| C stacked + Dixon-Coles (rho -0.0294) | same rates as B | 1.1140 | 0.9567 | 55.4% | 0 |
| ELO2 Logistic Regression | n/a | n/a | 0.9582 | 55.4% | 0 |
| Bookmaker | n/a | n/a | 0.9400 | 56.8% | 0 |

A vs B by deviance: B better by -0.0017 [-0.0114, +0.0075] (cannot tell, so the simpler B). C vs B by outcome log loss: -0.0002
[-0.0012, +0.0010] (cannot tell, so rho = 0). All three Poisson variants vs ELO2 LR: about +0.0016 [-0.0038, +0.0072]
(cannot tell); vs the bookmaker the bookmaker is clearly better by about 0.0166 [0.0055, 0.0276].

**Dixon-Coles premise, tested.** Observed / expected counts under independent Poisson, fit seasons (in sample): 0-0 1.056, 1-0 1.024,
0-1 0.876, 1-1 0.971: close to 1, and rho = -0.029 improves the fit-season log-likelihood by only 0.77 (likelihood-ratio statistic
1.53, below 3.84). On the validation seasons the default model expected far more 0-0 than happened (27 observed against 54 expected)
because its rates were too low for the higher-scoring seasons (see b). Dixon-Coles is not adopted.

| Experiment (vs the chosen B) | Result (improvement; positive = better) | Verdict |
|---|---|---|
| a. Features | raw goal averages: -0.0020 [-0.0077, +0.0035] (deviance) | Cannot tell: the drift of raw goal averages does not hurt this model by itself (keep level-neutral) |
| b. League scoring level | left out: +0.0324 [+0.0227, +0.0430]; as a fixed offset (log league level, coefficient 1): +0.0670 [+0.0280, +0.1097] | **Including `league_avg_goals_prev380` as a free feature HURTS** (its fitted weight is slightly negative, and it extrapolates 4.5+ standard deviations in validation); the **offset is clearly best** |
| c. Training window (extra row last 3 included) | last 3: +0.0717 [+0.0445, +0.1005]; last 5: +0.0510 [+0.0350, +0.0674]; recency: +0.0499 [+0.0344, +0.0665] | **Last 3 seasons clearly better than the default, by the rule** (see the confound below) |
| d. Dixon-Coles rho | fitted on the fit seasons: -0.0002 [-0.0012, +0.0010]; diagnostic fit + validation: rho -0.0313, same log loss | **rho = 0** |
| e. Team fixed effects (L2, alpha 1) | +0.0051 [-0.0211, +0.0306] deviance, but outcome log loss worse (0.9725 vs 0.9566); the same penalised model without team effects 1.1116 vs 1.1115 with them | **Elo and relative-goal features alone suffice**; alpha is at the edge of its grid and was chosen on validation |

**Chosen by the rule:** stacked Poisson (log link), level-neutral `_rel` features, league level as a fixed offset, trained on the last
3 seasons, rho = 0. Combined validation deviance 1.0772 against 1.1140 for the default (+0.0736 [+0.0303, +0.1190]); outcome log loss
0.9570 (default 0.9566). Average predicted total goals on validation 3.163 against 3.107 real (default model 2.662).

**Confound found by a diagnostic (not used for selection).** Experiments b and c both repair the same problem (the scoring level).
Repeating the window comparison under the offset configuration: all 8 seasons deviance 1.0805; last 3 +0.0067 [-0.0029, +0.0172];
last 5 +0.0059 [-0.0002, +0.0117]; recency +0.0037 [-0.0000, +0.0076]: **no window is clearly better once the level is handled.**
So the clear win of "last 3" in experiment c is mostly the offset's job. The rule's model uses the last 3 seasons; all seasons gives almost the same validation score with 3,040 instead of 1,140 fit rows.
**Resolved in the review below: the deployed model uses all seasons.**

**What the model learned.** Per 100 Elo points of advantage a team's expected goals are multiplied by 1.17; at equal strength the home team
scores 1.19 times as many goals as the away team; the relative attack/defence features add very little beyond Elo (some weights are
tiny or have the "wrong" sign because the features overlap). The log link makes rates grow exponentially with the Elo gap: the largest
validation rate is 4.3 goals (Man City v Southampton, Elo gap +449), and the top calibration group overshoots (predicted 2.74, actual 2.35
home goals).

**Draws and scores.** The highest draw probability is 0.265 (mean 0.212, real rate 0.230), so Draw is never the top outcome: 0 draws
predicted, like the classifiers (macro-F1 0.4152). The model's most likely SCORELINE is a draw (usually 1-1) in 445 of 760 matches;
taking the outcome of that scoreline gives 106 of 175 draws found (recall 0.61, precision 0.24) but accuracy only 41.4%. Exact
scoreline right in 11.6% of matches (rounded expected score 9.7%); the right outcome with the wrong exact score in 51.4% of matches.

**Stage 7 handoff (as first delivered; the numbers below describe the last-3 model and are superseded by "decisions approved after
review" further down: all seasons, cap 3.5, two Elo snapshots).** `models/poisson_stacked.joblib`, `models/poisson_config.json`, `src/poisson_model.py`,
`data/processed/poisson_pre_season_2025-26.csv` (380 fixtures; lambda_home 0.97 to 2.34, lambda_away 0.81 to 1.96; average 2.88 goals per
match; Elo from `elo_start_of_2025-26.csv`, neutral form features, 2024-25 league level 2.934) and the README section "Poisson goal
model: how Stage 7 should call it" (inputs, outputs, the two Stage 7 modes). The choice between the frozen pre-season mode and the
matchweek-by-matchweek mode is left to you in Stage 7 (CLAUDE.md pitfall).

### Stage 5d: decisions approved after review (2026-11-14)

| # | Decision | Status |
|---|---|---|
| 1 | **The final Poisson model is fitted on all 10 training seasons (2015-16 to 2024-25, 3,800 matches, 7,600 stacked rows).** The rule's last-3 model is kept in the analysis as "what the rule picked"; the confound diagnostic showed no window was clearly better once the league level is an offset (all 8 seasons 1.0805; last 3 1.0772, +0.0067 [-0.0029, +0.0172] within noise). | settled |
| 2 | **Predicted lambda is capped at 3.5, inside `PoissonGoalModel.predict()`.** `lambda_cap` is saved in `models/poisson_config.json`, so Stage 7 and the app get it automatically; `predict_uncapped()` returns the raw rates. | settled |
| 3 | **Dixon-Coles is dropped (rho = 0) and team fixed effects are not used.** Dixon-Coles did not help, probably because Elo-based rates already describe low-score matches well and our sample is small. (The stage's own premise test is consistent with this: on the fit seasons the observed/expected counts of the four lowest scorelines were 1.056, 1.024, 0.876, 0.971, and rho = -0.029 gained only 0.77 log-likelihood. The "because Elo already describes them" part is the group's explanation and was not tested separately.) | settled |
| 4 | **Two pre-season Elo snapshots for 2025-26** (Stage 7 picks between them): `poisson_pre_season_2025-26.csv` (keep = 0.9, default) and `poisson_pre_season_2025-26_keep075.csv` (keep = 0.75). `features.csv` and `elo_start_of_2025-26.csv` are unchanged. | settled |
| 5 | **Stage 8 player data source: the vaastav FPL archive** (github.com/vaastav/Fantasy-Premier-League), with two guards: (a) map its team names to the names in our match data (and fail loudly on any team that does not map); (b) use strictly only data dated before each match (no end-of-season totals for earlier matches). Licence and terms to be checked when Stage 8 starts (CLAUDE.md rule 10). | settled |

**What the cap did on validation (model fitted on 2015-16 to 2022-23, scored on 2023-24 and 2024-25, `reports/poisson_final_5d.csv`).**
7 of 760 matches (0.9%) had a rate above 3.5 (largest uncapped rate 4.34): Arsenal v Luton, Liverpool v Sheffield United, Man City v Ipswich,
Arsenal v Southampton, Man City v Southampton, Liverpool v Leicester and Liverpool v Southampton, all with an Elo gap of +343 or more (the rule's last-3 model had 5 such matches). In all 7 the favourite scored 1 to 4 goals; the cap
changed no other prediction.

| Validation (760 matches) | Deviance | Outcome log loss | Accuracy | MAE total goals | Predicted total goals (real 3.107) |
|---|---|---|---|---|---|
| Rule's pick: last 3 seasons, uncapped | 1.0772 | 0.9570 | 55.7% | 1.3302 | 3.163 |
| All 8 seasons, uncapped | 1.0805 | 0.9581 | 55.4% | 1.3251 | 3.125 |
| **All 8 seasons, capped at 3.5 (deployed configuration)** | **1.0795** | **0.9583** | **55.4%** | **1.3231** | **3.122** |

The cap improves the deviance by 0.0021 [+0.00003, +0.0061]: a tiny gain, at the edge of the noise rule, and not the reason it was chosen
(it is a safety limit for Stage 7 and the app). The all-8-seasons model scores slightly worse than the last-3 model on validation (1.0805 v
1.0772), a difference inside the noise. The figures and hand examples in the lesson (32 to 36) describe the rule's last-3 model on the fit
seasons; the deployed model differs from it only in the window and the cap.

**Deployed model (refit on all 10 training seasons, never scored on 2025-26).** Stacked, level-neutral features, league level as an offset,
rho = 0, cap 3.5. Coefficients per +1 SD: intercept -0.0484, att 0.0112, opp_def 0.0060, att_venue 0.0634, opp_def_venue 0.0269,
elo_diff_own 0.2490, abs_elo_diff -0.0124, season_no_fans -0.0023 (now varies, because 2020-21 is in the window; the pre-season rows use 0),
is_home 0.0971. As before, Elo carries the model: the other weights are small.

**Two Elo snapshots.** Both are computed from matches before 2025-26 only (the 2025-26 rows are loaded without any score). keep = 0.9 equals
the Stage 4 file exactly (asserted). keep = 0.75 pulls ratings further toward 1500 at every season start, so the spread of the pre-season
table shrinks: Liverpool 1680 to 1631, Man City 1672 to 1620, Arsenal 1671 to 1621; promoted teams stay at 1400.

| Pre-season table | lambda_home | lambda_away | Average total goals | Rates changed by the cap |
|---|---|---|---|---|
| keep 0.9 (default) | 0.94 to 2.41 | 0.78 to 1.98 | 2.879 | 0 |
| keep 0.75 | 1.03 to 2.23 | 0.85 to 1.84 | 2.865 | 0 |

**Caveat for Stage 7 when choosing between them.** The Poisson model learned its Elo weight from the Elo ratings built with keep = 0.9
(those are the `elo_diff` values in the training rows). Feeding it keep = 0.75 ratings is a change of input scale, not a retrained model: the
two snapshots differ only in how strongly last season's strength is carried over, but the model's Elo coefficient was calibrated on keep = 0.9
ratings. Both are legitimate pre-season inputs; neither uses any 2025-26 result. If Stage 7 wants a like-for-like comparison, the matchweek
mode should update Elo with the same `keep` that the features used.

**Stage 7 handoff (updated).** `models/poisson_config.json` (now with `lambda_cap`), `models/poisson_stacked.joblib`, `src/poisson_model.py`,
the two pre-season tables, and the README section. A rate of 3.5 or more can still appear only if the Stage 7 simulation builds its own features
outside `predict()`; call `model.predict` and the cap is applied.

---

## Open decisions

1. **Stage 7 mode:** frozen pre-season rates, or matchweek-by-matchweek updating from simulated results (to be chosen in Stage 7, with the
   explanation of the trade-off, CLAUDE.md pitfall).
2. **Stage 7: which Elo snapshot** (keep 0.9 or 0.75) to use for the pre-season table, see the caveat above. Low priority, tied within noise.
3. **Stage 8:** confirm the licence and terms of the vaastav archive, and build the team-name map, when the stage starts.

(Closed in the Stage 5d review: final window (all seasons), the lambda cap (3.5), Dixon-Coles and team fixed effects (not adopted), the two Elo
snapshots, and the Stage 8 data source.)

(Closed in Stage 5d: the level-neutral form of the league scoring level. Tested: raw vs relative goal averages made no difference; the league
level works as a fixed offset, not as a free feature.)

(Closed in the Stage 5b review: the forest to carry to Stage 6, the saved reference forest, and the leaf-size grid. Closed in the
Stage 5c review: the XGBoost to carry to Stage 6, the DIFF diagnostic, the grid, the monotone construction and the extras.)

**Stage 6 test plan so far (all scored once, together, on 2025-26):** primary models: ELO2 Logistic Regression, ELO2 Random Forest,
ELO2 XGBoost; reference rows: FULL Logistic Regression, FULL_v2 Random Forest, FULL_v2 XGBoost; baselines: always-home-win, class
frequencies, bookmaker; diagnostics (labelled as such, not competing primaries): DIFF Random Forest and DIFF XGBoost, plus the
drift check on the test features.

---

## Stage 6: honest model comparison on 2025-26 (2026-11-15)

The plan above was binding. The pre-registration (`reports/stage6_preregistration.md`, hash in `reports/stage6_run_log.json`) was written
before the test set was loaded. Readings of the prompt, recorded before the run: the "no refit / no early stopping" bullets cover the 10
scored models and the test season, while the plan itself requires the DIFF refit (early stopping on 2023-24 and 2024-25); "4,000" means
4,000 bootstrap resamples on one shared table (seed 42); the test loader lives in `stage6.py` and `modeling_utils.py` is unchanged
(`get_slice("test")` still raises); debugging was done by a dry run on the validation seasons in a scratch folder.

**What was run.** Two diagnostics were refit on 2015-16 to 2024-25 (DIFF Random Forest, 500 trees; DIFF XGBoost, 41 rounds by early stopping on
2023-24 and 2024-25; audit in `reports/xgb_diff_early_stopping_audit.json`, purpose `final_refit_diff`). The 20 earlier model files are
byte-identical (`reports/stage6_models_before.json`). The test season (380 matches, 15 Aug 2025 to 24 May 2026, all complete, odds for all)
was loaded once and 12 predictors were scored.

**Result (log loss, 380 matches).** Bookmaker 1.0185; DIFF RF (diagnostic) 1.0283; Poisson 1.0303; FULL LR 1.0327; **ELO2 LR 1.0335
(accuracy 49.5%)**; FULL_v2 RF 1.0345; DIFF XGB (diagnostic) 1.0367; FULL_v2 XGB 1.0383; ELO2 XGB 1.0385; ELO2 RF 1.0421; class frequencies
1.0838; always-home-win 42.6% accuracy. No competitor was significantly better than ELO2 LR (all eight intervals contain zero), so the
pre-registered primary stands: **a tie, not a victory.** ELO2 LR vs bookmaker +0.0149 [-0.0048, +0.0345] (cannot tell; on validation +0.0182
[0.0066, 0.0292]). ELO2 vs FULL: LR +0.0008 [-0.0126, +0.0146], RF +0.0077, XGB +0.0002 (ties). Nothing needs the label "post-hoc best".

**Observations (not decisions).** Accuracy 46.8% to 50.0%, below the 50 to 55% expectation (the bookmaker 48.9%): 27.4% of matches were draws
(training 23.3%); no model picks draws. Drift: 4 of 34 features beyond 3 standard deviations (22 of 34 on validation); the validation drift
pattern did not repeat (league level +1.08 mean z, against +4.51), so the test season did not stress the Stage 5a amendment. The 2024-25 pattern
(LR edge in a low-home-win season) repeated in direction only (ELO2 LR best of the three ELO2 primaries, gaps +0.005 and +0.009, inside the noise).
Poisson on the test season: MAE 0.948 home, 0.843 away, 1.245 total, deviance 1.0724, exact score 12.1%, no rate reached the 3.5 cap.

**Decisions for Stage 7 (recorded, not settled by test results).**
| # | Decision | Status |
|---|---|---|
| 1 | Use the deployed `poisson_stacked` model as saved (all 10 seasons, rho 0, cap 3.5) | settled |
| 2 | Elo snapshot: Stage 6 holds no evidence about keep 0.9 vs 0.75 (neither table was scored; choosing on the test set would be tuning). Default: keep 0.9, which matches the scale the Poisson weights were learned on | open, for the user in Stage 7 |
| 3 | Stage 7 mode (frozen pre-season rates or matchweek-by-matchweek updating from simulated results) | open, for the user in Stage 7 |
| 4 | The test numbers Stage 7 is compared with use real in-season form; a pre-season simulation has less information, so compare like with like | settled |

**Post-run edits (cosmetic only).** Figure 41's legend was redrawn from the saved `drift_test.csv` (`stage6.py replot-drift`); no test data read.
Report: `reports/stage6_results.md`. Lesson: `docs/learning/06_model_comparison.md`. Checks: `src/check_stage6.py`.

---

## Stage 7: season simulation, matchweek by matchweek (2026-11-15)

**Decided upstream (not re-decided):** Elo `keep` 0.9; matchweek-by-matchweek updating with simulated results; team-level only. **Stage 8 (player
leaders) is dropped from the project**, so the earlier "Stage 8 data source: vaastav FPL archive" entries (Stage 5d review and the open decisions) are
void.

**Readings and deviations from the prompt (flagged before the work started).**
- The league level 2.83 is not "the Stage 5d training-end value": it is the Stage 4 neutral default `league_goals` (the 10-season training mean).
  The Stage 5d pre-season table used 2.934 (2024-25). Held constant at 2.83 as specified; it carries no 2025-26 information.
- Only the features the saved Poisson model consumes are built (Elo difference, the 8 relative goal features, the league level, `season_no_fans`).
  Points, shots, head-to-head and matches-played features would never reach the model, so they are not built.
- There is no matchweek column: matchweek numbering is derived from the fixture list (each team's game number).
- Implementation: all 1,000 seasons advance in lockstep (the schedule is the same in every sim); simulation i uses `default_rng(i)` uniforms turned into
  Poisson counts by inverse-CDF sampling, so a season is identical alone or in a batch (tested). Outputs are CSV (1.3 MB and 5.8 MB), not parquet.
- `reports/sim_preregistration.md` fixed the metrics before the simulation ran; the real results were read once, in `compare()`.

**Bug found in Stage 5d code (for you to decide).** In `src/features_goals.py` (`build_goal_features`, the `rel[i] = [...]` row and the `pd.DataFrame(rel,
columns=[...])` labels) the columns `home_ga_10_rel` and `away_gf_10_rel` hold each other's values: `home_ga_10_rel` is the AWAY team's goals scored (corr 0.975
with `away_gf_10`) and `away_gf_10_rel` is the HOME team's goals conceded (0.970 with `home_ga_10`). The Stage 5d parent-window check tested a different
array and missed it. Effect: in the stacked Poisson model the away row's attack and defence inputs were swapped; training, validation, Stage 6 and the
simulation all used the same layout, so the saved model and every reported number are internally consistent. Measured on the validation seasons only
(`reports/poisson_layout_diagnostic.csv`, same configuration refit on the fit seasons): deviance 1.0805 (as saved) vs 1.0802 (corrected), +0.0006
[-0.0012, +0.0024]; outcome log loss 0.9581 vs 0.9577, +0.0003 [-0.0003, +0.0009]. **Immaterial.** The simulator reproduces the saved layout
(`LEGACY_SWAPPED_LABELS = True`, verified against `features_goals.py` on a real season). Options: (A, recommended) keep everything as is and document the bug
in the report; (B) fix the labels, regenerate `features_goals.csv`, refit the Poisson on all seasons and re-simulate, which changes the pre-registered Stage 6
model and cannot be re-scored on the test season without a second look at it; (C) fix the code and the labels only in the files, with no refit (then
the saved model and the features disagree: not recommended).

**Results.** 1,000 seasons, average 2.831 goals per match, Elo 1210 to 1854, largest rate 3.5 (the cap). Simulated top 6 by mean points (P title / P top 4):
Liverpool 71.0 (0.214 / 0.607), Man City 70.2 (0.187 / 0.566), Arsenal 70.0 (0.195 / 0.561), Chelsea 61.9 (0.075 / 0.340), Aston Villa 61.5 (0.073 / 0.330),
Newcastle 58.7 (0.048 / 0.274). Bottom 3: Leeds 36.2 (P relegation 0.452), Burnley 36.5 (0.441), Sunderland 36.9 (0.432). Against the real 2025-26 table
(Arsenal 85, Man City 78, Man United 71, Aston Villa 65, Liverpool 60, ..., West Ham 39, Burnley 22, Wolves 20): mean absolute position error 3.83,
Spearman 0.564, mean absolute points error 9.78 (signed +0.83), real top 4 captured 2 of 4, real relegated captured 1 of 3, 18 of 20 real totals inside
the simulated 5th to 95th percentile (outside: Man United 71 vs 69, Wolves 20 vs 25). The simulated champion averages 86.1 points (real 85); the expected table
gives Arsenal 70.0 because it averages over the seasons Arsenal did not win. Ceiling: one-match-ahead Poisson log loss 1.0303 / accuracy 48.4% vs simulation 1.0500 /
46.1%: gap +0.0197 [-0.0005, +0.0395] (at the edge of the noise); total-goals deviance 1.0417 vs 1.0345 (the simulation is as good on goal counts).

**Open for review (none changes a number above; resolved in "Stage 7: decisions approved after review" below).**
1. The bug options A / B / C above.
2. League level: 2.83 (training mean, held constant) vs the 2024-25 value 2.934 used by the Stage 5d pre-season table. Real 2025-26 scored 2.75 (not used).
3. Start-of-season form: neutral for all teams (as specified) vs carrying each team's real last-10 matches of 2024-25 (leakage-free, closer to how the model was trained).
4. Independent home/away goal draws (standard) vs the small negative correlation of -0.14 measured on the training seasons.
5. Report framing: the simulation ranks Liverpool as favourite; the real champion Arsenal had a 19.5% simulated title chance (3rd by P(title)).

### Stage 7: decisions approved after review (2026-11-15)

| # | Decision | Status |
|---|---|---|
| 1 | **The `_rel` column swap in `src/features_goals.py` is documented, NOT fixed** (option A). Why: the bug was consistent across training, validation, Stage 6 and the simulation; fixing it would retroactively break the pre-registered Stage 6 result; its measured effect on the validation seasons is +0.0006 (deviance, [-0.0012, +0.0024]), immaterial. The simulator keeps `LEGACY_SWAPPED_LABELS = True`. Details: the Stage 7 section above and `reports/poisson_layout_diagnostic.csv` | settled |
| 2 | **Future work** (not done in this project): (a) start the simulator from each team's real 2024-25 form (last-10 matches carried over) instead of neutral defaults: it is information available at the start of the season and might improve the forecasts; (b) model the correlated home/away goals (observed correlation -0.14 on the training seasons) with a bivariate copula or the Dixon-Coles approach; independence is kept for simplicity | recorded |
| 3 | League level 2.83, Elo `keep` 0.9 and the other Stage 7 choices stay as run | settled |

---

## Stage 9: Streamlit app (2026-11-15)

Stage 8 (player leaders) is dropped, so the app has two pages (Match Predictor, Season Table). The app only SERVES saved artifacts; nothing is retrained or re-simulated.
Checks: `src/check_app.py`. Screenshots: figures 46 and 47.

**Deviations from the prompt, and readings (for review).**
| # | What | Why |
|---|---|---|
| 1 | Python **3.12** on Streamlit Community Cloud, not 3.11 | `app/requirements.txt` pins the versions the models were pickled with (numpy 2.5.4, scipy 1.18.1, xgboost 3.4.1, scikit-learn 1.9.1); numpy, scipy and xgboost have no Python 3.11 builds at those versions (checked with `pip download --python-version`). Pinning keeps the pickles loading identically |
| 2 | `.streamlit/config.toml` lives at the repo ROOT, not in `app/` | Streamlit reads the theme from the folder where `streamlit run` is started (the repo root, also on Cloud); a file in `app/.streamlit/` would be ignored |
| 3 | Crests are 128 x 128 PNG (not 64 x 64) and come from the English Wikipedia club infoboxes, not Commons | The 96 px matchup card needs more than 64 px; commons.wikimedia.org is unreachable from here, and these crests are local "non-free" Wikipedia files (trademarked logos), not freely licensed Commons files. **The PNGs are git-ignored** until you decide whether to publish them; without them the app shows coloured 3-letter badges (tested). Sources: `data/app/team_crests/SOURCES.md` (local) |
| 4 | Hero image is an original football icon (`data/app/hero_ball.svg`), not a downloaded silhouette | No licensing question at all |
| 5 | The scoreboard shows Poisson expected goals to one decimal ("LIV 1.4 : 1.3 BOU"), plus the most likely exact score underneath | An expected value is a mean, not a score; rounding 1.4 : 1.3 to "1 : 1" would hide that |
| 6 | The Streamlit header is made transparent, not removed | Removing it also removes the sidebar toggle on phones |
| 7 | Bookmaker row = the B365 odds of the REAL 2025-26 fixture of that pair (margin removed, Stage 6 method), shown with a note | The models describe each team at the END of the season; the bookmaker odds were set before the real match (August to May). They are not like-for-like, and the page says so. Every ordered pair of 2025-26 teams exists exactly once, so the row is always filled |
| 8 | Page 1 uses the end-of-2025-26 state: Elo after the last match and last-10 form after the last match, with the league level = the last 380 matches (2.75) | Built by running the project's own feature code over 380 hypothetical fixtures on one day after the season (`app/team_state_builder.py`), so the features match the training code exactly (check_app compares all 380 pairs: max difference 0). The Poisson inputs keep the documented `_rel` label swap, as the saved model expects |
| 9 | The fan-chart median line uses the team colour, blended toward white when it is too dark for the purple page (Fulham black, Newcastle, Tottenham, Chelsea) | Black on dark purple is invisible |
| 10 | The team state and the Season Table are built from 2025-26 results; the app itself reads no score column | `check_app.py` scans the runtime files for raw-data names and score columns |

**Addition (requested after review): "What really happened" on Page 1.** Every ordered team pair was played once in 2025-26, so Page 1 now shows the real
score and date of the chosen fixture, and the chance each model gave to the real outcome. Decisions: (a) this deliberately relaxes the Stage 9 rule "the app reads
no 2025-26 score": the scores come from one small precomputed display file, `data/app/real_results_2025-26.csv` (columns renamed home_score / away_score;
written by `app/team_state_builder.py`), never from raw data, and `src/check_app.py` still forbids raw files and raw score column names, and proves
predictions are identical when the scores are zeroed; (b) the page says it is a sanity check, not a test, because the end-of-2025-26 state already includes that
very match (the fair test is Stage 6); (c) Liverpool 4-2 Bournemouth equals the Stage 6 worked example (checked).

**Update (requested after deployment): crests are now committed.** The deployed app showed only initials because the crest PNGs were git-ignored. The project owner
chose to publish them, so `.gitignore` no longer excludes `data/app/team_crests/*.png` (20 files, 432 KB). They remain trademarked, non-free club logos used for
this non-commercial university project only; sources are listed in `data/app/team_crests/SOURCES.md`. The badge fallback stays in place and is still tested.
