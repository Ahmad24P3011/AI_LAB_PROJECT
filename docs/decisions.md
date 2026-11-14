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

## Open decisions

1. **Stage 5c, for you:** which XGBoost goes to Stage 6? (a) the rule's choice (ELO2 features, 0.9714), clearly worse than the ELO2 Logistic
   Regression; (b) the default FULL_v2 model (0.9699, saved as the reference row). Recommendation: score (a) and (b) together in Stage 6,
   as in 5a and 5b.
2. **DIFF for XGBoost:** DIFF scored best of the four sets again (0.9651, +0.0048 [-0.0034, +0.0135], within noise; best in both seasons).
   In 5b the DIFF forest was fixed as a Stage 6 test diagnostic. Should the DIFF XGBoost also be a diagnostic? Its configuration would be:
   DIFF set (16 features), `max_depth` 3, `learning_rate` 0.1, `min_child_weight` 20, `subsample` 0.8, `colsample_bytree` 0.8, trees chosen by
   early stopping, all ten training seasons.
3. All three XGBoost hyperparameters are at the edge of the grid (depth 3, `learning_rate` 0.1, `min_child_weight` 20). Extend, or accept?
4. Monotone constraint: keep it out (no effect and it does not give a monotone Home-win probability in multi-class mode), or try the
   two-model (binary) construction?
5. Elo `keep` constant: 0.9 (tuned, at the edge of the grid) or 0.75 (EDA-based)? Tied within noise. Low priority.
6. Stage 5d: test a level-neutral form of the league scoring level (team goals relative to the league average).
7. Stage 8: player data source (recommendation: the vaastav FPL archive).

(Closed in the Stage 5b review: the forest to carry to Stage 6, the saved reference forest, and the leaf-size grid.)
