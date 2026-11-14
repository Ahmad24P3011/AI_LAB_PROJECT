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

## Open decisions

1. **Stage 5b, for you:** which Random Forest goes to Stage 6? (a) the rule's choice (ELO2 features, validation 0.9733),
   which is clearly worse than the ELO2 Logistic Regression; (b) the default FULL_v2 forest (0.9649, saved as the reference row);
   (c) DIFF (0.9586, lowest, but picking it now is post-hoc). Recommendation: score (a) and (b) together in Stage 6, as in 5a, and
   treat DIFF only as a documented alternative unless you decide otherwise before Stage 6.
2. `min_samples_leaf` = 80 is the edge of the grid. Extend the grid in a later stage?
3. Elo `keep` constant: 0.9 (tuned, at the edge of the grid) or 0.75 (EDA-based)? Tied within noise. Low priority.
4. Stage 5c: which feature set is primary for XGBoost (same noise rule, same v3 sets)?
5. Stage 5d: test a level-neutral form of the league scoring level (team goals relative to the league average).
6. Stage 8: player data source (recommendation: the vaastav FPL archive).
