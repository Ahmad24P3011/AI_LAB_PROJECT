# Stage 6 results: honest model comparison on the 2025-26 test season

## Pre-registration (written before the 2025-26 test set is loaded)

Binding. Nothing in the post-test analysis changes any of the following.

- **Primary model of the project:** ELO2 Logistic Regression (`models/logreg.joblib`).
- **Three competing families, each as "primary ELO2" + "FULL reference":** Logistic Regression (`logreg`, `logreg_full_reference`), Random Forest (`random_forest`, `random_forest_full_reference`), XGBoost (`xgboost`, `xgboost_full_reference`).
- **Poisson** (`poisson_stacked`) scored as a result classifier via its bivariate grid and as a goals model (MAE, deviance).
- **Three baselines:** always-home-win, class frequencies, bookmaker.
- **Two diagnostics (not competing primaries, labelled as such):** DIFF Random Forest and DIFF XGBoost. Refit now on all 10 training seasons with the configurations frozen in `docs/decisions.md` (no re-tuning, test set untouched).
- **Primary metric:** log loss. **Secondary:** accuracy, macro-F1, per-class F1, confusion matrix. For Poisson also: MAE on goals (home, away, total) and Poisson deviance.
- **Noise rule unchanged:** differences are called significant only when the 95% paired-bootstrap interval of the log-loss difference excludes zero.
- **Scoring stage:** no model is refit, re-tuned or re-selected by this stage (the only refits are the two DIFF diagnostics, before the test set is loaded). The test season is loaded once, used once, and the results are published once.

### Readings of the rule, fixed before the test set is loaded
These clarify wording only; they were written down before any test number existed.

1. **Who counts as a "competitor" for the winner rule:** the 8 other model predictors (the two other primaries, the four FULL references, Poisson, and the two diagnostics). The bookmaker and the baselines are benchmarks, not competitors, and are reported separately.
2. **If a diagnostic (DIFF) is significantly better than ELO2 LR,** it is reported as "post-hoc best on the test set, not pre-registered", exactly like any other model, and ELO2 LR stays the pre-registered primary.
3. **Bootstrap:** one table of 4,000 resamples of the test matches (fixed seed 42) is drawn once and used for every comparison. Difference = log loss of A minus log loss of B, so a negative number means A is better. "Significant" means the 95% interval excludes zero.
4. **Always-home-win** makes hard predictions only, so it has no log loss and is not part of the bootstrap comparisons.
5. **Class frequencies** are taken from the 10 training seasons (2015-16 to 2024-25), the same data the final models use.
6. **Poisson's role** in the table is "primary" (a goals model, the fourth family); it is scored as a classifier for the comparison, but it is used in Stage 7 regardless of what happens here.
7. **Early stopping of the DIFF XGBoost** (the only early stopping in this stage) uses 2023-24 and 2024-25 as the stopping seasons, as recorded in `docs/decisions.md`.
8. **Matchweek handling:** the test season is loaded and its matches counted; both the full set and the completed-matchweeks slice are reported. Which matches are missing, if any, is stated.
9. **Dry run:** the pipeline is first exercised on the validation season (outputs to a scratch folder, nothing written to `reports/`) to find bugs without touching the test set. The real run is executed once.

---

## Headline table: all predictors on the 2025-26 test season (380 matches, ordered by log loss)

Every predictor scored the same 380 matches, once. **Bold = the pre-registered primary.** "Draws predicted" is how often the model's most probable outcome was a Draw (104 of the 380 matches really were draws). Always-home-win makes hard predictions only, so it has no log loss.

| Predictor | Role | Log loss | Accuracy | Macro-F1 | Home F1 | Draw F1 | Away F1 | Draws predicted |
|---|---|---|---|---|---|---|---|---|
| Bookmaker (Bet365) | baseline | 1.0185 | 48.9% | 0.365 | 0.626 | 0.000 | 0.470 | 0 |
| DIFF Random Forest (diagnostic) | diagnostic | 1.0283 | 48.7% | 0.364 | 0.628 | 0.000 | 0.463 | 0 |
| Poisson (scoreline grid) | primary | 1.0303 | 48.4% | 0.364 | 0.615 | 0.000 | 0.479 | 0 |
| Logistic Regression FULL (reference) | reference | 1.0327 | 46.8% | 0.347 | 0.610 | 0.000 | 0.430 | 1 |
| **Logistic Regression ELO2** | primary | **1.0335** | 49.5% | 0.371 | 0.630 | 0.000 | 0.482 | 0 |
| Random Forest FULL_v2 (reference) | reference | 1.0345 | 47.4% | 0.350 | 0.615 | 0.000 | 0.436 | 0 |
| DIFF XGBoost (diagnostic) | diagnostic | 1.0367 | 47.4% | 0.349 | 0.623 | 0.000 | 0.423 | 1 |
| XGBoost FULL_v2 (reference) | reference | 1.0383 | 48.2% | 0.355 | 0.625 | 0.000 | 0.439 | 0 |
| XGBoost ELO2 | primary | 1.0385 | 49.2% | 0.364 | 0.635 | 0.000 | 0.458 | 0 |
| Random Forest ELO2 | primary | 1.0421 | 50.0% | 0.374 | 0.637 | 0.000 | 0.484 | 0 |
| Class frequencies (training) | baseline | 1.0838 | 42.6% | 0.199 | 0.598 | 0.000 | 0.000 | 0 |
| Always predict home win | baseline | n/a | 42.6% | 0.199 | 0.598 | 0.000 | 0.000 | 0 |

*How to read it:* lower log loss is better. With 380 matches, a log-loss difference smaller than about 0.012 to 0.02 (the width of the bootstrap intervals below) cannot be told from luck, and an accuracy difference smaller than about 5 points cannot either (one standard error of an accuracy near 50% on 380 matches is 2.6 points). Figure 39 (`reports/figures/39_log_loss_bar.png`) shows the same ranking with 95% intervals: they overlap almost entirely.

## Key numbers

- **ELO2 Logistic Regression (the pre-registered primary):** test log loss **1.0335**, accuracy **49.5%** (188 of 380), macro-F1 0.371. For context, on the validation seasons it scored 0.9582 and 55.4%. Accuracy fell about 6 points: this season was harder for everyone (see "Draws" below), including the bookmaker (48.9%). The expectation of 50 to 55% in `CLAUDE.md` was therefore **not met on this season**: the best accuracy in the table is 50.0% (ELO2 Random Forest), and the bookmaker is at 48.9%.
- **Against the baselines:** always-home-win gets 42.6% (162 home wins in 380); the class frequencies give log loss 1.0838. ELO2 LR beats the class-frequency log loss by 0.0503 [0.0134, 0.0871] (clear), and its accuracy is 6.8 points above always-home-win (more than the 5-point noise level, so a genuine but modest gain).
- **Against the bookmaker:** the bookmaker's log loss is 1.0185. ELO2 LR is behind by 0.0149 (95% interval of the log-loss difference ELO2 LR minus bookmaker: -0.0048 to +0.0345). The interval contains zero, so **this test cannot say that the bookmaker is better** (on the validation seasons it clearly was: 0.0182 [0.0066, 0.0292]). Two other models are clearly behind the bookmaker on the test season: the ELO2 Random Forest (+0.0236 [+0.0006, +0.0460]) and the FULL_v2 XGBoost (+0.0198 [+0.0016, +0.0379]).
- **Against its FULL reference:** ELO2 minus FULL = +0.0008 [-0.0126, +0.0146]. Two features (`elo_diff`, `abs_elo_diff`) do as well as the 33-feature model, as they did on validation. (Random Forest: +0.0077 [-0.0060, +0.0215]; XGBoost: +0.0002 [-0.0136, +0.0137]: also ties.)
- **Against the other families' primaries** (difference = model minus ELO2 LR; negative = model better): Random Forest ELO2 +0.0086 [-0.0039, +0.0207]; XGBoost ELO2 +0.0051 [-0.0065, +0.0162]; Poisson -0.0032 [-0.0106, +0.0045]. All intervals contain zero: the four families cannot be separated on this test season.

### Pre-registered comparisons (paired bootstrap, 4,000 resamples of the same 380 matches, seed 42)
*Difference = log loss of A minus log loss of B. A negative number means A is better. "Significant" only if the 95% interval excludes zero.*

**Every model against ELO2 LR (the primary):**

| A | B | Mean difference | 95% interval | Verdict |
|---|---|---|---|---|
| Logistic Regression FULL (reference) | Logistic Regression ELO2 | -0.0008 | [-0.0146, +0.0126] | cannot tell |
| Random Forest ELO2 | Logistic Regression ELO2 | +0.0086 | [-0.0039, +0.0207] | cannot tell |
| Random Forest FULL_v2 (reference) | Logistic Regression ELO2 | +0.0010 | [-0.0086, +0.0106] | cannot tell |
| XGBoost ELO2 | Logistic Regression ELO2 | +0.0051 | [-0.0065, +0.0162] | cannot tell |
| XGBoost FULL_v2 (reference) | Logistic Regression ELO2 | +0.0049 | [-0.0077, +0.0171] | cannot tell |
| Poisson (scoreline grid) | Logistic Regression ELO2 | -0.0032 | [-0.0106, +0.0045] | cannot tell |
| DIFF Random Forest (diagnostic) | Logistic Regression ELO2 | -0.0052 | [-0.0170, +0.0057] | cannot tell |
| DIFF XGBoost (diagnostic) | Logistic Regression ELO2 | +0.0032 | [-0.0108, +0.0163] | cannot tell |
| Class frequencies (training) | Logistic Regression ELO2 | +0.0503 | [+0.0134, +0.0871] | A clearly worse |
| Bookmaker (Bet365) | Logistic Regression ELO2 | -0.0149 | [-0.0345, +0.0048] | cannot tell |

**Every model against the bookmaker:**

| A | B | Mean difference | 95% interval | Verdict |
|---|---|---|---|---|
| Logistic Regression ELO2 | Bookmaker (Bet365) | +0.0149 | [-0.0048, +0.0345] | cannot tell |
| Logistic Regression FULL (reference) | Bookmaker (Bet365) | +0.0142 | [-0.0038, +0.0315] | cannot tell |
| Random Forest ELO2 | Bookmaker (Bet365) | +0.0236 | [+0.0006, +0.0460] | A clearly worse |
| Random Forest FULL_v2 (reference) | Bookmaker (Bet365) | +0.0159 | [-0.0029, +0.0347] | cannot tell |
| XGBoost ELO2 | Bookmaker (Bet365) | +0.0200 | [-0.0017, +0.0411] | cannot tell |
| XGBoost FULL_v2 (reference) | Bookmaker (Bet365) | +0.0198 | [+0.0016, +0.0379] | A clearly worse |
| Poisson (scoreline grid) | Bookmaker (Bet365) | +0.0117 | [-0.0076, +0.0305] | cannot tell |
| DIFF Random Forest (diagnostic) | Bookmaker (Bet365) | +0.0097 | [-0.0083, +0.0277] | cannot tell |
| DIFF XGBoost (diagnostic) | Bookmaker (Bet365) | +0.0182 | [-0.0004, +0.0362] | cannot tell |
| Class frequencies (training) | Bookmaker (Bet365) | +0.0652 | [+0.0304, +0.1001] | A clearly worse |

**ELO2 primary vs its FULL reference, and vs its DIFF diagnostic:**

| A | B | Mean difference | 95% interval | Verdict |
|---|---|---|---|---|
| Logistic Regression ELO2 | Logistic Regression FULL (reference) | +0.0008 | [-0.0126, +0.0146] | cannot tell |
| Random Forest ELO2 | Random Forest FULL_v2 (reference) | +0.0077 | [-0.0060, +0.0215] | cannot tell |
| XGBoost ELO2 | XGBoost FULL_v2 (reference) | +0.0002 | [-0.0136, +0.0137] | cannot tell |
| Random Forest ELO2 | DIFF Random Forest (diagnostic) | +0.0139 | [-0.0013, +0.0289] | cannot tell |
| XGBoost ELO2 | DIFF XGBoost (diagnostic) | +0.0018 | [-0.0130, +0.0166] | cannot tell |

## Draws

- 104 of the 380 matches (27.4%) were draws, more than the 23.3% in the training seasons and the second-highest share of any of the 11 seasons. The models gave Draw an average probability of 23.4% to 24.6% (the highest probability any model gave a match was 0.27 to 0.41), so **no model ever picked Draw as its most likely outcome, except one match each for the FULL Logistic Regression and the DIFF XGBoost, both wrong.** Draw F1 is 0 for every predictor. This is expected (draws are close calls between two similar teams), and a high-draw season made it more costly: the class-frequency baseline itself rose from 1.0675 (validation) to 1.0838.
- **ELO2 LR confusion matrix** (rows = what happened; columns = Home, Draw, Away): Home 127 / 0 / 35; Draw 61 / 0 / 43; Away 53 / 0 / 61. Home win: precision 0.527, recall 0.784, F1 0.630. Away win: precision 0.439, recall 0.535, F1 0.482. Draw: 0 / 0 / 0.
- The Poisson model's most likely **scoreline** is a draw (usually 1-1) in 263 of 380 matches. Taking the outcome of that scoreline finds 74 of the 104 draws (recall 0.71, precision 0.28) but accuracy falls to 38.2%, so it is not a better classifier; it only shows that a goals model sees the draws spread over 0-0, 1-1 and 2-2. All six confusion matrices are in `reports/figures/37_test_confusion_matrices.png`.

## Poisson as a goals model

The deployed Poisson model (stacked, level-neutral features, league level as an offset, rho 0, fitted on all 10 training seasons, rates capped at 3.5) was scored as it was saved. No rate reached the cap on this season (0 of 380 matches).

| Goal metric (test, 380 matches) | Value | Validation (Stage 5d) |
|---|---|---|
| MAE home goals | 0.948 | not recorded separately |
| MAE away goals | 0.843 | not recorded separately |
| MAE total goals | 1.245 | 1.323 |
| Poisson deviance (mean per goal count) | 1.0724 | 1.0795 |
| Exact scoreline right (most likely scoreline) | 12.1% (46 matches) | 11.6% |
| Rounded expected score right | 8.7% | 9.7% |
| Outcome right, taking the outcome of the most likely scoreline | 38.2% | 41.4% |
| Outcome right, taking the highest summed probability | 48.4% | 55.4% |
| Average predicted total goals / real | 2.81 / 2.75 | 3.12 / 3.11 |

*Validation column:* MAE total, deviance and the average predicted total are for the deployed configuration fitted on the fit seasons; the two scoreline rows (11.6%, 9.7%) and the 41.4% are from the rule's last-3-seasons model, which Stage 5d analysed in detail (the deployed all-seasons model was not analysed that way); 55.4% is the deployed model's validation accuracy.

*As a classifier* the Poisson model scores log loss 1.0303 and accuracy 48.4%, level with ELO2 LR (difference -0.0032 [-0.0106, +0.0045]).

**Worked example: Liverpool v Bournemouth, 15 August 2025 (the opening match, picked in advance, not for how it looks).** The model gives expected goals 2.203 for Liverpool and 0.988 for Bournemouth, so Home 64.9%, Draw 19.3%, Away 15.8%. The five most likely scores are 2-0 (10.0%), 2-1 (9.9%), 1-0 (9.1%), 1-1 (9.0%) and 3-0 (7.3%). The real score was **4-2**: a home win (the model's top outcome, correct), but not its top scoreline. The probability the model gave 4-2 was 2.0%: a typical football result is a long way from the single most likely score.

**Numbers for Stage 7.** These are the test-season numbers the season simulation will have to match: MAE on total goals 1.245, deviance 1.0724, an average predicted total of 2.81 goals against 2.75 real. One caution, stated plainly: these test numbers use form features built from the real earlier matches of 2025-26 (the usual one-match-at-a-time prediction). A Stage 7 simulation that uses only pre-season information, or only simulated results, has less information, so it should not be expected to reach these per-match numbers. Stage 7 should compare itself with the pre-season-only rates in the same way, not with these.

## Drift on the test features (diagnostic only; no model was changed because of it)

Measured in fit-season (2015-16 to 2022-23) standard deviations, the same way as the Stage 5a validation table (`reports/drift_test.csv`, figure 41). **On the test season only 4 of the 34 features have any row beyond 3 standard deviations (home_venue_ga 3.84, away_venue_gf 3.31, home_ga_10 3.24, away_shots_for_10 3.01), against 22 of 34 on the validation seasons; the average size of a feature's mean shift is 0.10 standard deviations, against 0.36.** The validation drift pattern did not repeat: `league_avg_goals_prev380`, which sat 4.5 standard deviations above the training mean on validation, is at +1.08 (largest row 2.59). The feature is the average of the previous 380 matches, so for 2025-26 it mixes the 2024-25 level (2.93) with the lower 2025-26 level (2.75 goals per match), and averages 2.83 against a training mean of 2.75 (on validation the underlying seasons scored 3.28 and 2.93). So the test season did not stress the amendment that removed that feature; its benefit (or cost) cannot be judged from this season. Nothing was re-tuned.

## The 2024-25 pattern, revisited

In 2024-25 (home teams won 40.8%) the ELO2 Logistic Regression clearly beat all four forests. 2025-26 was also a low-home-win season (**42.6%**, against 46.1% in 2023-24), so it is a fair second look. **The direction repeated, the clear edge did not.** ELO2 LR has the lowest log loss of the three ELO2 primaries (1.0335 against 1.0385 for XGBoost and 1.0421 for the forest), but the gaps (+0.0051 and +0.0086) are inside the noise; at FULL features the three families are within 0.006 of each other (1.0327, 1.0345, 1.0383). Every model again over-predicted home wins, but by less: average predicted home-win probability was 43.7% to 45.5% against 42.6% real (in 2024-25: 44.6% to 45.4% against 40.8%); the five reliability curves in `38_test_calibration.png` follow the diagonal loosely, with wide error bars (about 380 matches in 5 groups). In the third quarter of the season (matchweeks 21 to 30, only 33% home wins) all models were at their worst (1.09 for the bookmaker, 1.12 to 1.16 for the models); the ELO2 LR was 1.1433 there.

## Missing matches

None. The test season was loaded once: 380 matches from 15 August 2025 to 24 May 2026, all 20 teams with 38 games, Bet365 odds present for all 380. The completed-matchweeks slice is therefore the full set (the two rows of `reports/stage6_periods.csv` are identical).

## Honest conclusion: did the pre-registered primary win?

**It won by the pre-registered rule, which means it tied.** No competitor (the other two primaries, the four FULL references, Poisson, and the two diagnostics) has a 95% interval entirely below zero against ELO2 LR, so by pre-registration ELO2 LR is the project's model. This is a successful pre-registration, but it is not evidence that ELO2 LR is better: it is not significantly better than any competitor either (except the class-frequency baseline). The point estimates put the bookmaker first (1.0185), then the DIFF Random Forest diagnostic (1.0283), Poisson (1.0303), FULL LR (1.0327) and ELO2 LR (1.0335) in fifth place, with everything between them within noise. **Nothing was significantly better than ELO2 LR on the test log loss, so no model needs the label "post-hoc best on the test set".** The DIFF diagnostics did not become anything more than diagnostics: DIFF Random Forest was the best model of the forests on validation and is again the best by point estimate (1.0283, a gain of 0.0139 [-0.0013, +0.0289] over the ELO2 forest) but inside the noise; the DIFF XGBoost is level with its ELO2 counterpart (+0.0018).

What this means for the report (requirement 6, "suggest the best model and explain why"): **the suggested model is the ELO2 Logistic Regression**, because it is the pre-registered primary, it ties with everything that is more complicated, it is the simplest to explain (two inputs, one formula: the Elo gap and its absolute value), and it is not clearly behind the bookmaker on this season. The Poisson model is the project's goals model and is used in Stage 7 whatever the comparison showed. The honest summary of Stage 6 is: **four very different model families, two years of validation tuning and 33 features end in a tie on a 380-match season at about 47 to 50% accuracy, with the bookmaker at the same level.**

## What this means for Stage 7

- **Poisson configuration:** use the deployed `poisson_stacked` model exactly as saved (stacked, all 10 seasons, rho 0, cap 3.5). It was scored here without change (cap never triggered; test deviance 1.0724) and is level with ELO2 LR as a classifier.
- **Elo snapshot:** Stage 6 contains **no information** about keep = 0.9 against keep = 0.75, because neither pre-season table was scored, and choosing between them on test results would be using the test set to tune. Unless you decide otherwise on other grounds, keep the default (`poisson_pre_season_2025-26.csv`, keep 0.9): it matches the Elo scale the Poisson weights were learned on.
- **Stage 7 mode** (frozen pre-season rates, or matchweek-by-matchweek updating) is still yours to choose at the start of Stage 7.
- The real 2025-26 results are never used to build features for later fixtures in the simulation.

## What was and was not done after the test run

Nothing about any model, the pre-registration or the comparison changed after the test numbers appeared. The only post-run edit was cosmetic: figure 41's legend (the colour swatches were both blue) was redrawn from the saved `drift_test.csv`, with no test data read. The dry-run on validation seasons used to find bugs wrote only to a scratch folder.
