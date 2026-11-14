# Lesson 06: Honest model comparison on the 2025-26 test season

**Stage goal in one sentence:** Score every model once, on the one season none of them ever saw (2025-26), under rules written down before we looked, and report what the numbers can and cannot tell us.

**Files created or changed:**
- `reports/stage6_preregistration.md`: the rules, written BEFORE the test set was loaded
- `src/stage6.py`: the scoring script (refit-diff, dry-run and run commands; the only code that reads the test season)
- `src/check_stage6.py`: 11 checks of the stage's promises (run it after the test run)
- `src/try_bootstrap.py`: the "Try it yourself" script of this lesson
- `models/diff_rf.joblib`, `models/diff_xgb.joblib` / `.ubj` and their `*_config.json` files: the two DIFF diagnostics (the only models fitted in this stage, before the test set was loaded)
- `reports/xgb_diff_early_stopping_audit.json`: which seasons the DIFF XGBoost used to choose its number of rounds (purpose `final_refit_diff`)
- `reports/stage6_results.csv` (the headline table), `stage6_bootstrap.csv` (25 paired comparisons), `stage6_periods.csv`, `stage6_pattern.csv`, `stage6_poisson_goals.csv`, `stage6_worked_example.json`, `drift_test.csv`, `test_predictions_stage6.csv` (every test prediction), `stage6_run_log.json`, `stage6_models_before.json`
- `reports/stage6_results.md`: the report-ready write-up (pre-registration, headline table, paragraphs)
- `reports/figures/37_test_confusion_matrices.png` to `41_drift_test_bar.png`
- `docs/decisions.md` (updated) and this lesson

---

## 1. The big picture
For five stages we built models and chose between them using the **validation seasons** (2023-24 and 2024-25). Think of validation as practice exams: we are allowed to look at the marks, change our study plan, and try again. The **test season** (2025-26) is the real exam, and the rule of an honest exam is that you sit it once. Stage 6 is that exam. If we peeked at the test marks and then went back to change a model, the test would stop being a test, and the number we report would be flattering and untrue. So this stage does not train or tune anything (apart from two diagnostic models whose settings were fixed in advance). It only *scores*, and the most important thing we did came before the scoring: we wrote down, in advance, what we would count as winning.

## 2. What we built
In the order it happens:

1. **The pre-registration** (`reports/stage6_preregistration.md`). We wrote down the primary model (ELO2 Logistic Regression), all 12 predictors, the metrics, the noise rule, and a few readings of ambiguous wording, and recorded its hash in a log. *Why:* so we cannot move the goalposts after seeing results. *Example:* "the bookmaker is a benchmark, not a competitor" was decided before any test number existed.
2. **Fingerprints of the frozen models.** Each saved model is loaded as it is and we print its file size, a short code for its config, its number of features and its settings (for example, `xgboost.joblib`: 2 features, 62 rounds, depth 3). We checked that the features the model remembers equal the ones in its config. We also checked that the 20 model files from earlier stages are byte-for-byte unchanged (a hash of each file).
3. **Refit of two diagnostics** (DIFF Random Forest and DIFF XGBoost) on the 10 training seasons, with settings fixed in `docs/decisions.md` during Stages 5b and 5c, before any test result existed. For the DIFF XGBoost the number of rounds (41) was found by early stopping on 2023-24 and 2024-25, never on the test season (the audit file says so).
4. **A dry run on validation seasons** in a scratch folder, to find bugs without touching the test season. It was not a result (the final models had already seen those seasons), and it also reproduced the Stage 5a drift table exactly, which proved the drift code was right.
5. **The one real run**: the test season is loaded once (a counter makes a second load raise an error), 380 matches, all 20 teams, 38 games each, odds for all. Every predictor is scored on those same 380 matches.
6. **The paired bootstrap** (explained below), 25 comparisons.
7. **Diagnostics and write-up**: periods of the season, drift, calibration, Poisson goal metrics, five figures, the report.

## 3. New concepts (explained simply)

- **Pre-registration:** writing down, before the exam, exactly what you will measure and what will count as a win. *Analogy:* a bet. If you decide the rules after the match, you can always find a rule you won. *Link to Stage 2:* we split the data by time into train, validation and test so the test season stays unseen. Pre-registration protects the *decisions* in the same way the split protects the *data*. *Link to KNN:* choosing K by looking at the test set is the same mistake; we choose K (and everything else) on validation only.
- **Tuning phase vs honest test:** the tuning phase is everything on validation (choosing features, depth, windows). The honest test is one scoring of the frozen result. After the honest test, you may *report* anything, but you may not go back and change the model and report the new test score as if it were fresh.
- **Paired bootstrap:** a way to measure how much a difference between two models is just luck. We have 380 matches. Imagine the season replayed 4,000 times, each pretend season made by drawing 380 matches *with replacement* from the real 380 (so some matches appear twice, some never). In each pretend season we recompute the average difference in log loss between model A and model B. "Paired" means both models are compared on the *same* matches in every pretend season, which removes the match-to-match difficulty (a derby is hard for both). *Analogy:* asking "if the season had been slightly different, would the ranking have changed?" *Link to K-means:* running K-means again from different starting points tells you whether the clusters are stable; the bootstrap does the same for a comparison.
- **Tiny example with real numbers:** the first five test matches. The per-match log loss of ELO2 LR is 0.417, 1.298, 1.362, 1.060, 0.765, and of the FULL LR 0.471, 1.225, 1.213, 1.158, 1.030, so the difference (ELO2 minus FULL) per match is -0.054, +0.073, +0.148, -0.099, -0.265, with average -0.039. One pretend season of 5 matches could pick matches 5, 1, 1, 2, 1 (with repeats): the differences -0.265, -0.054, -0.054, +0.073, -0.054 average -0.071. Repeat thousands of times and you get a spread of averages.
- **Confidence interval (here a "95% interval") around a difference:** the range that contains the middle 95% of the pretend-season averages. In plain words: "if we replayed the season, the true gap would usually sit in here". **If the interval contains 0, we cannot say the models differ**: with the data we have, "A is better", "B is better" and "no difference" are all plausible. If the whole interval is below 0, model A is better and it was probably not luck. Real example: ELO2 LR minus FULL LR = +0.0008 [-0.0126, +0.0146]. The average difference is almost zero, and the interval comfortably contains zero: a tie.
- **Significant (our rule):** the 95% interval excludes zero. Never "the number is lower, so it wins".
- **Post-hoc:** Latin for "after this". A choice or a finding made after seeing results. Post-hoc findings are not wrong, but they are weaker evidence, because with enough looking you can find something that looks good by chance. We label them. *Example:* the DIFF models were chosen because they did best on validation among the forests; they are called "diagnostics", their settings were frozen before the test result, and if one had beaten the primary it would have been called "post-hoc best on the test set, not pre-registered", not "the new best model".
- **Why beating the bookmaker is hard:** the bookmaker sees everything we see (form, strength) and also injuries, line-ups, news and the money people bet, so its odds already contain far more than our two Elo numbers; and thousands of bettors correct it. Its odds also carry a margin (the "overround", 5.5% on average in our data: the three implied probabilities add up to 105.5%), which we remove by dividing by the sum. Even so, it is the benchmark that models rarely beat clearly. On the test season it was ahead of ELO2 LR by 0.0149 in log loss, but the interval [-0.0048, +0.0345] contains zero, so we cannot say it was better on these 380 matches.

## 4. How the code works
- `load_test_set(seasons)` in `src/stage6.py`: the only function that reads test rows (features, the level-neutral goal features, and the three Bet365 odds columns). A counter raises an error on a second call, and `check_stage6.py` verifies there is exactly one call site. The Stage 5 guards (`get_slice("test")` raises) were left untouched: `modeling_utils.py` is byte-identical to its Stage 5d version, and the check proves it and proves the old guards still raise errors.
- `refit_diff()`: refits the two diagnostics on `get_slice("train_all")` (training rows only), saves them, writes the early-stopping audit, and asserts that the 20 earlier model files did not change.
- `predict_all(...)`: calls `predict_proba` on each frozen model with the features listed in its own config. Nothing is refit. The class-frequency baseline uses the 10 training seasons' frequencies. The bookmaker turns odds into probabilities (`bookmaker_baseline`, the only function that converts odds).
- `bootstrap_table(...)`: ONE table of 4,000 resamples (seed 42) used for every comparison; the difference is "loss of A minus loss of B", so negative means A is better.
- `decide_winner(...)`: the pre-registered rule as code: the primary wins unless some competitor's interval (against the primary) lies entirely below zero.
- Reading the winner rule in code: `block[block["verdict"] == "A better"]` selects the competitors that are clearly better than ELO2 LR; there were none.

## 5. Results and how to read them

**Headline table** (380 test matches, ordered by log loss; lower is better; ELO2 LR is the pre-registered primary):

| Predictor | Role | Log loss | Accuracy | Macro-F1 |
|---|---|---|---|---|
| Bookmaker (Bet365) | baseline | 1.0185 | 48.9% | 0.365 |
| DIFF Random Forest | diagnostic | 1.0283 | 48.7% | 0.364 |
| Poisson (scoreline grid) | primary | 1.0303 | 48.4% | 0.365 |
| Logistic Regression FULL | reference | 1.0327 | 46.8% | 0.347 |
| **Logistic Regression ELO2** | **primary** | **1.0335** | **49.5%** | **0.371** |
| Random Forest FULL_v2 | reference | 1.0345 | 47.4% | 0.350 |
| DIFF XGBoost | diagnostic | 1.0367 | 47.4% | 0.349 |
| XGBoost FULL_v2 | reference | 1.0383 | 48.2% | 0.355 |
| XGBoost ELO2 | primary | 1.0385 | 49.2% | 0.364 |
| Random Forest ELO2 | primary | 1.0421 | 50.0% | 0.374 |
| Class frequencies | baseline | 1.0838 | 42.6% | 0.199 |
| Always predict home win | baseline | n/a | 42.6% | 0.199 |

- *What am I looking at?* Every predictor on the same 380 matches. *What does it tell us?* Ten predictors lie within 0.025 of each other in log loss and within about 3 points of accuracy; only the two baselines are clearly worse. *What would good look like?* A clear gap: an interval below zero. We did not get one. Figure 39 shows the 95% interval of each bar: they overlap almost completely.
- **The verdict:** the rule says ELO2 LR wins unless something is *significantly* better. **Nothing was** (all eight competitors have intervals that contain zero), so the pre-registered primary stands: a successful pre-registration. But read it correctly: it is a **tie**, not a victory. ELO2 LR is fifth by point estimate, and we cannot separate it from the bookmaker, from Poisson, or from any model with 33 features.
- **Accuracy is lower than expected.** 49.5% against the 50 to 55% in our notes (validation: 55.4%). The bookmaker was also at 48.9%, so this season was harder: 27.4% of matches were draws (training: 23.3%) and models never pick draws. We report this plainly instead of hunting for a better-looking number.
- **Confusion matrices (figure 37).** Rows = what happened, columns = predicted. For ELO2 LR: Home 127 / 0 / 35; Draw 61 / 0 / 43; Away 53 / 0 / 61. The Draw column is all zeros: not one match was predicted as a draw (the other models predicted at most one). A good result would show a strong diagonal including the middle cell; the Draw row shows that all 104 draws were assigned to Home or Away.
- **Calibration (figure 38).** Each dot groups the matches by predicted home-win probability (5 equal groups) and shows how often the home team really won; the diagonal is perfect. All five predictors follow the diagonal loosely, with wide error bars, and all over-predict home wins slightly overall (43.7% to 45.5% predicted against 42.6% real).
- **Poisson goal metrics** (figure 40: predicted vs actual goals): MAE 0.948 (home), 0.843 (away), 1.245 (total); deviance 1.0724; exact scoreline right in 12.1% of matches. These are the test numbers Stage 7 will have to match; remember they use real earlier-2025-26 form, which a pre-season simulation does not have.
- **Drift (figure 41, diagnostic only):** only 4 of 34 features reached beyond 3 standard deviations on the test season (22 of 34 on validation), and the league scoring-level feature that drifted by 4.5 standard deviations on validation drifted by only 1.1. The validation drift pattern did not repeat, so the test season did not really test the Stage 5a amendment. No model was changed because of it.
- **The 2024-25 pattern:** in 2024-25 the Logistic Regression clearly beat the forests. On 2025-26 (also a low-home-win season, 42.6%) the direction repeated (ELO2 LR is the best of the three ELO2 primaries), but the gaps (+0.005 and +0.009) are inside the noise. Honest summary: the clear edge did not repeat.

## 6. Common mistakes and gotchas
- **Peeking and re-tuning.** The classic mistake: look at the test result, adjust a setting, score again. We prevented it by a one-shot loader, by hashing all model files before and after, and by a dry run on validation for debugging. The test table was never loaded in any debugging step.
- **Calling the lowest number the winner.** DIFF Random Forest has a lower log loss than ELO2 LR, but by -0.0052 with an interval of [-0.0170, +0.0057]. That is noise.
- **Reading "no significant difference" as "they are equal".** It means we do not have enough data to tell. 380 matches can only detect gaps of about 0.012 to 0.02 in log loss.
- **Choosing the Elo snapshot with the test season.** Stage 7 needs a choice between two pre-season tables (Elo `keep` 0.9 or 0.75). The test results contain no information about that (neither table was scored), so choosing on them would be tuning on the test set. We kept the default.
- **Odds as a feature.** The bookmaker is only a benchmark: `check_stage6.py` scans every script and every model's feature list to prove odds are touched only by the baseline.
- **Mixing up per-match and aggregate comparisons.** The bootstrap resamples whole matches, because the two models' losses on the same match belong together.

## 7. Check yourself
1. What does it mean for a result to be "pre-registered", and which part of the Stage 2 train/validation/test split does it protect?
2. In your own words, what is a paired bootstrap, and why "paired"?
3. ELO2 LR minus the bookmaker is +0.0149 with the interval [-0.0048, +0.0345]. Can we say the bookmaker is better? What would we need to see?
4. What does "post-hoc" mean, and why is DIFF Random Forest labelled a diagnostic and not a winner?
5. Why is it so hard for a model with only team-strength and form numbers to beat the bookmaker?

## 8. Try it yourself
Open `src/try_bootstrap.py`. It computes, in about 100 bootstrap resamples, the interval of the 2025-26 log-loss difference between ELO2 LR and the FULL LR reference. **Before running it, predict:** will the interval contain zero? Roughly how wide will it be (more than 0.001? more than 0.05?)? Then run it from the project folder (`python src/try_bootstrap.py`) and compare with the 4,000-resample interval in the report ([-0.0126, +0.0146]). Extension: change `range(100)` to `range(10)`, then to `range(2000)`, and see how the interval's ends move; then change the seed from 1 to 2. Which of the two changes moves the interval more, the number of resamples or the seed?

## 9. How this connects to the report
This stage feeds the **Model comparison** section (requirement 6: "suggest the best model and explain why") and the **Evaluation** section. *Draft paragraph:* "We compared four model families (Logistic Regression, Random Forest, XGBoost and Poisson regression), each on the same 380 matches of the 2025-26 season, after fixing the rules in writing before the test season was loaded. The best model by the pre-registered rule is the Logistic Regression on the Elo difference (log loss 1.0335, accuracy 49.5%): none of the more complex models, nor the Poisson model, was significantly better, and its log loss was not significantly different from the bookmaker's (1.0185; difference +0.0149, 95% interval -0.0048 to +0.0345). We therefore recommend the simplest model, whose two inputs and single formula are easy to explain. The Poisson model, which predicts expected goals (MAE 1.245 on total goals), is used for the score predictions and the season simulation. The test season was harder than the validation seasons (27.4% draws), and every model scored about 47 to 50% accuracy."

---

## Answers to "Check yourself"
1. It means writing down the models, the metrics and the rule for declaring a winner *before* looking at the test results. It protects the decisions made after the data split: Stage 2 keeps the test *data* unseen, pre-registration keeps the test *decisions* (which model wins) from being bent after seeing the scores.
2. Draw many pretend seasons by picking 380 matches with replacement from the real 380, and in each one compute the average difference in loss between two models; the spread of those averages shows how much the difference depends on luck. "Paired" means both models are scored on the same matches in each pretend season, so the difficulty of the matches cancels out.
3. No. The interval contains zero, so on these matches the bookmaker could be better, equal, or slightly worse. To say the bookmaker is better we would need an interval entirely above zero for ELO2 LR minus the bookmaker (as on validation, where the gap was +0.0182 with the interval [0.0066, 0.0292]) or more matches (a longer season or more seasons) to narrow the interval.
4. "Post-hoc" means decided or found after seeing results. DIFF Random Forest was selected because it did best on validation (a post-hoc choice), and it is not part of the pre-registered primaries, so even if it beat the primary it could only be called "post-hoc best on the test set, not pre-registered". In fact it did not beat it significantly.
5. The bookmaker sees injuries, line-ups, news and betting volume as well as form, and the market keeps correcting its prices; our models only see past results.
