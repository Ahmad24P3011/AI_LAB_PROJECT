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
