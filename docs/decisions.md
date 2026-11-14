# Decisions log

A running record of the choices made in this project, why, and their status.
(This file did not exist before Stage 5a. It was created then and seeded from the
decisions of Stages 1 to 4, so earlier entries are a summary.)

Status: **settled** = decided and used; **open** = waiting for the team's decision.

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
| 4 | Head-to-head features kept in the table | Plan said to test them on validation data | settled in 5a (see below) |

---

## Stage 5a: baselines and Logistic Regression

All choices below were made on the validation seasons (2023-24, 2024-25, 760 matches) with
models fitted on 2015-16 to 2022-23. The 2025-26 test season was not touched.

**Noise rule used:** adopt a change only if the 95% paired-bootstrap interval of the
validation log-loss improvement excludes zero. If it does not, keep the simpler option and
say the data cannot tell the options apart. One standard error of accuracy on 760 matches is
about 1.8 points, so accuracy alone never decides anything. With about 9 comparisons, roughly
one "clearly better" result is expected by chance at the 95% level, so a borderline hit is weak
evidence.

| Question | Result (improvement in validation log loss vs the default; positive = better) | Decision |
|---|---|---|
| Feature set (tuned C each) | FULL (34, default) 0.9608; REDUCED (27) +0.0016 [-0.0007, +0.0040]; SMALL (3) -0.0055 [-0.0163, +0.0045] | The data cannot tell them apart. By the noise rule the simplest, **SMALL**, is chosen. **Open for review** (see below) |
| Head-to-head features | Without h2h: +0.0025 [+0.0004, +0.0048] | Interval just excludes zero, so "without" is clearly better by the rule, but it is borderline. Moot for the chosen SMALL set |
| `season_no_fans` | Exclude the feature: -0.0033 [-0.0069, -0.0002] (clearly worse). Keep it but drop 2020-21 rows: +0.0017 [-0.0013, +0.0047] (cannot tell) | **Keep the feature as it is** (the default). Dropping 2020-21 rows is not needed |
| Training window (from 2022-23 back) | Last 3: -0.0254 [-0.0375, -0.0140] (clearly worse); last 5: -0.0018 [-0.0096, +0.0054] (cannot tell); all 8 with recency weights: -0.0153 [-0.0241, -0.0071] (clearly worse) | **Use all seasons, no weights** |
| Warm-up rows in training | Exclude: -0.0140 [-0.0211, -0.0075] (clearly worse) | **Keep warm-up rows.** But see the diagnostic: the effect comes from `league_avg_goals_prev380`, not from the warm-up rows themselves |
| Regularization C (FULL) | C = 0.001, 0.01, 0.1, 1, 10 gave validation log loss 0.9615, 0.9608, 0.9669, 0.9690, 0.9699 | C = 0.01 for FULL; SMALL needs almost no regularization (C = 10) |
| `class_weight="balanced"` | Predicts 104 draws instead of 1, accuracy 53.3% vs 55.4%, log loss 0.9673 vs 0.9608 | Keep the unweighted model (honest probabilities) |

**Final Logistic Regression configuration (from the rule):** the SMALL feature set
(`elo_diff`, `abs_elo_diff`, `league_avg_goals_prev380`), C = 10, all seasons, no weights, warm-up rows
included. Validation log loss 0.9663, accuracy 55.3%. Refit on 2015-16 to 2024-25 and saved
(`models/logreg.joblib`). Not scored on the test season.

**Diagnostic finding (not used for any decision).** `league_avg_goals_prev380` is far outside its
training range in the validation seasons: training mean 2.75 and standard deviation 0.08, validation mean 3.11,
so the typical validation value is 5.3 training standard deviations away (maximum 6.7). A linear model
extrapolates from such values. Evidence: the warm-up exclusion result vanishes once the feature is removed
(-0.0003 [-0.0021, +0.0015]); FULL without it scores 0.9577 vs 0.9608 (+0.0032 [-0.0009, +0.0074]); SMALL without it
(just `elo_diff` and `abs_elo_diff`) scores 0.9582 vs 0.9663 for SMALL. None of these differences passes the noise rule on its own,
but they all point the same way.

**Open decisions for the team (Stage 5a):**
1. Keep SMALL (rule result, simplest, easy to explain), or use REDUCED (27 features, lowest validation
   log loss 0.9592 but not clearly better), or SMALL without `league_avg_goals_prev380` (2 features, 0.9582)?
   Recommendation: keep the rule's result for Stage 6, report the others, and decide about `league_avg_goals_prev380`
   together, because tree models (Stages 5b and 5c) are not hurt by this extrapolation but Logistic Regression is.
2. Whether `league_avg_goals_prev380` should be redefined or dropped for linear models.
