# Premier League Prediction (AI Lab, Group #8)

We build a machine learning system that predicts Premier League match results
and simulates a full season table. We train on past seasons (up to 2024-25) and
test on the 2025-26 season, which has already been played, so we can compare our
predictions with what really happened. The final model is deployed in a simple
Streamlit web app with a match predictor, a simulated season table and player
leaderboards. Everything uses free tools and free data.

## Folder layout

```
CLAUDE.md
README.md
requirements.txt
template/           reference files (requirements, plan, lesson template), do not edit
data/raw/           original downloaded files, never edited
data/processed/     cleaned data and features
notebooks/          optional exploration notebooks
src/                reusable Python code
models/             saved trained models
reports/figures/    saved plots
app/                Streamlit app
docs/learning/      lesson files (one per stage)
```

## How to set up

Windows (PowerShell):
```
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/check_setup.py
```

Mac / Linux:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/check_setup.py
```

The last command should end with `Setup OK`.

## Run the app (later, Stage 9)

```
streamlit run app/app.py
```

## Poisson goal model: how Stage 7 (season simulation) should call it

The Stage 5d model turns a match into two expected-goal rates, `lambda_home` and `lambda_away`. Stage 7 samples scores
from them.

**Files**
- `models/poisson_config.json`: the model type, the feature columns, the scaler (mean and scale), the coefficients, `rho`
  (Dixon-Coles) and the training window. `models/poisson_stacked.joblib`: the fitted statsmodels result (log link).
- `src/poisson_model.py`: `PoissonGoalModel`, `score_grid`, `outcome_probs`, `top_scorelines`. Stage 7 only needs this file.
- `data/processed/poisson_pre_season_2025-26.csv` (start-of-season Elo with `keep` = 0.9, the default) and
  `data/processed/poisson_pre_season_2025-26_keep075.csv` (`keep` = 0.75): ready-made rates for all 380 fixtures of 2025-26
  (see mode 1). Stage 7 picks one. Their `elo_home_start`/`elo_away_start` columns say which snapshot they use.

**Inputs (one row per match)**: the columns in `PoissonGoalModel.load("models/poisson_config.json").required_columns`:
`home_gf_10_rel`, `away_gf_10_rel`, `home_ga_10_rel`, `away_ga_10_rel`, `home_venue_gf_rel`, `home_venue_ga_rel`,
`away_venue_gf_rel`, `away_venue_ga_rel` (each 1.0 = league average), `elo_diff` (home Elo minus away Elo, before the match),
`abs_elo_diff`, `league_avg_goals_prev380` (average total goals in the previous 380 matches; the chosen model uses it as a
fixed offset, log(level / 2)) and `season_no_fans` (0).

**Outputs**
```python
import sys; sys.path.insert(0, "src")                           # run from the project folder
from poisson_model import PoissonGoalModel, score_grid, outcome_probs
model = PoissonGoalModel.load("models/poisson_config.json")
lambda_home, lambda_away = model.predict(features_table)        # expected goals, one pair per row
grid, mass = score_grid(lambda_home, lambda_away, model.rho)    # grid[n, i, j] = P(home scores i, away scores j)
p_home_draw_away = outcome_probs(grid)                          # summed Home / Draw / Away probabilities
```
`model.predict` already limits both rates to `model.lambda_cap` (3.5, saved in the config); `model.predict_uncapped(...)` gives the raw
rates. Do not cap again in Stage 7.
To simulate one match: `rng.poisson(lambda_home), rng.poisson(lambda_away)` (exactly the model when `model.rho` is 0, which it is
in the current configuration). If `rho` is not 0, sample a cell of `grid` instead (`rng.choice(121, p=grid[n].ravel())`).
Over 1,000 simulations of a 380-match season, add the points and read the table, title / top 4 / relegation shares.

**The two Stage 7 modes** (you choose in Stage 7; do not use real 2025-26 results to build features and call it a simulation):
1. *Frozen pre-season.* Use `poisson_pre_season_2025-26.csv` as it is: one fixed pair of rates per fixture, computed from
   the start-of-2025-26 Elo snapshot (`data/processed/elo_start_of_2025-26.csv`, built from data before 2025-26 only), neutral
   form features (all `_rel` = 1.0) and the 2024-25 league scoring level. Simple and honest, but team strengths never
   change during the simulated season, so final tables are less spread out than real ones. Use the `keep075` file for the
   alternative Elo snapshot. (The model's Elo weight was learned on keep 0.9 ratings, so keep 0.75 is a changed input scale, not a
   retrained model.)
2. *Matchweek by matchweek.* After each simulated matchweek, update the features from the SIMULATED results: Elo with the
   Stage 4 rule and constants (`data/processed/feature_params.json`), the rolling `_rel` features with
   `features_goals.build_goal_features` (or an incremental version of it), and the league level; then recompute the rates for
   the next matchweek. More realistic, but it needs a feature updater and simulated results feed back into later rates.

**Known limits.** The log link grows exponentially with the Elo gap, so rates for extreme mismatches can be too high (the
largest validation rate was 4.3 goals), which is why the model caps lambda at 3.5 (7 of 760 validation matches were affected). The
model is trained on all ten training seasons (2015-16 to 2024-25).

## Stage 6 results (model comparison on the 2025-26 test season)

All predictors were scored once on the same 380 test matches, under rules written down before the test set was loaded
(`reports/stage6_preregistration.md`). The pre-registered primary, the Logistic Regression on the Elo difference, was not beaten by any
competitor (a statistical tie: log loss 1.0335, accuracy 49.5%; bookmaker 1.0185, 48.9%). Full table, intervals and figures:
`reports/stage6_results.md`, `reports/stage6_results.csv`, `reports/stage6_bootstrap.csv`, figures 37 to 41. Check with
`python src/check_stage6.py`.
