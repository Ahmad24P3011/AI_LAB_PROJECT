"""Checks for Stage 5d (Poisson goal model). Run after:
  python src/features_goals.py ; python src/poisson.py

Each check prints PASS only after its assertion succeeded.
"""
import hashlib
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from features import MATCHES_CSV, PARAMS_JSON
from features_goals import (GOAL_FEATURES_CSV, GOAL_FEATURE_COLUMNS, GOAL_PARAMS_JSON, REL_COLUMNS,
                            build_goal_features)
from modeling_utils import (FEATURE_SETS_V1_JSON, FEATURE_SETS_V2_JSON, FEATURE_SETS_V3_JSON, FIT_SEASONS,
                            TEST_SEASONS, TRAIN_SEASONS, TestSetAccessError, assert_no_test_rows, get_slice)
from poisson import (ABLATIONS_CSV, CONFIG_PATH, ELO_START_CSV, FINAL_CSV, FINAL_END, FIT_END, LAMBDA_CAP, PRE_SEASON_075_CSV,
                     PRE_SEASON_CSV, VAL_LAMBDAS_FINAL_CSV, elo_start_snapshot,
                     evaluate_model, fit_cfg, load_fixtures_2025_26, load_goal_rows, make_cfg)
from poisson_model import MAX_GOALS, PoissonGoalModel, outcome_probs, score_grid

V_SHA = {"v1": "bc63b1f3f02615457f83834a5a4fe0b07033750e8ad08e5b26c2e839a5fea87b",
         "v2": "68280d590d5f42510f994e233b8b4608655dae7dd26ffaa97fc9a774744eecc0",
         "v3": "9ca210478c2e9fa25c254e6db557ac7c578a014e366cf4a686eb72fc86b4e946"}
OUTCOME_COLUMNS = {"FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "home_goals", "away_goals", "result", "HS", "AS",
                   "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"}


def passed(message):
    print(f"PASS  {message}")


rows = load_goal_rows()
fit, val, train_all = get_slice("fit", rows), get_slice("validation", rows), get_slice("train_all", rows)
config = json.loads(CONFIG_PATH.read_text())
params = json.loads(PARAMS_JSON.read_text())
goal_defaults = json.loads(GOAL_PARAMS_JSON.read_text())["goal_defaults"]

# ------------------------------------------------ the test set is never used for fitting or scoring
assert not (rows["split"] == "test").any() and not rows["season"].isin(TEST_SEASONS).any()
for table in (fit, val, train_all):
    assert not table["season"].isin(TEST_SEASONS).any() and (table["Date"] < "2025-07-01").all()
for forbidden in ("test", "2025-26"):
    try:
        get_slice(forbidden)
        raise AssertionError(f"get_slice('{forbidden}') should have raised")
    except TestSetAccessError:
        pass
fake = fit.iloc[:5].assign(season="2025-26", split="test")
for action in (lambda: assert_no_test_rows(fake), lambda: evaluate_model(PoissonGoalModel.load(CONFIG_PATH, load_results=False), fake),
               lambda: fit_cfg(make_cfg(config["spec"]), fake, FIT_END)):
    try:
        action()
        raise AssertionError("a test row was accepted")
    except TestSetAccessError:
        pass
assert set(config["window_seasons"]) <= set(TRAIN_SEASONS) and not set(config["window_seasons"]) & set(TEST_SEASONS)
passed("no 2025-26 row in any slice (fit, validation, train_all, the saved model's window); evaluating or fitting on a test row raises TestSetAccessError")

assert fit["Date"].max() < val["Date"].min()
passed(f"latest fit date {fit['Date'].max().date()} is before earliest validation date {val['Date'].min().date()}")

for name, path in (("v1", FEATURE_SETS_V1_JSON), ("v2", FEATURE_SETS_V2_JSON), ("v3", FEATURE_SETS_V3_JSON)):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == V_SHA[name]
passed("the frozen feature-set files v1, v2 and v3 are unchanged (hashes match)")

# ------------------------------------------------------------ the new feature table
n_goals = sum(len(c) for c in pd.read_csv(GOAL_FEATURES_CSV, usecols=["match_id"], chunksize=1000))
n_matches = sum(len(c) for c in pd.read_csv(MATCHES_CSV, usecols=["match_id"], chunksize=1000))
n_features = sum(len(c) for c in pd.read_csv(MATCHES_CSV.parent / "features.csv", usecols=["match_id"], chunksize=1000))
assert n_goals == n_matches == n_features == 4180
passed(f"features_goals.csv has {n_goals} rows = matches_clean.csv = features.csv (features.csv itself was not changed)")

assert rows[GOAL_FEATURE_COLUMNS].notna().all().all() and np.isfinite(rows[GOAL_FEATURE_COLUMNS]).all().all()
assert (rows[REL_COLUMNS] >= 0).all().all() and rows[REL_COLUMNS].max().max() < 6
assert abs(rows[REL_COLUMNS].mean().mean() - 1.0) < 0.02
by_season = rows.groupby("season")["home_gf_10_rel"].mean()
assert by_season.between(0.95, 1.05).all() and rows.groupby("season")["home_gf_10"].mean().max() - rows.groupby("season")["home_gf_10"].mean().min() > 0.2
passed("the _rel features have no missing values, average 1.0, and stay near 1.0 in every season while the raw goal averages "
       f"differ by {rows.groupby('season')['home_gf_10'].mean().max() - rows.groupby('season')['home_gf_10'].mean().min():.2f} goals between seasons")

# --------------------------------------------------------- perturbation test on the _rel features
matches = pd.concat([c[c["split"] == "train"] for c in pd.read_csv(MATCHES_CSV, parse_dates=["Date"], chunksize=500)], ignore_index=True)
assert not matches["season"].isin(TEST_SEASONS).any()
base = build_goal_features(matches, params["elo"], params["defaults"], goal_defaults)     # also asserts every parent feature is identical
assert np.allclose(base[GOAL_FEATURE_COLUMNS].to_numpy(float), rows[GOAL_FEATURE_COLUMNS].to_numpy(float), rtol=0, atol=1e-9)
passed("rebuilding the goal features reproduces features_goals.csv, and each _rel feature is computed from the SAME prior "
       "matches as its parent feature (all 8 parents recomputed and compared with features.csv)")


def perturb(frame, i):
    changed = frame.copy()
    changed.loc[i, "home_goals"] = frame.loc[i, "home_goals"] + 3
    changed.loc[i, "away_goals"] = max(frame.loc[i, "away_goals"] - 1, 0)
    hg, ag = changed.loc[i, "home_goals"], changed.loc[i, "away_goals"]
    changed.loc[i, "result"] = 0 if hg > ag else (1 if hg == ag else 2)
    for column, extra in [("HS", 7), ("AS", 5), ("HST", 3), ("AST", 2)]:
        changed.loc[i, column] = frame.loc[i, column] + extra
    return changed


print("\nPerturbation tests for the _rel features (change one match's score and shots, rebuild everything):")
rng = np.random.default_rng(11)
for i in rng.choice(matches.index[matches["Date"] < matches["Date"].max()], size=4, replace=False):
    date = matches.loc[i, "Date"]
    rebuilt = build_goal_features(perturb(matches, i), params["elo"], params["defaults"], goal_defaults)
    not_later = (matches["Date"] <= date).to_numpy()
    assert np.array_equal(base.loc[not_later, GOAL_FEATURE_COLUMNS].to_numpy(), rebuilt.loc[not_later, GOAL_FEATURE_COLUMNS].to_numpy()), f"match {i} leaks"
    later = int((base.loc[~not_later, GOAL_FEATURE_COLUMNS].to_numpy() != rebuilt.loc[~not_later, GOAL_FEATURE_COLUMNS].to_numpy()).any(axis=1).sum())
    assert later > 0
    print(f"  match {i:4} ({matches.loc[i, 'HomeTeam']} v {matches.loc[i, 'AwayTeam']}, {date.date()}): goal features unchanged for all "
          f"{int(not_later.sum())} matches on or before this date; {later} later matches changed")
passed("changing a later match's result never changes the goal features of that match or of any earlier match")

# ------------------------------------------------------------ the scoreline grid and Dixon-Coles
sample = val.sample(60, random_state=3)
lam_h, lam_a = np.exp(np.random.default_rng(1).uniform(-0.3, 1.2, 200)), np.exp(np.random.default_rng(2).uniform(-0.4, 1.0, 200))
for rho in (0.0, -0.0294, 0.05):
    grid, mass = score_grid(lam_h, lam_a, rho)
    assert np.allclose(grid.sum(axis=(1, 2)), 1.0, atol=1e-12) and (grid >= 0).all()
    assert np.allclose(outcome_probs(grid).sum(axis=1), 1.0, atol=1e-12)
independent, mass0 = score_grid(lam_h, lam_a, 0.0)
assert (mass0 > 0.99).all()
assert mass0.min() > 0.999 or lam_h.max() > 3
saved_val = pd.read_csv(VAL_LAMBDAS_FINAL_CSV)       # the deployed configuration, fitted on the fit seasons
_, mass_val = score_grid(saved_val["lambda_home"].to_numpy(), saved_val["lambda_away"].to_numpy(), config["rho"])
assert mass_val.min() > 0.995
passed(f"the {MAX_GOALS + 1} x {MAX_GOALS + 1} grids sum to 1 (after normalising) for 200 random rate pairs and 3 values of rho; "
       f"before normalising the grid holds at least {mass_val.min():.4f} of the probability on all 760 validation matches")

adjusted, _ = score_grid(lam_h, lam_a, -0.0294)
assert np.allclose(adjusted.sum(axis=2), independent.sum(axis=2), atol=1e-12)      # home-goals marginal
assert np.allclose(adjusted.sum(axis=1), independent.sum(axis=1), atol=1e-12)      # away-goals marginal
k = np.arange(MAX_GOALS + 1)
assert np.allclose((adjusted.sum(axis=2) * k).sum(axis=1), (independent.sum(axis=2) * k).sum(axis=1), atol=1e-12)
assert not np.allclose(adjusted, independent)
passed("the Dixon-Coles adjustment changes the four lowest scorelines but leaves both marginal goal distributions (and so "
       "the expected goals lambda_home and lambda_away) unchanged")

# ------------------------------------------------------------ saved models reproduce their predictions
model = PoissonGoalModel.load(CONFIG_PATH)
for part in model.parts.values():
    assert part["results"] is not None and isinstance(part["results"].model.family.link, sm.families.links.Log)
    assert np.allclose(part["results"].params[0], part["params"][0])
assert config["spec"] == model.spec
final_cfg = make_cfg(config["spec"], window=None if len(config["window_seasons"]) == len(TRAIN_SEASONS) else len(config["window_seasons"]),
                     recency=config["recency_weights"], rho_mode=config["rho_mode"])
refit, used = fit_cfg(final_cfg, train_all, FINAL_END)
assert len(used) == config["n_training_rows"] and abs(refit.rho - model.rho) < 1e-9
assert config["lambda_cap"] == LAMBDA_CAP == 3.5 and model.lambda_cap == 3.5 and model.rho == 0.0
assert list(config["window_seasons"]) == list(TRAIN_SEASONS)        # decided after review: the final model uses all training seasons
refit.lambda_cap = config["lambda_cap"]
lh_saved, la_saved = model.predict(val)
lh_new, la_new = refit.predict(val)
assert np.allclose(lh_saved, lh_new, atol=1e-9) and np.allclose(la_saved, la_new, atol=1e-9)
assert np.allclose(outcome_probs(score_grid(lh_saved, la_saved, model.rho)[0]), outcome_probs(score_grid(lh_new, la_new, refit.rho)[0]), atol=1e-9)
assert (lh_saved > 0).all() and (la_saved > 0).all()
fit_only, _ = fit_cfg(final_cfg, fit, FIT_END)
fit_only.lambda_cap = config["lambda_cap"]
lh_fit_only, la_fit_only = fit_only.predict(val)
proba_fit_only = outcome_probs(score_grid(lh_fit_only, la_fit_only, fit_only.rho)[0])
assert np.allclose(lh_fit_only, saved_val["lambda_home"], atol=1e-9) and np.allclose(la_fit_only, saved_val["lambda_away"], atol=1e-9)
assert np.allclose(proba_fit_only, saved_val[["poisson_p_home", "poisson_p_draw", "poisson_p_away"]].to_numpy(), atol=1e-9)
passed("the saved model (loaded from JSON and joblib, log link) equals a fresh refit from its config; refitting on the fit seasons "
       "reproduces the saved validation lambdas and the bivariate-grid outcome probabilities")

# ------------------------------------------------------------ the safety cap on lambda
raw_h, raw_a = model.predict_uncapped(val)
cap_h, cap_a = model.predict(val)
assert np.allclose(cap_h, np.minimum(raw_h, 3.5)) and np.allclose(cap_a, np.minimum(raw_a, 3.5))
assert cap_h.max() <= 3.5 and cap_a.max() <= 3.5 and max(raw_h.max(), raw_a.max()) > 3.5     # the cap really does something
below = (raw_h <= 3.5) & (raw_a <= 3.5)
assert np.allclose(cap_h[below], raw_h[below]) and np.allclose(cap_a[below], raw_a[below])      # and changes nothing else
n_over = int((~below).sum())
assert n_over <= 15
loaded_no_cap = PoissonGoalModel.load(CONFIG_PATH)
loaded_no_cap.lambda_cap = None
assert loaded_no_cap.predict(val)[0].max() > 3.5                                                # the cap lives in the model, not in a caller
passed(f"the lambda cap of 3.5 is saved in the config and applied inside predict(): both rates are min(raw, 3.5), rates at or below 3.5 are "
       f"untouched; the logged count of capped validation matches ({int(pd.read_csv(FINAL_CSV)['val_matches_capped'].iloc[-1])} of 760) comes from the model "
       f"fitted on the fit seasons (the final model, refit on all ten seasons, would cap {n_over} of them in-sample)")

# ------------------------------------------------------------ the two start-of-season Elo snapshots
pre9 = pd.read_csv(PRE_SEASON_CSV, parse_dates=["Date"])
pre75 = pd.read_csv(PRE_SEASON_075_CSV, parse_dates=["Date"])
assert len(pre75) == len(pre9) == 380 and (pre9[["fixture_id", "HomeTeam", "AwayTeam"]] == pre75[["fixture_id", "HomeTeam", "AwayTeam"]]).all().all()
assert not set(pre75.columns) & OUTCOME_COLUMNS and list(pre75.columns) == list(pre9.columns)
elo9, elo75 = elo_start_snapshot(0.9), elo_start_snapshot(0.75)
saved_elo9 = pd.read_csv(ELO_START_CSV).set_index("team")["elo_start"]
assert np.allclose(elo9.sort_index(), saved_elo9.sort_index(), atol=1e-9)                          # keep 0.9 = the Stage 4 file
assert np.allclose(pre75["elo_home_start"], pre75["HomeTeam"].map(elo75)) and np.allclose(pre75["elo_away_start"], pre75["AwayTeam"].map(elo75))
assert not np.allclose(elo9.sort_index(), elo75.sort_index())
assert elo75.max() < elo9.max() and elo75.min() >= elo9.min() - 1e-9                                  # keep 0.75 pulls ratings toward 1500
assert (pre75[["lambda_home", "lambda_away"]] <= 3.5).all().all() and (pre75["lambda_home"] > 0).all()
assert np.allclose(pre75["league_avg_goals_prev380"], pre9["league_avg_goals_prev380"])
frame75 = pre75[["fixture_id"]].copy()
for column in [c for c in model.required_columns if c.endswith("_rel")]:
    frame75[column] = 1.0
frame75["elo_diff"], frame75["abs_elo_diff"] = pre75["elo_diff"], pre75["abs_elo_diff"]
frame75["league_avg_goals_prev380"], frame75["season_no_fans"] = pre75["league_avg_goals_prev380"], 0
lh75, la75 = model.predict(frame75)
assert np.allclose(lh75, pre75["lambda_home"], atol=1e-9) and np.allclose(la75, pre75["lambda_away"], atol=1e-9)
fh, fa = fit_only.predict_uncapped(val)       # the model fitted on the fit seasons only: its count is the one logged in poisson_final_5d.csv
assert int(((fh > 3.5) | (fa > 3.5)).sum()) == int(pd.read_csv(FINAL_CSV)["val_matches_capped"].iloc[-1])
passed("two pre-season tables (keep 0.9 and keep 0.75): same 380 fixtures, no result columns, keep 0.9 equals the Stage 4 Elo file, keep 0.75 is "
       "recomputed from matches before 2025-26 only, and each table's lambdas equal what the saved model predicts from its own Elo snapshot")

# ------------------------------------------------------------ the Stage 7 handoff table
pre = pd.read_csv(PRE_SEASON_CSV, parse_dates=["Date"])
elo = pd.read_csv(ELO_START_CSV).set_index("team")["elo_start"]
fixtures = load_fixtures_2025_26()
assert list(fixtures.columns) == ["fixture_id", "Date", "HomeTeam", "AwayTeam"]       # no outcome column is ever read
assert len(pre) == len(fixtures) == 380 and not set(pre.columns) & OUTCOME_COLUMNS
teams = pd.concat([pre["HomeTeam"], pre["AwayTeam"]]).value_counts()
assert len(teams) == 20 and (teams == 38).all() and set(teams.index) == set(elo.index)
assert np.allclose(pre["elo_home_start"], pre["HomeTeam"].map(elo)) and np.allclose(pre["elo_away_start"], pre["AwayTeam"].map(elo))
assert np.allclose(pre["elo_diff"], pre["elo_home_start"] - pre["elo_away_start"])
level = (train_all.sort_values("Date").tail(380)[["home_goals", "away_goals"]].sum(axis=1)).mean()
assert np.allclose(pre["league_avg_goals_prev380"], level)
assert (pre["lambda_home"] > 0).all() and (pre["lambda_away"] > 0).all() and np.allclose(pre[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0, atol=1e-9)
neutral = pre[["fixture_id"]].copy()
for column in REL_COLUMNS[:-4]:
    neutral[column] = 1.0
for column in ["home_venue_gf_rel", "home_venue_ga_rel", "away_venue_gf_rel", "away_venue_ga_rel"]:
    neutral[column] = 1.0
neutral["elo_diff"], neutral["abs_elo_diff"] = pre["elo_diff"], pre["abs_elo_diff"]
neutral["league_avg_goals_prev380"], neutral["season_no_fans"] = pre["league_avg_goals_prev380"], 0
lh_pre, la_pre = model.predict(neutral)
assert np.allclose(lh_pre, pre["lambda_home"], atol=1e-9) and np.allclose(la_pre, pre["lambda_away"], atol=1e-9)
passed("poisson_pre_season_2025-26.csv: 380 fixtures, 20 teams with 38 matches each, no result columns, Elo from the Stage 4 start-of-season "
       "snapshot, all lambdas positive, and the lambdas equal what the saved model predicts from the neutral start-of-season features")

# ------------------------------------------------------------ results table
results = pd.read_csv(ABLATIONS_CSV)
assert {"a. features", "b. league scoring level", "c. training window", "d. Dixon-Coles rho", "e. team fixed effects",
        "final combined"} <= set(results["experiment"])
assert {"improvement_vs_default", "ci_low", "ci_high", "decision", "val_deviance", "val_log_loss"} <= set(results.columns)
assert set(results[results["decision"] == "CHOSEN"]["experiment"]) == {"a. features", "b. league scoring level", "c. training window",
                                                                        "d. Dixon-Coles rho", "e. team fixed effects"}
passed("ablations_poisson.csv has experiments a to e, the combination and the diagnostics, with improvement, interval and decision columns")

print("\nAll checks passed.")
