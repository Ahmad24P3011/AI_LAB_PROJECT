"""Checks for Stage 5b (Random Forest). Run after:
  python src/modeling_utils.py ; python src/rf.py

Each check prints PASS only after its assertion succeeded.
"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from features import FEATURE_COLUMNS, MATCHES_CSV, PARAMS_JSON, build_features
from modeling_utils import (CLASS_CODES, DIFF_FEATURES, DIFF_PAIRS, FEATURE_SETS_V1_JSON,
                            FEATURE_SETS_V2_JSON, FEATURE_SETS_V3_JSON, FIT_SEASONS, LEAGUE_AVG,
                            PROBA_COLUMNS, TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS, TestSetAccessError,
                            add_diff_features, get_slice, load_feature_sets, load_features, make_xy,
                            verify_diff_is_row_local)
from rf import (ABLATIONS_CSV, CONFIG_PATH, FIT_END, FINAL_END, MAX_MODEL_MB, MODEL_PATH,
                REFERENCE_CONFIG_PATH, REFERENCE_MODEL_PATH, VAL_PREDICTIONS_CSV, fit_rf, make_rf_config,
                rf_proba)

# SHA-256 hashes recorded when each file was frozen.
V1_SHA = "bc63b1f3f02615457f83834a5a4fe0b07033750e8ad08e5b26c2e839a5fea87b"
V2_SHA = "68280d590d5f42510f994e233b8b4608655dae7dd26ffaa97fc9a774744eecc0"
V3_SHA = "9ca210478c2e9fa25c254e6db557ac7c578a014e366cf4a686eb72fc86b4e946"
DIFF_COLUMNS = [f"diff_{x}" for x in DIFF_PAIRS]


def passed(message):
    print(f"PASS  {message}")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


raw = load_features()                              # training rows only
rows = add_diff_features(raw)
fit, val, train_all = get_slice("fit", rows), get_slice("validation", rows), get_slice("train_all", rows)
v2, v3 = load_feature_sets(version=2), load_feature_sets(version=3)
configs = {"primary": json.loads(CONFIG_PATH.read_text()), "reference": json.loads(REFERENCE_CONFIG_PATH.read_text())}
models = {"primary": joblib.load(MODEL_PATH), "reference": joblib.load(REFERENCE_MODEL_PATH)}
saved = pd.read_csv(VAL_PREDICTIONS_CSV)

# ------------------------------------------------ the test set is never touched
assert not (rows["split"] == "test").any() and not rows["season"].isin(TEST_SEASONS).any()
for table in (fit, val, train_all):
    assert not table["season"].isin(TEST_SEASONS).any() and (table["Date"] < "2025-07-01").all()
for forbidden in ("test", "2025-26"):
    try:
        get_slice(forbidden)
        raise AssertionError(f"get_slice('{forbidden}') should have raised")
    except TestSetAccessError:
        pass
try:
    make_xy(fit.iloc[[0]].assign(season="2025-26", split="test"), v3["DIFF"])
    raise AssertionError("make_xy accepted a test row")
except TestSetAccessError:
    pass
for config in configs.values():
    assert set(config["window_seasons"]) <= set(TRAIN_SEASONS) and not set(config["window_seasons"]) & set(TEST_SEASONS)
passed("no 2025-26 row in any fitted or scored slice (fit, validation, train_all, both saved models' configs); "
       "the guards raise TestSetAccessError when asked for test rows")

assert fit["Date"].max() < val["Date"].min()
passed(f"latest fit date {fit['Date'].max().date()} is before earliest validation date {val['Date'].min().date()}")

# --------------------------------------------------------- frozen feature sets
assert sha(FEATURE_SETS_V1_JSON) == V1_SHA and sha(FEATURE_SETS_V2_JSON) == V2_SHA
assert sha(FEATURE_SETS_V3_JSON) == V3_SHA
assert FEATURE_SETS_V3_JSON.stat().st_mtime < ABLATIONS_CSV.stat().st_mtime
assert FEATURE_SETS_V3_JSON.stat().st_mtime < VAL_PREDICTIONS_CSV.stat().st_mtime
passed("feature_sets_v3.json hash matches the frozen value and predates reports/ablations_rf.csv; "
       "v1 and v2 are byte-for-byte unchanged")

assert v3["FULL_v2"] == v2["FULL"] and v3["REDUCED"] == v2["REDUCED"] and v3["ELO2"] == v2["ELO2"]
assert v3["DIFF"] == DIFF_FEATURES and len(v3["DIFF"]) == 16
for name in ("FULL_v2", "REDUCED", "ELO2", "DIFF"):
    assert LEAGUE_AVG not in v3[name]
assert any(f.startswith("h2h") for f in v3["FULL_v2"])
assert not any(f.startswith("h2h") for f in v3["REDUCED"] + v3["ELO2"] + v3["DIFF"])
assert set(v3["DIFF"]) <= set(rows.columns)
passed("v3 sets: FULL_v2 (33), REDUCED (24), ELO2 (2) equal the v2 sets; DIFF has 16 features; "
       "league_avg_goals_prev380 is in none; head-to-head only in FULL_v2")

# ---------------------------------------------------- DIFF = same-row arithmetic
assert verify_diff_is_row_local(rows)
for stat in DIFF_PAIRS:
    assert np.array_equal(rows[f"diff_{stat}"].to_numpy(), (rows[f"home_{stat}"] - rows[f"away_{stat}"]).to_numpy())
passed("DIFF features are same-row arithmetic: the same values after shuffling the rows and when computed "
       "one row at a time, and equal home_X minus away_X exactly")

params = json.loads(PARAMS_JSON.read_text())
parts = [c[c["split"] == "train"] for c in pd.read_csv(MATCHES_CSV, parse_dates=["Date"], chunksize=500)]
matches = pd.concat(parts, ignore_index=True)              # training matches only
assert not matches["season"].isin(TEST_SEASONS).any()
base = build_features(matches, params["elo"], params["defaults"])
assert np.allclose(base[FEATURE_COLUMNS].to_numpy(float), raw[FEATURE_COLUMNS].to_numpy(float), rtol=0, atol=1e-9)
base_diff = add_diff_features(base)[DIFF_COLUMNS]


def perturb(frame, i):
    """Change the final score AND the shot counts of match i."""
    changed = frame.copy()
    changed.loc[i, "home_goals"] = frame.loc[i, "home_goals"] + 3
    changed.loc[i, "away_goals"] = max(frame.loc[i, "away_goals"] - 1, 0)
    hg, ag = changed.loc[i, "home_goals"], changed.loc[i, "away_goals"]
    changed.loc[i, "result"] = 0 if hg > ag else (1 if hg == ag else 2)
    for column, extra in [("HS", 7), ("AS", 5), ("HST", 3), ("AST", 2)]:
        changed.loc[i, column] = frame.loc[i, column] + extra
    return changed


print("\nPerturbation tests for the DIFF features (change one match's score and shots, rebuild everything):")
rng = np.random.default_rng(7)
for i in rng.choice(matches.index[matches["Date"] < matches["Date"].max()], size=4, replace=False):
    date = matches.loc[i, "Date"]
    rebuilt = build_features(perturb(matches, i), params["elo"], params["defaults"])
    new_diff = add_diff_features(rebuilt)[DIFF_COLUMNS]
    not_later = (matches["Date"] <= date).to_numpy()
    assert np.array_equal(base_diff.loc[not_later].to_numpy(), new_diff.loc[not_later].to_numpy()), f"match {i} leaks"
    later_changed = int((base_diff.loc[~not_later].to_numpy() != new_diff.loc[~not_later].to_numpy()).any(axis=1).sum())
    assert later_changed > 0
    # and each match's DIFF still equals its own home_X - away_X in the rebuilt table
    assert np.array_equal(new_diff.to_numpy(), (rebuilt[[f"home_{x}" for x in DIFF_PAIRS]].to_numpy()
                                                 - rebuilt[[f"away_{x}" for x in DIFF_PAIRS]].to_numpy()))
    print(f"  match {i:4} ({matches.loc[i, 'HomeTeam']} v {matches.loc[i, 'AwayTeam']}, {date.date()}): "
          f"DIFF unchanged for all {int(not_later.sum())} matches on or before this date; "
          f"{later_changed} later matches changed")
passed("changing another match's result never changes the DIFF features of that match or of any earlier match")

# -------------------------------------------------- probabilities and class order
probabilities, features_by_model = {}, {}
for name, model in models.items():
    assert isinstance(model, RandomForestClassifier), "trees need no scaler: the saved model is a bare forest"
    assert list(model.classes_) == CLASS_CODES == [0, 1, 2] and model.n_estimators == 500
    features_by_model[name] = configs[name]["features"]
    probabilities[name] = rf_proba(model, val, features_by_model[name])
    assert np.allclose(probabilities[name].sum(axis=1), 1.0, atol=1e-9) and (probabilities[name] >= 0).all()
assert (saved["match_id"].to_numpy() == val["match_id"].to_numpy()).all() and len(saved) == 760
for prefix in ["rf_default", "rf_chosen", "elo2_primary", "full_reference", "class_freq", "bookmaker"]:
    assert np.allclose(saved[[f"{prefix}_{c}" for c in PROBA_COLUMNS]].sum(axis=1), 1.0, atol=1e-9)
earlier = pd.read_csv(VAL_PREDICTIONS_CSV.parent / "val_predictions_5a_v2.csv")
for prefix in ["elo2_primary", "full_reference", "class_freq", "bookmaker"]:
    cols = [f"{prefix}_{c}" for c in PROBA_COLUMNS]
    assert np.allclose(saved[cols].to_numpy(), earlier[cols].to_numpy(), atol=1e-12)
passed("both forests: classes are [0, 1, 2] = Home win, Draw, Away win, 500 trees, no scaler; probabilities sum to 1; "
       "all predictors in val_predictions_5b.csv sum to 1, and the Stage 5a predictors are copied unchanged")

# ------------------------------------------ saved models reproduce their predictions
for name, config in configs.items():
    cfg = make_rf_config(config["features"], config["min_samples_leaf"], config["max_features"],
                         config["max_depth"], window=None if len(config["window_seasons"]) == len(TRAIN_SEASONS)
                         else len(config["window_seasons"]), recency=config["recency_weights"],
                         include_warmup=config["include_warmup_rows"], drop_2020_21=config["drop_2020_21_rows"])
    refit, used = fit_rf(cfg, train_all, FINAL_END)
    assert np.allclose(rf_proba(refit, val, config["features"]), probabilities[name], atol=1e-12)
    assert len(used) == config["n_training_rows"]
    fit_only, _ = fit_rf(cfg, fit, FIT_END)
    prefix = "rf_chosen" if name == "primary" else "rf_default"
    assert np.allclose(rf_proba(fit_only, val, config["features"]),
                       saved[[f"{prefix}_{c}" for c in PROBA_COLUMNS]].to_numpy(), atol=1e-12)
passed("each saved forest equals a fresh refit from its config, and refitting on the fit seasons reproduces "
       "the saved validation predictions")

for path in (MODEL_PATH, REFERENCE_MODEL_PATH):
    size_mb = path.stat().st_size / 1024 / 1024
    assert size_mb < MAX_MODEL_MB, f"{path.name} is {size_mb:.1f} MB"
passed(f"model files are {MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB and "
       f"{REFERENCE_MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB, far under the {MAX_MODEL_MB} MB limit")

# ------------------------------------------------------------ results table
results = pd.read_csv(ABLATIONS_CSV)
assert {"a. feature set", "b. head-to-head", "c. season_no_fans", "d. training window",
        "e. warm-up rows", "final combined"} <= set(results["experiment"])
assert {"improvement_vs_default", "ci_low", "ci_high", "decision", "val_log_loss"} <= set(results.columns)
default_row = results[results["option"] == "FULL_v2 (default)"].iloc[0]
default_accuracy = float(np.mean(saved[[f"rf_default_{c}" for c in PROBA_COLUMNS]].to_numpy().argmax(axis=1) == val["result"].to_numpy()))
assert abs(default_row["val_accuracy"] - default_accuracy) < 1e-12
chosen = results[results["decision"] == "CHOSEN"]
assert set(chosen["experiment"]) == {"a. feature set", "b. head-to-head", "c. season_no_fans",
                                     "d. training window", "e. warm-up rows"}
passed("ablations_rf.csv has all experiments with improvement, interval and decision columns; the default row's "
       "accuracy equals a hand count from the saved probabilities")

print("\nAll checks passed.")
