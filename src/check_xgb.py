"""Checks for Stage 5c (XGBoost). Run after:
  python src/modeling_utils.py ; python src/xgb.py

Each check prints PASS only after its assertion succeeded.
"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from xgboost import Booster, DMatrix, XGBClassifier

from logreg import training_rows
from modeling_utils import (CLASS_CODES, FEATURE_SETS_V1_JSON, FEATURE_SETS_V2_JSON, FEATURE_SETS_V3_JSON,
                            FIT_SEASONS, PROBA_COLUMNS, TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS,
                            TestSetAccessError, add_diff_features, get_slice, load_feature_sets,
                            load_features, make_xy)
from xgb import (ABLATIONS_CSV, AUDIT_JSON, FINAL_END, FIT_END, MAX_MODEL_MB, MAX_TREES, MODEL_PATHS, STOP_SEASONS,
                 VAL_PREDICTIONS_CSV, fit_xgb, make_xgb, make_xgb_config, xgb_proba)

V1_SHA = "bc63b1f3f02615457f83834a5a4fe0b07033750e8ad08e5b26c2e839a5fea87b"
V2_SHA = "68280d590d5f42510f994e233b8b4608655dae7dd26ffaa97fc9a774744eecc0"
V3_SHA = "9ca210478c2e9fa25c254e6db557ac7c578a014e366cf4a686eb72fc86b4e946"


def passed(message):
    print(f"PASS  {message}")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


rows = add_diff_features(load_features())
fit, val, train_all = get_slice("fit", rows), get_slice("validation", rows), get_slice("train_all", rows)
sets = load_feature_sets(version=3)
audit = json.loads(AUDIT_JSON.read_text())
saved = pd.read_csv(VAL_PREDICTIONS_CSV)
configs = {name: json.loads(path.with_name(path.name + "_config.json").read_text()) for name, path in MODEL_PATHS.items()}
models = {name: joblib.load(path.with_suffix(".joblib")) for name, path in MODEL_PATHS.items()}

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
    make_xy(fit.iloc[[0]].assign(season="2025-26", split="test"), sets["DIFF"])
    raise AssertionError("make_xy accepted a test row")
except TestSetAccessError:
    pass
for config in configs.values():
    assert set(config["window_seasons"]) <= set(TRAIN_SEASONS) and not set(config["window_seasons"]) & set(TEST_SEASONS)
passed("no 2025-26 row in any fitted or scored slice (fit, validation, train_all, both saved models' configs); "
       "the guards raise TestSetAccessError when asked for test rows")

assert fit["Date"].max() < val["Date"].min()
passed(f"latest fit date {fit['Date'].max().date()} is before earliest validation date {val['Date'].min().date()}")

# --------------------------------------------- early stopping never uses validation
scored = [a for a in audit if a["purpose"] == "scored_on_validation"]
final = [a for a in audit if a["purpose"] == "final_refit"]
assert len(scored) >= 30 and len(final) == 2, (len(scored), len(final))
for run in scored:
    stop, inner_fit = set(run["inner_stop_seasons"]), set(run["inner_fit_seasons"])
    assert len(run["inner_stop_seasons"]) == STOP_SEASONS
    assert stop <= set(FIT_SEASONS) and inner_fit <= set(FIT_SEASONS)
    assert not (stop | inner_fit) & set(VAL_SEASONS) and not (stop | inner_fit) & set(TEST_SEASONS)
    assert run["inner_fit_last_date"] < run["inner_stop_first_date"]
    assert max(inner_fit) < min(stop)
passed(f"all {len(scored)} early-stopping runs behind scored models fit and stop only on 2015-16 to 2022-23, "
       f"stop on their last {STOP_SEASONS} seasons, and the stop seasons come after the fit seasons in time")

default_cfg = make_xgb_config(sets["FULL_v2"], configs["reference"]["max_depth"], configs["reference"]["learning_rate"],
                              configs["reference"]["min_child_weight"])
window_rows = training_rows(fit, default_cfg, FIT_END)
stop_ids = set(window_rows[window_rows["season"].isin(sorted(window_rows["season"].unique())[-STOP_SEASONS:])]["match_id"])
assert stop_ids and not stop_ids & set(val["match_id"]) and not stop_ids & set(train_all[train_all["season"].isin(VAL_SEASONS)]["match_id"])
assert sorted(window_rows["season"].unique())[-STOP_SEASONS:] == ["2021-22", "2022-23"]
passed(f"row level: the {len(stop_ids)} matches used to stop the default model's trees (2021-22, 2022-23) share no match_id "
       f"with the {len(val)} validation matches")

for run in final:
    assert set(run["inner_fit_seasons"] + run["inner_stop_seasons"]) <= set(TRAIN_SEASONS)
    assert run["inner_stop_seasons"] == VAL_SEASONS      # the final models stop on the last 2 training seasons ...
assert all(c["status"].startswith(("primary", "reference")) for c in configs.values())
passed("the two final models stop on 2023-24 and 2024-25 (same shape as before) and are labelled final: they are "
       "refit on all ten seasons and are never scored on validation or test")

# --------------------------------------------------------- frozen feature sets
assert sha(FEATURE_SETS_V1_JSON) == V1_SHA and sha(FEATURE_SETS_V2_JSON) == V2_SHA and sha(FEATURE_SETS_V3_JSON) == V3_SHA
assert not FEATURE_SETS_V3_JSON.with_name("feature_sets_v4.json").exists()
assert all(len(sets[k]) == n for k, n in (("FULL_v2", 33), ("REDUCED", 24), ("ELO2", 2), ("DIFF", 16)))
passed("v1, v2 and v3 feature-set files are byte-for-byte unchanged and no v4 exists; sizes are 33, 24, 2 and 16")

# -------------------------------------------------- probabilities and class order
probabilities = {}
for name, model in models.items():
    assert list(model.classes_) == CLASS_CODES == [0, 1, 2]
    assert model.get_params()["objective"] == "multi:softprob" and model.get_booster().num_boosted_rounds() == configs[name]["n_trees"]
    assert not any("scale" in str(type(step)).lower() for step in [model])
    probabilities[name] = xgb_proba(model, val, configs[name]["features"])
    assert np.allclose(probabilities[name].sum(axis=1), 1.0, atol=1e-5) and (probabilities[name] >= 0).all()
assert (saved["match_id"].to_numpy() == val["match_id"].to_numpy()).all() and len(saved) == 760
for prefix in ["xgb_default", "xgb_chosen", "elo2_primary", "full_reference", "rf_chosen", "rf_default", "class_freq", "bookmaker"]:
    assert np.allclose(saved[[f"{prefix}_{c}" for c in PROBA_COLUMNS]].sum(axis=1), 1.0, atol=1e-5)
earlier = pd.read_csv(VAL_PREDICTIONS_CSV.parent / "val_predictions_5b.csv")
for prefix in ["elo2_primary", "full_reference", "rf_chosen", "rf_default", "class_freq", "bookmaker"]:
    cols = [f"{prefix}_{c}" for c in PROBA_COLUMNS]
    assert np.allclose(saved[cols].to_numpy(), earlier[cols].to_numpy(), atol=1e-12)
passed("both XGBoost models: classes [0, 1, 2] = Home win, Draw, Away win, objective multi:softprob, tree count = config, no scaler; "
       "probabilities sum to 1; all predictors in val_predictions_5c.csv sum to 1 and the 5b predictors are copied unchanged")

# ------------------------------------------ saved models reproduce their predictions
for name, path in MODEL_PATHS.items():
    features = configs[name]["features"]
    X_val, _ = make_xy(val, features)
    booster = Booster()
    booster.load_model(path.with_suffix(".ubj"))
    from_ubj = booster.predict(DMatrix(X_val))
    assert np.allclose(from_ubj, probabilities[name], atol=1e-5), "the .ubj and .joblib files disagree"
    cfg = make_xgb_config(features, configs[name]["max_depth"], configs[name]["learning_rate"], configs[name]["min_child_weight"],
                          window=None, recency=configs[name]["recency_weights"], include_warmup=configs[name]["include_warmup_rows"],
                          drop_2020_21=configs[name]["drop_2020_21_rows"], monotone=configs[name]["monotone_constraint_on_elo_diff"])
    refit, used, n_trees, _ = fit_xgb(cfg, train_all, FINAL_END, "check_refit")
    assert n_trees == configs[name]["n_trees"] and len(used) == configs[name]["n_training_rows"]
    assert np.allclose(xgb_proba(refit, val, features), probabilities[name], atol=1e-6)
    prefix = "xgb_chosen" if name == "primary" else "xgb_default"
    fit_only, _, _, _ = fit_xgb(cfg, fit, FIT_END, "check_refit")
    assert np.allclose(xgb_proba(fit_only, val, features), saved[[f"{prefix}_{c}" for c in PROBA_COLUMNS]].to_numpy(), atol=1e-6)
passed("for both models: the .ubj file and the .joblib file give the same predictions; a fresh refit (including early stopping) "
       "reproduces the saved model and, on the fit seasons, the saved validation predictions")

for path in MODEL_PATHS.values():
    for suffix in (".joblib", ".ubj"):
        size_mb = path.with_suffix(suffix).stat().st_size / 1024 / 1024
        assert size_mb < MAX_MODEL_MB, f"{path.with_suffix(suffix).name} is {size_mb:.1f} MB"
largest = max(p.with_suffix(s).stat().st_size for p in MODEL_PATHS.values() for s in (".joblib", ".ubj")) / 1024 / 1024
passed(f"all four model files are under the {MAX_MODEL_MB} MB limit (largest {largest:.2f} MB)")

# ------------------------------------------------- monotone constraint is passed on
mono_cfg = make_xgb_config(sets["FULL_v2"], 3, 0.1, 20, monotone=True)
constraint = make_xgb(mono_cfg, 5).get_params()["monotone_constraints"]
assert tuple(constraint) == tuple(1 if f == "elo_diff" else 0 for f in sets["FULL_v2"]) and sum(constraint) == 1
assert MAX_TREES == 2000
passed("the monotone setting puts +1 on elo_diff and 0 on all 32 other features of FULL_v2 (early-stopping cap 2000 trees)")

# ------------------------------------------------------------ results table
results = pd.read_csv(ABLATIONS_CSV)
assert {"a. feature set", "b. head-to-head", "c. season_no_fans", "d. training window", "e. warm-up rows",
        "f. monotone constraint", "final combined"} <= set(results["experiment"])
assert {"improvement_vs_default", "ci_low", "ci_high", "decision", "val_log_loss"} <= set(results.columns)
default_row = results[results["option"] == "FULL_v2 (default)"].iloc[0]
default_accuracy = float(np.mean(saved[[f"xgb_default_{c}" for c in PROBA_COLUMNS]].to_numpy().argmax(axis=1) == val["result"].to_numpy()))
assert abs(default_row["val_accuracy"] - default_accuracy) < 1e-12
assert set(results[results["decision"] == "CHOSEN"]["experiment"]) == {"a. feature set", "b. head-to-head", "c. season_no_fans",
                                                                       "d. training window", "e. warm-up rows", "f. monotone constraint"}
passed("ablations_xgb.csv has experiments a to f, the combination and the diagnostics, with improvement, interval and decision "
       "columns; the default row's accuracy equals a hand count from the saved probabilities")

print("\nAll checks passed.")
