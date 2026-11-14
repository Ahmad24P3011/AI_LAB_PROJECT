"""Checks for Stage 5a. Run after src/modeling_utils.py and src/logreg.py.

Each check prints PASS only after its assertion succeeded.
"""
import json

import joblib
import numpy as np
import pandas as pd

from logreg import (CONFIG_PATH, FIT_END, FINAL_END, MODEL_PATH, RESULTS_CSV, VAL_PREDICTIONS_CSV,
                    fit_config, make_config)
from modeling_utils import (CLASS_CODES, FEATURE_SETS_JSON, FIT_SEASONS, ODDS_CSV, PROBA_COLUMNS,
                            TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS, TestSetAccessError,
                            bookmaker_probabilities, build_feature_sets, evaluate, get_slice,
                            load_feature_sets, load_features, make_xy, paired_bootstrap,
                            per_match_log_loss, predict_proba)


def passed(message):
    print(f"PASS  {message}")


rows = load_features()
fit, val, train_all = get_slice("fit", rows), get_slice("validation", rows), get_slice("train_all", rows)
config = json.loads(CONFIG_PATH.read_text())
model = joblib.load(MODEL_PATH)

# ------------------------------------------------ the test set is never touched
assert not (rows["split"] == "test").any() and not rows["season"].isin(TEST_SEASONS).any()
for table in (fit, val, train_all):
    assert not table["season"].isin(TEST_SEASONS).any() and (table["Date"] < "2025-07-01").all()
passed("every slice loaded in Stage 5 (fit, validation, train_all) contains no 2025-26 row")

for forbidden in ("test", "2025-26", "all"):
    try:
        get_slice(forbidden)
        raise AssertionError(f"get_slice('{forbidden}') should have raised")
    except TestSetAccessError:
        pass
fake_test_row = fit.iloc[[0]].assign(season="2025-26", split="test")
try:
    make_xy(fake_test_row, config["features"])
    raise AssertionError("make_xy accepted a test row")
except TestSetAccessError:
    pass
passed("asking for test rows raises TestSetAccessError (get_slice and make_xy guards work)")

used = set(config["window_seasons"])
assert used <= set(TRAIN_SEASONS) and not used & set(TEST_SEASONS)
assert set(fit["season"]) == set(FIT_SEASONS) and set(val["season"]) == set(VAL_SEASONS)
passed("the saved model's config uses only training seasons; fit and validation slices are as defined")

# ----------------------------------------------------- time order and frozen sets
assert fit["Date"].max() < val["Date"].min()
passed(f"latest fit date {fit['Date'].max().date()} is before earliest validation date "
       f"{val['Date'].min().date()}")

frozen = load_feature_sets()
fresh = build_feature_sets(fit)
assert frozen["REDUCED"] == fresh["REDUCED"] and frozen["FULL"] == fresh["FULL"]
assert len(frozen["FULL"]) == 34 and len(frozen["SMALL"]) == 3 and "away_matches_played" not in frozen["REDUCED"]
assert FEATURE_SETS_JSON.stat().st_mtime < RESULTS_CSV.stat().st_mtime < MODEL_PATH.stat().st_mtime + 3600
assert FEATURE_SETS_JSON.stat().st_mtime < VAL_PREDICTIONS_CSV.stat().st_mtime
passed("feature_sets.json equals the mechanical rule's output and was written BEFORE any model result")

# ------------------------------------------------------ probabilities and classes
assert list(model.classes_) == CLASS_CODES == [0, 1, 2]
proba = predict_proba(model, val, config["features"])
assert proba.shape == (len(val), 3) and np.allclose(proba.sum(axis=1), 1.0, atol=1e-9) and (proba >= 0).all()
passed("saved model: classes are [0, 1, 2] = Home win, Draw, Away win; probabilities sum to 1")

saved = pd.read_csv(VAL_PREDICTIONS_CSV)
assert (saved["match_id"].to_numpy() == val["match_id"].to_numpy()).all() and len(saved) == 760
for prefix in ["logreg_default", "logreg_chosen", "class_freq", "bookmaker"]:
    block = saved[[f"{prefix}_{c}" for c in PROBA_COLUMNS]].to_numpy()
    assert np.allclose(block.sum(axis=1), 1.0, atol=1e-9)
home_lean = saved["bookmaker_p_home"] > saved["bookmaker_p_away"]
assert (saved.loc[home_lean, "result"] == 0).mean() > (saved.loc[~home_lean, "result"] == 0).mean()
passed("saved validation predictions of all four models sum to 1, and the column order is right "
       "(home-leaning odds really go with more home wins)")

# --------------------------------------------------- baselines by hand
y = val["result"].to_numpy()
hand_home = sum(1 for r in y if r == 0) / len(y)
assert abs(evaluate(y, pred=np.zeros(len(y), dtype=int))["accuracy"] - hand_home) < 1e-12
counts = fit["result"].value_counts().reindex(CLASS_CODES).to_numpy()
shares = counts / counts.sum()
hand_loss = -sum(np.log(shares[r]) for r in y) / len(y)
frequency_result = evaluate(y, np.tile(shares, (len(y), 1)))
assert abs(frequency_result["log_loss"] - hand_loss) < 1e-12
passed(f"always-home-win accuracy {hand_home:.4f} and class-frequency log loss {hand_loss:.4f} "
       f"match hand computations")

wanted = set(val["match_id"])
odds = pd.concat([c[c["match_id"].isin(wanted)] for c in
                  pd.read_csv(ODDS_CSV, usecols=["match_id", "B365H", "B365D", "B365A"], chunksize=500)])
odds = odds.set_index("match_id").loc[val["match_id"]]
favourite = np.argmin(odds[["B365H", "B365D", "B365A"]].to_numpy(), axis=1)   # lowest odds = favourite
probabilities, _ = bookmaker_probabilities(val["match_id"])
assert np.array_equal(probabilities.argmax(axis=1), favourite)
assert abs(np.mean(favourite == y) - evaluate(y, probabilities)["accuracy"]) < 1e-12
passed("bookmaker probabilities: the favourite is the lowest-odds outcome and accuracy matches a hand count")

# --------------------------------------------------------------- bootstrap
same = np.random.default_rng(0).random(500)
zero = paired_bootstrap(same, same)
assert zero["mean_diff"] == zero["ci_low"] == zero["ci_high"] == 0
shifted = paired_bootstrap(same + 0.05, same)
assert abs(shifted["mean_diff"] - 0.05) < 1e-12 and shifted["ci_low"] > 0.049 and shifted["ci_high"] < 0.051
passed("paired bootstrap: identical losses give 0, a constant 0.05 gap gives an interval around 0.05")

# --------------------------------------------- saved model reproduces predictions
cfg = make_config(config["features"], config["C"],
                  window=None if len(config["window_seasons"]) == len(TRAIN_SEASONS) else len(config["window_seasons"]),
                  recency=config["recency_weights"], include_warmup=config["include_warmup_rows"],
                  drop_2020_21=config["drop_2020_21_rows"])
refit, refit_rows = fit_config(cfg, train_all, FINAL_END)
assert np.allclose(predict_proba(refit, val, config["features"]), proba, atol=1e-9)
assert len(refit_rows) == config["n_training_rows"]
scaler = model.named_steps["scale"]
assert np.allclose(scaler.mean_, refit_rows[config["features"]].mean().to_numpy(), atol=1e-9)
passed("the saved model loads and equals a fresh refit from its config; its scaler was fitted on "
       "those training rows only")

fit_only, _ = fit_config(cfg, fit, FIT_END)
assert np.allclose(predict_proba(fit_only, val, config["features"]),
                   saved[[f"logreg_chosen_{c}" for c in PROBA_COLUMNS]].to_numpy(), atol=1e-9)
passed("refitting the chosen configuration on the fit seasons reproduces the saved validation predictions")

results = pd.read_csv(RESULTS_CSV)
assert {"a. feature set", "b. head-to-head", "c. season_no_fans", "d. training window",
        "e. warm-up rows", "final combined"} <= set(results["experiment"])
assert results.loc[results["experiment"] != "final combined", "val_log_loss"].between(0.9, 1.1).all()
passed("ablations_logreg.csv has all five experiments and the combined configuration")

print("\nAll checks passed.")
