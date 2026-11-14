"""Checks for Stage 5a, including the amendment. Run after:
  python src/modeling_utils.py ; python src/logreg.py ; python src/amendment_5a.py

Each check prints PASS only after its assertion succeeded.
"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from amendment_5a import (DRIFT_CSV, DRIFT_LIMIT, PRIMARY_CONFIG, PRIMARY_MODEL, REFERENCE_CONFIG,
                          REFERENCE_MODEL, VAL_PREDICTIONS_CSV as VAL_V2_CSV)
from features import FEATURE_COLUMNS, FEATURES_CSV
from logreg import FIT_END, FINAL_END, RESULTS_CSV as ABLATIONS_CSV, fit_config, make_config
from modeling_utils import (CLASS_CODES, ELO2_FEATURES, FEATURE_SETS_V1_JSON, FEATURE_SETS_V2_JSON,
                            FIT_SEASONS, H2H_FEATURES, LEAGUE_AVG, ODDS_CSV, PROBA_COLUMNS,
                            TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS, TestSetAccessError,
                            bookmaker_probabilities, build_feature_sets, build_feature_sets_v2,
                            evaluate, get_slice, load_feature_sets, load_features, make_xy,
                            paired_bootstrap, predict_proba)

# SHA-256 of feature_sets_v1.json when it was frozen (before any model result was seen).
V1_SHA256 = "bc63b1f3f02615457f83834a5a4fe0b07033750e8ad08e5b26c2e839a5fea87b"


def passed(message):
    print(f"PASS  {message}")


rows = load_features()
fit, val, train_all = get_slice("fit", rows), get_slice("validation", rows), get_slice("train_all", rows)
v1, v2 = load_feature_sets(version=1), load_feature_sets(version=2)
configs = {"primary": json.loads(PRIMARY_CONFIG.read_text()), "reference": json.loads(REFERENCE_CONFIG.read_text())}
models = {"primary": joblib.load(PRIMARY_MODEL), "reference": joblib.load(REFERENCE_MODEL)}

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
    make_xy(fake_test_row, v2["ELO2"])
    raise AssertionError("make_xy accepted a test row")
except TestSetAccessError:
    pass
passed("asking for test rows raises TestSetAccessError (get_slice and make_xy guards work)")

for config in configs.values():
    assert set(config["window_seasons"]) <= set(TRAIN_SEASONS) and not set(config["window_seasons"]) & set(TEST_SEASONS)
assert set(fit["season"]) == set(FIT_SEASONS) and set(val["season"]) == set(VAL_SEASONS)
passed("both saved models' configs use only training seasons; fit and validation slices are as defined")

assert fit["Date"].max() < val["Date"].min()
passed(f"latest fit date {fit['Date'].max().date()} is before earliest validation date {val['Date'].min().date()}")

# ----------------------------------------------- frozen v1 and the v2 amendment
assert hashlib.sha256(FEATURE_SETS_V1_JSON.read_bytes()).hexdigest() == V1_SHA256
assert v1["FULL"] == build_feature_sets(fit)["FULL"] and v1["REDUCED"] == build_feature_sets(fit)["REDUCED"]
assert FEATURE_SETS_V1_JSON.stat().st_mtime < ABLATIONS_CSV.stat().st_mtime
passed("feature_sets_v1.json is byte-for-byte unchanged (hash matches), equals the mechanical rule's "
       "output, and predates the experiments")

assert {k: v2[k] for k in ("FULL", "REDUCED", "ELO2")} == {k: build_feature_sets_v2(v1)[k] for k in ("FULL", "REDUCED", "ELO2")}
assert v2["date"] == "2026-11-14" and "post-hoc" in v2["reason"].lower() and "feature_sets_v1" in v2["amends"]
assert FEATURE_SETS_V1_JSON.stat().st_mtime < FEATURE_SETS_V2_JSON.stat().st_mtime
assert v2["ELO2"] == ELO2_FEATURES == ["elo_diff", "abs_elo_diff"]
passed("feature_sets_v2.json is derived mechanically from v1, carries its date and reason, and is newer than v1")

for name in ("FULL", "REDUCED", "ELO2"):
    assert LEAGUE_AVG not in v2[name]
assert LEAGUE_AVG in FEATURE_COLUMNS and LEAGUE_AVG in pd.read_csv(FEATURES_CSV, nrows=1).columns
assert any(f in v2["FULL"] for f in H2H_FEATURES)
assert not any(f in v2["REDUCED"] + v2["ELO2"] for f in H2H_FEATURES)
passed(f"{LEAGUE_AVG} is in no classifier set but is still in features.csv (for Stage 5d); "
       f"head-to-head appears only in FULL")

# ------------------------------------------------------ the saved models
assert configs["primary"]["features"] == v2["ELO2"] and configs["primary"]["status"] == "primary"
assert configs["reference"]["features"] == v2["FULL"]
for config in configs.values():
    assert LEAGUE_AVG not in config["features"]
assert not any(f in configs["primary"]["features"] for f in H2H_FEATURES)
passed("primary model = ELO2 (2 features, no head-to-head); reference model = FULL v2 (33 features); "
       "neither contains league_avg_goals_prev380")

probabilities = {}
for name, model in models.items():
    assert list(model.classes_) == CLASS_CODES == [0, 1, 2]
    probabilities[name] = predict_proba(model, val, configs[name]["features"])
    assert np.allclose(probabilities[name].sum(axis=1), 1.0, atol=1e-9) and (probabilities[name] >= 0).all()
passed("saved models: classes are [0, 1, 2] = Home win, Draw, Away win; probabilities sum to 1")

saved = pd.read_csv(VAL_V2_CSV)
assert (saved["match_id"].to_numpy() == val["match_id"].to_numpy()).all() and len(saved) == 760
for prefix in ["elo2_primary", "full_reference", "class_freq", "bookmaker"]:
    assert np.allclose(saved[[f"{prefix}_{c}" for c in PROBA_COLUMNS]].sum(axis=1), 1.0, atol=1e-9)
home_lean = saved["bookmaker_p_home"] > saved["bookmaker_p_away"]
assert (saved.loc[home_lean, "result"] == 0).mean() > (saved.loc[~home_lean, "result"] == 0).mean()
passed("saved validation predictions of all four predictors sum to 1, and the column order is right")

# --------------------------------------------------------- baselines by hand
y = val["result"].to_numpy()
hand_home = sum(1 for r in y if r == 0) / len(y)
assert abs(evaluate(y, pred=np.zeros(len(y), dtype=int))["accuracy"] - hand_home) < 1e-12
counts = fit["result"].value_counts().reindex(CLASS_CODES).to_numpy()
shares = counts / counts.sum()
hand_loss = -sum(np.log(shares[r]) for r in y) / len(y)
assert abs(evaluate(y, np.tile(shares, (len(y), 1)))["log_loss"] - hand_loss) < 1e-12
passed(f"always-home-win accuracy {hand_home:.4f} and class-frequency log loss {hand_loss:.4f} match hand computations")

wanted = set(val["match_id"])
odds = pd.concat([c[c["match_id"].isin(wanted)] for c in
                  pd.read_csv(ODDS_CSV, usecols=["match_id", "B365H", "B365D", "B365A"], chunksize=500)])
odds = odds.set_index("match_id").loc[val["match_id"]]
favourite = np.argmin(odds[["B365H", "B365D", "B365A"]].to_numpy(), axis=1)
book, _ = bookmaker_probabilities(val["match_id"])
assert np.array_equal(book.argmax(axis=1), favourite)
assert abs(np.mean(favourite == y) - evaluate(y, book)["accuracy"]) < 1e-12
passed("bookmaker probabilities: the favourite is the lowest-odds outcome and accuracy matches a hand count")

same = np.random.default_rng(0).random(500)
zero, shifted = paired_bootstrap(same, same), paired_bootstrap(same + 0.05, same)
assert zero["mean_diff"] == zero["ci_low"] == zero["ci_high"] == 0
assert abs(shifted["mean_diff"] - 0.05) < 1e-12 and shifted["ci_low"] > 0.049 and shifted["ci_high"] < 0.051
passed("paired bootstrap: identical losses give 0, a constant 0.05 gap gives an interval around 0.05")

# ----------------------------------- saved models reproduce their predictions
for name, config in configs.items():
    cfg = make_config(config["features"], config["C"],
                      window=None if len(config["window_seasons"]) == len(TRAIN_SEASONS) else len(config["window_seasons"]),
                      recency=config["recency_weights"], include_warmup=config["include_warmup_rows"],
                      drop_2020_21=config["drop_2020_21_rows"])
    refit, used = fit_config(cfg, train_all, FINAL_END)
    assert np.allclose(predict_proba(refit, val, config["features"]), probabilities[name], atol=1e-9)
    assert len(used) == config["n_training_rows"]
    assert np.allclose(models[name].named_steps["scale"].mean_, used[config["features"]].mean().to_numpy(), atol=1e-9)
    prefix = "elo2_primary" if name == "primary" else "full_reference"
    fit_only, _ = fit_config(cfg, fit, FIT_END)
    assert np.allclose(predict_proba(fit_only, val, config["features"]),
                       saved[[f"{prefix}_{c}" for c in PROBA_COLUMNS]].to_numpy(), atol=1e-9)
passed("both saved models equal a fresh refit from their configs (scalers fitted on those rows only), and "
       "refitting on the fit seasons reproduces the saved validation predictions")

# ------------------------------------------------------------ drift table
drift = pd.read_csv(DRIFT_CSV, index_col="feature")
assert len(drift) == 34 and set(drift.index) == set(FEATURE_COLUMNS)
assert (drift["flagged_above_3"] == (drift["max_abs_z"] > DRIFT_LIMIT)).all()
mean, std = fit[LEAGUE_AVG].mean(), fit[LEAGUE_AVG].std(ddof=0)
hand_z = (val[LEAGUE_AVG] - mean) / std
assert abs(drift.loc[LEAGUE_AVG, "max_abs_z"] - hand_z.abs().max()) < 1e-9
assert abs(drift.loc[LEAGUE_AVG, "val_mean_z"] - hand_z.mean()) < 1e-9
assert drift.loc[LEAGUE_AVG, "share_of_val_rows_above_3"] > 0.5
others = drift.drop(index=LEAGUE_AVG)
assert (others["val_mean_z"].abs() < 1).all() and (others["share_of_val_rows_above_3"] < 0.1).all()
passed(f"drift table: 34 features, league_avg_goals_prev380 matches a hand computation "
       f"(mean z {drift.loc[LEAGUE_AVG, 'val_mean_z']:+.2f}, {drift.loc[LEAGUE_AVG, 'share_of_val_rows_above_3'] * 100:.0f}% "
       f"of validation rows above 3); no other feature drifts systematically (mean |z| < 1, under 10% of rows above 3); "
       f"{int(drift['flagged_above_3'].sum())} of 34 have at least one value above 3")

# ----------------------------------------------- home-win check per season
for season, expected_rate in (("2023-24", 175 / 380), ("2024-25", 155 / 380)):
    mask = (val["season"] == season).to_numpy()
    assert abs((val.loc[mask, "result"] == 0).mean() - expected_rate) < 1e-12
    predicted = saved.loc[mask, "elo2_primary_p_home"].mean()
    assert 0.40 < predicted < 0.50
assert abs((val.loc[val["season"] == "2024-25", "result"] == 0).mean() - 0.408) < 0.0006   # EDA 02
passed("per-season home-win rates are 46.1% (2023-24) and 40.8% (2024-25, as in EDA 02); "
       "the ELO2 average predictions come from the saved probabilities")

results = pd.read_csv(ABLATIONS_CSV)
assert {"a. feature set", "b. head-to-head", "c. season_no_fans", "d. training window",
        "e. warm-up rows"} <= set(results["experiment"])
passed("the original (v1) ablation table is still present with all five experiments")

print("\nAll checks passed.")
