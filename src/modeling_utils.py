"""Shared tools for Stage 5 (5a to 5d): data slices, metrics, bootstrap, feature sets.

THE TEST-SET RULE: the 2025-26 test season is never loaded or scored in Stage 5.
Everything here is built so that asking for test rows raises an error:
  - load_features() keeps only training rows while reading the file;
  - get_slice() only knows the slices "fit", "validation" and "train_all";
  - make_xy() refuses any table that contains a test row.

Run `python src/modeling_utils.py` once to write data/processed/feature_sets.json.
That file is FROZEN before any model result is seen.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS
from preprocess import TEST_SEASONS, TRAIN_SEASONS, VALIDATION_SCHEME

ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "data" / "processed" / "features.csv"
ODDS_CSV = ROOT / "data" / "processed" / "odds.csv"
FEATURE_SETS_JSON = ROOT / "data" / "processed" / "feature_sets.json"

FIT_SEASONS = VALIDATION_SCHEME["fit"]        # 2015-16 ... 2022-23
VAL_SEASONS = VALIDATION_SCHEME["validate"]   # 2023-24, 2024-25
SLICES = {"fit": FIT_SEASONS, "validation": VAL_SEASONS, "train_all": TRAIN_SEASONS}

# One fixed class order used everywhere (it is the order of the result codes).
CLASS_CODES = [0, 1, 2]
CLASS_NAMES = ["Home win", "Draw", "Away win"]
PROBA_COLUMNS = ["p_home", "p_draw", "p_away"]

CORRELATION_LIMIT = 0.85   # pairs above this are "redundant twins" (see build_feature_sets)
SMALL_FEATURES = ["elo_diff", "abs_elo_diff", "league_avg_goals_prev380"]
SEED = 42


class TestSetAccessError(RuntimeError):
    """Raised whenever Stage 5 code asks for 2025-26 (test) rows."""


# ------------------------------------------------------------------ data slices
def load_features():
    """All TRAINING rows of features.csv. Test rows are dropped chunk by chunk
    while reading, so no table in memory ever holds them."""
    parts = []
    for chunk in pd.read_csv(FEATURES_CSV, parse_dates=["Date"], chunksize=500):
        parts.append(chunk[chunk["split"] == "train"])
    rows = pd.concat(parts, ignore_index=True)
    assert_no_test_rows(rows)
    return rows


def assert_no_test_rows(table):
    """Stop if any row belongs to the test season."""
    if "split" in table and (table["split"] == "test").any():
        raise TestSetAccessError("table contains test rows")
    if "season" in table and table["season"].isin(TEST_SEASONS).any():
        raise TestSetAccessError("table contains a test season")


def get_slice(name, rows=None):
    """Return the 'fit', 'validation' or 'train_all' rows. 'test' is refused."""
    if name not in SLICES:
        raise TestSetAccessError(f"slice '{name}' is not allowed in Stage 5 "
                                 f"(allowed: {list(SLICES)}); the test set is off limits")
    rows = load_features() if rows is None else rows
    result = rows[rows["season"].isin(SLICES[name])].reset_index(drop=True)
    assert_no_test_rows(result)
    return result


def make_xy(table, features):
    """Every fit and every prediction goes through here, so every one is guarded."""
    assert_no_test_rows(table)
    return table[features], table["result"].to_numpy()


def window_seasons(last_n, end_season):
    """The `last_n` seasons counted back from (and including) `end_season`."""
    ordered = [s for s in TRAIN_SEASONS if s <= end_season]
    return ordered if last_n is None else ordered[-last_n:]


def recency_weights(table, end_season, half_life_seasons=2):
    """Sample weights that halve every `half_life_seasons` seasons back from end_season.
    Rescaled to average 1 so the regularisation strength keeps the same meaning."""
    ordered = [s for s in TRAIN_SEASONS if s <= end_season]
    age = table["season"].map({s: len(ordered) - 1 - i for i, s in enumerate(ordered)})
    weights = 0.5 ** (age / half_life_seasons)
    return (weights / weights.mean()).to_numpy()


# ------------------------------------------------------------- feature sets
def build_feature_sets(fit_rows):
    """FULL, REDUCED and SMALL feature sets. REDUCED uses a mechanical rule on the
    FIT seasons only (2015-16 to 2022-23): validation and test rows are not used."""
    corr = fit_rows[FEATURE_COLUMNS].corr().abs()
    average_corr = (corr.sum() - 1) / (len(corr) - 1)   # average |r| against all other features

    keep = list(FEATURE_COLUMNS)
    removed = []
    # Always drop away_matches_played (r = 1.00 with home_matches_played).
    keep.remove("away_matches_played")
    removed.append({"dropped": "away_matches_played", "kept_twin": "home_matches_played",
                    "r": round(float(corr.loc["home_matches_played", "away_matches_played"]), 3),
                    "why": "forced: identical to home_matches_played"})

    pairs = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(FEATURE_COLUMNS)
             for b in FEATURE_COLUMNS[i + 1:] if corr.loc[a, b] > CORRELATION_LIMIT]
    for a, b, r in sorted(pairs, key=lambda p: -p[2]):          # strongest pair first
        if a in keep and b in keep:
            drop, stay = (a, b) if average_corr[a] >= average_corr[b] else (b, a)
            keep.remove(drop)
            removed.append({"dropped": drop, "kept_twin": stay, "r": round(r, 3),
                            "why": f"larger average |r| with other features "
                                   f"({average_corr[drop]:.3f} vs {average_corr[stay]:.3f})"})
    return {"FULL": list(FEATURE_COLUMNS), "REDUCED": keep, "SMALL": list(SMALL_FEATURES),
            "removed_by_rule": removed,
            "rule": (f"computed on seasons {FIT_SEASONS[0]} to {FIT_SEASONS[-1]} only. Drop "
                     f"away_matches_played. Then for each pair with |r| > {CORRELATION_LIMIT} "
                     "(strongest first): if both features are still present, drop the one with the "
                     "larger average |r| against all other features.")}


def load_feature_sets():
    return json.loads(FEATURE_SETS_JSON.read_text())


# ------------------------------------------------------------------- models
def make_logreg(C, class_weight=None):
    """Scaler + multinomial Logistic Regression in ONE pipeline: when the pipeline is
    fitted, the scaler learns its means/stds from the training rows only."""
    return Pipeline([("scale", StandardScaler()),
                     ("model", LogisticRegression(C=C, class_weight=class_weight,
                                                  max_iter=5000, random_state=SEED))])


def fit_model(model, train_rows, features, sample_weight=None):
    X, y = make_xy(train_rows, features)
    if sample_weight is None:
        model.fit(X, y)
    else:
        model.fit(X, y, model__sample_weight=sample_weight)
    assert list(model.classes_) == CLASS_CODES   # columns of predict_proba follow CLASS_CODES
    return model


def predict_proba(model, rows, features):
    X, _ = make_xy(rows, features)
    return model.predict_proba(X)


# ------------------------------------------------------------------ metrics
def per_match_log_loss(y_true, proba):
    """-log(probability given to what really happened), one number per match."""
    proba = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0)
    return -np.log(proba[np.arange(len(y_true)), np.asarray(y_true)])


def evaluate(y_true, proba=None, pred=None):
    """All metrics. Pass `proba` (n x 3) for a probability model, or only `pred`
    (class codes) for a model that makes hard predictions (log loss is then undefined)."""
    y_true = np.asarray(y_true)
    if proba is not None:
        proba = np.asarray(proba, dtype=float)
        assert proba.shape == (len(y_true), 3) and np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        pred = proba.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, labels=CLASS_CODES, zero_division=0)
    onehot_pred = np.eye(3)[pred]
    probabilities = proba if proba is not None else onehot_pred
    return {
        "accuracy": float(np.mean(pred == y_true)),
        "log_loss": float(per_match_log_loss(y_true, proba).mean()) if proba is not None else np.nan,
        "macro_f1": float(f1.mean()),
        # Brier score: average squared distance between the probabilities and what happened
        "brier": float(((probabilities - np.eye(3)[y_true]) ** 2).sum(axis=1).mean()),
        "confusion": confusion_matrix(y_true, pred, labels=CLASS_CODES),   # rows = truth
        "per_class": pd.DataFrame({"precision": precision, "recall": recall, "f1": f1},
                                  index=CLASS_NAMES),
        "predicted_counts": np.bincount(pred, minlength=3),
    }


def paired_bootstrap(loss_a, loss_b, n_boot=2000, seed=SEED):
    """Compare two options scored on the SAME matches. Resample the matches with
    replacement n_boot times. Returns the mean of (loss_a - loss_b) and a 95% interval:
    a positive number means option B has the lower loss (is better)."""
    difference = np.asarray(loss_a) - np.asarray(loss_b)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(difference), size=(n_boot, len(difference)))
    means = difference[picks].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(difference.mean()), "ci_low": float(low), "ci_high": float(high)}


# --------------------------------------------------------------- bookmaker
def bookmaker_probabilities(match_ids):
    """Bet365 odds -> probabilities for the given (validation) matches only.
    Implied probability = 1 / odds. The three add up to MORE than 1 (the bookmaker's
    margin, the overround), so we divide by their sum to make them add up to 1."""
    wanted = set(match_ids)
    parts = []
    for chunk in pd.read_csv(ODDS_CSV, usecols=["match_id", "B365H", "B365D", "B365A"],
                             chunksize=500):
        parts.append(chunk[chunk["match_id"].isin(wanted)])   # other rows are dropped at once
    odds = pd.concat(parts).set_index("match_id").loc[list(match_ids)]
    raw = 1 / odds[["B365H", "B365D", "B365A"]].to_numpy()
    return raw / raw.sum(axis=1, keepdims=True), float((raw.sum(axis=1) - 1).mean())


def main():
    fit_rows = get_slice("fit")
    sets = build_feature_sets(fit_rows)
    FEATURE_SETS_JSON.write_text(json.dumps(sets, indent=2))
    print(f"Saved {FEATURE_SETS_JSON.relative_to(ROOT)} (frozen before any model is trained)")
    print(f"FULL: {len(sets['FULL'])} features | REDUCED: {len(sets['REDUCED'])} | "
          f"SMALL: {len(sets['SMALL'])}")
    print(f"Rule: {sets['rule']}")
    print("\nRemoved by the rule:")
    for item in sets["removed_by_rule"]:
        print(f"  {item['dropped']:24} (r = {item['r']:.2f} with {item['kept_twin']}; {item['why']})")
    print("\nREDUCED feature list:")
    print("  " + ", ".join(sets["REDUCED"]))
    print("\nSMALL feature list:")
    print("  " + ", ".join(sets["SMALL"]))


if __name__ == "__main__":
    main()
