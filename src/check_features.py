"""Leakage and sanity tests for the Stage 4 features. Run after src/features.py.

Each check prints PASS only after its assertion succeeded; an assertion error
stops the script before 'All checks passed' can be printed.
"""
import json

import numpy as np
import pandas as pd

from features import (FEATURE_COLUMNS, FEATURES_CSV, MATCHES_CSV, META_COLUMNS, PARAMS_JSON,
                      build_features, compute_defaults, warmup_flags)

COMPARE_COLUMNS = FEATURE_COLUMNS + ["defaults_used"]
# Columns that describe the match itself (or odds). None may be an input feature.
FORBIDDEN = {"FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "home_goals", "away_goals", "result",
             "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"}

matches = pd.read_csv(MATCHES_CSV, parse_dates=["Date"])
saved = pd.read_csv(FEATURES_CSV, parse_dates=["Date"])
settings = json.loads(PARAMS_JSON.read_text())
params, defaults = settings["elo"], settings["defaults"]


def passed(message):
    print(f"PASS  {message}")


def leaking_columns(before, after, date_mask):
    """Columns whose values changed on rows that must NOT be affected."""
    changed = []
    for column in before.columns:
        if not np.array_equal(before.loc[date_mask, column].to_numpy(),
                              after.loc[date_mask, column].to_numpy()):
            changed.append(column)
    return changed


def perturb(frame, i):
    """Change the final score AND the shot counts of match i (a different result)."""
    changed = frame.copy()
    changed.loc[i, "home_goals"] = frame.loc[i, "home_goals"] + 3
    changed.loc[i, "away_goals"] = max(frame.loc[i, "away_goals"] - 1, 0)
    hg, ag = changed.loc[i, "home_goals"], changed.loc[i, "away_goals"]
    changed.loc[i, "result"] = 0 if hg > ag else (1 if hg == ag else 2)
    for column, extra in [("HS", 7), ("AS", 5), ("HST", 3), ("AST", 2)]:
        changed.loc[i, column] = frame.loc[i, column] + extra
    return changed


# ------------------------------------------------------- structure and content
assert len(saved) == len(matches) == 4180
assert saved["Date"].is_monotonic_increasing and (saved["match_id"] == matches["match_id"]).all()
passed("features.csv has 4180 rows, in date order, same matches as matches_clean.csv")

assert list(saved.columns) == META_COLUMNS + FEATURE_COLUMNS
assert len(FEATURE_COLUMNS) == 34 and len(set(FEATURE_COLUMNS)) == 34
passed("columns are exactly the metadata plus the 34-feature whitelist")

assert not set(FEATURE_COLUMNS) & FORBIDDEN
assert not any(c.startswith(("B365", "BW", "PS", "Avg", "Max", "Bb")) for c in FEATURE_COLUMNS)
passed("no same-match goals, shots, fouls, corners, cards, results or odds among the features")

assert saved[FEATURE_COLUMNS].notna().all().all() and np.isfinite(saved[FEATURE_COLUMNS]).all().all()
passed("no missing or infinite values in any feature column")

f = saved
assert f[["home_ppg_5", "away_ppg_5", "home_ppg_10", "away_ppg_10", "h2h_home_ppg_3"]].stack().between(0, 3).all()
assert (f[[c for c in FEATURE_COLUMNS if c.endswith(("_gf_5", "_ga_5", "_gf_10", "_ga_10", "_for_10", "_against_10"))]] >= 0).all().all()
assert f["home_elo"].between(1000, 2000).all() and f["away_elo"].between(1000, 2000).all()
assert f["h2h_n"].between(0, 3).all() and f["season_no_fans"].isin([0, 1]).all()
assert f["home_matches_played"].between(0, 37).all() and f["away_matches_played"].between(0, 37).all()
assert f["league_avg_goals_prev380"].between(2, 4).all()
assert (f["abs_elo_diff"] == (f["home_elo"] - f["away_elo"]).abs()).all()
passed("values are in sensible ranges (points 0-3, Elo 1000-2000, matches played 0-37, ...)")

assert set(f.loc[f["season_no_fans"] == 1, "season"]) == {"2020-21"}
assert (f.loc[f["split"] == "test", "season_no_fans"] == 0).all()
passed("season_no_fans is 1 only in 2020-21 and 0 for every test match")

first = f.iloc[0]
assert first["home_elo"] == 1500 and first["away_elo"] == 1500 and first["h2h_n"] == 0
assert first["home_ppg_5"] == defaults["ppg"] and first["home_shots_for_10"] == defaults["shots"]
assert first["home_matches_played"] == 0 and first["defaults_used"] == 1 and first["warmup"] == 1
passed("the very first match uses Elo 1500 and the neutral defaults (no history exists)")

# ------------------------------------------------ choices come from training rows
train_rows = matches[matches["split"] == "train"].reset_index(drop=True)
expected_defaults = compute_defaults(train_rows)
assert all(abs(defaults[k] - expected_defaults[k]) < 1e-12 for k in defaults)
assert (saved["warmup"].to_numpy() == warmup_flags(matches).astype(int)).all()
passed("neutral defaults equal values recomputed from training rows only; warmup flag reproduces")

rebuilt = build_features(matches, params, defaults)
assert np.allclose(rebuilt[COMPARE_COLUMNS].to_numpy(float), saved[COMPARE_COLUMNS].to_numpy(float),
                   rtol=0, atol=1e-9)
passed("rebuilding all features from matches_clean.csv reproduces features.csv")

train_only = build_features(train_rows, params, defaults)
saved_train = saved[saved["split"] == "train"]
assert np.allclose(train_only[COMPARE_COLUMNS].to_numpy(float),
                   saved_train[COMPARE_COLUMNS].to_numpy(float), rtol=0, atol=1e-9)
passed("training-row features are identical with and without the test season in the data")

# ------------------------------------------------------------ perturbation tests
rng = np.random.default_rng(42)
last_date = matches["Date"].max()
candidates = matches.index[matches["Date"] < last_date]
train_ids = rng.choice(candidates[matches.loc[candidates, "split"] == "train"], 4, replace=False)
test_ids = rng.choice(candidates[matches.loc[candidates, "split"] == "test"], 2, replace=False)
print("\nPerturbation tests (change the final score and shots of one match, recompute everything):")
for i in list(train_ids) + list(test_ids):
    date = matches.loc[i, "Date"]
    perturbed = build_features(perturb(matches, i), params, defaults)
    not_later = (matches["Date"] <= date).to_numpy()   # this match, same-day matches, all earlier ones
    leaks = leaking_columns(rebuilt[COMPARE_COLUMNS], perturbed[COMPARE_COLUMNS], not_later)
    assert not leaks, f"match {i} leaks into: {leaks}"
    later_changed = int((rebuilt.loc[~not_later, COMPARE_COLUMNS].to_numpy()
                         != perturbed.loc[~not_later, COMPARE_COLUMNS].to_numpy()).any(axis=1).sum())
    assert later_changed > 0, "the change should reach later matches, otherwise the test proves nothing"
    print(f"  match {i:4} ({matches.loc[i, 'HomeTeam']} v {matches.loc[i, 'AwayTeam']}, "
          f"{date.date()}, {matches.loc[i, 'split']}): {int(not_later.sum())} rows on or before this "
          f"date unchanged; {later_changed} later rows changed")
passed("changing a match's score and shots never changes its own or any earlier/same-day features")

# The test itself must be able to catch a leak: add a deliberately leaky column.
i = int(train_ids[0])
leaky_before = rebuilt[COMPARE_COLUMNS].copy()
leaky_before["leaky_same_match_goals"] = matches["home_goals"].to_numpy()
leaky_after = build_features(perturb(matches, i), params, defaults)[COMPARE_COLUMNS]
leaky_after["leaky_same_match_goals"] = perturb(matches, i)["home_goals"].to_numpy()
mask = (matches["Date"] <= matches.loc[i, "Date"]).to_numpy()
assert leaking_columns(leaky_before, leaky_after, mask) == ["leaky_same_match_goals"]
passed("detector self-test: a deliberately leaky column (same-match goals) IS caught by this test")

print("\nAll checks passed.")
