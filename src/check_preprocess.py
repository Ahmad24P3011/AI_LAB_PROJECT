"""Assertions that check the Stage 2 outputs. Run after src/preprocess.py.

Every check prints PASS only after its assertion succeeded. If an assertion
fails, Python stops with an error and the final 'All checks passed' is NOT printed.
"""
import re

import joblib
import pandas as pd

from preprocess import (CORE_COLUMNS, MODELS_DIR, PROCESSED_DIR, RESULT_CODES,
                        STAT_COLUMNS, TEST_SEASONS, TRAIN_SEASONS,
                        VALIDATION_SCHEME)

dates = ["Date"]
matches = pd.read_csv(PROCESSED_DIR / "matches_clean.csv", parse_dates=dates)
train = pd.read_csv(PROCESSED_DIR / "train.csv", parse_dates=dates)
test = pd.read_csv(PROCESSED_DIR / "test.csv", parse_dates=dates)
odds = pd.read_csv(PROCESSED_DIR / "odds.csv", parse_dates=dates)


def passed(message):
    print(f"PASS  {message}")


# --- row counts
assert len(matches) == 4180, len(matches)
assert (matches.groupby("season").size() == 380).all()
assert matches["season"].nunique() == 11
passed("4180 matches: 11 seasons x 380")
assert len(train) == 3800 and len(test) == 380 and len(train) + len(test) == len(matches)
assert set(train["season"]) == set(TRAIN_SEASONS) and set(test["season"]) == set(TEST_SEASONS)
passed("train = 3800 (2015-16..2024-25), test = 380 (2025-26)")

# --- duplicates and missing values
assert not matches.duplicated(subset=["Date", "HomeTeam", "AwayTeam"]).any()
passed("no duplicate matches")
assert matches[CORE_COLUMNS].isna().sum().sum() == 0
passed("no missing values in core columns")
assert matches[STAT_COLUMNS].isna().sum().sum() == 0
passed("no missing values in match stats")

# --- dates
assert matches["Date"].is_monotonic_increasing
assert matches["match_id"].tolist() == list(range(len(matches)))
passed("dates are sorted (oldest first) and match_id runs 0..4179")
for season, group in matches.groupby("season"):
    start_year = int(season[:4])
    assert group["Date"].min() >= pd.Timestamp(f"{start_year}-08-01"), season
    assert group["Date"].max() <= pd.Timestamp(f"{start_year + 1}-07-31"), season
passed("every date falls inside its own season (so 2-digit years were read right)")

# --- team names
teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
simplified = {}
for name in teams:
    simplified.setdefault(re.sub(r"[^a-z]", "", name.lower()), []).append(name)
assert all(len(names) == 1 for names in simplified.values()), simplified
assert len(teams) == 34
for season, group in matches.groupby("season"):
    assert len(set(group["HomeTeam"]) | set(group["AwayTeam"])) == 20, season
passed("exactly one spelling per club: 34 clubs, 20 per season")

# --- targets
assert set(matches["result"]) == set(RESULT_CODES.values())
assert (matches["result"] == matches["FTR"].map(RESULT_CODES)).all()
assert (matches["home_goals"] == matches["FTHG"]).all()
assert (matches["away_goals"] == matches["FTAG"]).all()
assert matches["home_goals"].dtype.kind == "i" and matches["away_goals"].dtype.kind == "i"
sign = (matches["home_goals"] - matches["away_goals"]).apply(lambda d: 0 if d > 0 else (2 if d < 0 else 1))
assert (sign == matches["result"]).all()
passed("result codes (0 H, 1 D, 2 A) agree with FTR and with the score; goals are integers")

# --- time split
assert train["Date"].max() < test["Date"].min()
passed(f"train ends {train['Date'].max().date()}, test starts {test['Date'].min().date()}: no overlap")
fit = train[train["season"].isin(VALIDATION_SCHEME["fit"])]
validate = train[train["season"].isin(VALIDATION_SCHEME["validate"])]
assert fit["Date"].max() < validate["Date"].min()
assert len(fit) == 3040 and len(validate) == 760
assert not set(validate["season"]) & set(TEST_SEASONS)
passed("validation scheme: fit 3040 before validate 760, test season never used")

# --- other outputs
assert (odds["match_id"] == matches["match_id"]).all()
assert not any(c in odds.columns for c in ["result", "home_goals", "away_goals"])
passed("odds.csv has the same matches in the same order and no target columns")
encoder = joblib.load(MODELS_DIR / "team_encoder.joblib")
scaler = joblib.load(MODELS_DIR / "stats_scaler.joblib")
assert len(encoder.categories_[0]) == 34 and list(scaler.feature_names_in_) == STAT_COLUMNS
# The scaler's mean must equal the TRAINING mean (not the test or whole-table mean).
assert (abs(scaler.mean_ - train[STAT_COLUMNS].mean().values) < 1e-9).all()
assert (abs(scaler.mean_ - matches[STAT_COLUMNS].mean().values) > 1e-9).any()
passed("saved encoder and scaler load, and the scaler was fitted on training data only")

print("\nAll checks passed.")
