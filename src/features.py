"""Stage 4: feature engineering.

Turns data/processed/matches_clean.csv into data/processed/features.csv:
one row per match, using ONLY information available before kick-off.

Run:  python src/features.py

How leakage is prevented
- Matches are processed in date order. For every date we first compute the
  features of ALL matches on that date, and only then add those results to the
  history. So a match never sees its own result or any same-day result.
- Every constant we choose (Elo constants, neutral defaults) is chosen from
  training seasons only. The test season is excluded while tuning.
"""
import itertools
import json
from collections import defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from preprocess import TEST_SEASONS, VALIDATION_SCHEME

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MATCHES_CSV = PROCESSED_DIR / "matches_clean.csv"
FEATURES_CSV = PROCESSED_DIR / "features.csv"
PARAMS_JSON = PROCESSED_DIR / "feature_params.json"
ELO_START_CSV = PROCESSED_DIR / "elo_start_of_2025-26.csv"
SCALER_PATH = ROOT / "models" / "feature_scaler.joblib"
TUNING_CSV = ROOT / "reports" / "elo_tuning_grid.csv"

# ---------------------------------------------------------------- constants
ELO_START = 1500
# Candidate values for the Elo constants (chosen on training seasons only).
ELO_GRID = {"K": [10, 20, 30, 40], "H": [0, 50, 100],
            "keep": [0.6, 0.75, 0.9], "offset": [0, -50, -100, -150]}
FORM_WINDOWS = (5, 10)      # last 5 and last 10 matches
VENUE_WINDOW = 10           # last 10 home (or away) matches
SHOT_WINDOW = 10
H2H_MEETINGS = 3            # last 3 meetings between the two teams
LEAGUE_WINDOW = 380         # previous 380 matches = one season's worth
SHRINK_K = 5                # shrinkage strength (see lesson)
WARMUP_MATCHES = 5          # fewer earlier matches than this -> warmup = 1
NO_FANS_SEASONS = ["2020-21"]

# The whitelist: the ONLY columns models may use as inputs.
ELO_COLUMNS = ["home_elo", "away_elo", "elo_diff", "abs_elo_diff"]
FORM_COLUMNS = [f"{side}_{stat}_{w}" for side in ("home", "away") for w in FORM_WINDOWS
                for stat in ("ppg", "gf", "ga")]
VENUE_COLUMNS = ["home_venue_gf", "home_venue_ga", "away_venue_gf", "away_venue_ga"]
SHOT_COLUMNS = [f"{side}_{stat}_10" for side in ("home", "away")
                for stat in ("shots_for", "shots_against", "sot_for", "sot_against")]
H2H_COLUMNS = ["h2h_home_ppg_3", "h2h_n"]
CONTEXT_COLUMNS = ["league_avg_goals_prev380", "season_no_fans",
                   "home_matches_played", "away_matches_played"]
FEATURE_COLUMNS = (ELO_COLUMNS + FORM_COLUMNS + VENUE_COLUMNS + SHOT_COLUMNS
                   + H2H_COLUMNS + CONTEXT_COLUMNS)
META_COLUMNS = ["match_id", "Date", "season", "split", "HomeTeam", "AwayTeam",
                "result", "home_goals", "away_goals", "warmup", "defaults_used"]

POINTS_HOME = {0: 3, 1: 1, 2: 0}   # result code -> points for the home team
POINTS_AWAY = {0: 0, 1: 1, 2: 3}
ELO_SCORE_HOME = {0: 1.0, 1: 0.5, 2: 0.0}  # Elo "actual score" for the home team


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


# ------------------------------------------------ seasons and promoted teams
def season_membership(matches):
    """Which teams played each season, and which were 'new' (no match the season before)."""
    seasons = list(dict.fromkeys(matches["season"]))  # chronological order
    teams = {s: set(matches.loc[matches["season"] == s, "HomeTeam"])
             | set(matches.loc[matches["season"] == s, "AwayTeam"]) for s in seasons}
    new_teams = {seasons[0]: set()}  # in the first season nobody has a "previous season"
    for previous, current in zip(seasons, seasons[1:]):
        new_teams[current] = teams[current] - teams[previous]
    return seasons, teams, new_teams


def warmup_flags(matches):
    """1 if either team has fewer than 5 EARLIER matches anywhere in the data."""
    seen = defaultdict(int)
    flags = np.zeros(len(matches), dtype=bool)
    for i, (home, away) in enumerate(zip(matches["HomeTeam"], matches["AwayTeam"])):
        flags[i] = seen[home] < WARMUP_MATCHES or seen[away] < WARMUP_MATCHES
        seen[home] += 1
        seen[away] += 1
    return flags


# --------------------------------------------------------------------- Elo
def apply_season_start(ratings, season, teams, new_teams, params):
    """Between seasons: pull ratings toward 1500, and give new teams a lower start."""
    for team in teams[season]:
        if team in new_teams[season]:
            ratings[team] = ELO_START + params["offset"]  # promoted or returning team
        elif team in ratings:
            ratings[team] = ELO_START + params["keep"] * (ratings[team] - ELO_START)
        else:
            ratings[team] = ELO_START  # very first season: everyone starts equal


def elo_expected(rating_home, rating_away, home_bonus):
    """Expected score of the home team (between 0 and 1)."""
    return 1 / (1 + 10 ** (-(rating_home + home_bonus - rating_away) / 400))


def compute_elo(matches, params):
    """Walk through matches in order. Returns ratings BEFORE each match, the
    expected home score, and the final ratings."""
    seasons, teams, new_teams = season_membership(matches)
    ratings = {}
    n = len(matches)
    home_elo, away_elo, expected = np.zeros(n), np.zeros(n), np.zeros(n)
    current_season = None
    columns = zip(matches["season"], matches["HomeTeam"], matches["AwayTeam"], matches["result"])
    for i, (season, home, away, result) in enumerate(columns):
        if season != current_season:
            apply_season_start(ratings, season, teams, new_teams, params)
            current_season = season
        rating_home, rating_away = ratings[home], ratings[away]
        home_elo[i], away_elo[i] = rating_home, rating_away
        expected[i] = elo_expected(rating_home, rating_away, params["H"])
        change = params["K"] * (ELO_SCORE_HOME[result] - expected[i])
        ratings[home] = rating_home + change   # the winner gains what the loser loses
        ratings[away] = rating_away - change
    return home_elo, away_elo, expected, ratings


def tune_elo(train):
    """Pick K, H, keep-fraction and promoted offset by the squared error between
    Elo's expected score and the real score. Training seasons only."""
    section("Elo tuning (training seasons only)")
    actual = train["result"].map(ELO_SCORE_HOME).to_numpy()
    warmup = warmup_flags(train)
    in_fit = train["season"].isin(VALIDATION_SCHEME["fit"]).to_numpy()
    in_validate = train["season"].isin(VALIDATION_SCHEME["validate"]).to_numpy()
    fit_mask = in_fit & ~warmup
    validate_mask = in_validate & ~warmup
    no_first_season = fit_mask & (train["season"] != train["season"].iloc[0]).to_numpy()

    def mse(expected, mask):
        return float(np.mean((expected[mask] - actual[mask]) ** 2))

    rows = []
    for values in itertools.product(*ELO_GRID.values()):
        params = dict(zip(ELO_GRID.keys(), values))
        expected = compute_elo(train, params)[2]
        rows.append({**params, "fit_mse": mse(expected, fit_mask),
                     "validate_mse": mse(expected, validate_mask),
                     "fit_mse_without_first_season": mse(expected, no_first_season)})
    grid = pd.DataFrame(rows).sort_values("fit_mse").reset_index(drop=True)
    TUNING_CSV.parent.mkdir(exist_ok=True)
    grid.to_csv(TUNING_CSV, index=False)

    baseline = np.full(len(actual), actual[fit_mask].mean())
    print(f"Rows used: {int(fit_mask.sum())} fit (2015-16 to 2022-23, warm-up rows skipped), "
          f"{int(validate_mask.sum())} validation (2023-24, 2024-25)")
    print(f"Baseline (always predict the average score {actual[fit_mask].mean():.3f}): "
          f"fit error {mse(baseline, fit_mask):.4f}, validation error {mse(baseline, validate_mask):.4f}")
    print(f"\nBest 8 of {len(grid)} combinations (error = mean squared error, lower is better):")
    print(grid.head(8).round(4).to_string())
    print("\nBest fit error for each single value (all other constants free):")
    for name in ELO_GRID:
        best = grid.groupby(name)["fit_mse"].min().round(4)
        print(f"  {name:6}", "  ".join(f"{k}: {v}" for k, v in best.items()))

    best = grid.iloc[0]
    params = {"K": int(best["K"]), "H": int(best["H"]), "keep": float(best["keep"]),
              "offset": int(best["offset"])}
    robust = grid.sort_values("fit_mse_without_first_season").iloc[0]
    print(f"\nChosen: K={params['K']}, H={params['H']}, keep={params['keep']}, "
          f"offset={params['offset']}")
    print(f"  fit error {best['fit_mse']:.4f}   validation error {best['validate_mse']:.4f}")
    print(f"Robustness check (ignore 2015-16, where every team starts at 1500): best would be "
          f"K={int(robust['K'])}, H={int(robust['H'])}, keep={robust['keep']}, "
          f"offset={int(robust['offset'])}")
    for name, values in ELO_GRID.items():
        if params[name] in (values[0], values[-1]):
            print(f"  note: {name}={params[name]} is at the edge of the grid {values}, "
                  f"so the true best value may lie outside it")
    return params


def elo_at_season_start(matches, season, params):
    """Ratings at the start of `season`, from matches BEFORE it only.
    Only the team names of the new season are needed (known from the fixture list)."""
    before = matches[matches["season"] != season]
    before = before[before["Date"] < matches.loc[matches["season"] == season, "Date"].min()]
    final_ratings = compute_elo(before.reset_index(drop=True), params)[3]
    _, teams, new_teams = season_membership(matches[matches["Date"] <=
                                                    matches.loc[matches["season"] == season, "Date"].max()])
    ratings = dict(final_ratings)
    apply_season_start(ratings, season, teams, new_teams, params)
    table = pd.DataFrame({"team": sorted(teams[season])})
    table["elo_start"] = table["team"].map(ratings).round(1)
    table["promoted_or_returning"] = table["team"].isin(new_teams[season])
    return table.sort_values("elo_start", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------- defaults
def compute_defaults(train):
    """Neutral values for 'no history yet', from TRAINING rows only."""
    points = pd.concat([train["result"].map(POINTS_HOME), train["result"].map(POINTS_AWAY)])
    goals = pd.concat([train["home_goals"], train["away_goals"]])
    return {"ppg": float(points.mean()), "goals": float(goals.mean()),
            "shots": float(pd.concat([train["HS"], train["AS"]]).mean()),
            "sot": float(pd.concat([train["HST"], train["AST"]]).mean()),
            "league_goals": float((train["home_goals"] + train["away_goals"]).mean())}


# ---------------------------------------------------------- rolling helpers
def mean_of(records, index, window, default):
    """Mean of column `index` over the last `window` records (default if none)."""
    recent = list(records)[-window:]
    if not recent:
        return default
    return sum(r[index] for r in recent) / len(recent)


def blended_venue(venue_records, overall_gf, overall_ga):
    """Shrink venue averages toward the team's overall averages:
    (n x venue_mean + k x overall_mean) / (n + k). With n = 0 we get the overall mean."""
    n = len(venue_records)
    if n == 0:
        return overall_gf, overall_ga
    venue_gf = sum(r[0] for r in venue_records) / n
    venue_ga = sum(r[1] for r in venue_records) / n
    return ((n * venue_gf + SHRINK_K * overall_gf) / (n + SHRINK_K),
            (n * venue_ga + SHRINK_K * overall_ga) / (n + SHRINK_K))


def head_to_head(meetings, home_team, default_ppg):
    """Home team's points per game in the last (up to 3) meetings, and how many there were."""
    if not meetings:
        return default_ppg, 0
    points = [POINTS_HOME[result] if meeting_home == home_team else POINTS_AWAY[result]
              for meeting_home, _, result in meetings]
    return sum(points) / len(points), len(points)


# --------------------------------------------------------- the main feature pass
def build_features(matches, params, defaults):
    """One chronological pass. Returns FEATURE_COLUMNS plus `defaults_used`."""
    matches = matches.reset_index(drop=True)
    assert matches["Date"].is_monotonic_increasing, "matches must be sorted by date"
    home_elo, away_elo, _, _ = compute_elo(matches, params)
    _, _, new_teams = season_membership(matches)

    # Per-team memory. Records are tuples:
    #   overall: (points, goals for, goals against, shots for, shots against, SOT for, SOT against)
    #   venue:   (goals for, goals against)
    overall = defaultdict(lambda: deque(maxlen=max(FORM_WINDOWS)))
    home_games = defaultdict(lambda: deque(maxlen=VENUE_WINDOW))
    away_games = defaultdict(lambda: deque(maxlen=VENUE_WINDOW))
    played_this_season = defaultdict(int)
    meetings = defaultdict(lambda: deque(maxlen=H2H_MEETINGS))
    league_goals = deque(maxlen=LEAGUE_WINDOW)

    seasons = matches["season"].to_numpy()
    homes, aways = matches["HomeTeam"].to_numpy(), matches["AwayTeam"].to_numpy()
    results = matches["result"].to_numpy()
    hg, ag = matches["home_goals"].to_numpy(), matches["away_goals"].to_numpy()
    shots_h, shots_a = matches["HS"].to_numpy(), matches["AS"].to_numpy()
    sot_h, sot_a = matches["HST"].to_numpy(), matches["AST"].to_numpy()

    rows = [None] * len(matches)
    current_season = None
    groups = matches.groupby("Date", sort=True).indices
    for date in sorted(groups):
        positions = groups[date]
        season = seasons[positions[0]]
        if season != current_season:
            for team in new_teams[season]:   # promoted/returning teams start with no history
                overall.pop(team, None)
                home_games.pop(team, None)
                away_games.pop(team, None)
            played_this_season.clear()
            current_season = season

        # 1) features for every match on this date, from history BEFORE this date
        for i in positions:
            home, away = homes[i], aways[i]
            row = {"home_elo": home_elo[i], "away_elo": away_elo[i],
                   "elo_diff": home_elo[i] - away_elo[i],
                   "abs_elo_diff": abs(home_elo[i] - away_elo[i])}
            for side, team in (("home", home), ("away", away)):
                for window in FORM_WINDOWS:
                    row[f"{side}_ppg_{window}"] = mean_of(overall[team], 0, window, defaults["ppg"])
                    row[f"{side}_gf_{window}"] = mean_of(overall[team], 1, window, defaults["goals"])
                    row[f"{side}_ga_{window}"] = mean_of(overall[team], 2, window, defaults["goals"])
                row[f"{side}_shots_for_10"] = mean_of(overall[team], 3, SHOT_WINDOW, defaults["shots"])
                row[f"{side}_shots_against_10"] = mean_of(overall[team], 4, SHOT_WINDOW, defaults["shots"])
                row[f"{side}_sot_for_10"] = mean_of(overall[team], 5, SHOT_WINDOW, defaults["sot"])
                row[f"{side}_sot_against_10"] = mean_of(overall[team], 6, SHOT_WINDOW, defaults["sot"])
            row["home_venue_gf"], row["home_venue_ga"] = blended_venue(
                home_games[home], row["home_gf_10"], row["home_ga_10"])
            row["away_venue_gf"], row["away_venue_ga"] = blended_venue(
                away_games[away], row["away_gf_10"], row["away_ga_10"])
            row["h2h_home_ppg_3"], row["h2h_n"] = head_to_head(
                meetings[frozenset((home, away))], home, defaults["ppg"])
            row["league_avg_goals_prev380"] = (sum(league_goals) / len(league_goals)
                                               if league_goals else defaults["league_goals"])
            row["season_no_fans"] = int(season in NO_FANS_SEASONS)
            row["home_matches_played"] = played_this_season[home]
            row["away_matches_played"] = played_this_season[away]
            row["defaults_used"] = int(len(overall[home]) == 0 or len(overall[away]) == 0)
            rows[i] = row

        # 2) only now add this date's results to the history
        for i in positions:
            home, away, result = homes[i], aways[i], results[i]
            overall[home].append((POINTS_HOME[result], hg[i], ag[i], shots_h[i], shots_a[i],
                                  sot_h[i], sot_a[i]))
            overall[away].append((POINTS_AWAY[result], ag[i], hg[i], shots_a[i], shots_h[i],
                                  sot_a[i], sot_h[i]))
            home_games[home].append((hg[i], ag[i]))
            away_games[away].append((ag[i], hg[i]))
            played_this_season[home] += 1
            played_this_season[away] += 1
            meetings[frozenset((home, away))].append((home, away, result))
            league_goals.append(hg[i] + ag[i])

    return pd.DataFrame(rows)[FEATURE_COLUMNS + ["defaults_used"]]


# ------------------------------------------------------------------ scaling
def fit_feature_scaler(features, split):
    """Standardise features with mean/std from TRAINING rows only."""
    train_features = features.loc[split == "train", FEATURE_COLUMNS]
    scaler = StandardScaler().fit(train_features)
    report = pd.DataFrame({
        "mean before": train_features.mean(), "std before": train_features.std(ddof=0),
        "mean after": 0.0, "std after": 1.0}).round(3)
    scaled = pd.DataFrame(scaler.transform(train_features), columns=FEATURE_COLUMNS)
    report["mean after"] = scaled.mean().round(3) + 0.0
    report["std after"] = scaled.std(ddof=0).round(3)
    return scaler, report


# --------------------------------------------------------------------- main
def main():
    matches = pd.read_csv(MATCHES_CSV, parse_dates=["Date"])
    assert matches["Date"].is_monotonic_increasing
    train = matches[matches["split"] == "train"].reset_index(drop=True)
    print(f"Loaded {len(matches)} matches ({len(train)} training, "
          f"{int((matches['split'] == 'test').sum())} test)")

    section("Neutral defaults (from training rows only)")
    defaults = compute_defaults(train)
    for name, value in defaults.items():
        print(f"  {name:13} {value:.3f}")

    # Tuning sees only training matches: the test season is not even loaded into it.
    params = tune_elo(train)
    PARAMS_JSON.write_text(json.dumps({"elo": params, "defaults": defaults}, indent=2))

    section("Building features")
    features = build_features(matches, params, defaults)
    out = matches[["match_id", "Date", "season", "split", "HomeTeam", "AwayTeam", "result",
                   "home_goals", "away_goals"]].copy()
    out["warmup"] = warmup_flags(matches).astype(int)
    out["defaults_used"] = features.pop("defaults_used")
    out = pd.concat([out, features], axis=1)[META_COLUMNS + FEATURE_COLUMNS]
    out.to_csv(FEATURES_CSV, index=False)
    print(f"Saved {FEATURES_CSV.relative_to(ROOT)}: {len(out)} rows, "
          f"{len(FEATURE_COLUMNS)} features, {FEATURES_CSV.stat().st_size / 1024:.0f} KB")
    print(f"Missing values in features: {int(out[FEATURE_COLUMNS].isna().sum().sum())}")

    section("Rows with no or short history")
    for name, subset in [("all", out), ("train", out[out["split"] == "train"]),
                         ("test", out[out["split"] == "test"])]:
        print(f"{name:6} {len(subset):5} rows | warmup=1: {int(subset['warmup'].sum()):4} "
              f"| neutral defaults used (a team had no history): {int(subset['defaults_used'].sum()):4} "
              f"| h2h_n=0 (no previous meeting): {int((subset['h2h_n'] == 0).sum()):4}")
    print("warmup=1 by season:", out.groupby("season")["warmup"].sum().to_dict())

    section("Elo at the start of 2025-26 (from data before 2025-26 only)")
    start = elo_at_season_start(matches, TEST_SEASONS[0], params)
    start.to_csv(ELO_START_CSV, index=False)
    print(start.to_string())

    section("Feature scaling (fit on training rows only)")
    scaler, report = fit_feature_scaler(out, out["split"])
    SCALER_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(report.to_string())
    print(f"Saved {SCALER_PATH.relative_to(ROOT)}. features.csv itself stays unscaled.")


if __name__ == "__main__":
    main()
