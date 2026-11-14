"""Stage 5d: level-neutral goal features (a second table; features.csv is NOT changed).

Run:  python src/features_goals.py     ->  data/processed/features_goals.csv  (same rows as matches_clean.csv)

Each "_rel" feature = a team's rolling goal average divided by the LEAGUE average of the same quantity,
so 1.0 means "league average". Dividing by the league average removes the scoring level of the moment
(for example the 2023-24 jump), so the model only sees how strong a team is RELATIVE to the league.

No leakage: a _rel feature uses (a) the SAME prior matches as its parent feature in features.csv (the team's
last 10 matches, or last 10 home/away matches) and (b) the league average over the matches played between the
start of that window and the day before this match. Matches on the same date are treated as simultaneous,
exactly as in Stage 4. build_goal_features() recomputes every parent feature and asserts it equals the one
built by src/features.py.
"""
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from features import (FORM_WINDOWS, MATCHES_CSV, PARAMS_JSON, SHRINK_K, VENUE_WINDOW, build_features,
                      compute_defaults, season_membership)

ROOT = Path(__file__).resolve().parent.parent
GOAL_FEATURES_CSV = ROOT / "data" / "processed" / "features_goals.csv"
GOAL_PARAMS_JSON = ROOT / "data" / "processed" / "goal_feature_params.json"
WINDOW = max(FORM_WINDOWS)          # 10 matches
MIN_LEAGUE_MATCHES = 20             # fewer league matches than this in a window -> use the neutral default

REL_COLUMNS = ["home_gf_10_rel", "away_gf_10_rel", "home_ga_10_rel", "away_ga_10_rel",
               "home_venue_gf_rel", "home_venue_ga_rel", "away_venue_gf_rel", "away_venue_ga_rel"]
KEPT_COLUMNS = ["elo_diff", "abs_elo_diff", "league_avg_goals_prev380", "season_no_fans"]
GOAL_FEATURE_COLUMNS = REL_COLUMNS + KEPT_COLUMNS
META_COLUMNS = ["match_id", "Date", "season", "split", "HomeTeam", "AwayTeam", "result", "home_goals",
                "away_goals", "warmup"]
# parent columns in features.csv that we recompute to prove the windows are identical
PARENT_COLUMNS = ["home_gf_10", "home_ga_10", "away_gf_10", "away_ga_10",
                  "home_venue_gf", "home_venue_ga", "away_venue_gf", "away_venue_ga"]


def compute_goal_defaults(train):
    """Neutral league averages for 'not enough league matches yet', from TRAINING rows only."""
    pooled = pd.concat([train["home_goals"], train["away_goals"]])
    return {"per_team": float(pooled.mean()), "home": float(train["home_goals"].mean()),
            "away": float(train["away_goals"].mean())}


def build_goal_features(matches, params, defaults, goal_defaults, check_parents=True):
    """One chronological pass. Returns the goal features (REL_COLUMNS + KEPT_COLUMNS) in row order."""
    matches = matches.reset_index(drop=True)
    assert matches["Date"].is_monotonic_increasing
    base = build_features(matches, params, defaults)            # Stage 4 features (Elo, parents, league level)
    _, _, new_teams = season_membership(matches)

    dates = matches["Date"].to_numpy().astype("datetime64[ns]").astype("int64")
    hg, ag = matches["home_goals"].to_numpy(), matches["away_goals"].to_numpy()
    homes, aways, seasons = matches["HomeTeam"].to_numpy(), matches["AwayTeam"].to_numpy(), matches["season"].to_numpy()
    cum_home, cum_away = np.concatenate([[0], np.cumsum(hg)]), np.concatenate([[0], np.cumsum(ag)])

    def league_means(start, end):
        """League averages over matches with start <= date < end (all strictly before this match)."""
        lo, hi = np.searchsorted(dates, start, side="left"), np.searchsorted(dates, end, side="left")
        n = hi - lo
        if n < MIN_LEAGUE_MATCHES:
            return goal_defaults["per_team"], goal_defaults["home"], goal_defaults["away"]
        home_avg, away_avg = (cum_home[hi] - cum_home[lo]) / n, (cum_away[hi] - cum_away[lo]) / n
        return (home_avg + away_avg) / 2, home_avg, away_avg

    overall = defaultdict(lambda: deque(maxlen=WINDOW))          # (date, gf, ga) of the last 10 matches
    home_games = defaultdict(lambda: deque(maxlen=VENUE_WINDOW))
    away_games = defaultdict(lambda: deque(maxlen=VENUE_WINDOW))

    def side(team, date, is_home):
        """The four relative features and the four recomputed parent values for one team."""
        records = overall[team]
        if len(records) == 0:
            return [1.0] * 4, [np.nan] * 4                       # no history: neutral
        venue = home_games[team] if is_home else away_games[team]
        gf10, ga10 = np.mean([r[1] for r in records]), np.mean([r[2] for r in records])
        per_team = league_means(records[0][0], date)[0]
        n = len(venue)
        if n == 0:
            venue_gf, venue_ga, base_gf, base_ga = gf10, ga10, per_team, per_team
        else:
            v_gf, v_ga = np.mean([r[1] for r in venue]), np.mean([r[2] for r in venue])
            _, v_home_avg, v_away_avg = league_means(venue[0][0], date)
            own_avg, opp_avg = (v_home_avg, v_away_avg) if is_home else (v_away_avg, v_home_avg)
            venue_gf = (n * v_gf + SHRINK_K * gf10) / (n + SHRINK_K)        # the Stage 4 shrinkage
            venue_ga = (n * v_ga + SHRINK_K * ga10) / (n + SHRINK_K)
            base_gf = (n * own_avg + SHRINK_K * per_team) / (n + SHRINK_K)  # same blend for the league baseline
            base_ga = (n * opp_avg + SHRINK_K * per_team) / (n + SHRINK_K)  # goals CONCEDED = what the other side scores
        return ([gf10 / per_team, ga10 / per_team, venue_gf / base_gf, venue_ga / base_ga],
                [gf10, ga10, venue_gf, venue_ga])

    rel = np.zeros((len(matches), 8))
    parents = np.full((len(matches), 8), np.nan)
    current_season = None
    groups = matches.groupby("Date", sort=True).indices
    for date_key in sorted(groups):
        positions = groups[date_key]
        season = seasons[positions[0]]
        if season != current_season:
            for team in new_teams[season]:                       # promoted / returning teams start fresh
                overall.pop(team, None)
                home_games.pop(team, None)
                away_games.pop(team, None)
            current_season = season
        for i in positions:                                      # 1) features from history BEFORE this date
            h_rel, h_par = side(homes[i], dates[i], True)
            a_rel, a_par = side(aways[i], dates[i], False)
            rel[i] = [h_rel[0], a_rel[0], h_rel[1], a_rel[1], h_rel[2], h_rel[3], a_rel[2], a_rel[3]]
            parents[i] = [h_par[0], h_par[1], a_par[0], a_par[1], h_par[2], h_par[3], a_par[2], a_par[3]]
        for i in positions:                                      # 2) only now add this date's results
            overall[homes[i]].append((dates[i], hg[i], ag[i]))
            overall[aways[i]].append((dates[i], ag[i], hg[i]))
            home_games[homes[i]].append((dates[i], hg[i], ag[i]))
            away_games[aways[i]].append((dates[i], ag[i], hg[i]))

    if check_parents:   # the windows behind every _rel feature are exactly the ones behind its parent
        for j, column in enumerate(["home_gf_10", "home_ga_10", "away_gf_10", "away_ga_10",
                                    "home_venue_gf", "home_venue_ga", "away_venue_gf", "away_venue_ga"]):
            has_history = ~np.isnan(parents[:, j])
            assert np.allclose(parents[has_history, j], base[column].to_numpy(float)[has_history], rtol=0, atol=1e-9), column

    out = pd.DataFrame(rel, columns=["home_gf_10_rel", "home_ga_10_rel", "away_gf_10_rel", "away_ga_10_rel",
                                     "home_venue_gf_rel", "home_venue_ga_rel", "away_venue_gf_rel", "away_venue_ga_rel"])
    out = out[REL_COLUMNS]
    for column in KEPT_COLUMNS:
        out[column] = base[column].to_numpy()
    return out


def main():
    matches = pd.read_csv(MATCHES_CSV, parse_dates=["Date"])
    params = json.loads(PARAMS_JSON.read_text())
    train = matches[matches["split"] == "train"].reset_index(drop=True)
    goal_defaults = compute_goal_defaults(train)        # training rows only
    GOAL_PARAMS_JSON.write_text(json.dumps({"goal_defaults": goal_defaults, "min_league_matches": MIN_LEAGUE_MATCHES}, indent=2))
    print(f"Neutral league averages (training rows only): {goal_defaults}")

    goals = build_goal_features(matches, params["elo"], params["defaults"], goal_defaults)
    saved = pd.read_csv(ROOT / "data" / "processed" / "features.csv", usecols=["match_id", "warmup"])
    assert (saved["match_id"].to_numpy() == matches["match_id"].to_numpy()).all()
    meta = matches[["match_id", "Date", "season", "split", "HomeTeam", "AwayTeam", "result", "home_goals", "away_goals"]].copy()
    meta["warmup"] = saved["warmup"].to_numpy()
    table = pd.concat([meta, goals], axis=1)
    table.to_csv(GOAL_FEATURES_CSV, index=False)
    print(f"Saved {GOAL_FEATURES_CSV.relative_to(ROOT)}: {len(table)} rows (matches_clean.csv has {len(matches)}), "
          f"{len(GOAL_FEATURE_COLUMNS)} features, {GOAL_FEATURES_CSV.stat().st_size / 1024:.0f} KB")
    print(f"Missing values: {int(table[GOAL_FEATURE_COLUMNS].isna().sum().sum())}")
    print("\nThe _rel features on training rows (1.0 = league average):")
    print(table.loc[table["split"] == "train", REL_COLUMNS].describe().loc[["mean", "std", "min", "max"]].round(3).to_string())
    print("\nMean of home_gf_10_rel by season (should stay near 1 although scoring levels differ):")
    print(table.groupby("season")["home_gf_10_rel"].mean().round(3).to_string())
    raw = pd.read_csv(ROOT / "data" / "processed" / "features.csv", usecols=["season", "home_gf_10"])
    print("\nFor comparison, mean of the RAW home_gf_10 by season:")
    print(raw.groupby("season")["home_gf_10"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
