"""Stage 9, step 1: build the small data files the web app reads. Run ONCE (the app never runs this):

    python app/team_state_builder.py

It writes
  data/processed/team_state_end_of_2025-26.csv   one row per team: Elo and last-10 goal form AFTER its last 2025-26 match
  data/app/bookmaker_2025-26.csv                 the bookmaker's (margin-removed) probabilities for the 380 team pairs
  data/app/sim_position_probs.csv                P(team finishes in position 1..20) over the 1,000 simulated seasons
  data/app/real_results_2025-26.csv              the real score of each 2025-26 fixture (shown on Page 1 for comparison only)

HOW THE TEAM STATE IS BUILT (the "end of 2025-26" trick). The models were trained on features that describe each team
BEFORE a match. To ask "what would the models say about Liverpool v Bournemouth right now?" we need those same features
for a match that has not been played yet. So we append 380 hypothetical fixtures (every ordered pair of teams, one
shared date after the season ended) to the real matches and run the project's own feature code (Stage 4 and 5d) over
the lot. Features of a match use only matches on EARLIER dates, so each hypothetical fixture gets exactly the state of
each team after its last real match, with no new code to get wrong. The hypothetical fixtures carry dummy results, but
those are only added to the history AFTER their own features are built, so they never leak into one another.
Elo is the one exception (the Elo walk is row by row), so it is taken from the real matches alone.

This script is data preparation, not training: it reads the 2025-26 results that the project's own features already
used. The app itself reads only the files it writes.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import PARAMS_JSON, build_features, compute_elo           # noqa: E402
from features_goals import GOAL_PARAMS_JSON, build_goal_features        # noqa: E402

MATCHES_CSV = ROOT / "data" / "processed" / "matches_clean.csv"
ODDS_CSV = ROOT / "data" / "processed" / "odds.csv"
TEAM_STATE_CSV = ROOT / "data" / "processed" / "team_state_end_of_2025-26.csv"
BOOKMAKER_CSV = ROOT / "data" / "app" / "bookmaker_2025-26.csv"
POSITION_PROBS_CSV = ROOT / "data" / "app" / "sim_position_probs.csv"
REAL_RESULTS_CSV = ROOT / "data" / "app" / "real_results_2025-26.csv"
SIM_TABLES_CSV = ROOT / "reports" / "sim_tables.csv"
SEASON = "2025-26"


def hypothetical_fixtures(matches):
    """380 fixtures (every home/away pair of the 2025-26 teams), all on one day after the season ended."""
    teams = sorted(set(matches.loc[matches["season"] == SEASON, "HomeTeam"]))
    day = matches["Date"].max() + pd.Timedelta(days=7)
    pairs = [(h, a) for h in teams for a in teams if h != a]
    fixtures = pd.DataFrame({"Date": day, "season": SEASON, "split": "test", "HomeTeam": [p[0] for p in pairs],
                             "AwayTeam": [p[1] for p in pairs]})
    for column in ("result", "home_goals", "away_goals", "HS", "AS", "HST", "AST"):
        fixtures[column] = 1 if column == "result" else 0          # dummy: only used AFTER the features are built
    return fixtures


def pair_inputs(matches, params, goal_defaults):
    """For all 380 hypothetical fixtures: the Poisson model's input table, built by the project's own feature code.
    Returns (pairs, goal_features, stage4_features, final_elo)."""
    fixtures = hypothetical_fixtures(matches)
    both = pd.concat([matches, fixtures], ignore_index=True)
    n = len(fixtures)
    goal_features = build_goal_features(both, params["elo"], params["defaults"], goal_defaults).iloc[-n:].reset_index(drop=True)
    stage4 = build_features(both, params["elo"], params["defaults"]).iloc[-n:].reset_index(drop=True)
    final_elo = compute_elo(matches, params["elo"])[3]              # real matches only (see the module note)
    elo_diff = fixtures["HomeTeam"].map(final_elo).to_numpy() - fixtures["AwayTeam"].map(final_elo).to_numpy()
    goal_features["elo_diff"], goal_features["abs_elo_diff"] = elo_diff, np.abs(elo_diff)
    return fixtures[["HomeTeam", "AwayTeam"]].reset_index(drop=True), goal_features, stage4, final_elo


def build_team_state(matches, params, goal_defaults):
    """One row per team. Each value is read from a column whose label is correct (see the note on the _rel swap)."""
    pairs, goals, stage4, final_elo = pair_inputs(matches, params, goal_defaults)
    played = matches[matches["season"] == SEASON]
    rows = []
    for team in sorted(final_elo.keys() & set(pairs["HomeTeam"])):
        as_home = (pairs["HomeTeam"] == team).to_numpy().nonzero()[0][0]       # any fixture where `team` is at home
        as_away = (pairs["AwayTeam"] == team).to_numpy().nonzero()[0][0]       # any fixture where `team` is away
        # NOTE: src/features_goals.py stores two _rel columns under swapped labels (documented in docs/decisions.md).
        # Only the correctly labelled ones are used here: home_gf_10_rel / home_venue_* for a home side and
        # away_ga_10_rel / away_venue_* for an away side.
        rows.append({
            "team": team,
            "elo": final_elo[team],
            "gf_10_rel": goals.loc[as_home, "home_gf_10_rel"],
            "ga_10_rel": goals.loc[as_away, "away_ga_10_rel"],
            "home_venue_gf_rel": goals.loc[as_home, "home_venue_gf_rel"],
            "home_venue_ga_rel": goals.loc[as_home, "home_venue_ga_rel"],
            "away_venue_gf_rel": goals.loc[as_away, "away_venue_gf_rel"],
            "away_venue_ga_rel": goals.loc[as_away, "away_venue_ga_rel"],
            "ppg_10": stage4.loc[as_home, "home_ppg_10"],
            "gf_10": stage4.loc[as_home, "home_gf_10"],
            "ga_10": stage4.loc[as_home, "home_ga_10"],
            "matches_played_2025_26": int(((played["HomeTeam"] == team) | (played["AwayTeam"] == team)).sum()),
            "last_match_date": played.loc[(played["HomeTeam"] == team) | (played["AwayTeam"] == team), "Date"].max().date().isoformat(),
        })
    state = pd.DataFrame(rows)
    state["league_goals_per_match"] = float(goals["league_avg_goals_prev380"].iloc[0])   # goals per match over the last 380 matches
    state["season_no_fans"] = 0
    return state


def build_bookmaker_table():
    """B365 odds -> probabilities, divided by their sum to remove the bookmaker's margin (the Stage 6 method)."""
    odds = pd.read_csv(ODDS_CSV, usecols=["season", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"])
    odds = odds[odds["season"] == SEASON].dropna()
    raw = 1 / odds[["B365H", "B365D", "B365A"]].to_numpy()
    table = odds[["HomeTeam", "AwayTeam"]].reset_index(drop=True)
    table[["p_home", "p_draw", "p_away"]] = raw / raw.sum(axis=1, keepdims=True)
    return table


def build_real_results(matches):
    """The real 2025-26 score of every fixture. Columns are renamed home_score / away_score on purpose: the app then never
    reads a raw score column (src/check_app.py enforces that), only this small, labelled, display-only table."""
    played = matches[matches["season"] == SEASON]
    table = played[["Date", "HomeTeam", "AwayTeam", "home_goals", "away_goals"]].rename(columns={"home_goals": "home_score", "away_goals": "away_score"})
    assert len(table) == 380 and not table.duplicated(["HomeTeam", "AwayTeam"]).any()
    return table.reset_index(drop=True)


def build_position_probs():
    """P(team finishes in each position 1..20) from the 1,000 simulated final tables."""
    tables = pd.read_csv(SIM_TABLES_CSV, usecols=["sim_id", "team", "final_position"])
    counts = pd.crosstab(tables["team"], tables["final_position"])
    return (counts / tables["sim_id"].nunique()).round(4)


def main():
    matches = pd.read_csv(MATCHES_CSV, parse_dates=["Date"])
    params = json.loads(PARAMS_JSON.read_text())
    goal_defaults = json.loads(GOAL_PARAMS_JSON.read_text())["goal_defaults"]

    state = build_team_state(matches, params, goal_defaults)
    state.to_csv(TEAM_STATE_CSV, index=False)
    print(f"Saved {TEAM_STATE_CSV.relative_to(ROOT)}: {state.shape}")
    print(state[["team", "elo", "gf_10_rel", "ga_10_rel", "ppg_10", "last_match_date"]].round(3).to_string(index=False))

    BOOKMAKER_CSV.parent.mkdir(parents=True, exist_ok=True)
    bookmaker = build_bookmaker_table()
    bookmaker.to_csv(BOOKMAKER_CSV, index=False)
    print(f"\nSaved {BOOKMAKER_CSV.relative_to(ROOT)}: {len(bookmaker)} fixtures")

    results = build_real_results(matches)
    results.to_csv(REAL_RESULTS_CSV, index=False)
    print(f"Saved {REAL_RESULTS_CSV.relative_to(ROOT)}: {len(results)} fixtures")

    positions = build_position_probs()
    positions.to_csv(POSITION_PROBS_CSV)
    print(f"Saved {POSITION_PROBS_CSV.relative_to(ROOT)}: {positions.shape}")


if __name__ == "__main__":
    main()
