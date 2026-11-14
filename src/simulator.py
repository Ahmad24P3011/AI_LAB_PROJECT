"""Stage 7: season simulation, matchweek by matchweek.

We play the 380 fixtures of 2025-26 in date order, 1,000 times. For every simulated match:
  1. build its features from the SIMULATED history only (Elo and the rolling goal windows);
  2. ask the saved Poisson model for the two expected-goal rates (lambda_home, lambda_away);
  3. draw the two goal counts from independent Poisson distributions;
  4. update Elo (K = 30, H = 50) and the rolling windows with the simulated score.

THE LEAKAGE RULES
  * The simulator never reads a 2025-26 score. The fixture list (id, date, teams) is read through
    read_features_columns(), which refuses any column outside FIXTURE_COLUMNS: the real features.csv rows of
    2025-26 were built from real results, so they are never loaded.
  * Every feature of a match uses only matches played on an EARLIER date in the same simulated season.
  * The real 2025-26 results are read exactly once, in compare(), the comparison step, after the simulation is saved.

HOW IT IS FAST. The schedule is the same in every simulation, so "which earlier matches are in a team's last-10
window" is the same everywhere; only the simulated scores differ. All 1,000 seasons therefore advance in lockstep:
one date at a time, with numpy arrays that hold one value per simulation. Simulation i uses its own random
stream default_rng(i), drawn in advance and turned into Poisson counts by inverse-CDF sampling, so simulation i gives
the same season whether it is run alone or inside the batch of 1,000.

THE LEGACY COLUMN LAYOUT (read this). src/features_goals.py (Stage 5d) stored two of its eight relative goal
features under swapped labels: the column called home_ga_10_rel holds the AWAY team's goals-for, and away_gf_10_rel
holds the HOME team's goals-against. The saved Poisson model was trained and scored with that layout, so to keep the
pre-registered model exactly as it is, build_model_inputs() reproduces it (LEGACY_SWAPPED_LABELS = True). Set it to
False only after features_goals.py is fixed and the Poisson model is refit.

Commands:  python src/simulator.py simulate | compare | all
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import poisson, spearmanr

from features import ELO_SCORE_HOME, PARAMS_JSON, SHRINK_K, VENUE_WINDOW, elo_expected
from features_goals import GOAL_PARAMS_JSON, MIN_LEAGUE_MATCHES, WINDOW
from logreg import BLUE, GREEN, GREY, ORANGE, VERMILION, setup_style
from modeling_utils import CLASS_NAMES, FEATURES_CSV, PROBA_COLUMNS, evaluate, paired_bootstrap, per_match_log_loss
from poisson_model import PoissonGoalModel, poisson_deviance
from preprocess import TEST_SEASONS

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures"
ELO_START_CSV = ROOT / "data" / "processed" / "elo_start_of_2025-26.csv"
POISSON_CONFIG = ROOT / "models" / "poisson_config.json"
STAGE6_PREDICTIONS = REPORTS / "test_predictions_stage6.csv"
SEASON = TEST_SEASONS[0]                         # "2025-26"
N_SIMS = 1000
FIXTURE_COLUMNS = ["match_id", "Date", "season", "HomeTeam", "AwayTeam"]     # the ONLY columns read before the comparison
REAL_RESULT_COLUMNS = ["match_id", "season", "HomeTeam", "AwayTeam", "home_goals", "away_goals", "result"]
LEGACY_SWAPPED_LABELS = True
ELO_BOUNDS = (1000, 2300)                        # sanity bounds for Elo, checked in check_sim.py
SMOOTHING = 0.5                                  # (count + 0.5) / (n_sims + 1.5) for the match-level log loss
TOP_N, RELEGATED_N = 4, 3


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


# ------------------------------------------------------------------ 1. fixtures (the only 2025-26 data used)
def read_features_columns(usecols):
    """Read features.csv for the 2025-26 rows, but only the columns asked for, and only if they are fixture columns."""
    forbidden = set(usecols) - set(FIXTURE_COLUMNS)
    assert not forbidden, f"the simulator may read fixture columns only; refused: {sorted(forbidden)}"
    parts = [c[c["season"] == SEASON] for c in pd.read_csv(FEATURES_CSV, usecols=usecols, parse_dates=["Date"], chunksize=500)]
    return pd.concat(parts, ignore_index=True)


def load_fixtures():
    """The 380 fixtures in date order (ties by match id). Matchweek-like numbering is derived from the list itself."""
    fixtures = read_features_columns(FIXTURE_COLUMNS).sort_values(["Date", "match_id"]).reset_index(drop=True)
    assert len(fixtures) == 380 and set(fixtures["season"]) == {SEASON}
    return fixtures


def build_schedule(fixtures):
    """Everything about the fixture list that does not depend on any score."""
    teams = sorted(set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"]))     # alphabetical = the last tie-break
    index = {t: i for i, t in enumerate(teams)}
    home, away = fixtures["HomeTeam"].map(index).to_numpy(), fixtures["AwayTeam"].map(index).to_numpy()
    dates = fixtures["Date"].to_numpy().astype("datetime64[ns]").astype("int64")
    first_of_date = np.searchsorted(dates, dates, side="left")               # first fixture played on the same date
    games_of = {t: [f for f in range(len(fixtures)) if home[f] == t or away[f] == t] for t in range(len(teams))}
    assert all(len(g) == 38 for g in games_of.values()), "every team must play 38 fixtures"
    game_no = np.zeros((len(fixtures), 2), dtype=int)                        # each side's game number (its own matchweek)
    for t, games in games_of.items():
        for k, f in enumerate(games, start=1):
            game_no[f, 0 if home[f] == t else 1] = k
    groups, start = [], 0
    for f in range(1, len(fixtures) + 1):                                    # fixtures that share a date are simultaneous
        if f == len(fixtures) or dates[f] != dates[start]:
            groups.append((start, f))
            start = f
    return {"fixtures": fixtures, "teams": teams, "index": index, "home": home, "away": away, "dates": dates,
            "first_of_date": first_of_date, "games_of": games_of, "game_no": game_no, "groups": groups}


# ------------------------------------------------------------------ 2. the initial state
def load_initial_state(schedule):
    """One shared starting point: the keep = 0.9 Elo snapshot, neutral form (no history yet) and the league level."""
    elo = pd.read_csv(ELO_START_CSV).set_index("team")
    stage4 = json.loads(PARAMS_JSON.read_text())
    goal_defaults = json.loads(GOAL_PARAMS_JSON.read_text())["goal_defaults"]
    assert set(elo.index) == set(schedule["teams"])
    state = {"elo_start": elo.loc[schedule["teams"], "elo_start"].to_numpy(float),
             "promoted": elo.loc[schedule["teams"], "promoted_or_returning"].to_numpy(bool),
             "elo_params": stage4["elo"], "league_level": float(stage4["defaults"]["league_goals"]),
             "goal_defaults": goal_defaults}
    assert state["elo_params"]["keep"] == 0.9 and state["elo_params"]["K"] == 30 and state["elo_params"]["H"] == 50
    return state


def initial_state_report(schedule, state):
    teams = {}
    for i, team in enumerate(schedule["teams"]):
        teams[team] = {"elo_start": float(state["elo_start"][i]), "promoted_or_returning": bool(state["promoted"][i]),
                       "form_start": "neutral: every relative goal feature is 1.0 until the team has simulated matches",
                       "source": "data/processed/elo_start_of_2025-26.csv (keep 0.9, built from matches before 2025-26 only)"}
    return {"description": "Starting state shared by all 1,000 simulations. No 2025-26 result is part of it.",
            "elo_constants": state["elo_params"],
            "league_goals_per_match": {"value": state["league_level"], "held_constant": True,
                                       "source": "Stage 4 neutral default league_goals = mean total goals per match over the 10 training seasons "
                                                 "(data/processed/feature_params.json). The Stage 5d pre-season table used 2.934 (2024-25); "
                                                 "2.83 was specified for Stage 7."},
            "neutral_goal_defaults": {**state["goal_defaults"], "source": "data/processed/goal_feature_params.json (training rows only); "
                                      "used for a league average over fewer than 20 simulated matches"},
            "features_built": ["elo_diff", "abs_elo_diff", "8 relative goal features (last 10 overall, last 10 at the venue)",
                               "league_avg_goals_prev380 (constant)", "season_no_fans = 0"],
            "features_not_built": "points, shots, head-to-head and matches-played features: the saved Poisson model does not use them",
            "season_no_fans": 0, "n_fixtures": len(schedule["fixtures"]), "teams": teams}


# ------------------------------------------------------------------ 3. features from the simulated history
def league_means(cum, lo, hi, goal_defaults, n_sims):
    """League average goals (per team, home side, away side) over simulated fixtures lo..hi-1 (all strictly before now)."""
    n = hi - lo
    if n < MIN_LEAGUE_MATCHES:                                               # too little league data: the neutral default
        full = lambda v: np.full(n_sims, v)
        return full(goal_defaults["per_team"]), full(goal_defaults["home"]), full(goal_defaults["away"])
    home_avg, away_avg = (cum[0][:, hi] - cum[0][:, lo]) / n, (cum[1][:, hi] - cum[1][:, lo]) / n
    return (home_avg + away_avg) / 2, home_avg, away_avg


def side_plan(schedule, f, team, is_home):
    """Which EARLIER fixtures sit in this team's last-10 window and its last-10 venue window (same for every simulation)."""
    dates, home = schedule["dates"], schedule["home"]
    prior = [g for g in schedule["games_of"][team] if dates[g] < dates[f]]
    overall = prior[-WINDOW:]
    venue = [g for g in prior if (home[g] == team) == is_home][-VENUE_WINDOW:]
    return {"overall": np.array(overall, dtype=int), "overall_was_home": np.array([home[g] == team for g in overall], dtype=bool),
            "venue": np.array(venue, dtype=int)}


def relative_goal_features(plan, is_home, f, schedule, goals, cum, goal_defaults, n_sims):
    """The four relative features of one side: (goals for, goals against, venue goals for, venue goals against),
    each divided by the league average of the same window. Same formulas as Stage 5d (src/features_goals.py)."""
    home_goals, away_goals = goals
    overall = plan["overall"]
    if len(overall) == 0:                                                    # no history yet: neutral
        one = np.ones(n_sims)
        return one, one, one, one
    first, now = schedule["first_of_date"], schedule["first_of_date"][f]
    flag = plan["overall_was_home"][None, :]
    gf10 = np.where(flag, home_goals[:, overall], away_goals[:, overall]).mean(axis=1)
    ga10 = np.where(flag, away_goals[:, overall], home_goals[:, overall]).mean(axis=1)
    per_team = league_means(cum, first[overall[0]], now, goal_defaults, n_sims)[0]
    venue = plan["venue"]
    n = len(venue)
    if n == 0:
        venue_gf, venue_ga, base_gf, base_ga = gf10, ga10, per_team, per_team
    else:
        own, opp = (home_goals, away_goals) if is_home else (away_goals, home_goals)
        v_gf, v_ga = own[:, venue].mean(axis=1), opp[:, venue].mean(axis=1)
        _, league_home, league_away = league_means(cum, first[venue[0]], now, goal_defaults, n_sims)
        own_avg, opp_avg = (league_home, league_away) if is_home else (league_away, league_home)
        venue_gf = (n * v_gf + SHRINK_K * gf10) / (n + SHRINK_K)             # the Stage 4 shrinkage toward the overall mean
        venue_ga = (n * v_ga + SHRINK_K * ga10) / (n + SHRINK_K)
        base_gf = (n * own_avg + SHRINK_K * per_team) / (n + SHRINK_K)
        base_ga = (n * opp_avg + SHRINK_K * per_team) / (n + SHRINK_K)
    return gf10 / per_team, ga10 / per_team, venue_gf / base_gf, venue_ga / base_ga


def build_model_inputs(home_side, away_side, elo_diff, league_level):
    """The table the saved Poisson model expects (column names of features_goals.csv), one row per simulated match."""
    h_gf, h_ga, h_vgf, h_vga = home_side
    a_gf, a_ga, a_vgf, a_vga = away_side
    table = {"home_gf_10_rel": h_gf, "away_ga_10_rel": a_ga, "home_venue_gf_rel": h_vgf, "home_venue_ga_rel": h_vga,
             "away_venue_gf_rel": a_vgf, "away_venue_ga_rel": a_vga,
             "home_ga_10_rel": h_ga, "away_gf_10_rel": a_gf}
    if LEGACY_SWAPPED_LABELS:                                                # reproduce the layout the model was trained on
        table["home_ga_10_rel"], table["away_gf_10_rel"] = a_gf, h_ga
    table.update({"elo_diff": elo_diff, "abs_elo_diff": np.abs(elo_diff),
                  "league_avg_goals_prev380": np.full(len(elo_diff), league_level), "season_no_fans": np.zeros(len(elo_diff))})
    return pd.DataFrame(table)


# ------------------------------------------------------------------ 4. the simulation
def draw_uniforms(seeds, n_fixtures):
    """One random stream per simulation, drawn in advance: simulation i always uses default_rng(i)."""
    return np.stack([np.random.default_rng(int(s)).random((n_fixtures, 2)) for s in seeds])


def update_elo(elo, home, away, home_goals, away_goals, params):
    """Stage 4 rule: K * (actual - expected), the winner gains what the loser loses. Returns nothing; changes `elo`."""
    actual = np.where(home_goals > away_goals, ELO_SCORE_HOME[0], np.where(home_goals == away_goals, ELO_SCORE_HOME[1], ELO_SCORE_HOME[2]))
    expected = elo_expected(elo[:, home], elo[:, away], params["H"])
    change = params["K"] * (actual - expected)
    elo[:, home] += change
    elo[:, away] -= change


def make_plans(schedule):
    """The window plans of every fixture and side. They depend only on the fixture list, never on a score."""
    n_fix = len(schedule["fixtures"])
    return {(f, side): side_plan(schedule, f, schedule["home" if side == 0 else "away"][f], side == 0)
            for f in range(n_fix) for side in (0, 1)}


def group_inputs(schedule, plans, start, end, goals, cum, elo, state):
    """Model inputs for the fixtures start..end-1 (one date), for every simulation, from history BEFORE this date."""
    n_sims = goals[0].shape[0]
    frames = []
    for f in range(start, end):
        h, a = schedule["home"][f], schedule["away"][f]
        sides = [relative_goal_features(plans[(f, side)], side == 0, f, schedule, goals, cum, state["goal_defaults"], n_sims)
                 for side in (0, 1)]
        frames.append(build_model_inputs(sides[0], sides[1], elo[:, h] - elo[:, a], state["league_level"]))
    return pd.concat(frames, ignore_index=True)


def run_simulation(seeds, model, schedule, state):
    """Simulate the whole season once per seed, in lockstep. No 2025-26 score is read anywhere in here."""
    n_sims, n_fix = len(seeds), len(schedule["fixtures"])
    uniforms = np.clip(draw_uniforms(seeds, n_fix), 1e-12, 1 - 1e-12)
    home_goals, away_goals = np.zeros((n_sims, n_fix), dtype=np.int16), np.zeros((n_sims, n_fix), dtype=np.int16)
    lam_home, lam_away = np.zeros((n_sims, n_fix)), np.zeros((n_sims, n_fix))
    cum = (np.zeros((n_sims, n_fix + 1)), np.zeros((n_sims, n_fix + 1)))     # running league goal totals (simulated)
    elo = np.tile(state["elo_start"], (n_sims, 1))
    elo_min, elo_max = elo.min(axis=1).copy(), elo.max(axis=1).copy()
    plans = make_plans(schedule)
    for start, end in schedule["groups"]:                                    # one date at a time, in date order
        inputs = group_inputs(schedule, plans, start, end, (home_goals, away_goals), cum, elo, state)   # 1) features from history BEFORE this date
        lam_h, lam_a = model.predict(inputs)                                 # 2) rates (the cap at 3.5 is inside predict)
        lam_h, lam_a = lam_h.reshape(end - start, n_sims), lam_a.reshape(end - start, n_sims)
        for k, f in enumerate(range(start, end)):                            # 3) goals, then 4) only now update the state
            lam_home[:, f], lam_away[:, f] = lam_h[k], lam_a[k]
            home_goals[:, f] = poisson.ppf(uniforms[:, f, 0], lam_h[k])
            away_goals[:, f] = poisson.ppf(uniforms[:, f, 1], lam_a[k])
            update_elo(elo, schedule["home"][f], schedule["away"][f], home_goals[:, f], away_goals[:, f], state["elo_params"])
            cum[0][:, f + 1] = cum[0][:, f] + home_goals[:, f]
            cum[1][:, f + 1] = cum[1][:, f] + away_goals[:, f]
        elo_min, elo_max = np.minimum(elo_min, elo.min(axis=1)), np.maximum(elo_max, elo.max(axis=1))
    return {"seeds": np.asarray(seeds), "home_goals": home_goals, "away_goals": away_goals, "lam_home": lam_home, "lam_away": lam_away,
            "elo_final": elo, "elo_min": elo_min, "elo_max": elo_max}


# ------------------------------------------------------------------ 5. tables from the simulated scores
def team_incidence(schedule):
    """(fixtures x teams) one-hot matrices: which team is at home / away in each fixture."""
    n_fix, n_teams = len(schedule["fixtures"]), len(schedule["teams"])
    home_matrix, away_matrix = np.zeros((n_fix, n_teams)), np.zeros((n_fix, n_teams))
    home_matrix[np.arange(n_fix), schedule["home"]] = 1
    away_matrix[np.arange(n_fix), schedule["away"]] = 1
    return home_matrix, away_matrix


def rank_teams(points, goal_diff, goals_for):
    """Final positions (1 = champion), ties broken by goal difference, then goals scored, then alphabetical order."""
    n_teams = points.shape[1]
    key = (points.astype(np.int64) * 1_000_000 + (goal_diff.astype(np.int64) + 500) * 1_000 + goals_for.astype(np.int64)) * n_teams
    key = key + (n_teams - 1 - np.arange(n_teams))[None, :]                  # alphabetical: the earlier name ranks higher
    return np.argsort(np.argsort(-key, axis=1), axis=1) + 1


def league_table(schedule, home_goals, away_goals):
    """Points, goals and positions for every simulation (rows) and team (columns)."""
    home_matrix, away_matrix = team_incidence(schedule)
    home_win, away_win, draw = (home_goals > away_goals), (away_goals > home_goals), (home_goals == away_goals)
    wins = home_win @ home_matrix + away_win @ away_matrix
    draws = draw @ home_matrix + draw @ away_matrix
    losses = away_win @ home_matrix + home_win @ away_matrix
    goals_for = home_goals @ home_matrix + away_goals @ away_matrix
    goals_against = away_goals @ home_matrix + home_goals @ away_matrix
    points = 3 * wins + draws
    position = rank_teams(points, goals_for - goals_against, goals_for)
    return {"played": (wins + draws + losses).astype(int), "wins": wins.astype(int), "draws": draws.astype(int), "losses": losses.astype(int),
            "gf": goals_for.astype(int), "ga": goals_against.astype(int), "gd": (goals_for - goals_against).astype(int),
            "points": points.astype(int), "position": position}


def long_tables(schedule, result):
    """reports/sim_tables.csv: one row per (sim_id, team)."""
    table = league_table(schedule, result["home_goals"], result["away_goals"])
    n_sims, n_teams = table["points"].shape
    out = pd.DataFrame({"sim_id": np.repeat(result["seeds"], n_teams), "team": np.tile(schedule["teams"], n_sims)})
    for name in ("played", "wins", "draws", "losses", "gf", "ga", "gd", "points"):
        out[name] = table[name].ravel()
    out = out.rename(columns={"gf": "GF", "ga": "GA", "gd": "GD"})
    out["final_position"] = table["position"].ravel()
    out["title_winner"] = out["final_position"] == 1
    out["top4"] = out["final_position"] <= TOP_N
    out["relegated"] = out["final_position"] > len(schedule["teams"]) - RELEGATED_N
    out["final_elo"] = result["elo_final"].ravel().round(2)
    return out


def scoreline_table(schedule, result):
    """reports/sim_scorelines.csv: one row per (sim_id, match_id)."""
    n_sims, n_fix = result["home_goals"].shape
    hg, ag = result["home_goals"].ravel(), result["away_goals"].ravel()
    return pd.DataFrame({"sim_id": np.repeat(result["seeds"], n_fix), "match_id": np.tile(schedule["fixtures"]["match_id"].to_numpy(), n_sims),
                         "home_goals": hg, "away_goals": ag, "result": np.where(hg > ag, 0, np.where(hg == ag, 1, 2))})


def summary_by_team(tables, state, schedule):
    rows = []
    for team, block in tables.groupby("team", sort=True):
        rows.append({"team": team, "elo_start": float(state["elo_start"][schedule["index"][team]]), "mean_points": block["points"].mean(),
                     "median_points": block["points"].median(), "p5_points": block["points"].quantile(0.05),
                     "p95_points": block["points"].quantile(0.95), "mean_position": block["final_position"].mean(),
                     "P_title": block["title_winner"].mean(), "P_top4": block["top4"].mean(), "P_relegation": block["relegated"].mean(),
                     "expected_GD": block["GD"].mean(), "expected_GF": block["GF"].mean(), "expected_GA": block["GA"].mean()})
    return pd.DataFrame(rows).sort_values(["mean_points", "expected_GD"], ascending=False).reset_index(drop=True)


def points_by_game(schedule, result):
    """Points after each team's k-th game (k = 1..38) in every simulation, summarised: the 'matchweek by matchweek' picture."""
    hg, ag = result["home_goals"], result["away_goals"]
    home_points, away_points = 3 * (hg > ag) + (hg == ag), 3 * (ag > hg) + (hg == ag)
    rows = []
    for t, team in enumerate(schedule["teams"]):
        games = schedule["games_of"][t]
        pts = np.stack([home_points[:, f] if schedule["home"][f] == t else away_points[:, f] for f in games], axis=1).cumsum(axis=1)
        q = np.percentile(pts, [5, 25, 50, 75, 95], axis=0)
        for k in range(38):
            rows.append({"team": team, "game_no": k + 1, "mean": pts[:, k].mean(), "p5": q[0, k], "p25": q[1, k], "p50": q[2, k],
                         "p75": q[3, k], "p95": q[4, k]})
    return pd.DataFrame(rows)


def write_sim_outputs(result, schedule, state, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = long_tables(schedule, result)
    tables.to_csv(out_dir / "sim_tables.csv", index=False)
    scorelines = scoreline_table(schedule, result)
    scorelines.to_csv(out_dir / "sim_scorelines.csv", index=False)
    summary = summary_by_team(tables, state, schedule)
    summary.to_csv(out_dir / "sim_summary_by_team.csv", index=False)
    expected = summary.copy()
    expected.insert(0, "expected_position", np.arange(1, len(expected) + 1))
    expected[["expected_position", "team", "mean_points", "expected_GD", "expected_GF", "expected_GA", "mean_position", "P_title", "P_top4",
              "P_relegation"]].to_csv(out_dir / "sim_expected_table.csv", index=False)
    points_by_game(schedule, result).to_csv(out_dir / "sim_points_by_game.csv", index=False)
    return tables, scorelines, summary, expected


# ------------------------------------------------------------------ 6. figures that need no real result
def plot_points_fan(by_game, summary):
    teams = list(summary["team"].head(6)) + list(summary["team"].tail(3))
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    for ax, team in zip(axes.ravel(), teams):
        block = by_game[by_game["team"] == team]
        ax.fill_between(block["game_no"], block["p5"], block["p95"], color=BLUE, alpha=0.18, label="5th to 95th percentile")
        ax.fill_between(block["game_no"], block["p25"], block["p75"], color=BLUE, alpha=0.30, label="25th to 75th")
        ax.plot(block["game_no"], block["p50"], color=BLUE, linewidth=2, label="median")
        ax.set_title(f"{team} (final median {block['p50'].iloc[-1]:.0f})", fontsize=11)
        ax.grid(axis="x", visible=True)
    for ax in axes[-1]:
        ax.set_xlabel("Games played")
    for ax in axes[:, 0]:
        ax.set_ylabel("Points")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Simulated points after each game, 1,000 seasons: top 6 and bottom 3 by mean points", fontweight="bold")
    save_figure(fig, "42_sim_points_fan.png")


def plot_position_histograms(tables, summary):
    teams = list(summary["team"].head(6))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, team in zip(axes.ravel(), teams):
        positions = tables.loc[tables["team"] == team, "final_position"]
        share = positions.value_counts(normalize=True).reindex(range(1, 21), fill_value=0) * 100
        ax.bar(share.index, share.values, color=[VERMILION if p == 1 else (ORANGE if p <= 4 else BLUE) for p in share.index])
        ax.set_title(f"{team}: mean position {positions.mean():.1f}", fontsize=11)
        ax.set_xticks([1, 4, 8, 12, 16, 20])
        ax.grid(axis="x", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("Final position (red = champion, orange = top 4)")
    for ax in axes[:, 0]:
        ax.set_ylabel("% of the 1,000 seasons")
    fig.suptitle("Where the six strongest simulated teams finish", fontweight="bold")
    save_figure(fig, "43_sim_position_histograms.png")


def plot_position_heatmap(tables, summary):
    order = list(summary.sort_values("mean_position")["team"])
    grid = pd.crosstab(tables["team"], tables["final_position"], normalize="index").reindex(order) * 100
    fig, ax = plt.subplots(figsize=(13, 8.5))
    annotations = grid.map(lambda v: f"{v:.0f}" if v >= 3 else "")
    sns.heatmap(grid, annot=annotations, fmt="", cmap="Blues", cbar_kws={"label": "% of the 1,000 seasons"}, linewidths=0.3, ax=ax,
                annot_kws={"size": 8})
    ax.set_xlabel("Final position")
    ax.set_ylabel("")
    ax.set_title("How often each team finishes in each position (numbers shown where 3% or more)")
    save_figure(fig, "45_sim_position_heatmap.png")


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print(f"[saved reports/figures/{name}]")


# ------------------------------------------------------------------ the commands
def simulate(n_sims=N_SIMS, out_dir=REPORTS):
    """Run and save the simulation. Reads fixtures only; no real result."""
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    section("Fixtures (the only 2025-26 data the simulator reads) and the initial state")
    schedule = build_schedule(load_fixtures())
    state = load_initial_state(schedule)
    fixtures = schedule["fixtures"]
    print(f"{len(fixtures)} fixtures, {fixtures['Date'].min().date()} to {fixtures['Date'].max().date()}, {len(schedule['teams'])} teams, "
          f"{len(schedule['groups'])} distinct dates; columns read: {FIXTURE_COLUMNS}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "sim_initial_state.json").write_text(json.dumps(initial_state_report(schedule, state), indent=1))
    print(f"Saved sim_initial_state.json; start Elo from {min(state['elo_start']):.1f} to {max(state['elo_start']):.1f}; league level {state['league_level']}")
    model = PoissonGoalModel.load(POISSON_CONFIG)
    assert model.lambda_cap == 3.5 and model.rho == 0.0 and model.spec["league"] == "offset"
    section(f"Simulating {n_sims} seasons (seeds 0 to {n_sims - 1})")
    result = run_simulation(list(range(n_sims)), model, schedule, state)
    total_goals = result["home_goals"].astype(int) + result["away_goals"]
    print(f"Average simulated goals per match {total_goals.mean():.3f} (home {result['home_goals'].mean():.3f}, away {result['away_goals'].mean():.3f}); "
          f"Elo range over all sims and times {result['elo_min'].min():.0f} to {result['elo_max'].max():.0f}; "
          f"largest rate used {max(result['lam_home'].max(), result['lam_away'].max()):.3f}")
    tables, scorelines, summary, expected = write_sim_outputs(result, schedule, state, out_dir)
    (Path(out_dir) / "sim_run_info.json").write_text(json.dumps({
        "n_sims": n_sims, "seeds": f"0 to {n_sims - 1}", "avg_goals_per_match": float(total_goals.mean()),
        "elo_min": float(result["elo_min"].min()), "elo_max": float(result["elo_max"].max()),
        "max_lambda_used": float(max(result["lam_home"].max(), result["lam_away"].max())), "lambda_cap": model.lambda_cap,
        "legacy_swapped_labels": LEGACY_SWAPPED_LABELS, "league_level": state["league_level"], "n_fixtures": len(fixtures)}, indent=1))
    for name in ("sim_tables.csv", "sim_scorelines.csv"):
        print(f"Saved {name}: {(Path(out_dir) / name).stat().st_size / 1024 / 1024:.1f} MB")
    section("Simulated expected table (ordered by mean points)")
    view = expected[["expected_position", "team", "mean_points", "p5_points", "p95_points", "mean_position", "P_title", "P_top4", "P_relegation",
                     "expected_GD"]]
    print(view.round(3).to_string(index=False))
    plot_points_fan(pd.read_csv(Path(out_dir) / "sim_points_by_game.csv"), summary)
    plot_position_histograms(tables, summary)
    plot_position_heatmap(tables, summary)
    return result


_REAL_READS = {"count": 0}


def load_real_results_once():
    """THE COMPARISON STEP: the only function that reads real 2025-26 scores. It runs once, after the simulation is saved."""
    assert _REAL_READS["count"] == 0, "the real 2025-26 results may be read only once"
    _REAL_READS["count"] += 1
    parts = [c[c["season"] == SEASON] for c in pd.read_csv(FEATURES_CSV, usecols=REAL_RESULT_COLUMNS, chunksize=500)]
    return pd.concat(parts, ignore_index=True)


def real_table(real, teams):
    """The real final table, with the same tie-breaks as the simulated ones."""
    home_idx, away_idx = real["HomeTeam"].map({t: i for i, t in enumerate(teams)}).to_numpy(), real["AwayTeam"].map({t: i for i, t in enumerate(teams)}).to_numpy()
    hg, ag = real["home_goals"].to_numpy(), real["away_goals"].to_numpy()
    table = league_table({"fixtures": real, "teams": teams, "home": home_idx, "away": away_idx}, hg[None, :], ag[None, :])
    return pd.DataFrame({"team": teams, "real_pos": table["position"][0], "real_pts": table["points"][0], "real_GD": table["gd"][0],
                         "real_GF": table["gf"][0]}).sort_values("real_pos").reset_index(drop=True)


def compare(out_dir=REPORTS):
    """Compare with the real 2025-26 results (read once) and with the one-match-ahead Stage 6 numbers."""
    setup_style()
    out_dir = Path(out_dir)
    section("THE COMPARISON STEP: reading the real 2025-26 results (once)")
    real = load_real_results_once()
    summary = pd.read_csv(out_dir / "sim_summary_by_team.csv")
    expected = pd.read_csv(out_dir / "sim_expected_table.csv")
    teams = sorted(summary["team"])
    actual = real_table(real, teams)
    merged = actual.merge(summary[["team", "mean_position", "mean_points", "P_title", "P_top4", "P_relegation", "expected_GD"]], on="team")
    merged = merged.rename(columns={"mean_position": "sim_mean_pos", "mean_points": "sim_mean_pts", "P_title": "sim_P_title",
                                    "P_top4": "sim_P_top4", "P_relegation": "sim_P_relegation", "expected_GD": "sim_expected_GD"})
    merged["position_error"] = merged["sim_mean_pos"] - merged["real_pos"]          # positive: the simulator put the team too low
    merged["abs_position_error"] = merged["position_error"].abs()
    merged["points_error"] = merged["sim_mean_pts"] - merged["real_pts"]
    merged = merged[["team", "real_pos", "real_pts", "real_GD", "sim_mean_pos", "sim_mean_pts", "sim_expected_GD", "sim_P_title", "sim_P_top4",
                     "sim_P_relegation", "position_error", "abs_position_error", "points_error"]].sort_values("real_pos")
    merged.to_csv(out_dir / "sim_vs_real_table.csv", index=False)

    sim_top4, sim_bottom3 = list(expected["team"].head(TOP_N)), list(expected["team"].tail(RELEGATED_N))
    real_top4, real_bottom3 = list(actual["team"].head(TOP_N)), list(actual["team"].tail(RELEGATED_N))
    champion = actual["team"].iloc[0]
    metrics = {"mean_abs_position_error": float(merged["abs_position_error"].mean()),
               "spearman_rank_correlation": float(spearmanr(merged["sim_mean_pos"], merged["real_pos"])[0]),
               "mean_abs_points_error": float(merged["points_error"].abs().mean()),
               "mean_signed_points_error": float(merged["points_error"].mean()),
               "real_top4": real_top4, "sim_top4": sim_top4, "n_real_top4_captured": len(set(sim_top4) & set(real_top4)),
               "real_bottom3": real_bottom3, "sim_bottom3": sim_bottom3, "n_real_relegated_captured": len(set(sim_bottom3) & set(real_bottom3)),
               "real_champion": champion, "real_champion_pts": int(actual["real_pts"].iloc[0]),
               "sim_P_title_of_real_champion": float(merged.loc[merged["team"] == champion, "sim_P_title"].iloc[0]),
               "sim_mean_pts_of_real_champion": float(merged.loc[merged["team"] == champion, "sim_mean_pts"].iloc[0]),
               "sim_favourite": expected["team"].iloc[0], "sim_P_title_of_favourite": float(expected["P_title"].iloc[0])}
    section("Simulated vs real final table")
    print(merged.round(2).to_string(index=False))
    for key, value in metrics.items():
        print(f"  {key}: {round(value, 4) if isinstance(value, float) else value}")

    section("Ceiling: one-match-ahead with real history (Stage 6) vs the simulation, same 380 fixtures")
    ceiling = pd.read_csv(STAGE6_PREDICTIONS)
    scorelines = pd.read_csv(out_dir / "sim_scorelines.csv", usecols=["sim_id", "match_id", "home_goals", "away_goals", "result"])
    fixtures = ceiling[["match_id", "result", "home_goals", "away_goals"]]
    assert set(fixtures["match_id"]) == set(scorelines["match_id"]) and (fixtures.sort_values("match_id")["result"].to_numpy()
                                                                        == real.sort_values("match_id")["result"].to_numpy()).all()
    n_sims = scorelines["sim_id"].nunique()
    counts = scorelines.groupby(["match_id", "result"]).size().unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0).loc[fixtures["match_id"]]
    raw_share = counts.to_numpy() / n_sims
    smoothed = (counts.to_numpy() + SMOOTHING) / (n_sims + 3 * SMOOTHING)
    y = fixtures["result"].to_numpy()
    one_step = ceiling[[f"poisson_{c}" for c in PROBA_COLUMNS]].to_numpy()
    sim_total = scorelines.assign(total=scorelines["home_goals"] + scorelines["away_goals"]).groupby("match_id")["total"].mean().loc[fixtures["match_id"]].to_numpy()
    step_total = (ceiling["poisson_lambda_home"] + ceiling["poisson_lambda_away"]).to_numpy()
    real_total = (fixtures["home_goals"] + fixtures["away_goals"]).to_numpy()
    rows = []
    for name, proba, mu in (("One-match-ahead Poisson (real history, Stage 6)", one_step, step_total),
                            ("Simulation, pre-season only (smoothed shares of 1,000 seasons)", smoothed, sim_total)):
        m = evaluate(y, proba=proba)
        f1 = m["per_class"]["f1"]
        rows.append({"predictor": name, "log_loss": m["log_loss"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "home_f1": f1["Home win"],
                     "draw_f1": f1["Draw"], "away_f1": f1["Away win"], "draws_predicted": int(m["predicted_counts"][1]),
                     "total_goals_deviance": float(poisson_deviance(real_total, mu).mean()), "total_goals_mae": float(np.abs(mu - real_total).mean())})
    raw_m = evaluate(y, proba=np.clip(raw_share, 1e-15, 1))
    gap = paired_bootstrap(per_match_log_loss(y, smoothed), per_match_log_loss(y, one_step))     # positive: one-step is better
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "sim_ceiling_comparison.csv", index=False)
    print(table.round(4).to_string(index=False))
    metrics["ceiling_log_loss"], metrics["sim_log_loss_smoothed"] = rows[0]["log_loss"], rows[1]["log_loss"]
    metrics["log_loss_gap_sim_minus_ceiling"] = rows[1]["log_loss"] - rows[0]["log_loss"]
    metrics["log_loss_gap_interval"] = [gap["ci_low"], gap["ci_high"]]
    metrics["sim_log_loss_unsmoothed_clipped_1e-15"] = raw_m["log_loss"]
    metrics["n_matches_where_sim_gave_real_outcome_zero_share"] = int((raw_share[np.arange(len(y)), y] == 0).sum())
    print(f"Gap (simulation minus one-step log loss): {metrics['log_loss_gap_sim_minus_ceiling']:+.4f}, 95% interval "
          f"[{gap['ci_low']:+.4f}, {gap['ci_high']:+.4f}] (positive = the one-step model is better)")
    print(f"Matches where the real outcome had a zero share in the 1,000 sims: {metrics['n_matches_where_sim_gave_real_outcome_zero_share']}")
    (out_dir / "sim_headline_metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    plot_sim_vs_real(merged)
    return merged, metrics


def plot_sim_vs_real(merged):
    fig, ax = plt.subplots(figsize=(8.8, 7.4))
    ax.plot([15, 90], [15, 90], "--", color=GREY, label="Simulated = real")
    ax.scatter(merged["sim_mean_pts"], merged["real_pts"], s=55, color=BLUE, zorder=3)
    placed = []
    for _, r in merged.iterrows():                                               # nudge labels that would overlap
        dx, dy = 0.6, 0.6
        while any(abs(r["sim_mean_pts"] + dx - px) < 5.5 and abs(r["real_pts"] + dy - py) < 1.7 for px, py in placed):
            dy -= 1.8
        placed.append((r["sim_mean_pts"] + dx, r["real_pts"] + dy))
        ax.annotate(r["team"], (r["sim_mean_pts"], r["real_pts"]), xytext=(r["sim_mean_pts"] + dx, r["real_pts"] + dy), fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": "#999999", "linewidth": 0.6})
    ax.set_xlabel("Simulated mean points (1,000 seasons, pre-season start)")
    ax.set_ylabel("Real points, 2025-26")
    ax.set_title("Simulated vs real final points")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=True)
    save_figure(fig, "44_sim_vs_real_scatter.png")


def main(command="all"):
    if command == "replot":          # redraw figure 44 from the SAVED comparison table (no real result is read again)
        setup_style()
        plot_sim_vs_real(pd.read_csv(REPORTS / "sim_vs_real_table.csv"))
    if command in ("simulate", "all"):
        simulate()
    if command in ("compare", "all"):
        compare()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
