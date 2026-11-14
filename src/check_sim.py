"""Checks for Stage 7 (season simulation). Run AFTER `python src/simulator.py simulate` and `compare`.

The simulator's own purity is tested without reading any real 2025-26 score: the fixtures come from the whitelist loader,
and the real-result files are only inspected in the last checks (they are outputs of the comparison step).
Each check prints PASS only after its assertions succeeded.
"""
import ast
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import simulator as S
from features import MATCHES_CSV, PARAMS_JSON, compute_elo
from features_goals import REL_COLUMNS, build_goal_features, compute_goal_defaults
from poisson_model import PoissonGoalModel

ROOT, REPORTS = S.ROOT, S.REPORTS
FORBIDDEN = ["FTHG", "FTAG", "FTR", "home_goals", "away_goals", "result"]


def passed(message):
    print(f"PASS  {message}")


schedule = S.build_schedule(S.load_fixtures())
state = S.load_initial_state(schedule)
model = PoissonGoalModel.load(S.POISSON_CONFIG)

# ------------------------------------------------ the loader refuses outcome columns and everything but fixture columns
for column in FORBIDDEN + ["elo_diff", "home_gf_10", "home_elo"]:
    try:
        S.read_features_columns(S.FIXTURE_COLUMNS + [column])
        raise AssertionError(f"the loader accepted {column}")
    except AssertionError as error:
        assert "refused" in str(error), error
fixtures = S.load_fixtures()
assert list(fixtures.columns) == S.FIXTURE_COLUMNS and len(fixtures) == 380 and fixtures["Date"].is_monotonic_increasing
real_files_read = []
original_read_csv = pd.read_csv


def spy_read_csv(path, *args, **kwargs):
    real_files_read.append((str(path), kwargs.get("usecols")))
    return original_read_csv(path, *args, **kwargs)


pd.read_csv = spy_read_csv
try:
    S.run_simulation([0], model, schedule, state)
finally:
    pd.read_csv = original_read_csv
assert real_files_read == [], f"the simulation core read files: {real_files_read}"
tree = ast.parse((ROOT / "src" / "simulator.py").read_text())
functions = {n.name: ast.get_source_segment((ROOT / "src" / "simulator.py").read_text(), n) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
users = lambda token: sorted(name for name, text in functions.items() if token in text and name != "load_real_results_once")
assert [n for n, t in functions.items() if "REAL_RESULT_COLUMNS" in t] == ["load_real_results_once"]
assert users("load_real_results_once()") == ["compare"], users("load_real_results_once()")
assert [n for n, t in functions.items() if "STAGE6_PREDICTIONS" in t] == ["compare"]
assert not any("load_real_results_once" in functions[n] or "STAGE6_PREDICTIONS" in functions[n] for n in ("simulate", "run_simulation", "group_inputs", "load_fixtures"))
S._REAL_READS["count"] = 1
try:
    S.load_real_results_once()
    raise AssertionError("a second read of the real results was allowed")
except AssertionError as error:
    assert "only once" in str(error)
finally:
    S._REAL_READS["count"] = 0
passed("the fixture loader refuses FTHG, FTAG, FTR, home_goals, away_goals, result and every feature column (usecols whitelist: match_id, Date, season, "
       "HomeTeam, AwayTeam); the simulation core reads no file at all; real results are read only by load_real_results_once, called only from compare(), and a second read raises")

# ------------------------------------------------ same seed, same output (and a simulation does not depend on the batch it runs in)
def table_bytes(seeds):
    with tempfile.TemporaryDirectory() as folder:
        result = S.run_simulation(seeds, model, schedule, state)
        S.write_sim_outputs(result, schedule, state, folder)
        return [(Path(folder) / n).read_bytes() for n in ("sim_tables.csv", "sim_scorelines.csv")]


first, second = table_bytes([0]), table_bytes([0])
assert first[0] == second[0] and first[1] == second[1] and len(first[0]) > 500
batch = S.run_simulation([0, 1, 2], model, schedule, state)
alone = S.run_simulation([0], model, schedule, state)
assert np.array_equal(batch["home_goals"][0], alone["home_goals"][0]) and np.array_equal(batch["away_goals"][0], alone["away_goals"][0])
assert not np.array_equal(batch["home_goals"][0], batch["home_goals"][1])
other = S.run_simulation([1], model, schedule, state)
assert np.array_equal(batch["home_goals"][1], other["home_goals"][0])
passed("running seed 0 twice gives byte-identical sim_tables.csv and sim_scorelines.csv; simulation 0 is the same season alone or inside a batch of 3, "
       "and different seeds give different seasons")

# ------------------------------------------------ invariants of a simulated season, against a hand count
sims = S.run_simulation(list(range(5)), model, schedule, state)
tables, scorelines = S.long_tables(schedule, sims), S.scoreline_table(schedule, sims)
teams = schedule["teams"]
fixtures_df = schedule["fixtures"]
for sim_id in range(5):
    block = scorelines[scorelines["sim_id"] == sim_id].set_index("match_id").loc[fixtures_df["match_id"]]
    played = dict.fromkeys(teams, 0)
    wins, draws, losses, gf, ga = ({t: 0 for t in teams} for _ in range(5))
    for (_, fx), (_, sc) in zip(fixtures_df.iterrows(), block.iterrows()):
        h, a, hg, ag = fx["HomeTeam"], fx["AwayTeam"], int(sc["home_goals"]), int(sc["away_goals"])
        played[h] += 1; played[a] += 1
        gf[h] += hg; ga[h] += ag; gf[a] += ag; ga[a] += hg
        if hg > ag:
            wins[h] += 1; losses[a] += 1
        elif hg < ag:
            wins[a] += 1; losses[h] += 1
        else:
            draws[h] += 1; draws[a] += 1
    mine = tables[tables["sim_id"] == sim_id].set_index("team")
    for t in teams:
        assert played[t] == 38 == mine.loc[t, "played"] and mine.loc[t, "wins"] + mine.loc[t, "draws"] + mine.loc[t, "losses"] == 38
        assert mine.loc[t, "points"] == 3 * wins[t] + draws[t] + 0 * losses[t]
        assert (mine.loc[t, "wins"], mine.loc[t, "draws"], mine.loc[t, "losses"]) == (wins[t], draws[t], losses[t])
        assert (mine.loc[t, "GF"], mine.loc[t, "GA"], mine.loc[t, "GD"]) == (gf[t], ga[t], gf[t] - ga[t])
    order = sorted(teams, key=lambda t: (-mine.loc[t, "points"], -mine.loc[t, "GD"], -mine.loc[t, "GF"], t))      # points, GD, GF, alphabetical
    assert list(mine.loc[order, "final_position"]) == list(range(1, 21)), "positions do not follow the tie-break rule"
    assert mine["title_winner"].sum() == 1 and mine["top4"].sum() == 4 and mine["relegated"].sum() == 3
    decisive = int((block["result"] != 1).sum())
    assert mine["points"].sum() == 3 * decisive + 2 * (380 - decisive) and mine["GF"].sum() == mine["GA"].sum()
passed("in 5 simulated seasons every team played exactly 38 matches; points = 3 x wins + draws equals a hand count from the simulated scorelines "
       "(also W/D/L, GF, GA, GD); positions follow points, GD, GF, alphabetical; exactly 1 champion, 4 in the top 4, 3 relegated; total points = 3 x decisive + 2 x draws")

# ------------------------------------------------ no information from the future (perturbation test on the whole pipeline)
class RecordingModel:
    """Wraps the real model and records every table of inputs it is asked about."""
    def __init__(self, inner):
        self.inner, self.tables = inner, []

    def predict(self, table):
        self.tables.append(table.copy())
        return self.inner.predict(table)


def run_with_forced_goals(forced_fixture, forced_goals):
    recorder, calls = RecordingModel(model), {"n": 0}
    original_ppf = S.poisson.ppf

    def ppf(u, mu):
        out = original_ppf(u, mu)
        call = calls["n"]
        calls["n"] += 1
        if forced_fixture is not None and call // 2 == forced_fixture:
            out = np.full_like(out, forced_goals[call % 2])
        return out

    S.poisson.ppf = ppf
    try:
        S.run_simulation([0], recorder, schedule, state)
    finally:
        S.poisson.ppf = original_ppf
    return {f: row for f, row in zip(range(380), pd.concat(recorder.tables, ignore_index=True).itertuples(index=False))}


baseline = run_with_forced_goals(None, None)
for target in (40, 150, 300):                                           # three fixtures at different points of the season
    perturbed = run_with_forced_goals(target, (6, 0))
    date = schedule["dates"][target]
    same, changed = 0, 0
    for f in range(380):
        if schedule["dates"][f] <= date:                                 # on or before the changed match: must be unchanged
            assert baseline[f] == perturbed[f], f"fixture {f} changed although it is not later than fixture {target}"
            same += 1
        else:
            changed += baseline[f] != perturbed[f]
    assert changed > 20, "later fixtures should react to the changed match"
    passed_note = (target, same, changed)
passed(f"changing one simulated score (6-0 forced at fixtures 40, 150 and 300) never changes the model inputs (Elo, form windows, league level) of that or any "
       f"earlier fixture, and does change later ones (last test: {passed_note[1]} fixtures unchanged, {passed_note[2]} later fixtures changed)")

# ------------------------------------------------ the incremental builder reproduces the Stage 4 / 5d reference on a real season
matches = pd.read_csv(MATCHES_CSV, parse_dates=["Date"])
season = matches[matches["season"] == "2024-25"].reset_index(drop=True)          # a training season, treated as a standalone season
params = json.loads(PARAMS_JSON.read_text())
goal_defaults = compute_goal_defaults(matches[matches["split"] == "train"])
reference = build_goal_features(season, params["elo"], params["defaults"], goal_defaults, check_parents=False)
home_elo, away_elo, _, _ = compute_elo(season, {**params["elo"]})
real_schedule = S.build_schedule(season[["match_id", "Date", "season", "HomeTeam", "AwayTeam"]].sort_values(["Date", "match_id"]).reset_index(drop=True))
order = real_schedule["fixtures"]["match_id"].to_numpy()
by_id = season.set_index("match_id").loc[order]
goals = (by_id["home_goals"].to_numpy()[None, :].astype(float), by_id["away_goals"].to_numpy()[None, :].astype(float))
cum = (np.concatenate([[0], np.cumsum(goals[0][0])])[None, :], np.concatenate([[0], np.cumsum(goals[1][0])])[None, :])
elo = np.full((1, 20), 1500.0)                                                     # compute_elo starts every team of a first season at 1500
plans, rows, elo_before = S.make_plans(real_schedule), [], []
real_state = {"goal_defaults": goal_defaults, "league_level": 2.83}
for start, end in real_schedule["groups"]:
    table = S.group_inputs(real_schedule, plans, start, end, goals, cum, elo, real_state)
    rows.append(table)
    for k, f in enumerate(range(start, end)):
        elo_before.append(elo[0, real_schedule["home"][f]] - elo[0, real_schedule["away"][f]])
        hg, ag = goals[0][0, f], goals[1][0, f]
        S.update_elo(elo, real_schedule["home"][f], real_schedule["away"][f], np.array([hg]), np.array([ag]), params["elo"])
mine = pd.concat(rows, ignore_index=True)
# group_inputs returns the rows grouped per date, with one row per fixture of the group (n_sims = 1): fixture order = schedule order
reference = reference.set_axis(season["match_id"]).loc[order].reset_index(drop=True)
for column in REL_COLUMNS:
    assert np.allclose(mine[column].to_numpy(), reference[column].to_numpy(), atol=1e-9), column
ref_elo = pd.Series(home_elo - away_elo, index=season["match_id"]).loc[order].to_numpy()
assert np.allclose(np.array(elo_before), ref_elo, atol=1e-9)
assert S.LEGACY_SWAPPED_LABELS is True
passed("on the real 2024-25 season (standalone, history empty at the start) the simulator's incremental feature builder reproduces the 8 relative goal "
       "features of src/features_goals.py (including its legacy column layout and the neutral defaults) and the Elo ratings of src/features.py, for all 380 matches")

# ------------------------------------------------ Elo bounds and average goals, from the saved 1,000-season run
info = json.loads((REPORTS / "sim_run_info.json").read_text())
tables_all = pd.read_csv(REPORTS / "sim_tables.csv")
scorelines_all = pd.read_csv(REPORTS / "sim_scorelines.csv")
assert info["n_sims"] == 1000 and len(tables_all) == 20_000 and len(scorelines_all) == 380_000
assert S.ELO_BOUNDS[0] < info["elo_min"] and info["elo_max"] < S.ELO_BOUNDS[1] and S.ELO_BOUNDS[0] < tables_all["final_elo"].min() and tables_all["final_elo"].max() < S.ELO_BOUNDS[1]
average_goals = float((scorelines_all["home_goals"] + scorelines_all["away_goals"]).mean())
assert 2.4 < average_goals < 3.4 and abs(average_goals - info["avg_goals_per_match"]) < 1e-9
assert scorelines_all.groupby("sim_id").size().eq(380).all() and sorted(scorelines_all["sim_id"].unique()) == list(range(1000))
assert tables_all.groupby("sim_id")["title_winner"].sum().eq(1).all() and tables_all.groupby("sim_id")["top4"].sum().eq(4).all() and tables_all.groupby("sim_id")["relegated"].sum().eq(3).all()
assert tables_all["played"].eq(38).all() and np.isclose(tables_all.groupby("team")["title_winner"].mean().sum(), 1.0)
passed(f"1,000 simulated seasons (seeds 0 to 999): Elo stayed between {info['elo_min']:.0f} and {info['elo_max']:.0f} (bounds 1000 to 2300); average simulated "
       f"goals per match {average_goals:.3f} over 380,000 matches (allowed 2.4 to 3.4; trained level 2.83); every sim has 1 champion, 4 top-4 and 3 relegated")

# ------------------------------------------------ the initial state file
initial = json.loads((REPORTS / "sim_initial_state.json").read_text())
elo_file = pd.read_csv(S.ELO_START_CSV).set_index("team")
assert set(initial["teams"]) == set(elo_file.index) and len(initial["teams"]) == 20
assert all(abs(initial["teams"][t]["elo_start"] - elo_file.loc[t, "elo_start"]) < 1e-9 for t in initial["teams"])
assert {t for t, v in initial["teams"].items() if v["promoted_or_returning"]} == {"Burnley", "Leeds", "Sunderland"}
assert initial["league_goals_per_match"]["value"] == 2.83 and initial["elo_constants"] == {"K": 30, "H": 50, "keep": 0.9, "offset": -100}
assert not any(word in json.dumps(initial) for word in ("home_goals", "away_goals", "FTHG"))
passed("sim_initial_state.json lists all 20 teams (keep 0.9 Elo from elo_start_of_2025-26.csv, Burnley, Leeds and Sunderland as promoted), the league level 2.83 and "
       "the Elo constants K 30, H 50; it contains no score or result")

# ------------------------------------------------ outputs of the comparison step
merged = pd.read_csv(REPORTS / "sim_vs_real_table.csv")
metrics = json.loads((REPORTS / "sim_headline_metrics.json").read_text())
assert len(merged) == 20 and sorted(merged["real_pos"]) == list(range(1, 21))
assert abs(merged["abs_position_error"].mean() - metrics["mean_abs_position_error"]) < 1e-9
assert abs(merged["points_error"].abs().mean() - metrics["mean_abs_points_error"]) < 1e-9
assert merged["real_pts"].sum() == merged["real_pts"].sum() and abs(merged["sim_mean_pts"].sum() - tables_all["points"].sum() / 1000) < 1e-6
ceiling = pd.read_csv(REPORTS / "sim_ceiling_comparison.csv")
stage6 = pd.read_csv(REPORTS / "stage6_results.csv").set_index("predictor")
assert abs(ceiling.loc[0, "log_loss"] - stage6.loc["Poisson (scoreline grid)", "log_loss"]) < 1e-9          # the ceiling is the Stage 6 number
assert abs(ceiling.loc[0, "accuracy"] - stage6.loc["Poisson (scoreline grid)", "accuracy"]) < 1e-12
assert (S.FIG_DIR / "42_sim_points_fan.png").stat().st_size > 20_000 and (S.FIG_DIR / "44_sim_vs_real_scatter.png").stat().st_size > 20_000
passed("sim_vs_real_table.csv has 20 teams with real positions 1 to 20 and its averages equal the headline metrics; the ceiling row equals the Stage 6 Poisson "
       "log loss and accuracy; the simulated mean points add up to the simulated total; figures 42 to 45 exist")

print("\nAll checks passed.")
