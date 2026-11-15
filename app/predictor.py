"""Prediction layer of the app: loads the SAVED models and the saved team state, and answers "home team v away team".

No training and no re-simulation happens here, and no Streamlit code either (so src/check_app.py can test it).
Everything is read from files that earlier stages saved:
  models/logreg.joblib, random_forest.joblib, xgboost.joblib   the three ELO2 classifiers (input: Elo difference)
  models/poisson_config.json                                   the Poisson goal model (numbers only, no pickle needed)
  data/processed/team_state_end_of_2025-26.csv                 each team's Elo and last-10 goal form (app/team_state_builder.py)
  data/app/bookmaker_2025-26.csv                               bookmaker probabilities with the margin removed
  data/app/real_results_2025-26.csv                            the real score of each 2025-26 fixture (display only)
"""
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from poisson_model import PoissonGoalModel, outcome_probs, score_grid, top_scorelines   # noqa: E402

MODELS_DIR = ROOT / "models"
TEAM_STATE_CSV = ROOT / "data" / "processed" / "team_state_end_of_2025-26.csv"
BOOKMAKER_CSV = ROOT / "data" / "app" / "bookmaker_2025-26.csv"
REAL_RESULTS_CSV = ROOT / "data" / "app" / "real_results_2025-26.csv"
ELO2 = ["elo_diff", "abs_elo_diff"]
STATE_COLUMNS = ["team", "elo", "gf_10_rel", "ga_10_rel", "home_venue_gf_rel", "home_venue_ga_rel",
                 "away_venue_gf_rel", "away_venue_ga_rel", "league_goals_per_match", "season_no_fans"]
CLASSIFIERS = {"logreg": "Logistic Regression (ELO2, primary)", "random_forest": "Random Forest (ELO2)",
               "xgboost": "XGBoost (ELO2)"}


class AppDataError(Exception):
    """A saved file is missing or does not cover all the teams (the app shows a friendly message)."""


def poisson_inputs(home, away):
    """One-row table in the column layout the saved Poisson model was trained and scored on.

    LEGACY LAYOUT (see docs/decisions.md, Stage 7): src/features_goals.py stored two of the eight relative goal columns
    under swapped labels (home_ga_10_rel holds the AWAY team's goals-for; away_gf_10_rel holds the HOME team's
    goals-against). The saved model learned from that layout, so the app reproduces it. src/check_app.py proves this
    function gives the same numbers as the training-time feature code for all 380 team pairs."""
    elo_diff = home["elo"] - away["elo"]
    return pd.DataFrame([{
        "home_gf_10_rel": home["gf_10_rel"], "away_ga_10_rel": away["ga_10_rel"],
        "home_venue_gf_rel": home["home_venue_gf_rel"], "home_venue_ga_rel": home["home_venue_ga_rel"],
        "away_venue_gf_rel": away["away_venue_gf_rel"], "away_venue_ga_rel": away["away_venue_ga_rel"],
        "home_ga_10_rel": away["gf_10_rel"], "away_gf_10_rel": home["ga_10_rel"],          # the legacy swap
        "elo_diff": elo_diff, "abs_elo_diff": abs(elo_diff),
        "league_avg_goals_prev380": home["league_goals_per_match"], "season_no_fans": home["season_no_fans"]}])


class Predictor:
    def __init__(self):
        for path in [TEAM_STATE_CSV, BOOKMAKER_CSV, REAL_RESULTS_CSV, MODELS_DIR / "poisson_config.json"] + [MODELS_DIR / f"{m}.joblib" for m in CLASSIFIERS]:
            if not path.exists():
                raise AppDataError(f"app data is out of date: {path.name} is missing")
        state = pd.read_csv(TEAM_STATE_CSV)
        if list(state.columns[:1]) != ["team"] or not set(STATE_COLUMNS) <= set(state.columns) or len(state) != 20:
            raise AppDataError("app data is out of date: the team state file does not have the expected 20 teams and columns")
        self.state = state.set_index("team")
        self.teams = sorted(self.state.index)
        self.bookmaker = pd.read_csv(BOOKMAKER_CSV).set_index(["HomeTeam", "AwayTeam"])
        self.real_results = pd.read_csv(REAL_RESULTS_CSV).set_index(["HomeTeam", "AwayTeam"])     # shown for comparison; never a model input
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                    # sklearn version notes when unpickling; versions are pinned in app/requirements.txt
            self.classifiers = {m: joblib.load(MODELS_DIR / f"{m}.joblib") for m in CLASSIFIERS}
        self.poisson = PoissonGoalModel.load(MODELS_DIR / "poisson_config.json", load_results=False)

    def predict(self, home_team, away_team):
        """Everything the page shows for one match, as plain numbers."""
        if home_team == away_team:
            raise ValueError("a team cannot play itself")
        home, away = self.state.loc[home_team], self.state.loc[away_team]
        elo_diff = float(home["elo"] - away["elo"])
        row = pd.DataFrame([{"elo_diff": elo_diff, "abs_elo_diff": abs(elo_diff)}])[ELO2]
        probabilities = {m: self.classifiers[m].predict_proba(row)[0] for m in CLASSIFIERS}     # order: Home, Draw, Away

        inputs = poisson_inputs(home, away)
        lam_home, lam_away = (float(v[0]) for v in self.poisson.predict(inputs))                 # includes the 3.5 cap
        grid, _ = score_grid(lam_home, lam_away, self.poisson.rho)
        probabilities["poisson"] = outcome_probs(grid)[0]

        bookmaker = None
        if (home_team, away_team) in self.bookmaker.index:
            bookmaker = self.bookmaker.loc[(home_team, away_team), ["p_home", "p_draw", "p_away"]].to_numpy(float)
        return {"home": home_team, "away": away_team, "elo_home": float(home["elo"]), "elo_away": float(away["elo"]),
                "elo_diff": elo_diff, "probabilities": probabilities, "bookmaker": bookmaker,
                "lambda_home": lam_home, "lambda_away": lam_away, "grid": grid[0],
                "top_scorelines": top_scorelines(grid[0], k=10)}

    def real_result(self, home_team, away_team):
        """The real 2025-26 fixture of this pairing (date, score, outcome), or None. Display only: no model reads it."""
        if (home_team, away_team) not in self.real_results.index:
            return None
        row = self.real_results.loc[(home_team, away_team)]
        home, away = int(row["home_score"]), int(row["away_score"])
        return {"date": row["Date"], "home_score": home, "away_score": away, "outcome": 0 if home > away else 1 if home == away else 2}
