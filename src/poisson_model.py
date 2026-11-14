"""Poisson goal model tools (Stage 5d). Shared by src/poisson.py, src/check_poisson.py and, later, Stage 7.

A match gets two expected-goal rates, lambda_home and lambda_away:
    log(lambda) = intercept + sum(weight x standardised feature) [+ offset]     (a Poisson GLM, log link)
The rates give (1) a score prediction, (2) Home/Draw/Away probabilities by summing a grid of scorelines,
(3) the Poisson rates that Stage 7 samples scores from.

How Stage 7 calls it (see also README.md):
    model = PoissonGoalModel.load("models/poisson_config.json")
    lambda_home, lambda_away = model.predict(features_table)          # needs the columns listed in model.required_columns
    grid, mass = score_grid(lambda_home, lambda_away, model.rho)      # grid[i, j] = P(home scores i, away scores j)
    p_home_draw_away = outcome_probs(grid)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
MAX_GOALS = 10   # the scoreline grid covers 0 to 10 goals for each team (an 11 x 11 grid)

# The model sees each team "from its own point of view": own attack, opponent's defence, and so on.
OWN_COLUMNS = ["att", "opp_def", "att_venue", "opp_def_venue", "elo_diff_own", "abs_elo_diff",
               "league_avg", "season_no_fans"]
REL_SOURCE = {"home": {"att": "home_gf_10_rel", "opp_def": "away_ga_10_rel",
                       "att_venue": "home_venue_gf_rel", "opp_def_venue": "away_venue_ga_rel"},
              "away": {"att": "away_gf_10_rel", "opp_def": "home_ga_10_rel",
                       "att_venue": "away_venue_gf_rel", "opp_def_venue": "home_venue_ga_rel"}}
RAW_SOURCE = {"home": {"att": "home_gf_10", "opp_def": "away_ga_10",
                       "att_venue": "home_venue_gf", "opp_def_venue": "away_venue_ga"},
              "away": {"att": "away_gf_10", "opp_def": "home_ga_10",
                       "att_venue": "away_venue_gf", "opp_def_venue": "home_venue_ga"}}


def make_spec(kind="stacked", level="rel", league="feature"):
    """kind: 'independent' (one model per side) or 'stacked' (one model, is_home flag).
    level: 'rel' (level-neutral features) or 'raw' (plain goal averages).
    league: 'feature' (league_avg_goals_prev380 as an input), 'none', or 'offset' (log of the league level, coefficient fixed at 1)."""
    return {"kind": kind, "level": level, "league": league}


def model_columns(spec):
    columns = [c for c in OWN_COLUMNS if not (c == "league_avg" and spec["league"] != "feature")]
    return columns + (["is_home"] if spec["kind"] == "stacked" else [])


def side_frame(table, side, level):
    """The features of one side of every match, in the 'own point of view' layout."""
    source = (REL_SOURCE if level == "rel" else RAW_SOURCE)[side]
    frame = pd.DataFrame({name: table[column].to_numpy() for name, column in source.items()}, index=table.index)
    frame["elo_diff_own"] = (1 if side == "home" else -1) * table["elo_diff"].to_numpy()   # +Elo edge for home, -Elo edge for away
    frame["abs_elo_diff"] = table["abs_elo_diff"].to_numpy()
    frame["league_avg"] = table["league_avg_goals_prev380"].to_numpy()
    frame["season_no_fans"] = table["season_no_fans"].to_numpy()
    return frame


def design(table, spec, with_goals=True):
    """Model matrices. independent -> {'home': ..., 'away': ...}; stacked -> {'stacked': ...}
    (home rows first, then away rows). Each entry: (X, offset, goals or None)."""
    columns = model_columns(spec)
    offset = np.log(table["league_avg_goals_prev380"].to_numpy() / 2) if spec["league"] == "offset" else np.zeros(len(table))
    home, away = side_frame(table, "home", spec["level"]), side_frame(table, "away", spec["level"])
    goals = ((table["home_goals"].to_numpy(), table["away_goals"].to_numpy()) if with_goals else (None, None))
    if spec["kind"] == "independent":
        return {"home": (home[columns], offset, goals[0]), "away": (away[columns], offset, goals[1])}
    home["is_home"], away["is_home"] = 1.0, 0.0
    stacked = pd.concat([home[columns], away[columns]], ignore_index=True)
    y = None if goals[0] is None else np.concatenate(goals)
    return {"stacked": (stacked, np.concatenate([offset, offset]), y)}


def poisson_deviance(y, mu):
    """Poisson deviance of each observation: 2 x (y ln(y/mu) - (y - mu)). Lower is better; it is the
    Poisson analogue of squared error (and of log loss)."""
    y, mu = np.asarray(y, dtype=float), np.asarray(mu, dtype=float)
    ratio = np.where(y > 0, y / mu, 1.0)
    return 2 * (np.where(y > 0, y * np.log(ratio), 0.0) - (y - mu))


class PoissonGoalModel:
    """Poisson GLM(s) with a log link, standardised inputs and an optional Dixon-Coles rho."""

    def __init__(self, spec):
        self.spec = dict(spec)
        self.rho = 0.0
        self.lambda_cap = None   # None = no cap. The saved final model uses 3.5 (a safety limit on expected goals)
        self.parts = {}          # name -> {'mean', 'scale', 'params', 'results'}
        self.columns = model_columns(spec)

    @property
    def required_columns(self):
        """Columns of the goal-features table the model needs."""
        names = (REL_SOURCE if self.spec["level"] == "rel" else RAW_SOURCE)
        return sorted({c for side in names.values() for c in side.values()}
                      | {"elo_diff", "abs_elo_diff", "league_avg_goals_prev380", "season_no_fans"})

    def fit(self, table, weights=None):
        """Fit on `table` (training rows only). The scaler is fitted on these rows only."""
        for name, (X, offset, y) in design(table, self.spec).items():
            scaler = StandardScaler().fit(X)
            varies = scaler.var_ > 0     # a column that never changes in this window (e.g. season_no_fans) gets weight 0
            Z = sm.add_constant(scaler.transform(X)[:, varies], has_constant="add")
            w = None if weights is None else (np.concatenate([weights, weights]) if name == "stacked" else weights)
            result = sm.GLM(y, Z, family=sm.families.Poisson(), offset=offset, freq_weights=w).fit()
            params = np.zeros(1 + X.shape[1])
            params[0] = result.params[0]
            params[1:][varies] = np.asarray(result.params)[1:]
            self.parts[name] = {"mean": scaler.mean_, "scale": scaler.scale_, "params": params, "results": result}
        return self

    def _predict_part(self, name, X, offset):
        part = self.parts[name]
        Z = (X.to_numpy(float) - part["mean"]) / part["scale"]
        return np.exp(part["params"][0] + Z @ part["params"][1:] + offset)

    def predict(self, table):
        """(lambda_home, lambda_away) for every row of `table`, limited to `lambda_cap` if the model has one."""
        lh, la = self.predict_uncapped(table)
        if self.lambda_cap is not None:
            lh, la = np.minimum(lh, self.lambda_cap), np.minimum(la, self.lambda_cap)
        return lh, la

    def predict_uncapped(self, table):
        """The raw model rates, before any cap (used to count how many predictions the cap changes)."""
        n = len(table)
        parts = design(table, self.spec, with_goals=False)
        if self.spec["kind"] == "independent":
            return (self._predict_part("home", *parts["home"][:2]), self._predict_part("away", *parts["away"][:2]))
        both = self._predict_part("stacked", *parts["stacked"][:2])
        return both[:n], both[n:]

    def coefficients(self):
        """Standardised coefficients (log-lambda change per +1 standard deviation), as tables."""
        tables = {}
        for name, part in self.parts.items():
            tables[name] = pd.Series(part["params"], index=["intercept"] + self._part_columns(name))
        return tables

    def _part_columns(self, name):
        return self.columns

    # ---- saving and loading (Stage 7 reads these two files) ----
    def save(self, config_path, models_dir, extra=None):
        """One file per part: models/poisson_home.joblib and poisson_away.joblib (independent) or
        models/poisson_stacked.joblib (stacked), plus a JSON config with everything needed to predict."""
        files = {}
        for name, part in self.parts.items():
            part["results"].remove_data()           # smaller files; predictions use the stored numbers
            path = Path(models_dir) / f"poisson_{name}.joblib"
            joblib.dump(part["results"], path, compress=3)
            files[name] = path.name
        config = {"spec": self.spec, "columns": self.columns, "rho": self.rho, "lambda_cap": self.lambda_cap, "required_columns": self.required_columns,
                  "files": files, "is_home_convention": "stacked model: home rows have is_home = 1, away rows 0; "
                  "the away row uses the NEGATIVE of elo_diff (the Elo edge from that team's own point of view)",
                  "parts": {name: {"mean": part["mean"].tolist(), "scale": part["scale"].tolist(),
                                   "params": part["params"].tolist()} for name, part in self.parts.items()}}
        config.update(extra or {})
        Path(config_path).write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls, config_path, load_results=True):
        config = json.loads(Path(config_path).read_text())
        model = cls(config["spec"])
        model.rho = config["rho"]
        model.lambda_cap = config.get("lambda_cap")
        for name, part in config["parts"].items():
            model.parts[name] = {"mean": np.array(part["mean"]), "scale": np.array(part["scale"]),
                                 "params": np.array(part["params"]), "results": None}
            if load_results:
                model.parts[name]["results"] = joblib.load(Path(config_path).parent / config["files"][name])
        return model


# ------------------------------------------------------------- scoreline grid
def dixon_coles_tau(lh, la, rho):
    """Dixon-Coles factors for the cells (0,0), (0,1), (1,0), (1,1)."""
    return {(0, 0): 1 - lh * la * rho, (0, 1): 1 + lh * rho, (1, 0): 1 + la * rho, (1, 1): 1 - rho}


def score_grid(lh, la, rho=0.0, max_goals=MAX_GOALS):
    """grid[n, i, j] = P(home scores i, away scores j). Independent Poisson product, with the Dixon-Coles
    adjustment of the four lowest cells if rho != 0. Returns (grid normalised to sum 1, raw mass before normalising)."""
    lh, la = np.atleast_1d(np.asarray(lh, float)), np.atleast_1d(np.asarray(la, float))
    k = np.arange(max_goals + 1)
    grid = poisson.pmf(k[None, :, None], lh[:, None, None]) * poisson.pmf(k[None, None, :], la[:, None, None])
    if rho != 0.0:
        for (i, j), factor in dixon_coles_tau(lh, la, rho).items():
            grid[:, i, j] *= factor
    mass = grid.sum(axis=(1, 2))
    return grid / mass[:, None, None], mass


def outcome_probs(grid):
    """Summed Home / Draw / Away probabilities, in the project's fixed class order."""
    n = grid.shape[1]
    i, j = np.indices((n, n))
    return np.stack([(grid * (i > j)).sum(axis=(1, 2)), (grid * (i == j)).sum(axis=(1, 2)),
                     (grid * (i < j)).sum(axis=(1, 2))], axis=1)


def top_scorelines(grid_row, k=5):
    """The k most likely scorelines of one match as (home goals, away goals, probability)."""
    order = np.argsort(grid_row.ravel())[::-1][:k]
    return [(int(o // grid_row.shape[1]), int(o % grid_row.shape[1]), float(grid_row.ravel()[o])) for o in order]


# ------------------------------------------------------------------ Dixon-Coles rho
def rho_bounds(lh, la):
    """The range of rho for which every Dixon-Coles factor stays positive."""
    lo = -1.0 / max(np.max(lh), np.max(la))
    hi = min(1.0, 1.0 / np.max(lh * la))
    return lo + 1e-6, hi - 1e-6


def score_log_likelihood(rho, lh, la, hg, ag):
    """Sum over matches of log P(observed score) under the Dixon-Coles joint."""
    ll = poisson.logpmf(hg, lh) + poisson.logpmf(ag, la)
    for (i, j), factor in dixon_coles_tau(lh, la, rho).items():
        mask = (hg == i) & (ag == j)
        ll = ll + np.where(mask, np.log(factor), 0.0)
    return float(ll.sum())


def fit_rho(lh, la, hg, ag):
    """The rho that maximises the likelihood of the observed scores (given the rates)."""
    lo, hi = rho_bounds(lh, la)
    result = minimize_scalar(lambda r: -score_log_likelihood(r, lh, la, hg, ag), bounds=(lo, hi), method="bounded")
    return float(result.x)
