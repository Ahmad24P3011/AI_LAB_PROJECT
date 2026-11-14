"""Stage 5d: Poisson regression for goals (validation seasons only).

All choices use the VALIDATION seasons (2023-24, 2024-25) with models fitted on 2015-16 to 2022-23.
The 2025-26 test season is never used for fitting or scoring (the Stage 5 guards refuse test rows). The
single place where the 2025-26 file is opened is load_fixtures_2025_26(): it reads the FIXTURE LIST only
(date and the two team names, never a score), because Stage 7 needs the fixtures.

Run:  python src/features_goals.py ; python src/poisson.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import MATCHES_CSV, PARAMS_JSON, elo_at_season_start
from features_goals import GOAL_FEATURES_CSV
from logreg import (BLUE, FIG_DIR, GREEN, GREY, ORANGE, VERMILION, calibration_points, decide,
                    print_metrics_block, save_figure, section, setup_style)
from modeling_utils import (CLASS_NAMES, FIT_SEASONS, PROBA_COLUMNS, SEED, TRAIN_SEASONS, VAL_SEASONS,
                            assert_no_test_rows, evaluate, get_slice, paired_bootstrap, per_match_log_loss,
                            recency_weights, window_seasons)
from poisson_model import (MAX_GOALS, PoissonGoalModel, design, dixon_coles_tau, fit_rho, make_spec, outcome_probs,
                           poisson_deviance, rho_bounds, score_grid, score_log_likelihood, top_scorelines)

ROOT = Path(__file__).resolve().parent.parent
FIT_END, FINAL_END = FIT_SEASONS[-1], TRAIN_SEASONS[-1]
ABLATIONS_CSV = ROOT / "reports" / "ablations_poisson.csv"
VAL_LAMBDAS_CSV = ROOT / "reports" / "val_lambdas_5d.csv"
VAL_LAMBDAS_FINAL_CSV = ROOT / "reports" / "val_lambdas_5d_final.csv"
FINAL_CSV = ROOT / "reports" / "poisson_final_5d.csv"
CONFIG_PATH = ROOT / "models" / "poisson_config.json"
PRE_SEASON_CSV = ROOT / "data" / "processed" / "poisson_pre_season_2025-26.csv"
PRE_SEASON_075_CSV = ROOT / "data" / "processed" / "poisson_pre_season_2025-26_keep075.csv"
ELO_START_CSV = ROOT / "data" / "processed" / "elo_start_of_2025-26.csv"
LAMBDA_CAP = 3.5          # safety limit on expected goals per team (decided after the Stage 5d review)
RAW_COLUMNS = ["home_gf_10", "home_ga_10", "away_gf_10", "away_ga_10",
               "home_venue_gf", "home_venue_ga", "away_venue_gf", "away_venue_ga"]
ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0]


# ---------------------------------------------------------------- data loading
def load_goal_rows():
    """TRAINING rows of features_goals.csv (plus the raw goal averages from features.csv for experiment a).
    Test rows are dropped chunk by chunk while reading, so no table in memory holds them."""
    goals = pd.concat([c[c["split"] == "train"] for c in pd.read_csv(GOAL_FEATURES_CSV, parse_dates=["Date"], chunksize=500)],
                      ignore_index=True)
    ids = set(goals["match_id"])
    raw = pd.concat([c[c["match_id"].isin(ids)] for c in pd.read_csv(ROOT / "data" / "processed" / "features.csv",
                                                                       usecols=["match_id"] + RAW_COLUMNS, chunksize=500)],
                    ignore_index=True)
    assert (raw["match_id"].to_numpy() == goals["match_id"].to_numpy()).all()
    rows = pd.concat([goals, raw[RAW_COLUMNS]], axis=1)
    assert_no_test_rows(rows)
    return rows


def load_fixtures_2025_26():
    """The 2025-26 FIXTURE LIST only: match id, date and the two team names. No score or result column is
    read (usecols), so no outcome of the test season is ever loaded here."""
    columns = ["match_id", "season", "Date", "HomeTeam", "AwayTeam"]
    fixtures = pd.concat([c[c["season"] == "2025-26"] for c in
                          pd.read_csv(MATCHES_CSV, usecols=columns, parse_dates=["Date"], chunksize=500)], ignore_index=True)
    assert list(fixtures.columns) == columns
    return fixtures.rename(columns={"match_id": "fixture_id"}).drop(columns="season")


# ------------------------------------------------------- fitting and evaluation
def make_cfg(spec, window=None, recency=False, rho_mode="zero"):
    return {"spec": dict(spec), "window": window, "recency": recency, "rho_mode": rho_mode}


def fit_cfg(cfg, source_rows, end_season):
    assert_no_test_rows(source_rows)     # guard the INPUT first: a test row must never be silently filtered away
    rows = source_rows[source_rows["season"].isin(window_seasons(cfg["window"], end_season))].reset_index(drop=True)
    assert_no_test_rows(rows)
    weights = recency_weights(rows, end_season) if cfg["recency"] else None
    model = PoissonGoalModel(cfg["spec"]).fit(rows, weights)
    if cfg["rho_mode"] == "fit":      # rho is fitted on the SAME training rows only
        lh, la = model.predict(rows)
        model.rho = fit_rho(lh, la, rows["home_goals"].to_numpy(), rows["away_goals"].to_numpy())
    return model, rows


def evaluate_model(model, table):
    """Goal metrics and match-outcome metrics of a model (with its rho) on a table of matches."""
    assert_no_test_rows(table)
    lh, la = model.predict(table)
    hg, ag = table["home_goals"].to_numpy(), table["away_goals"].to_numpy()
    dev_home, dev_away = poisson_deviance(hg, lh), poisson_deviance(ag, la)
    grid, mass = score_grid(lh, la, model.rho)
    proba = outcome_probs(grid)
    y = table["result"].to_numpy()
    return {"lh": lh, "la": la, "grid": grid, "mass": mass, "proba": proba, "rho": model.rho,
            "dev_match": dev_home + dev_away, "dev_mean": float((dev_home + dev_away).mean() / 2),
            "mae_home": float(np.abs(lh - hg).mean()), "mae_away": float(np.abs(la - ag).mean()),
            "mae_total": float(np.abs(lh + la - (hg + ag)).mean()),
            "losses": per_match_log_loss(y, proba), "metrics": evaluate(y, proba)}


def score_cfg(cfg, fit_rows, val_rows):
    model, rows = fit_cfg(cfg, fit_rows, FIT_END)
    outcome = evaluate_model(model, val_rows)
    lh, la = model.predict(rows)
    outcome.update({"model": model, "rows": rows, "train_dev": float((poisson_deviance(rows["home_goals"], lh)
                                                                      + poisson_deviance(rows["away_goals"], la)).mean() / 2)})
    return outcome


class TeamFixedEffectsModel:
    """Experiment e: the stacked model plus one weight per attacking team and per defending team,
    with an L2 penalty (scikit-learn's PoissonRegressor)."""

    def __init__(self, spec, alpha, with_teams=True):
        self.spec, self.alpha, self.with_teams, self.rho = dict(spec), alpha, with_teams, 0.0

    def _matrix(self, table, fit=False):
        X, _, y = design(table, self.spec)["stacked"]
        attackers = np.concatenate([table["HomeTeam"], table["AwayTeam"]]).reshape(-1, 1)
        defenders = np.concatenate([table["AwayTeam"], table["HomeTeam"]]).reshape(-1, 1)
        if fit:
            self.scaler = StandardScaler().fit(X)
            self.att_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(attackers)
            self.def_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(defenders)
        parts = [self.scaler.transform(X)]
        if self.with_teams:
            parts += [self.att_encoder.transform(attackers), self.def_encoder.transform(defenders)]
        return np.hstack(parts), y

    def fit(self, table):
        X, y = self._matrix(table, fit=True)
        self.model = PoissonRegressor(alpha=self.alpha, max_iter=5000).fit(X, y)
        return self

    def predict(self, table):
        X, _ = self._matrix(table)
        mu = self.model.predict(X)
        return mu[:len(table)], mu[len(table):]


# ------------------------------------------------------------------ reporting
def model_row(ev):
    m = ev["metrics"]
    return {"MAE home": ev["mae_home"], "MAE away": ev["mae_away"], "MAE total": ev["mae_total"],
            "deviance": ev["dev_mean"], "log loss": m["log_loss"], "accuracy": m["accuracy"],
            "macro-F1": m["macro_f1"], "draws predicted": int(m["predicted_counts"][1])}


def reference_row(metrics):
    return {"MAE home": np.nan, "MAE away": np.nan, "MAE total": np.nan, "deviance": np.nan, "log loss": metrics["log_loss"],
            "accuracy": metrics["accuracy"], "macro-F1": metrics["macro_f1"], "draws predicted": int(metrics["predicted_counts"][1])}


def _poisson_pmf0(lam):
    """P(0 goals) = exp(-lambda)."""
    return np.exp(-np.asarray(lam, float))


def pair_table(losses):
    """Paired bootstrap of the match-outcome log loss for every pair: improvement of the ROW over the COLUMN."""
    names = list(losses)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            s = paired_bootstrap(losses[b], losses[a])          # positive = a is better than b
            rows.append({"a (row)": a, "b (column)": b, "a better than b by": s["mean_diff"], "ci_low": s["ci_low"],
                         "ci_high": s["ci_high"], "verdict": "a clearly better" if s["ci_low"] > 0 else
                         ("b clearly better" if s["ci_high"] < 0 else "cannot tell")})
    return pd.DataFrame(rows)


def premise_table(lh, la, hg, ag, label):
    """Dixon-Coles premise: observed vs expected number of the four lowest scorelines under independence."""
    grid, _ = score_grid(lh, la, 0.0)
    rows = []
    for i, j in ((0, 0), (1, 0), (0, 1), (1, 1)):
        observed, expected = int(((hg == i) & (ag == j)).sum()), float(grid[:, i, j].sum())
        rows.append({"score": f"{i}-{j}", "observed": observed, "expected (independent Poisson)": round(expected, 1),
                     "observed / expected": round(observed / expected, 3)})
    print(f"\n{label}:")
    print(pd.DataFrame(rows).to_string(index=False))


def score_stats(ev, table):
    """Task 5: score predictions on a table of matches."""
    grid, hg, ag, y = ev["grid"], table["home_goals"].to_numpy(), table["away_goals"].to_numpy(), table["result"].to_numpy()
    flat = grid.reshape(len(grid), -1).argmax(axis=1)
    top_h, top_a = flat // grid.shape[2], flat % grid.shape[2]
    exact = (top_h == hg) & (top_a == ag)
    expected_exact = (np.round(ev["lh"]) == hg) & (np.round(ev["la"]) == ag)
    top_outcome = np.where(top_h > top_a, 0, np.where(top_h == top_a, 1, 2))
    prob_outcome = ev["proba"].argmax(axis=1)
    return {"top_h": top_h, "top_a": top_a, "exact": exact, "expected_exact": expected_exact,
            "top_outcome": top_outcome, "prob_outcome": prob_outcome,
            "outcome_right": prob_outcome == y, "outcome_right_score_wrong": (prob_outcome == y) & ~exact,
            "top_outcome_right": top_outcome == y}


# ------------------------------------------------------------------ the plots
def plot_predicted_vs_actual(ev, table):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, side, lam, goals, color in (( axes[0], "Home team", ev["lh"], table["home_goals"].to_numpy(), BLUE),
                                        (axes[1], "Away team", ev["la"], table["away_goals"].to_numpy(), ORANGE)):
        hexes = ax.hexbin(lam, goals, gridsize=(14, 7), cmap="Greys", mincnt=1, extent=(0.3, 3.3, -0.5, 7.5))
        bins = pd.qcut(lam, 8)
        grouped = pd.DataFrame({"lam": lam, "goals": goals, "bin": bins}).groupby("bin", observed=True).mean()
        ax.plot(grouped["lam"], grouped["goals"], "o-", color=color, linewidth=2.5, label="Average actual goals in each group")
        ax.plot([0.3, 3.3], [0.3, 3.3], "--", color=VERMILION, label="Perfect: prediction = average actual")
        ax.set_xlabel("Predicted expected goals (lambda)")
        ax.set_title(side)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="x", visible=True)
    axes[0].set_ylabel("Actual goals scored")
    fig.colorbar(hexes, ax=axes.tolist(), label="Number of matches", shrink=0.8)
    fig.suptitle("Predicted vs actual goals, validation matches (grey = how many matches)", fontweight="bold")
    fig.savefig(FIG_DIR / "32_poisson_predicted_vs_actual_goals.png", bbox_inches="tight")
    plt.close(fig)
    print("[saved reports/figures/32_poisson_predicted_vs_actual_goals.png]")


def plot_lambda_calibration(ev, table):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot([0.5, 2.6], [0.5, 2.6], "--", color=GREY, label="Perfect calibration")
    for side, lam, goals, color, marker, shift in (("Home team", ev["lh"], table["home_goals"].to_numpy(), BLUE, "o", -0.01),
                                                    ("Away team", ev["la"], table["away_goals"].to_numpy(), VERMILION, "s", 0.01)):
        grouped = pd.DataFrame({"lam": lam, "goals": goals, "bin": pd.qcut(lam, 8)}).groupby("bin", observed=True).agg(
            predicted=("lam", "mean"), actual=("goals", "mean"), n=("goals", "size"))
        grouped["err"] = 1.96 * np.sqrt(grouped["actual"] / grouped["n"])      # Poisson standard error
        print(f"\nCalibration of lambda, {side}:")
        print(grouped.round(3).to_string())
        ax.errorbar(grouped["predicted"] + shift, grouped["actual"], yerr=grouped["err"], fmt=marker + "-", color=color, capsize=3, label=side)
    ax.set_xlabel("Predicted expected goals (average lambda in the group)")
    ax.set_ylabel("Average actual goals in the group")
    ax.set_title("Calibration of the goal rates (8 equal-sized groups, 95% error bars)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=True)
    save_figure(fig, "33_poisson_calibration_of_lambdas.png")


def plot_outcome_calibration(poisson_proba, lr_proba, book_proba, y):
    home_won = (y == 0).astype(float)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Perfect calibration")
    for (name, proba), (color, marker, shift) in zip({"Poisson (scoreline grid)": poisson_proba, "ELO2 Logistic Regression": lr_proba,
                                                       "Bookmaker": book_proba}.items(),
                                                      [(BLUE, "o", -0.006), (VERMILION, "s", 0.0), (GREEN, "^", 0.006)]):
        table = calibration_points(proba[:, 0], home_won)
        print(f"\nCalibration of home-win probability, {name}:")
        print(table.round(3).to_string())
        ax.errorbar(table["predicted"] + shift, table["actual"], yerr=table["err"], fmt=marker + "-", color=color, capsize=3, label=name)
    ax.set_xlim(0.1, 0.85)
    ax.set_ylim(0.0, 0.9)
    ax.set_xlabel("Predicted probability of a home win")
    ax.set_ylabel("How often the home team really won")
    ax.set_title("Calibration of the Home-win probability (5 equal groups, 95% error bars)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=True)
    save_figure(fig, "34_poisson_outcome_calibration.png")


def plot_scoreline_heatmap(examples):
    fig, axes = plt.subplots(1, len(examples), figsize=(7.2 * len(examples), 6.3))
    for ax, (title, grid_row, lh, la) in zip(np.atleast_1d(axes), examples):
        shown = grid_row[:6, :6] * 100
        sns.heatmap(shown, annot=True, fmt=".1f", cmap="Blues", cbar=False, linewidths=0.5, ax=ax, annot_kws={"size": 9},
                    xticklabels=range(6), yticklabels=range(6))
        ax.grid(False)
        for h, a, _ in top_scorelines(grid_row, 10):                # outline the ten most likely scorelines
            if h < 6 and a < 6:
                ax.add_patch(plt.Rectangle((a, h), 1, 1, fill=False, edgecolor=VERMILION, linewidth=2.2))
        ax.set_xlabel("Away team goals")
        ax.set_ylabel("Home team goals")
        probs = outcome_probs(grid_row[None])[0]
        ax.set_title(f"{title}\nlambda home {lh:.2f}, away {la:.2f}; Home {probs[0]:.0%} / Draw {probs[1]:.0%} / Away {probs[2]:.0%}", fontsize=11)
    fig.suptitle("Scoreline probabilities (%) from the Poisson grid; red boxes = the ten most likely scorelines", fontweight="bold")
    save_figure(fig, "35_poisson_scoreline_heatmap.png")


def plot_dixon_coles(lh_fit, la_fit, hg, ag, rho):
    lo, hi = rho_bounds(lh_fit, la_fit)
    rhos = np.linspace(max(lo, -0.3), min(hi, 0.3), 61)
    ll = np.array([score_log_likelihood(r, lh_fit, la_fit, hg, ag) for r in rhos])
    ll0 = score_log_likelihood(0.0, lh_fit, la_fit, hg, ag)
    mean_h, mean_a = float(lh_fit.mean()), float(la_fit.mean())
    independent, _ = score_grid([mean_h], [mean_a], 0.0)
    adjusted, _ = score_grid([mean_h], [mean_a], rho)
    cells = [(0, 0), (1, 1), (0, 1), (1, 0)]
    change = [(adjusted[0, i, j] / independent[0, i, j] - 1) * 100 for i, j in cells]
    print(f"\nDixon-Coles effect for an average match (lambda {mean_h:.2f} v {mean_a:.2f}, rho {rho:+.4f}):")
    for (i, j), c, a, b in zip(cells, change, independent[0, [c[0] for c in cells], [c[1] for c in cells]],
                               adjusted[0, [c[0] for c in cells], [c[1] for c in cells]]):
        print(f"  {i}-{j}: {a * 100:.2f}% -> {b * 100:.2f}% ({c:+.1f}% of itself)")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].plot(rhos, ll - ll0, color=BLUE, linewidth=2)
    axes[0].axvline(rho, color=VERMILION, linestyle="--", label=f"best rho = {rho:+.3f}")
    axes[0].axvline(0, color=GREY, linestyle=":", label="rho = 0 (no adjustment)")
    axes[0].set_xlabel("rho")
    axes[0].set_ylabel("Log-likelihood of the observed scores, relative to rho = 0")
    axes[0].set_title("Fitting rho on the fit seasons")
    axes[0].legend(loc="lower center")
    axes[0].grid(axis="x", visible=True)
    colors = [BLUE if c > 0 else VERMILION for c in change]
    bars = axes[1].bar([f"{i}-{j}" for i, j in cells], change, color=colors)
    for bar, c, (i, j) in zip(bars, change, cells):
        axes[1].text(bar.get_x() + bar.get_width() / 2, c + (0.3 if c >= 0 else -0.3), f"{c:+.1f}%\n({independent[0, i, j] * 100:.1f}% to {adjusted[0, i, j] * 100:.1f}%)",
                     ha="center", va="bottom" if c >= 0 else "top", fontsize=9)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylim(min(change) - 3, max(change) + 3)
    axes[1].set_xlabel("Scoreline (home-away)")
    axes[1].set_ylabel("Change in its probability (% of itself)")
    axes[1].set_title(f"What rho does to an average match (lambda {mean_h:.2f} v {mean_a:.2f})")
    save_figure(fig, "36_poisson_dixon_coles_effect.png")


# --------------------------------------------------------------- experiments
def goal_record(experiment, label, cfg, outcome, stats, metric, rank=np.nan, note="", n_features=None, n_rows=None):
    spec = cfg["spec"]
    return {"experiment": experiment, "option": label, "kind": spec["kind"], "level": spec["level"], "league": spec["league"],
            "n_features": n_features if n_features is not None else len(PoissonGoalModel(spec).columns),
            "window_seasons": cfg["window"] or len(FIT_SEASONS), "recency_weights": cfg["recency"], "rho": outcome["rho"],
            "n_train_rows": n_rows if n_rows is not None else len(outcome["rows"]),
            "train_deviance": outcome.get("train_dev", np.nan), "val_deviance": outcome["dev_mean"],
            "val_log_loss": outcome["metrics"]["log_loss"], "val_accuracy": outcome["metrics"]["accuracy"],
            "val_macro_f1": outcome["metrics"]["macro_f1"], "val_mae_total": outcome["mae_total"],
            "improvement_vs_default": stats["mean_diff"], "ci_low": stats["ci_low"], "ci_high": stats["ci_high"],
            "primary_metric": metric, "rank": rank, "note": note, "cfg": cfg}


def run_experiments(default_cfg, default, fit_rows, val_rows):
    """ONE change at a time from the chosen model. Primary metric: validation deviance per match for changes to the rates;
    match-outcome log loss for changes that only touch the joint (rho)."""
    section("Experiments (one change at a time from the chosen model)")
    records, decisions = [], {}
    spec = default_cfg["spec"]

    def add(experiment, label, cfg, outcome, rank, metric="deviance", note="", default_label=False, **kw):
        loss = "dev_match" if metric == "deviance" else "losses"
        stats = ({"mean_diff": np.nan, "ci_low": None, "ci_high": None} if default_label
                 else paired_bootstrap(default[loss], outcome[loss]))
        records.append(goal_record(experiment, label, cfg, outcome, stats, metric, rank, note, **kw))
        value = outcome["dev_mean"] if metric == "deviance" else outcome["metrics"]["log_loss"]
        return {"label": label, "rank": rank, "val_loss": value, "ci_low": stats["ci_low"], "ci_high": stats["ci_high"], "cfg": cfg}

    def run(name, options, metric="deviance"):
        entries = [add(name, *option[:4], metric=metric, **option[4]) if len(option) > 4 else add(name, *option, metric=metric)
                   for option in options]
        choice, reason = decide(entries)
        decisions[name] = {"choice": choice, "reason": reason}
        view = pd.DataFrame([r for r in records if r["experiment"] == name])[["option", "n_train_rows", "val_deviance", "val_log_loss",
                                                                              "val_accuracy", "improvement_vs_default", "ci_low", "ci_high"]]
        print(f"\n--- {name} (primary metric: {metric})")
        print(view.round(4).to_string(index=False))
        print(f"  -> {choice['label']}: {reason}")

    D = {"default_label": True}
    # a) level-neutral features vs raw goal averages (shows whether the drift problem is real)
    other = "raw" if spec["level"] == "rel" else "rel"
    cfg = make_cfg({**spec, "level": other}, rho_mode=default_cfg["rho_mode"])
    run("a. features", [(f"{spec['level']} features (default)", default_cfg, default, 0, D),
                        (f"{other} features", cfg, score_cfg(cfg, fit_rows, val_rows), 1)])
    # b) the league scoring level: as a feature (default), left out, or as a fixed offset (extra option)
    options = [(f"league_avg as {spec['league']} (default)", default_cfg, default, 1, D)]
    for league, rank, note in (("feature", 1, ""), ("none", 0, ""), ("offset", 2, "extra option: log league level as an offset, coefficient fixed at 1")):
        if league != spec["league"]:
            cfg = make_cfg({**spec, "league": league}, rho_mode=default_cfg["rho_mode"])
            label = {"feature": "league_avg_goals_prev380 as a feature", "none": "league average left out",
                     "offset": "league level as an offset (coefficient fixed at 1)"}[league]
            options.append((label, cfg, score_cfg(cfg, fit_rows, val_rows), rank, {"note": note}))
    run("b. league scoring level", options)
    # c) training window; "last 3" added because CLAUDE.md asks for it
    options = [("all 8 seasons (default)", default_cfg, default, 0, D)]
    for n in (3, 5):
        cfg = make_cfg(spec, window=n, rho_mode=default_cfg["rho_mode"])
        options.append((f"last {n} seasons", cfg, score_cfg(cfg, fit_rows, val_rows), 1,
                        {"note": "added to follow the CLAUDE.md pitfall"} if n == 3 else {}))
    cfg = make_cfg(spec, recency=True, rho_mode=default_cfg["rho_mode"])
    options.append(("all 8, recency weights", cfg, score_cfg(cfg, fit_rows, val_rows), 1, {}))
    run("c. training window", options)
    # d) Dixon-Coles rho: fitted on the fit seasons vs zero (outcome log loss is the metric: rho changes only the joint)
    fitted = make_cfg(spec, rho_mode="fit")
    zero = make_cfg(spec, rho_mode="zero")
    fitted_out, zero_out = score_cfg(fitted, fit_rows, val_rows), score_cfg(zero, fit_rows, val_rows)
    base_is_fit = default_cfg["rho_mode"] == "fit"
    default_for_d = fitted_out if base_is_fit else zero_out
    saved_default = default
    default = default_for_d       # the rho experiment compares against the model's own rho setting
    run("d. Dixon-Coles rho", [(f"rho {'fitted on the fit seasons' if base_is_fit else '= 0 (no adjustment)'} (default)",
                               fitted if base_is_fit else zero, default_for_d, 1 if base_is_fit else 0, D),
                              ("rho = 0 (no adjustment)" if base_is_fit else "rho fitted on the fit seasons",
                               zero if base_is_fit else fitted, zero_out if base_is_fit else fitted_out, 0 if base_is_fit else 1, {})],
        metric="log_loss")
    default = saved_default
    # d diagnostic: rho fitted on fit + validation (ONLY to see how much validation would shift it; never used for selection)
    model, rows = fit_cfg(zero, fit_rows, FIT_END)
    lh_tr, la_tr = model.predict(rows)
    lh_v, la_v = default["lh"], default["la"]
    rho_both = fit_rho(np.concatenate([lh_tr, lh_v]), np.concatenate([la_tr, la_v]),
                       np.concatenate([rows["home_goals"], val_rows["home_goals"]]), np.concatenate([rows["away_goals"], val_rows["away_goals"]]))
    model.rho = rho_both
    diag = evaluate_model(model, val_rows)
    diag.update({"rows": rows, "train_dev": default.get("train_dev", np.nan)})
    stats = paired_bootstrap(zero_out["losses"], diag["losses"])
    print(f"\nDiagnostic (NOT used for selection): rho fitted on fit + validation = {rho_both:+.4f} "
          f"(fit seasons only: {fitted_out['rho']:+.4f}); validation outcome log loss {diag['metrics']['log_loss']:.4f} "
          f"vs {zero_out['metrics']['log_loss']:.4f} with rho = 0 (improvement {stats['mean_diff']:+.4f} [{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}])")
    records.append(goal_record("diagnostic (not used for selection)", "rho fitted on fit + validation", zero, diag, stats, "log_loss",
                               note="shows how much validation would shift rho; never used to choose"))
    # e) team fixed effects with an L2 penalty (alpha chosen on validation deviance: flagged)
    stacked = make_spec("stacked", spec["level"], "feature")
    scored = {}
    for alpha in ALPHA_GRID:
        model = TeamFixedEffectsModel(stacked, alpha).fit(fit_rows)
        scored[alpha] = (model, evaluate_model(model, val_rows))
    best_alpha = min(scored, key=lambda a: scored[a][1]["dev_mean"])
    print("\nTeam fixed effects, L2 penalty alpha -> validation deviance: " + ", ".join(
        f"{a:g}: {scored[a][1]['dev_mean']:.4f}" for a in ALPHA_GRID) + f" (best alpha {best_alpha:g}, chosen on validation)")
    fe_out = scored[best_alpha][1]
    fe_out.update({"rows": fit_rows, "train_dev": np.nan})
    base_only = TeamFixedEffectsModel(stacked, best_alpha, with_teams=False).fit(fit_rows)
    base_out = evaluate_model(base_only, val_rows)
    print(f"  the same penalised model WITHOUT team effects: deviance {base_out['dev_mean']:.4f} "
          f"(with team effects {fe_out['dev_mean']:.4f}; the unpenalised stacked model {default['dev_mean']:.4f})")
    run("e. team fixed effects", [("Elo and relative-goal features only (default)", default_cfg, default, 0, D),
                                 (f"plus team attack and defence effects (L2 alpha {best_alpha:g})", default_cfg, fe_out, 1,
                                  {"note": "alpha chosen on validation deviance", "n_features": 8 + 2 * len(set(fit_rows['HomeTeam']))})])
    return records, decisions


def assemble_final(default_cfg, decisions):
    spec = dict(default_cfg["spec"])
    spec["level"] = ("raw" if decisions["a. features"]["choice"]["label"].startswith("raw") else "rel")
    league = decisions["b. league scoring level"]["choice"]["label"]
    spec["league"] = "none" if league.startswith("league average left out") else ("offset" if league.startswith("league level") else "feature")
    window_label = decisions["c. training window"]["choice"]["label"]
    return make_cfg(spec, window={"last 3 seasons": 3, "last 5 seasons": 5}.get(window_label),
                    recency=window_label.endswith("recency weights"),
                    rho_mode="fit" if "fitted on the fit seasons" in decisions["d. Dixon-Coles rho"]["choice"]["label"] else "zero")


# ------------------------------------------------------------ the Stage 7 handoff
def elo_start_snapshot(keep):
    """Start-of-2025-26 Elo ratings for a given `keep` constant (the other Elo constants stay as tuned in Stage 4).
    Only matches BEFORE 2025-26 are used for the ratings. The 2025-26 rows are loaded WITHOUT any result or score
    (only the team names, which tell us who is promoted), so no test outcome is ever in memory."""
    fixture_columns = ["season", "Date", "HomeTeam", "AwayTeam"]
    parts = []
    for chunk in pd.read_csv(MATCHES_CSV, parse_dates=["Date"], chunksize=500):
        is_test = chunk["season"] == "2025-26"
        parts.append(pd.concat([chunk[~is_test], chunk.loc[is_test, fixture_columns]]))
    matches = pd.concat(parts, ignore_index=True)
    assert matches.loc[matches["season"] == "2025-26", "result"].isna().all()
    params = dict(json.loads(PARAMS_JSON.read_text())["elo"], keep=keep)
    return elo_at_season_start(matches, "2025-26", params).set_index("team")["elo_start"]


def build_pre_season(model, train_all, elo=None):
    """lambda_home and lambda_away for every 2025-26 FIXTURE, from the start-of-season Elo snapshot and a
    start-of-season configuration (empty current-season history: the relative form features are neutral = 1.0)."""
    assert model.spec["level"] == "rel", "the pre-season table needs the level-neutral features"
    fixtures = load_fixtures_2025_26()
    if elo is None:
        elo = pd.read_csv(ELO_START_CSV).set_index("team")["elo_start"]
    last_380 = train_all.sort_values("Date").tail(380)                      # the 2024-25 season = the previous 380 matches
    league_level = float((last_380["home_goals"] + last_380["away_goals"]).mean())
    table = fixtures.copy()
    table["elo_home_start"], table["elo_away_start"] = table["HomeTeam"].map(elo), table["AwayTeam"].map(elo)
    assert table[["elo_home_start", "elo_away_start"]].notna().all().all()
    table["elo_diff"] = table["elo_home_start"] - table["elo_away_start"]
    table["abs_elo_diff"] = table["elo_diff"].abs()
    table["league_avg_goals_prev380"] = league_level
    table["season_no_fans"] = 0
    for column in ["home_gf_10_rel", "away_gf_10_rel", "home_ga_10_rel", "away_ga_10_rel", "home_venue_gf_rel",
                   "home_venue_ga_rel", "away_venue_gf_rel", "away_venue_ga_rel"]:
        table[column] = 1.0                                                  # neutral: no current-season history yet
    lh, la = model.predict(table)
    grid, _ = score_grid(lh, la, model.rho)
    proba = outcome_probs(grid)
    flat = grid.reshape(len(grid), -1).argmax(axis=1)
    out = table[["fixture_id", "Date", "HomeTeam", "AwayTeam", "elo_home_start", "elo_away_start", "elo_diff",
                 "abs_elo_diff", "league_avg_goals_prev380", "season_no_fans"]].copy()
    out["lambda_home"], out["lambda_away"] = lh, la
    out["p_home"], out["p_draw"], out["p_away"] = proba[:, 0], proba[:, 1], proba[:, 2]
    out["most_likely_home_goals"], out["most_likely_away_goals"] = flat // grid.shape[2], flat % grid.shape[2]
    return out


# --------------------------------------------------------------------- main
def main():
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_goal_rows()
    fit_rows, val_rows = get_slice("fit", rows), get_slice("validation", rows)
    y_val = val_rows["result"].to_numpy()
    print(f"Fit rows: {len(fit_rows)}; validation rows: {len(val_rows)}; test rows loaded: 0")

    section("Models A, B and C on the fit seasons, scored on the validation seasons")
    cfg_a, cfg_b = make_cfg(make_spec("independent")), make_cfg(make_spec("stacked"))
    A, B = score_cfg(cfg_a, fit_rows, val_rows), score_cfg(cfg_b, fit_rows, val_rows)
    cfg_c = make_cfg(make_spec("stacked"), rho_mode="fit")
    C = score_cfg(cfg_c, fit_rows, val_rows)
    print(f"Dixon-Coles rho (fitted on the fit seasons only): {C['rho']:+.4f}")
    print("\nModel B (stacked) standardised coefficients (change in log lambda per +1 standard deviation):")
    print(B["model"].coefficients()["stacked"].round(4).to_string())
    print("\nModel A (independent) coefficients, home model | away model:")
    ca = A["model"].coefficients()
    print(pd.DataFrame({"home goals": ca["home"], "away goals": ca["away"]}).round(4).to_string())

    saved = pd.read_csv(ROOT / "reports" / "val_predictions_5c.csv")
    assert (saved["match_id"].to_numpy() == val_rows["match_id"].to_numpy()).all()
    refs = {"ELO2 Logistic Regression": saved[[f"elo2_primary_{c}" for c in PROBA_COLUMNS]].to_numpy(),
            "Bookmaker": saved[[f"bookmaker_{c}" for c in PROBA_COLUMNS]].to_numpy()}
    ref_metrics = {name: evaluate(y_val, proba) for name, proba in refs.items()}
    section("Comparison on the validation seasons")
    table = pd.DataFrame({"A independent Poisson": model_row(A), "B stacked Poisson": model_row(B),
                          "C stacked + Dixon-Coles": model_row(C),
                          **{k: reference_row(v) for k, v in ref_metrics.items()}}).T
    print(table.round(4).to_string())
    print("(deviance = Poisson deviance per team-goal count, mean of the home and away sides; C has the same rates as B)")
    losses = {"A": A["losses"], "B": B["losses"], "C": C["losses"], "ELO2 LR": per_match_log_loss(y_val, refs["ELO2 Logistic Regression"]),
              "Bookmaker": per_match_log_loss(y_val, refs["Bookmaker"])}
    print("\nPaired bootstrap on match-outcome log loss (a better than b by; positive = a better):")
    print(pair_table(losses).round(4).to_string(index=False))
    print(f"\nGrid mass inside {MAX_GOALS} goals per team: smallest {B['mass'].min():.6f}")

    section("Dixon-Coles premise: observed vs expected low scorelines (independent Poisson with the model's rates)")
    premise_table(B["lh"], B["la"], val_rows["home_goals"].to_numpy(), val_rows["away_goals"].to_numpy(), "Validation seasons")
    lh_fit, la_fit = B["model"].predict(B["rows"])
    hg_fit, ag_fit = B["rows"]["home_goals"].to_numpy(), B["rows"]["away_goals"].to_numpy()
    premise_table(lh_fit, la_fit, hg_fit, ag_fit, "Fit seasons (in-sample)")
    zero_share = float((val_rows["home_goals"] == 0).mean()), float((val_rows["away_goals"] == 0).mean())
    from scipy.stats import poisson as _poisson
    print(f"\nShare of 0-goal sides on validation: home real {zero_share[0]:.3f} vs model {float(_poisson.pmf(0, B['lh']).mean()):.3f}; "
          f"away real {zero_share[1]:.3f} vs model {float(_poisson.pmf(0, B['la']).mean()):.3f}")

    section("Which model? (the noise rule: adopt a change only if the 95% interval excludes zero, else keep the simpler)")
    dev_ab = paired_bootstrap(A["dev_match"], B["dev_match"])    # positive = B has the lower deviance
    print(f"A (independent) vs B (stacked), Poisson deviance: B better by {dev_ab['mean_diff']:+.4f} [{dev_ab['ci_low']:+.4f}, {dev_ab['ci_high']:+.4f}]")
    base_name = "A" if dev_ab["ci_high"] < 0 else "B"     # B is the simpler (fewer parameters); A only if clearly better
    base_out, base_cfg = (A, cfg_a) if base_name == "A" else (B, cfg_b)
    dc = paired_bootstrap(base_out["losses"], C["losses"])
    use_dc = dc["ci_low"] > 0
    print(f"Base model: {base_name}. Dixon-Coles C vs B, outcome log loss: C better by {dc['mean_diff']:+.4f} [{dc['ci_low']:+.4f}, {dc['ci_high']:+.4f}] "
          f"-> {'adopt the adjustment' if use_dc else 'keep rho = 0 (cannot tell)'}")
    default_cfg = make_cfg(base_cfg["spec"], rho_mode="fit" if use_dc else "zero")
    default = C if (use_dc and base_name == "B") else (score_cfg(default_cfg, fit_rows, val_rows) if base_name == "A" or use_dc else B)
    print(f"Chosen model for the experiments: {base_name} ({base_cfg['spec']['kind']}), rho_mode = {default_cfg['rho_mode']}, "
          f"validation deviance {default['dev_mean']:.4f}, outcome log loss {default['metrics']['log_loss']:.4f}")

    records, decisions = run_experiments(default_cfg, default, fit_rows, val_rows)
    final_cfg = assemble_final(default_cfg, decisions)

    section("Confirmation of the combined configuration (one run, no grid)")
    final = score_cfg(final_cfg, fit_rows, val_rows)
    combined = paired_bootstrap(default["dev_match"], final["dev_match"])
    print(f"Chosen: {final_cfg['spec']}, window={final_cfg['window'] or 'all 8'}, recency={final_cfg['recency']}, rho {final['rho']:+.4f}")
    print(f"  validation deviance {final['dev_mean']:.4f} (default {default['dev_mean']:.4f}); outcome log loss {final['metrics']['log_loss']:.4f} "
          f"(default {default['metrics']['log_loss']:.4f}); accuracy {final['metrics']['accuracy']:.4f}")
    print(f"  deviance improvement vs default {combined['mean_diff']:+.4f} [{combined['ci_low']:+.4f}, {combined['ci_high']:+.4f}]")
    if combined["ci_high"] < 0:
        print("  The combination is clearly WORSE than the default: reverting to the default.")
        final_cfg, final = default_cfg, default
    records.append(goal_record("final combined", "chosen configuration", final_cfg, final, combined, "deviance"))

    # Diagnostic (NOT used for selection): experiments b and c both repair the same level problem. Repeat the window
    # comparison with the chosen league handling, to see whether "fewer seasons" still helps once the level is handled.
    section("Diagnostic (NOT used for selection): the window experiment repeated under the chosen league handling")
    diag_spec = final_cfg["spec"]
    diag_base = score_cfg(make_cfg(diag_spec, rho_mode=final_cfg["rho_mode"]), fit_rows, val_rows)
    print(f"League handling '{diag_spec['league']}', all 8 seasons: deviance {diag_base['dev_mean']:.4f}, outcome log loss {diag_base['metrics']['log_loss']:.4f}")
    for label, window, recency in (("last 3 seasons", 3, False), ("last 5 seasons", 5, False), ("all 8, recency weights", None, True)):
        cfg = make_cfg(diag_spec, window=window, recency=recency, rho_mode=final_cfg["rho_mode"])
        outcome = score_cfg(cfg, fit_rows, val_rows)
        stats = paired_bootstrap(diag_base["dev_match"], outcome["dev_match"])
        print(f"  {label}: deviance {outcome['dev_mean']:.4f}; improvement vs all 8 {stats['mean_diff']:+.4f} [{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]; "
              f"outcome log loss {outcome['metrics']['log_loss']:.4f}")
        records.append(goal_record("diagnostic (not used for selection)", f"window under {diag_spec['league']} league handling: {label}", cfg, outcome, stats,
                                   "deviance", note="reference = all 8 seasons with the same league handling"))
    premise_table(final["lh"], final["la"], val_rows["home_goals"].to_numpy(), val_rows["away_goals"].to_numpy(),
                  "Dixon-Coles premise again, for the CHOSEN model on the validation seasons")
    print(f"Share of 0-goal sides on validation, chosen model: home real {(val_rows['home_goals'] == 0).mean():.3f} vs model "
          f"{float(_poisson_pmf0(final['lh']).mean()):.3f}; away real {(val_rows['away_goals'] == 0).mean():.3f} vs model {float(_poisson_pmf0(final['la']).mean()):.3f}")
    print(f"Average predicted total goals on validation: {float((final['lh'] + final['la']).mean()):.3f} (real {float((val_rows['home_goals'] + val_rows['away_goals']).mean()):.3f}); "
          f"for the default model {float((default['lh'] + default['la']).mean()):.3f}")
    results = pd.DataFrame(records)
    results["decision"] = results.apply(lambda r: "CHOSEN" if r["experiment"] in decisions and
                                        decisions[r["experiment"]]["choice"]["label"] == r["option"] else "", axis=1)
    results.drop(columns=["cfg", "rank"]).to_csv(ABLATIONS_CSV, index=False)
    print(f"\nSaved {ABLATIONS_CSV.relative_to(ROOT)}")

    section("The draw problem revisited (chosen model, validation seasons)")
    stats = score_stats(final, val_rows)
    draw_p = final["proba"][:, 1]
    print(f"Bivariate-grid draw probability: mean {draw_p.mean():.3f}, min {draw_p.min():.3f}, max {draw_p.max():.3f} (real draw rate {(y_val == 1).mean():.3f})")
    print(f"Matches where Draw is the highest-probability outcome: {int((stats['prob_outcome'] == 1).sum())} of {len(y_val)}")
    print(f"Matches where the MOST LIKELY SCORELINE is a draw (1-1 or 0-0 ...): {int((stats['top_outcome'] == 1).sum())} of {len(y_val)}")
    print_metrics_block("Chosen Poisson model (outcome with the highest summed probability)", final["metrics"])
    from modeling_utils import evaluate as _evaluate
    by_score = _evaluate(y_val, pred=stats["top_outcome"])
    print(f"\nPredicting the outcome of the most likely SCORELINE instead: accuracy {by_score['accuracy']:.4f}, macro-F1 {by_score['macro_f1']:.4f}")
    print(pd.DataFrame(by_score["confusion"], index=CLASS_NAMES, columns=CLASS_NAMES).to_string())
    print(by_score["per_class"].round(3).to_string())
    for name in ("ELO2 Logistic Regression",):
        print(f"{name}: {int(ref_metrics[name]['predicted_counts'][1])} draws predicted (macro-F1 {ref_metrics[name]['macro_f1']:.4f})")

    section("Score predictions (chosen model, validation seasons)")
    print(f"Most likely scoreline exactly right: {stats['exact'].mean() * 100:.1f}% of {len(y_val)} matches "
          f"({int(stats['exact'].sum())} matches)")
    print(f"Rounded expected score (round(lambda_home), round(lambda_away)) exactly right: {stats['expected_exact'].mean() * 100:.1f}%")
    print(f"Outcome (H/D/A) right when we take the highest-probability outcome: {stats['outcome_right'].mean() * 100:.1f}%; "
          f"of these, {int(stats['outcome_right_score_wrong'].sum())} matches had the right outcome but the wrong exact score "
          f"({stats['outcome_right_score_wrong'].mean() * 100:.1f}% of all matches)")
    print(f"Outcome right when we take the outcome of the most likely scoreline: {stats['top_outcome_right'].mean() * 100:.1f}%")
    common = pd.Series(list(zip(stats["top_h"], stats["top_a"]))).value_counts().head(5)
    print("Most common predicted scorelines:", ", ".join(f"{h}-{a} ({n})" for (h, a), n in common.items()))

    section("Hand examples (validation matches)")
    diff = final["lh"] - final["la"]
    city_home = np.where(val_rows["HomeTeam"] == "Man City")[0]
    extreme_idx = int(city_home[np.argmax(diff[city_home])])
    r = val_rows.iloc[extreme_idx]
    print(f"For reference, the MOST extreme Man City home match: {r['HomeTeam']} v {r['AwayTeam']} {r['Date'].date()}, elo_diff {r['elo_diff']:.0f}: "
          f"lambda_home {final['lh'][extreme_idx]:.2f}, lambda_away {final['la'][extreme_idx]:.2f} (the log link grows exponentially with the Elo gap)")
    heavy_idx = int(city_home[np.argsort(np.abs(diff[city_home] - np.median(diff[city_home])))[0]])   # a typical Man City home match
    total = final["lh"] + final["la"]
    close_idx = int(np.argmin(np.where((total > 2.4) & (total < 3.3), np.abs(diff), np.inf)))
    examples = []
    for label, idx in (("Heavy favourite at home", heavy_idx), ("Close match", close_idx)):
        r = val_rows.iloc[idx]
        grid_row = final["grid"][idx]
        probs = final["proba"][idx]
        print(f"\n{label}: {r['HomeTeam']} v {r['AwayTeam']}, {r['Date'].date()} (real score {int(r['home_goals'])}-{int(r['away_goals'])})")
        print(f"  lambda_home {final['lh'][idx]:.3f}, lambda_away {final['la'][idx]:.3f}; elo_diff {r['elo_diff']:.1f}; Home {probs[0]:.3f}, Draw {probs[1]:.3f}, Away {probs[2]:.3f}")
        print("  top 5 scorelines:", ", ".join(f"{h}-{a} {p * 100:.1f}%" for h, a, p in top_scorelines(grid_row, 5)))
        print(f"  probability beyond 10 goals for either team (cut off): {1 - final['mass'][idx]:.2e}")
        print("  11 x 11 grid (percent; rows = home goals 0-10, columns = away goals 0-10):")
        print(pd.DataFrame(grid_row * 100).round(1).to_string())
        examples.append((f"{label}: {r['HomeTeam']} v {r['AwayTeam']}", grid_row, final["lh"][idx], final["la"][idx]))

    plot_predicted_vs_actual(final, val_rows)
    plot_lambda_calibration(final, val_rows)
    plot_outcome_calibration(final["proba"], refs["ELO2 Logistic Regression"], refs["Bookmaker"], y_val)
    plot_scoreline_heatmap(examples)
    # Figure 36 uses Model B's rates on the fit seasons and Model C's rho (fitted on those same rates), so curve and line agree
    plot_dixon_coles(lh_fit, la_fit, hg_fit, ag_fit, C["rho"])

    section("Saving validation rates, the final model and the Stage 7 handoff")
    out = pd.DataFrame({"match_id": val_rows["match_id"], "lambda_home": final["lh"], "lambda_away": final["la"]})
    for column, values in zip(PROBA_COLUMNS, final["proba"].T):
        out[f"poisson_{column}"] = values
    out.to_csv(VAL_LAMBDAS_CSV, index=False)
    print(f"Saved {VAL_LAMBDAS_CSV.relative_to(ROOT)} ({len(out)} validation matches; rates from the model fitted on the fit seasons)")
    train_all = get_slice("train_all", rows)

    section("Decisions after the review: all 8 seasons, rho = 0, lambda capped at 3.5")
    rule_final = final                                        # the rule's pick (last 3 seasons), analysed above
    deploy_cfg = make_cfg(final_cfg["spec"], window=None, recency=False, rho_mode="zero")
    deploy = score_cfg(deploy_cfg, fit_rows, val_rows)        # fitted on the fit seasons, scored on validation
    uncapped = deploy["model"].predict_uncapped(val_rows)
    deploy["model"].lambda_cap = LAMBDA_CAP
    capped = evaluate_model(deploy["model"], val_rows)
    over = (uncapped[0] > LAMBDA_CAP) | (uncapped[1] > LAMBDA_CAP)
    n_capped = int(over.sum())
    r_lh, r_la = rule_final["model"].predict_uncapped(val_rows)
    print(f"Validation matches with a rate above {LAMBDA_CAP} (all-8 model, fitted on the fit seasons): {n_capped} of {len(val_rows)}; "
          f"largest uncapped rate {max(uncapped[0].max(), uncapped[1].max()):.3f}")
    print(f"  (the rule's last-3 model had {int(((r_lh > LAMBDA_CAP) | (r_la > LAMBDA_CAP)).sum())} such matches, "
          f"largest {max(r_lh.max(), r_la.max()):.3f})")
    for _, r in val_rows[over].iterrows():
        print(f"    {r['Date'].date()} {r['HomeTeam']} v {r['AwayTeam']}: elo_diff {r['elo_diff']:.0f}, "
              f"real score {int(r['home_goals'])}-{int(r['away_goals'])}")
    print(f"{'':36} {'deviance':>9} {'log loss':>9} {'accuracy':>9} {'MAE total':>10} {'pred. total goals':>18}")
    variants = (("rule: last 3 seasons, uncapped", rule_final), ("all 8 seasons, uncapped", deploy),
                (f"all 8 seasons, capped at {LAMBDA_CAP} (deployed)", capped))
    for label, ev in variants:
        print(f"{label:36} {ev['dev_mean']:9.4f} {ev['metrics']['log_loss']:9.4f} {ev['metrics']['accuracy']:9.4f} "
              f"{ev['mae_total']:10.4f} {float((ev['lh'] + ev['la']).mean()):18.3f}")
    cap_stats = paired_bootstrap(deploy["dev_match"], capped["dev_match"])
    print(f"Cap effect on deviance (capped better by): {cap_stats['mean_diff']:+.5f} [{cap_stats['ci_low']:+.5f}, {cap_stats['ci_high']:+.5f}]")
    pd.DataFrame([{"model": label, "val_deviance": ev["dev_mean"], "val_log_loss": ev["metrics"]["log_loss"],
                   "val_accuracy": ev["metrics"]["accuracy"], "val_mae_total": ev["mae_total"],
                   "val_matches_capped": n_capped if "capped" in label else 0} for label, ev in variants]).to_csv(FINAL_CSV, index=False)

    section("Saving validation rates, the final model and the Stage 7 handoff")
    for path, ev, note in ((VAL_LAMBDAS_CSV, rule_final, "the rule's last-3 model (used for the analysis and figures above)"),
                           (VAL_LAMBDAS_FINAL_CSV, capped, "the deployed configuration (all 8 seasons, capped), fitted on the fit seasons")):
        out = pd.DataFrame({"match_id": val_rows["match_id"], "lambda_home": ev["lh"], "lambda_away": ev["la"]})
        for column, values in zip(PROBA_COLUMNS, ev["proba"].T):
            out[f"poisson_{column}"] = values
        out.to_csv(path, index=False)
        print(f"Saved {path.relative_to(ROOT)}: {note}")
    model, used = fit_cfg(deploy_cfg, train_all, FINAL_END)
    model.lambda_cap = LAMBDA_CAP
    model.save(CONFIG_PATH, ROOT / "models", extra={
        "status": "primary Poisson goal model; refit on all training seasons, not scored on the test season",
        "window_seasons": window_seasons(deploy_cfg["window"], FINAL_END), "recency_weights": deploy_cfg["recency"],
        "n_training_rows": len(used), "rho_mode": deploy_cfg["rho_mode"], "max_goals_in_grid": MAX_GOALS,
        "chosen_on": f"validation seasons {VAL_SEASONS} with models fitted on {FIT_SEASONS[0]} to {FIT_SEASONS[-1]}; "
                     "all-seasons window and the 3.5 cap were set after the review",
        "scaler": "StandardScaler fitted on these training rows only (mean and scale stored per part)",
        "goal_features_file": "data/processed/features_goals.csv", "link": "log",
        "lambda_cap_note": "predict() limits both expected-goal rates to lambda_cap; predict_uncapped() gives the raw rates"})
    print(f"Saved models/poisson_*.joblib and {CONFIG_PATH.relative_to(ROOT)}: spec {model.spec}, rho {model.rho:+.4f}, "
          f"cap {model.lambda_cap}, {len(used)} rows ({used['season'].min()} to {used['season'].max()})")
    print("Deployed coefficients (log lambda per +1 SD):")
    print(model.coefficients()["stacked"].round(4).to_string())
    elo_090, elo_075 = elo_start_snapshot(0.9), elo_start_snapshot(0.75)
    saved_090 = pd.read_csv(ELO_START_CSV).set_index("team")["elo_start"]
    assert (elo_090.sort_index() - saved_090.sort_index()).abs().max() < 1e-9, "keep=0.9 snapshot must equal the Stage 4 file"
    for label, elo, path in (("keep = 0.9 (default)", elo_090, PRE_SEASON_CSV), ("keep = 0.75", elo_075, PRE_SEASON_075_CSV)):
        pre = build_pre_season(model, train_all, elo)
        pre.to_csv(path, index=False)
        neutral = pre.assign(**{c: 1.0 for c in model.required_columns if c.endswith("_rel")})
        raw_h, raw_a = model.predict_uncapped(neutral)
        print(f"\nSaved {path.relative_to(ROOT)} [{label}]: {len(pre)} fixtures; lambda_home {pre['lambda_home'].min():.2f} to "
              f"{pre['lambda_home'].max():.2f}, lambda_away {pre['lambda_away'].min():.2f} to {pre['lambda_away'].max():.2f}; "
              f"average total goals {(pre['lambda_home'] + pre['lambda_away']).mean():.3f}; "
              f"rates changed by the cap: {int((raw_h > LAMBDA_CAP).sum() + (raw_a > LAMBDA_CAP).sum())}; "
              f"league level {pre['league_avg_goals_prev380'].iloc[0]:.3f}")
        print(pre.head(3).drop(columns="Date").round(3).to_string(index=False))
    both = pd.DataFrame({"keep 0.9": elo_090, "keep 0.75": elo_075}).sort_values("keep 0.9", ascending=False)
    print("\nStart-of-2025-26 Elo under the two keep values (top 4 and bottom 4 teams):")
    print(pd.concat([both.head(4), both.tail(4)]).round(1).to_string())


if __name__ == "__main__":
    main()
