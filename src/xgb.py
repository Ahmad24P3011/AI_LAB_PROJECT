"""Stage 5c: XGBoost (validation seasons only).

All choices use the VALIDATION seasons (2023-24, 2024-25) with models fitted on 2015-16 to 2022-23.
The 2025-26 test season is never loaded (see modeling_utils.py). Trees need no scaling.

EARLY STOPPING, in two steps, always on a split INSIDE the data a model is allowed to use:
  1. fit on the earlier seasons of the window, stop on its LAST TWO seasons -> the number of trees;
  2. refit on the whole window with exactly that number of trees (this refit is the scored model).
So the validation seasons are never used for stopping when a model is scored on them.
Only the final models (not scored on validation any more) stop on 2023-24 and 2024-25.

Run:  python src/modeling_utils.py   (once: frozen v3 feature sets)
      python src/xgb.py
"""
import json
import time
from itertools import product
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost
from sklearn.inspection import permutation_importance
from xgboost import DMatrix, XGBClassifier

from logreg import (BLUE, FIG_DIR, GREEN, GREY, ORANGE, VERMILION, calibration_points, decide,
                    fit_config, make_config, metrics_row, print_metrics_block, save_figure, section,
                    setup_style, training_rows)
from modeling_utils import (CLASS_CODES, CLASS_NAMES, FIT_SEASONS, PROBA_COLUMNS, SEED, TRAIN_SEASONS,
                            VAL_SEASONS, add_diff_features, evaluate, get_slice, load_feature_sets,
                            load_features, make_xy, paired_bootstrap, per_match_log_loss,
                            recency_weights, verify_diff_is_row_local, window_seasons)
from rf import fit_rf, make_rf_config, partial_dependence_curve

ROOT = Path(__file__).resolve().parent.parent
FIT_END, FINAL_END = FIT_SEASONS[-1], TRAIN_SEASONS[-1]
GRID = {"max_depth": [3, 5, 7], "learning_rate": [0.03, 0.1], "min_child_weight": [5, 20]}
FIXED = {"subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0}
MAX_TREES, STOP_ROUNDS, STOP_SEASONS = 2000, 50, 2
H2H = ["h2h_home_ppg_3", "h2h_n"]
MAX_MODEL_MB = 50
ABLATIONS_CSV = ROOT / "reports" / "ablations_xgb.csv"
VAL_PREDICTIONS_CSV = ROOT / "reports" / "val_predictions_5c.csv"
AUDIT_JSON = ROOT / "reports" / "xgb_early_stopping_audit.json"
MODEL_PATHS = {"primary": ROOT / "models" / "xgboost", "reference": ROOT / "models" / "xgboost_full_reference"}
AUDIT = []   # one entry per early-stopping run: which seasons were used to fit and to stop


# ------------------------------------------------------- configurations and fitting
def make_xgb_config(features, max_depth, learning_rate, min_child_weight, window=None, recency=False,
                    include_warmup=True, drop_2020_21=False, monotone=False):
    return {"features": list(features), "max_depth": max_depth, "learning_rate": learning_rate,
            "min_child_weight": min_child_weight, "window": window, "recency": recency,
            "include_warmup": include_warmup, "drop_2020_21": drop_2020_21, "monotone": monotone}


def make_xgb(cfg, n_estimators, early_stopping=False):
    """multi:softprob = one probability per class. n_jobs=1 keeps every run exactly reproducible."""
    params = dict(objective="multi:softprob", num_class=3, tree_method="hist", n_estimators=n_estimators,
                  max_depth=cfg["max_depth"], learning_rate=cfg["learning_rate"],
                  min_child_weight=cfg["min_child_weight"], random_state=SEED, n_jobs=1,
                  eval_metric="mlogloss", **FIXED)
    if early_stopping:
        params["early_stopping_rounds"] = STOP_ROUNDS
    if cfg["monotone"]:   # +1 on elo_diff, 0 on everything else
        params["monotone_constraints"] = tuple(1 if f == "elo_diff" else 0 for f in cfg["features"])
    return XGBClassifier(**params)


def find_n_trees(cfg, rows, end_season, purpose):
    """Step 1: early stopping on the LAST TWO seasons of `rows`, fitting on the earlier ones."""
    seasons = sorted(rows["season"].unique())
    stop_seasons, fit_seasons = seasons[-STOP_SEASONS:], seasons[:-STOP_SEASONS]
    inner_fit, inner_stop = rows[rows["season"].isin(fit_seasons)], rows[rows["season"].isin(stop_seasons)]
    weights = recency_weights(inner_fit, end_season) if cfg["recency"] else None
    X_fit, y_fit = make_xy(inner_fit, cfg["features"])
    X_stop, y_stop = make_xy(inner_stop, cfg["features"])
    model = make_xgb(cfg, MAX_TREES, early_stopping=True)
    model.fit(X_fit, y_fit, sample_weight=weights, eval_set=[(X_stop, y_stop)], verbose=False)
    n_trees = int(model.best_iteration) + 1
    AUDIT.append({"purpose": purpose, "n_features": len(cfg["features"]), "inner_fit_seasons": fit_seasons,
                  "inner_stop_seasons": stop_seasons, "inner_fit_last_date": str(inner_fit["Date"].max().date()),
                  "inner_stop_first_date": str(inner_stop["Date"].min().date()), "n_inner_fit": len(inner_fit),
                  "n_inner_stop": len(inner_stop), "n_trees": n_trees})
    return n_trees, model


def fit_xgb(cfg, source_rows, end_season, purpose):
    """Step 2: refit on the whole window with the number of trees that early stopping found."""
    rows = training_rows(source_rows, cfg, end_season)
    n_trees, inner = find_n_trees(cfg, rows, end_season, purpose)
    weights = recency_weights(rows, end_season) if cfg["recency"] else None
    X, y = make_xy(rows, cfg["features"])
    model = make_xgb(cfg, n_trees)
    model.fit(X, y, sample_weight=weights, verbose=False)
    assert list(model.classes_) == CLASS_CODES
    return model, rows, n_trees, inner


def xgb_proba(model, table, features):
    X, _ = make_xy(table, features)
    return model.predict_proba(X)


def score_xgb(cfg, fit_rows, val_rows):
    model, rows, n_trees, inner = fit_xgb(cfg, fit_rows, FIT_END, "scored_on_validation")
    y_val = val_rows["result"].to_numpy()
    X_val, _ = make_xy(val_rows, cfg["features"])
    val_proba = model.predict_proba(X_val)
    inner_proba = inner.predict_proba(X_val)            # the early-stopped inner model, for information only
    train_proba = xgb_proba(model, rows, cfg["features"])
    return {"model": model, "rows": rows, "n_trees": n_trees, "val_proba": val_proba,
            "metrics": evaluate(y_val, val_proba), "val_losses": per_match_log_loss(y_val, val_proba),
            "train_loss": float(per_match_log_loss(rows["result"], train_proba).mean()),
            "inner_val_loss": float(per_match_log_loss(y_val, inner_proba).mean())}


def tune_grid(features, fit_rows, val_rows):
    """The full grid of 3 x 2 x 2 = 12 combinations, each scored once on validation log loss."""
    records, outcomes = [], {}
    for depth, eta, mcw in product(*GRID.values()):
        started = time.time()
        cfg = make_xgb_config(features, depth, eta, mcw)
        outcome = score_xgb(cfg, fit_rows, val_rows)
        outcomes[(depth, eta, mcw)] = outcome
        records.append({"max_depth": depth, "learning_rate": eta, "min_child_weight": mcw,
                        "n_trees": outcome["n_trees"], "train_log_loss": outcome["train_loss"],
                        "val_log_loss": outcome["metrics"]["log_loss"],
                        "val_log_loss_inner_model": outcome["inner_val_loss"],
                        "val_accuracy": outcome["metrics"]["accuracy"], "seconds": round(time.time() - started, 1)})
    table = pd.DataFrame(records)
    best = table.loc[table["val_log_loss"].idxmin()]
    hyper = {"max_depth": int(best["max_depth"]), "learning_rate": float(best["learning_rate"]),
             "min_child_weight": int(best["min_child_weight"])}
    return table, hyper, outcomes


def tune_depth_only(features, hyper, fit_rows, val_rows, monotone=False):
    """Experiment a: re-tune only max_depth (and, through early stopping, n_estimators)."""
    best = None
    for depth in GRID["max_depth"]:
        cfg = make_xgb_config(features, depth, hyper["learning_rate"], hyper["min_child_weight"], monotone=monotone)
        outcome = score_xgb(cfg, fit_rows, val_rows)
        if best is None or outcome["metrics"]["log_loss"] < best[1]["metrics"]["log_loss"]:
            best = (cfg, outcome)
    return best


# ----------------------------------------------------------------- experiments
def xgb_record(experiment, label, cfg, outcome, stats, rank=np.nan, note=""):
    """One row of reports/ablations_xgb.csv (same layout as 5a/5b; hyperparameters are XGBoost's)."""
    return {"experiment": experiment, "option": label, "n_features": len(cfg["features"]),
            "max_depth": cfg["max_depth"], "learning_rate": cfg["learning_rate"],
            "min_child_weight": cfg["min_child_weight"], "n_trees": outcome["n_trees"],
            "window_seasons": cfg["window"] or len(FIT_SEASONS), "recency_weights": cfg["recency"],
            "include_warmup": cfg["include_warmup"], "drop_2020_21_rows": cfg["drop_2020_21"],
            "monotone_elo_diff": cfg["monotone"], "n_train_rows": len(outcome["rows"]),
            "train_log_loss": outcome["train_loss"], "val_log_loss": outcome["metrics"]["log_loss"],
            "val_accuracy": outcome["metrics"]["accuracy"], "val_macro_f1": outcome["metrics"]["macro_f1"],
            "improvement_vs_default": stats["mean_diff"], "ci_low": stats["ci_low"],
            "ci_high": stats["ci_high"], "rank": rank, "note": note, "cfg": cfg}


def run_experiments(sets, default_cfg, default, hyper, fit_rows, val_rows):
    """ONE change at a time from the default. No grid over combinations."""
    section("Experiments (one change at a time, validation seasons)")
    records, decisions, tuned = [], {}, {}
    default_losses = default["val_losses"]
    full = sets["FULL_v2"]
    base = (hyper["max_depth"], hyper["learning_rate"], hyper["min_child_weight"])

    def add(experiment, label, cfg, outcome, rank, note=""):
        is_default = label.endswith("(default)")
        stats = ({"mean_diff": np.nan, "ci_low": None, "ci_high": None} if is_default
                 else paired_bootstrap(default_losses, outcome["val_losses"]))
        records.append(xgb_record(experiment, label, cfg, outcome, stats, rank, note))
        return {"label": label, "rank": rank, "val_loss": outcome["metrics"]["log_loss"],
                "ci_low": stats["ci_low"], "ci_high": stats["ci_high"], "cfg": cfg}

    def run_experiment(name, options):
        entries = [add(name, *option) for option in options]
        choice, reason = decide(entries)
        decisions[name] = {"choice": choice, "reason": reason}
        view = pd.DataFrame([r for r in records if r["experiment"] == name])[
            ["option", "n_features", "max_depth", "n_trees", "val_log_loss", "val_accuracy",
             "improvement_vs_default", "ci_low", "ci_high"]]
        print(f"\n--- {name}")
        print(view.round(4).to_string(index=False))
        print(f"  -> {choice['label']}: {reason}")

    # a) feature set: simplest first (ELO2 < DIFF < REDUCED < FULL_v2); max_depth re-tuned for each
    options = [("FULL_v2 (default)", default_cfg, default, 3)]
    for rank, name in ((2, "REDUCED"), (0, "ELO2"), (1, "DIFF")):
        cfg, outcome = tune_depth_only(sets[name], hyper, fit_rows, val_rows)
        tuned[name] = (cfg, outcome)
        options.append((name, cfg, outcome, rank))
    run_experiment("a. feature set", options)

    # b) head-to-head
    cfg = make_xgb_config([f for f in full if f not in H2H], *base)
    run_experiment("b. head-to-head", [("with h2h (default)", default_cfg, default, 1),
                                       ("without h2h", cfg, score_xgb(cfg, fit_rows, val_rows), 0)])
    # c) season_no_fans
    cfg = make_xgb_config([f for f in full if f != "season_no_fans"], *base)
    run_experiment("c. season_no_fans", [("include (default)", default_cfg, default, 1),
                                         ("exclude the feature", cfg, score_xgb(cfg, fit_rows, val_rows), 0)])
    # d) training window; "last 3" added because CLAUDE.md asks for it
    options = [("all 8 seasons (default)", default_cfg, default, 0)]
    for n in (3, 5):
        cfg = make_xgb_config(full, *base, window=n)
        options.append((f"last {n} seasons", cfg, score_xgb(cfg, fit_rows, val_rows), 1,
                        "added to follow the CLAUDE.md pitfall" if n == 3 else ""))
    cfg = make_xgb_config(full, *base, recency=True)
    options.append(("all 8, recency weights", cfg, score_xgb(cfg, fit_rows, val_rows), 1))
    run_experiment("d. training window", options)
    # e) warm-up rows
    cfg = make_xgb_config(full, *base, include_warmup=False)
    run_experiment("e. warm-up rows", [("include (default)", default_cfg, default, 0),
                                       ("exclude from training", cfg, score_xgb(cfg, fit_rows, val_rows), 1)])
    # f) monotone constraint on elo_diff (positive), 0 on everything else
    cfg = make_xgb_config(full, *base, monotone=True)
    run_experiment("f. monotone constraint", [("unconstrained (default)", default_cfg, default, 0),
                                              ("elo_diff monotone increasing", cfg, score_xgb(cfg, fit_rows, val_rows), 1)])
    return records, decisions, tuned


def run_monotone_diagnostics(sets, tuned, hyper, fit_rows, val_rows):
    """NOT used to choose anything. In FULL_v2 the unconstrained home_elo / away_elo can bypass the
    elo_diff constraint, so we also try it where elo_diff is the only Elo feature."""
    section("Diagnostic (NOT used for any decision): monotone constraint on sets where elo_diff is the only Elo input")
    records = []
    for name in ("ELO2", "DIFF"):
        cfg, plain = tuned[name]
        mono_cfg = make_xgb_config(sets[name], cfg["max_depth"], cfg["learning_rate"], cfg["min_child_weight"], monotone=True)
        mono = score_xgb(mono_cfg, fit_rows, val_rows)
        stats = paired_bootstrap(plain["val_losses"], mono["val_losses"])
        print(f"  {name} (max_depth {cfg['max_depth']}): unconstrained {plain['metrics']['log_loss']:.4f} -> "
              f"monotone {mono['metrics']['log_loss']:.4f}; improvement {stats['mean_diff']:+.4f} "
              f"[{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]")
        records.append(xgb_record("diagnostic (not used for selection)", f"{name}: elo_diff monotone", mono_cfg, mono, stats,
                                  note=f"reference = the unconstrained {name} model"))
    return records


def assemble_final_config(sets, default_cfg, hyper, decisions, tuned):
    """Combine the per-experiment decisions into ONE configuration."""
    set_label = decisions["a. feature set"]["choice"]["label"]
    if set_label.startswith("FULL_v2"):
        features, depth = list(sets["FULL_v2"]), hyper["max_depth"]
    else:
        features, depth = list(sets[set_label]), tuned[set_label][0]["max_depth"]
    if decisions["b. head-to-head"]["choice"]["label"] == "without h2h":
        features = [f for f in features if f not in H2H]
    if decisions["c. season_no_fans"]["choice"]["label"] == "exclude the feature":
        features = [f for f in features if f != "season_no_fans"]
    window_label = decisions["d. training window"]["choice"]["label"]
    return make_xgb_config(features, depth, hyper["learning_rate"], hyper["min_child_weight"],
                           window={"last 3 seasons": 3, "last 5 seasons": 5}.get(window_label),
                           recency=window_label.endswith("recency weights"),
                           include_warmup=decisions["e. warm-up rows"]["choice"]["label"].startswith("include"),
                           monotone=decisions["f. monotone constraint"]["choice"]["label"].startswith("elo_diff monotone"))


# ----------------------------------------------- boosting, SHAP and importance
def boosting_curve(features, hyper, fit_rows, val_rows, n_rounds=500):
    """Stumps (depth 1) trained for 500 rounds with NO early stopping, scored after every round.
    This is an illustration of boosting; it is not used to select anything."""
    cfg = make_xgb_config(features, 1, hyper["learning_rate"], hyper["min_child_weight"])
    X, y = make_xy(fit_rows, features)
    Xv, yv = make_xy(val_rows, features)
    model = make_xgb(cfg, n_rounds)
    model.fit(X, y, verbose=False)
    rows = []
    for k in range(1, n_rounds + 1):
        rows.append({"trees": k,
                     "train": per_match_log_loss(y, model.predict_proba(X, iteration_range=(0, k))).mean(),
                     "validation": per_match_log_loss(yv, model.predict_proba(Xv, iteration_range=(0, k))).mean()})
    return pd.DataFrame(rows)


def shap_values(model, table, features, n=200):
    """Exact TreeSHAP from XGBoost itself (pred_contribs). Returns (rows x classes x features) in
    log-odds-like units, plus the sampled feature values."""
    sample = table.sample(n=min(n, len(table)), random_state=SEED)
    X, _ = make_xy(sample, features)
    contributions = model.get_booster().predict(DMatrix(X), pred_contribs=True)   # (n, 3, features + 1)
    return contributions[:, :, :-1], X


# ------------------------------------------------------------------- plots
def plot_heatmap(table, hyper):
    table = table.copy()
    make_label = lambda eta, mcw: f"learning_rate {eta:g}\nmin_child_weight {int(mcw)}"
    table["column"] = table.apply(lambda r: make_label(r["learning_rate"], r["min_child_weight"]), axis=1)
    order = [make_label(eta, mcw) for eta in sorted(table["learning_rate"].unique()) for mcw in sorted(table["min_child_weight"].unique())]
    values = table.pivot(index="max_depth", columns="column", values="val_log_loss")[order]
    trees = table.pivot(index="max_depth", columns="column", values="n_trees")[order]
    labels = values.copy().astype(object)
    for r in values.index:
        for c in values.columns:
            labels.loc[r, c] = f"{values.loc[r, c]:.4f}\n({int(trees.loc[r, c])} trees)"
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.heatmap(values, annot=labels, fmt="", cmap="viridis_r", linewidths=1, cbar_kws={"label": "Validation log loss (lower is better)"}, ax=ax)
    ax.grid(False)
    best_column = make_label(hyper["learning_rate"], hyper["min_child_weight"])
    row_i, col_i = list(values.index).index(hyper["max_depth"]), list(values.columns).index(best_column)
    ax.add_patch(plt.Rectangle((col_i, row_i), 1, 1, fill=False, edgecolor="black", linewidth=3))
    plt.setp(ax.get_xticklabels(), rotation=0)
    ax.set_xlabel("subsample = colsample_bytree = 0.8 in every cell")
    ax.set_ylabel("max_depth")
    ax.set_title("XGBoost grid: validation log loss (and trees chosen by early stopping); black box = best")
    save_figure(fig, "25_xgb_tuning_heatmap.png")


def plot_boosting_curve(curve, references):
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(curve["trees"], curve["train"], "-", color=BLUE, label="Training seasons")
    ax.plot(curve["trees"], curve["validation"], "-", color=VERMILION, linewidth=2, label="Validation seasons")
    marks = curve[curve["trees"].isin([1, 5, 25, 100, 500])]
    ax.plot(marks["trees"], marks["validation"], "s", color=VERMILION)
    for _, row in marks.iterrows():
        ax.annotate(f"{row['validation']:.3f}", (row["trees"], row["validation"]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color=VERMILION)
    best = curve.loc[curve["validation"].idxmin()]
    ax.axvline(best["trees"], color="black", linewidth=0.8, alpha=0.5)
    ax.annotate(f"validation lowest after {int(best['trees'])} trees\n(illustration only)", xy=(best["trees"], best["validation"]),
                xytext=(best["trees"] + 40, best["validation"] + 0.03), arrowprops={"arrowstyle": "->", "color": "black"}, fontsize=9)
    for name, value, color, style in references:
        ax.axhline(value, color=color, linestyle=style, label=f"{name} ({value:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("Number of trees (stumps, depth 1), log scale; no early stopping")
    ax.set_ylabel("Log loss (lower is better)")
    ax.set_title("Boosting round by round: log loss against the number of trees")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="x", visible=False)
    save_figure(fig, "26_xgb_boosting_curve.png")


def plot_confusion(left, right):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (title, metrics) in zip(axes, [left, right]):
        counts = metrics["confusion"]
        shares = counts / counts.sum(axis=1, keepdims=True) * 100
        labels = np.array([[f"{counts[i, j]}\n({shares[i, j]:.0f}%)" for j in range(3)] for i in range(3)])
        sns.heatmap(shares, annot=labels, fmt="", cmap="Blues", vmin=0, vmax=100, cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, linewidths=1, ax=ax)
        ax.grid(False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("What really happened")
        ax.set_title(f"{title}\naccuracy {metrics['accuracy']:.1%}, draws predicted: {int(metrics['predicted_counts'][1])}")
    fig.suptitle("Validation matches: confusion matrices (cell = count and % of the true row)", fontweight="bold")
    save_figure(fig, "27_xgb_confusion_matrix.png")


def plot_calibration(series, y):
    home_won = (y == 0).astype(float)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Perfect calibration")
    styles = [(BLUE, "o", -0.006), (VERMILION, "s", 0.0), (GREEN, "^", 0.006)]
    for (name, proba), (color, marker, shift) in zip(series.items(), styles):
        table = calibration_points(proba[:, 0], home_won)
        print(f"\nCalibration of home-win probability, {name}:")
        print(table.round(3).to_string())
        ax.errorbar(table["predicted"] + shift, table["actual"], yerr=table["err"], fmt=marker + "-", color=color, capsize=3, label=name)
    ax.set_xlim(0.1, 0.85)
    ax.set_ylim(0.0, 0.9)
    ax.set_xlabel("Predicted probability of a home win")
    ax.set_ylabel("How often the home team really won")
    ax.set_title("Calibration on validation matches (5 equal-sized groups, 95% error bars)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=True)
    save_figure(fig, "28_xgb_calibration.png")


def plot_beeswarm(panels, max_features=12):
    """Beeswarm of SHAP values for the Home-win output. Colour = the feature's value (percentile)."""
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, len(panels), figsize=(8.5 * len(panels), 7.2), squeeze=False)
    scatter = None
    for ax, (title, shap_home, X) in zip(axes[0], panels):
        order = np.argsort(-np.abs(shap_home).mean(axis=0))[:max_features][::-1]   # biggest on top
        for row, j in enumerate(order):
            percentile = X.iloc[:, j].rank(pct=True).to_numpy()
            scatter = ax.scatter(shap_home[:, j], row + rng.uniform(-0.3, 0.3, len(X)), c=percentile,
                                 cmap="viridis", s=12, vmin=0, vmax=1)
        ax.set_yticks(range(len(order)), [X.columns[j] for j in order], fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value for HOME WIN (push on the Home-win score; right = more likely)")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
    fig.subplots_adjust(wspace=0.55)   # room for the right panel's feature names
    fig.colorbar(scatter, ax=axes[0].tolist(), label="Feature value (percentile: low to high)", shrink=0.8)
    fig.suptitle("SHAP beeswarm, Home-win output (200 validation matches)", fontweight="bold")
    fig.savefig(FIG_DIR / "29_xgb_shap_beeswarm.png", bbox_inches="tight")
    plt.close(fig)
    print("[saved reports/figures/29_xgb_shap_beeswarm.png]")


def plot_importance_three_ways(table):
    order = table["permutation"].sort_values().index
    fig, axes = plt.subplots(1, 3, figsize=(17, 9.5), sharey=True)
    specs = [("gain", BLUE, "XGBoost gain importance\n(what the model used while training)", "Share of total gain (%)"),
             ("shap", GREEN, "Mean |SHAP|, all 3 outcomes\n(200 validation matches)", "Share of total mean |SHAP| (%)"),
             ("permutation", ORANGE, "Permutation importance on VALIDATION\n(log loss increase when shuffled)", "Log-loss increase (10 shuffles, +/- 1 std)")]
    for ax, (column, color, title, label) in zip(axes, specs):
        if column == "permutation":
            ax.barh(range(len(order)), table.loc[order, column], xerr=table.loc[order, "perm_std"], color=color, ecolor="black", capsize=2)
            ax.axvline(0, color="black", linewidth=0.8)
        else:
            ax.barh(range(len(order)), table.loc[order, column], color=color)
        ax.set_title(title)
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
    axes[0].set_yticks(range(len(order)), order, fontsize=8)
    fig.suptitle("XGBoost feature importance three ways (default model, FULL_v2)", fontweight="bold")
    save_figure(fig, "30_xgb_importance_three_ways.png")


def plot_partial_dependence(grid, curves, rug_values):
    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {"ELO2 Logistic Regression": (VERMILION, "-", 3), "Random Forest (ELO2)": (GREEN, "--", 2),
              "Random Forest (FULL_v2)": (GREEN, ":", 2), "XGBoost (chosen)": (BLUE, "-", 2.5),
              "XGBoost (default, FULL_v2)": (ORANGE, "-", 2), "XGBoost (default, monotone on elo_diff)": (GREY, "-.", 2)}
    for name, curve in curves.items():
        color, style, width = styles[name]
        ax.plot(grid, curve, style, color=color, linewidth=width, label=name)
    ax.plot(rug_values, np.full(len(rug_values), 0.015), "|", color=GREY, alpha=0.25)
    ax.set_xlabel("elo_diff (home Elo minus away Elo, points); grey ticks = where matches are")
    ax.set_ylabel("Average predicted probability of a HOME WIN")
    ax.set_ylim(0, 1)
    ax.set_title("Partial dependence of the Home-win probability on elo_diff")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="x", visible=True)
    save_figure(fig, "31_xgb_partial_dependence.png")


# -------------------------------------------------------------------- saving
def save_xgb(cfg, train_all, base_path, status):
    """Final model: choose the number of trees by early stopping on the LAST TWO training seasons
    (2023-24, 2024-25), then refit on all ten seasons with that number. Saved as .joblib and .ubj."""
    model, rows, n_trees, _ = fit_xgb(cfg, train_all, FINAL_END, "final_refit")
    joblib_path, ubj_path = base_path.with_suffix(".joblib"), base_path.with_suffix(".ubj")
    joblib.dump(model, joblib_path, compress=3)
    model.save_model(ubj_path)
    sizes = {p.name: round(p.stat().st_size / 1024 / 1024, 2) for p in (joblib_path, ubj_path)}
    print(f"Saved {joblib_path.name} and {ubj_path.name}: {len(cfg['features'])} features, {n_trees} trees, "
          f"{len(rows)} rows ({rows['season'].min()} to {rows['season'].max()}), sizes (MB) {sizes}")
    if max(sizes.values()) > MAX_MODEL_MB:
        raise SystemExit("Model file is over 50 MB: reduce n_estimators or raise min_child_weight, retest on validation.")
    base_path.with_name(base_path.name + "_config.json").write_text(json.dumps({
        "status": status, "features": cfg["features"], "n_trees": n_trees, "max_depth": cfg["max_depth"],
        "learning_rate": cfg["learning_rate"], "min_child_weight": cfg["min_child_weight"], **FIXED,
        "objective": "multi:softprob", "num_class": 3, "tree_method": "hist", "random_state": SEED, "n_jobs": 1,
        "monotone_constraint_on_elo_diff": cfg["monotone"], "window_seasons": window_seasons(cfg["window"], FINAL_END),
        "recency_weights": cfg["recency"], "recency_half_life_seasons": 2 if cfg["recency"] else None,
        "include_warmup_rows": cfg["include_warmup"], "drop_2020_21_rows": cfg["drop_2020_21"],
        "class_order": dict(zip(CLASS_CODES, CLASS_NAMES)), "n_training_rows": len(rows), "scaling": "none (trees do not need it)",
        "early_stopping": f"number of trees chosen by early stopping (rounds {STOP_ROUNDS}, cap {MAX_TREES}) fitting on the "
                          f"earlier seasons and stopping on the last {STOP_SEASONS} seasons of the training period; then refit on all",
        "chosen_on": f"validation seasons {VAL_SEASONS} with models fitted on {FIT_SEASONS[0]} to {FIT_SEASONS[-1]}",
        "feature_sets_file": "data/processed/feature_sets_v3.json", "file_sizes_mb": sizes, "xgboost_version": xgboost.__version__}, indent=2))
    return model


# --------------------------------------------------------------------- main
def main():
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sets = load_feature_sets(version=3)   # frozen v3, reused (no v4)
    all_training = add_diff_features(load_features())
    assert verify_diff_is_row_local(all_training)
    fit_rows, val_rows = get_slice("fit", all_training), get_slice("validation", all_training)
    y_val = val_rows["result"].to_numpy()
    print(f"Fit rows: {len(fit_rows)}; validation rows: {len(val_rows)}; test rows loaded: 0")
    print("Feature sets (feature_sets_v3.json): " + ", ".join(f"{k} {len(sets[k])}" for k in ("FULL_v2", "REDUCED", "ELO2", "DIFF")))
    inner_fit, inner_stop = FIT_SEASONS[:-STOP_SEASONS], FIT_SEASONS[-STOP_SEASONS:]
    print(f"Early stopping inside the fit seasons: fit on {inner_fit[0]} to {inner_fit[-1]}, stop on {', '.join(inner_stop)}")

    section("XGBoost: tuning on the default features (FULL_v2); 12 combinations, each scored once")
    grid, hyper, outcomes = tune_grid(sets["FULL_v2"], fit_rows, val_rows)
    print(grid.round(4).to_string(index=False))
    print(f"Chosen hyperparameters: {hyper}")
    default_cfg = make_xgb_config(sets["FULL_v2"], **hyper)
    default = outcomes[(hyper["max_depth"], hyper["learning_rate"], hyper["min_child_weight"])]
    print(f"Default: {default['n_trees']} trees from early stopping; train log loss {default['train_loss']:.4f}, "
          f"validation {default['metrics']['log_loss']:.4f} (the early-stopped inner model alone: {default['inner_val_loss']:.4f})")
    plot_heatmap(grid, hyper)

    saved = pd.read_csv(ROOT / "reports" / "val_predictions_5b.csv")
    assert (saved["match_id"].to_numpy() == val_rows["match_id"].to_numpy()).all()
    others = {name: saved[[f"{prefix}_{c}" for c in PROBA_COLUMNS]].to_numpy()
              for name, prefix in [("ELO2 Logistic Regression", "elo2_primary"), ("FULL reference Logistic Regression", "full_reference"),
                                   ("Random Forest ELO2", "rf_chosen"), ("Random Forest FULL_v2", "rf_default"),
                                   ("Class frequencies", "class_freq"), ("Bookmaker", "bookmaker")]}
    book_loss = per_match_log_loss(y_val, others["Bookmaker"]).mean()
    lr_loss = per_match_log_loss(y_val, others["ELO2 Logistic Regression"]).mean()
    freq_loss = per_match_log_loss(y_val, others["Class frequencies"]).mean()

    section("Boosting round by round (stumps, no early stopping; illustration)")
    curve = boosting_curve(sets["FULL_v2"], hyper, fit_rows, val_rows)
    for k in (1, 5, 25, 100, 500):
        row = curve[curve["trees"] == k].iloc[0]
        print(f"  {k:3} trees: training log loss {row['train']:.4f}, validation {row['validation']:.4f}")
    best_k = curve.loc[curve["validation"].idxmin()]
    print(f"  validation log loss is lowest after {int(best_k['trees'])} trees ({best_k['validation']:.4f})")
    plot_boosting_curve(curve, [("ELO2 LR", lr_loss, ORANGE, "-."), ("Bookmaker", book_loss, GREEN, "--"), ("Class frequencies", freq_loss, GREY, ":")])

    records, decisions, tuned = run_experiments(sets, default_cfg, default, hyper, fit_rows, val_rows)
    diagnostic_records = run_monotone_diagnostics(sets, tuned, hyper, fit_rows, val_rows)
    final_cfg = assemble_final_config(sets, default_cfg, hyper, decisions, tuned)

    section("Confirmation of the combined configuration (one run, no grid)")
    final = score_xgb(final_cfg, fit_rows, val_rows)
    combined = paired_bootstrap(default["val_losses"], final["val_losses"])
    print(f"Chosen: {len(final_cfg['features'])} features, max_depth={final_cfg['max_depth']}, learning_rate={final_cfg['learning_rate']}, "
          f"min_child_weight={final_cfg['min_child_weight']}, {final['n_trees']} trees (early stopping), window={final_cfg['window'] or 'all 8'}, "
          f"recency={final_cfg['recency']}, warmup included={final_cfg['include_warmup']}, monotone={final_cfg['monotone']}")
    print(f"  validation log loss {final['metrics']['log_loss']:.4f} (default {default['metrics']['log_loss']:.4f}), accuracy "
          f"{final['metrics']['accuracy']:.4f} (default {default['metrics']['accuracy']:.4f})")
    print(f"  improvement vs default {combined['mean_diff']:+.4f} [{combined['ci_low']:+.4f}, {combined['ci_high']:+.4f}]")
    reverted = combined["ci_high"] < 0
    if reverted:
        print("  The combination is clearly WORSE than the default: reverting to the default.")
        final_cfg, final = default_cfg, default
    records.append(xgb_record("final combined", "chosen configuration", final_cfg, final, combined, note="reverted" if reverted else ""))
    results = pd.DataFrame(records + diagnostic_records)
    results["decision"] = results.apply(lambda r: "CHOSEN" if r["experiment"] in decisions and
                                        decisions[r["experiment"]]["choice"]["label"] == r["option"] else "", axis=1)
    results.drop(columns=["cfg", "rank"]).to_csv(ABLATIONS_CSV, index=False)
    print(f"\nSaved {ABLATIONS_CSV.relative_to(ROOT)}")

    section("Comparison on the validation seasons")
    metrics = {"Always predict Home win": evaluate(y_val, pred=np.zeros(len(y_val), dtype=int))}
    for name, proba in others.items():
        metrics[name] = evaluate(y_val, proba)
    metrics["XGBoost default (FULL_v2)"] = default["metrics"]
    metrics["XGBoost chosen"] = final["metrics"]
    print(pd.DataFrame({k: metrics_row(v) for k, v in metrics.items()}).T.round(4).to_string())
    print_metrics_block("XGBoost default (FULL_v2)", default["metrics"])
    print_metrics_block("XGBoost chosen", final["metrics"])
    losses = {name: per_match_log_loss(y_val, proba) for name, proba in others.items()}
    for label, outcome in (("default (FULL_v2)", default), ("chosen", final)):
        for name in ("ELO2 Logistic Regression", "Bookmaker", "Random Forest ELO2", "Random Forest FULL_v2"):
            s = paired_bootstrap(losses[name], outcome["val_losses"])
            print(f"  XGBoost {label} vs {name}: log-loss improvement {s['mean_diff']:+.4f} [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}] (positive = XGBoost better)")
    for label, outcome in (("default", default), ("chosen", final)):
        d = outcome["metrics"]["per_class"].loc["Draw"]
        print(f"  Draws, {label}: predicted {int(outcome['metrics']['predicted_counts'][1])}; draw precision {d['precision']:.3f}, recall {d['recall']:.3f}, F1 {d['f1']:.3f}; "
              f"highest draw probability {outcome['val_proba'][:, 1].max():.3f}")
    print("\nOver-sharpening check (spread of predicted Home-win probability, validation):")
    for name, proba in (("ELO2 LR", others["ELO2 Logistic Regression"]), ("Random Forest FULL_v2", others["Random Forest FULL_v2"]),
                        ("XGBoost default", default["val_proba"]), ("XGBoost chosen", final["val_proba"]), ("Bookmaker", others["Bookmaker"])):
        p = proba[:, 0]
        print(f"  {name:22} min {p.min():.3f}, 5% {np.percentile(p, 5):.3f}, 95% {np.percentile(p, 95):.3f}, max {p.max():.3f}; "
              f"share above 0.70: {(p > 0.70).mean() * 100:.1f}%; largest single probability {proba.max():.3f}")

    print("\nPer validation season (log loss, average predicted home-win probability):")
    out = []
    for season in VAL_SEASONS:
        mask = (val_rows["season"] == season).to_numpy()
        row = {"season": season, "actual home-win rate": (y_val[mask] == 0).mean()}
        for name, loss, proba in (("XGB default", default["val_losses"], default["val_proba"]), ("XGB chosen", final["val_losses"], final["val_proba"]),
                                  ("ELO2 LR", losses["ELO2 Logistic Regression"], others["ELO2 Logistic Regression"]),
                                  ("RF ELO2", losses["Random Forest ELO2"], others["Random Forest ELO2"]),
                                  ("Bookmaker", losses["Bookmaker"], others["Bookmaker"])):
            row[f"{name} log loss"], row[f"{name} avg P(home)"] = loss[mask].mean(), proba[mask, 0].mean()
        out.append(row)
    print(pd.DataFrame(out).set_index("season").round(4).T.to_string())
    print("\nValidation log loss per season for each feature set (drift check):")
    by_set = [{"feature set": "FULL_v2", **{s: default["val_losses"][(val_rows["season"] == s).to_numpy()].mean() for s in VAL_SEASONS}}]
    for name in ("REDUCED", "ELO2", "DIFF"):
        by_set.append({"feature set": name, **{s: tuned[name][1]["val_losses"][(val_rows["season"] == s).to_numpy()].mean() for s in VAL_SEASONS}})
    print(pd.DataFrame(by_set).set_index("feature set").round(4).to_string())

    other_for_plot = ("Chosen" if final_cfg["features"] != default_cfg["features"] else "Default")
    plot_confusion(("XGBoost chosen", final["metrics"]), (("XGBoost default (FULL_v2)", default["metrics"]) if final_cfg != default_cfg
                                                           else ("ELO2 Logistic Regression (reference)", metrics["ELO2 Logistic Regression"])))
    plot_calibration({"XGBoost (chosen)": final["val_proba"], "ELO2 Logistic Regression": others["ELO2 Logistic Regression"],
                      "Bookmaker": others["Bookmaker"]}, y_val)

    section("Importance and SHAP")
    names = default_cfg["features"]
    shap_default, X_sample = shap_values(default["model"], val_rows, names)
    panels = [(f"Default model (FULL_v2, {len(names)} features)", shap_default[:, 0, :], X_sample)]
    if final_cfg["features"] != default_cfg["features"]:
        shap_chosen, X_chosen = shap_values(final["model"], val_rows, final_cfg["features"])
        panels.insert(0, (f"Chosen model ({len(final_cfg['features'])} features)", shap_chosen[:, 0, :], X_chosen))
    plot_beeswarm(panels)
    X_val_default, _ = make_xy(val_rows, names)
    started = time.time()
    perm = permutation_importance(default["model"], X_val_default, y_val, scoring="neg_log_loss", n_repeats=10, random_state=SEED, n_jobs=1)
    gain_raw = default["model"].get_booster().get_score(importance_type="gain")
    gain = pd.Series({f: gain_raw.get(f, 0.0) for f in names})
    shap_abs = np.abs(shap_default).mean(axis=(0, 1))
    table = pd.DataFrame({"gain": gain / gain.sum() * 100, "shap": pd.Series(shap_abs, index=names) / shap_abs.sum() * 100,
                          "permutation": pd.Series(perm.importances_mean, index=names), "perm_std": pd.Series(perm.importances_std, index=names)})
    print(f"(permutation importance took {time.time() - started:.0f} s)")
    for column, title in (("gain", "gain (%)"), ("shap", "mean |SHAP| (%)"), ("permutation", "permutation (log-loss increase)")):
        print(f"Top 8 by {title}:")
        print(table.sort_values(column, ascending=False).head(8)[["gain", "shap", "permutation"]].round(4).to_string())
    print(f"Features whose shuffling does NOT hurt (permutation <= 0): {int((table['permutation'] <= 0).sum())} of {len(names)}")
    print(f"Features never used in a split (gain 0): {int((gain == 0).sum())} of {len(names)}")
    plot_importance_three_ways(table)
    if final_cfg["features"] != default_cfg["features"]:
        shap_c = np.abs(shap_chosen).mean(axis=(0, 1))
        print("Chosen model mean |SHAP| (share %):", dict(zip(final_cfg["features"], np.round(shap_c / shap_c.sum() * 100, 1))))

    section("Partial dependence on elo_diff")
    grid_values = np.linspace(fit_rows["elo_diff"].quantile(0.05), fit_rows["elo_diff"].quantile(0.95), 25)
    elo2_lr_cfg = make_config(sets["ELO2"], json.loads((ROOT / "models" / "logreg_config.json").read_text())["C"])
    elo2_lr, _ = fit_config(elo2_lr_cfg, fit_rows, FIT_END)
    rf_elo2, _ = fit_rf(make_rf_config(sets["ELO2"], 80, 0.5, 8), fit_rows, FIT_END)
    rf_full, _ = fit_rf(make_rf_config(sets["FULL_v2"], 80, 0.5, 8), fit_rows, FIT_END)
    mono_default_cfg = make_xgb_config(sets["FULL_v2"], **hyper, monotone=True)
    mono_default = next(r for r in records if r["experiment"] == "f. monotone constraint" and r["option"].startswith("elo_diff monotone"))
    mono_model = fit_xgb(mono_default_cfg, fit_rows, FIT_END, "scored_on_validation")[0]
    curves = {"ELO2 Logistic Regression": partial_dependence_curve(elo2_lr, sets["ELO2"], fit_rows, grid_values),
              "Random Forest (ELO2)": partial_dependence_curve(rf_elo2, sets["ELO2"], fit_rows, grid_values),
              "Random Forest (FULL_v2)": partial_dependence_curve(rf_full, sets["FULL_v2"], fit_rows, grid_values),
              "XGBoost (default, FULL_v2)": partial_dependence_curve(default["model"], names, fit_rows, grid_values),
              "XGBoost (default, monotone on elo_diff)": partial_dependence_curve(mono_model, mono_default_cfg["features"], fit_rows, grid_values)}
    if final_cfg["features"] != default_cfg["features"]:
        curves["XGBoost (chosen)"] = partial_dependence_curve(final["model"], final_cfg["features"], fit_rows, grid_values)
    for name, curve_values in curves.items():
        steps = np.diff(curve_values)
        print(f"  {name:42} P(home) at {grid_values[0]:.0f}: {curve_values[0]:.3f}, at 0: {np.interp(0, grid_values, curve_values):.3f}, "
              f"at {grid_values[-1]:.0f}: {curve_values[-1]:.3f}; steps that go DOWN: {int((steps < -1e-9).sum())} of {len(steps)}")
    plot_partial_dependence(grid_values, curves, fit_rows["elo_diff"].to_numpy())

    section("Validation predictions saved for Stage 6")
    predictions = pd.DataFrame({"match_id": val_rows["match_id"], "result": y_val})
    for prefix, proba in [("xgb_default", default["val_proba"]), ("xgb_chosen", final["val_proba"]),
                          ("elo2_primary", others["ELO2 Logistic Regression"]), ("full_reference", others["FULL reference Logistic Regression"]),
                          ("rf_chosen", others["Random Forest ELO2"]), ("rf_default", others["Random Forest FULL_v2"]),
                          ("class_freq", others["Class frequencies"]), ("bookmaker", others["Bookmaker"])]:
        for column, values in zip(PROBA_COLUMNS, proba.T):
            predictions[f"{prefix}_{column}"] = values
    predictions.to_csv(VAL_PREDICTIONS_CSV, index=False)
    print(f"Saved {VAL_PREDICTIONS_CSV.relative_to(ROOT)} ({len(predictions)} matches)")

    section("Final XGBoost models: refit on 2015-16 to 2024-25 (NOT scored on the test season)")
    train_all = get_slice("train_all", all_training)
    save_xgb(final_cfg, train_all, MODEL_PATHS["primary"], "primary: configuration chosen by the noise rule")
    save_xgb(default_cfg, train_all, MODEL_PATHS["reference"], "reference row for Stage 6: the default configuration (FULL_v2)")
    AUDIT_JSON.write_text(json.dumps(AUDIT, indent=1))
    print(f"Saved {AUDIT_JSON.relative_to(ROOT)} ({len(AUDIT)} early-stopping runs). Neither model was scored on the test season.")


if __name__ == "__main__":
    main()
