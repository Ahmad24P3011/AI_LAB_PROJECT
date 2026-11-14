"""Stage 5b: Random Forest (validation seasons only).

All choices use the VALIDATION seasons (2023-24, 2024-25) with models fitted on 2015-16 to 2022-23.
The 2025-26 test season is never loaded (see modeling_utils.py). Trees need no scaling.
Run:  python src/modeling_utils.py   (once: writes the frozen v3 feature sets)
      python src/rf.py
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
from matplotlib.colors import to_rgb
from matplotlib.ticker import NullFormatter
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, plot_tree

from logreg import (BLUE, FIG_DIR, GREEN, GREY, ORANGE, VERMILION, calibration_points, decide,
                    metrics_row, print_metrics_block, save_figure, section, setup_style,
                    training_rows)
from modeling_utils import (CLASS_CODES, CLASS_NAMES, FIT_SEASONS, PROBA_COLUMNS, SEED,
                            TRAIN_SEASONS, VAL_SEASONS, add_diff_features, evaluate, get_slice,
                            load_feature_sets, load_features, make_xy, paired_bootstrap,
                            per_match_log_loss, recency_weights, verify_diff_is_row_local,
                            window_seasons)

ROOT = Path(__file__).resolve().parent.parent
FIT_END, FINAL_END = FIT_SEASONS[-1], TRAIN_SEASONS[-1]
N_TREES = 500
RF_GRID = {"min_samples_leaf": [5, 10, 20, 40, 80], "max_features": ["sqrt", 0.5], "max_depth": [None, 8]}
LEAF_GRID = RF_GRID["min_samples_leaf"]
ABLATIONS_CSV = ROOT / "reports" / "ablations_rf.csv"
VAL_PREDICTIONS_CSV = ROOT / "reports" / "val_predictions_5b.csv"
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
CONFIG_PATH = ROOT / "models" / "random_forest_config.json"
REFERENCE_MODEL_PATH = ROOT / "models" / "random_forest_full_reference.joblib"
REFERENCE_CONFIG_PATH = ROOT / "models" / "random_forest_full_reference_config.json"
MAX_MODEL_MB = 50
H2H = ["h2h_home_ppg_3", "h2h_n"]


# ------------------------------------------------------- configurations and fitting
def make_rf_config(features, min_samples_leaf, max_features, max_depth, window=None,
                   recency=False, include_warmup=True, drop_2020_21=False):
    return {"features": list(features), "min_samples_leaf": min_samples_leaf,
            "max_features": max_features, "max_depth": max_depth, "window": window,
            "recency": recency, "include_warmup": include_warmup, "drop_2020_21": drop_2020_21}


def make_rf(cfg):
    """500 trees. random_state is fixed, so the same data always gives the same forest."""
    return RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=cfg["min_samples_leaf"],
                                  max_features=cfg["max_features"], max_depth=cfg["max_depth"],
                                  random_state=SEED, n_jobs=-1)


def fit_rf(cfg, source_rows, end_season):
    rows = training_rows(source_rows, cfg, end_season)
    weights = recency_weights(rows, end_season) if cfg["recency"] else None
    X, y = make_xy(rows, cfg["features"])          # guarded: refuses test rows
    model = make_rf(cfg)
    model.fit(X, y, sample_weight=weights)
    assert list(model.classes_) == CLASS_CODES
    return model, rows


def rf_proba(model, rows, features):
    X, _ = make_xy(rows, features)
    return model.predict_proba(X)


def score_rf(cfg, fit_rows, val_rows):
    model, rows = fit_rf(cfg, fit_rows, FIT_END)
    y_val = val_rows["result"].to_numpy()
    val_proba = rf_proba(model, val_rows, cfg["features"])
    train_proba = rf_proba(model, rows, cfg["features"])   # optimistic: the trees saw these rows
    return {"model": model, "rows": rows, "val_proba": val_proba,
            "metrics": evaluate(y_val, val_proba), "val_losses": per_match_log_loss(y_val, val_proba),
            "train_loss": float(per_match_log_loss(rows["result"], train_proba).mean())}


def tune_grid(features, fit_rows, val_rows):
    """The full grid of 5 x 2 x 2 = 20 combinations, scored on validation log loss."""
    records = []
    for leaf, max_features, depth in product(*RF_GRID.values()):
        started = time.time()
        outcome = score_rf(make_rf_config(features, leaf, max_features, depth), fit_rows, val_rows)
        records.append({"min_samples_leaf": leaf, "max_features": max_features, "max_depth": depth,
                        "train_log_loss": outcome["train_loss"],
                        "val_log_loss": outcome["metrics"]["log_loss"],
                        "val_accuracy": outcome["metrics"]["accuracy"],
                        "seconds": round(time.time() - started, 1)})
    table = pd.DataFrame(records)
    best = table.loc[table["val_log_loss"].idxmin()]
    return table, {"min_samples_leaf": int(best["min_samples_leaf"]), "max_features": best["max_features"],
                   "max_depth": None if pd.isna(best["max_depth"]) else int(best["max_depth"])}


def tune_leaf_only(features, hyper, fit_rows, val_rows):
    """Experiment a: re-tune ONLY min_samples_leaf for a different feature set."""
    best_cfg, best_outcome = None, None
    for leaf in LEAF_GRID:
        cfg = make_rf_config(features, leaf, hyper["max_features"], hyper["max_depth"])
        outcome = score_rf(cfg, fit_rows, val_rows)
        if best_outcome is None or outcome["metrics"]["log_loss"] < best_outcome["metrics"]["log_loss"]:
            best_cfg, best_outcome = cfg, outcome
    return best_cfg, best_outcome


# ----------------------------------------------------------------- experiments
def rf_record(experiment, label, cfg, outcome, stats, rank=np.nan, note=""):
    """One row of reports/ablations_rf.csv (same layout as ablations_logreg.csv; the column C is
    replaced by the forest's hyperparameters)."""
    return {"experiment": experiment, "option": label, "n_features": len(cfg["features"]),
            "min_samples_leaf": cfg["min_samples_leaf"], "max_features": cfg["max_features"],
            "max_depth": cfg["max_depth"], "window_seasons": cfg["window"] or len(FIT_SEASONS),
            "recency_weights": cfg["recency"], "include_warmup": cfg["include_warmup"],
            "drop_2020_21_rows": cfg["drop_2020_21"], "n_train_rows": len(outcome["rows"]),
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

    def add(experiment, label, cfg, outcome, rank, note=""):
        is_default = label.endswith("(default)")
        stats = ({"mean_diff": np.nan, "ci_low": None, "ci_high": None} if is_default
                 else paired_bootstrap(default_losses, outcome["val_losses"]))
        records.append(rf_record(experiment, label, cfg, outcome, stats, rank, note))
        return {"label": label, "rank": rank, "val_loss": outcome["metrics"]["log_loss"],
                "ci_low": stats["ci_low"], "ci_high": stats["ci_high"], "cfg": cfg}

    def run_experiment(name, options):
        entries = [add(name, *option) for option in options]
        choice, reason = decide(entries)
        decisions[name] = {"choice": choice, "reason": reason}
        view = pd.DataFrame([r for r in records if r["experiment"] == name])[
            ["option", "n_features", "min_samples_leaf", "n_train_rows", "val_log_loss", "val_accuracy",
             "improvement_vs_default", "ci_low", "ci_high"]]
        print(f"\n--- {name}")
        print(view.round(4).to_string(index=False))
        print(f"  -> {choice['label']}: {reason}")

    # a) feature set: simplest first (ELO2 < DIFF < REDUCED < FULL_v2)
    options = [("FULL_v2 (default)", default_cfg, default, 3)]
    for rank, name in ((2, "REDUCED"), (0, "ELO2"), (1, "DIFF")):
        cfg, outcome = tune_leaf_only(sets[name], hyper, fit_rows, val_rows)
        tuned[name] = cfg
        options.append((name, cfg, outcome, rank))
    run_experiment("a. feature set", options)

    # b) head-to-head
    cfg = make_rf_config([f for f in full if f not in H2H], **hyper)
    run_experiment("b. head-to-head", [
        ("with h2h (default)", default_cfg, default, 1),
        ("without h2h", cfg, score_rf(cfg, fit_rows, val_rows), 0)])

    # c) season_no_fans
    cfg = make_rf_config([f for f in full if f != "season_no_fans"], **hyper)
    run_experiment("c. season_no_fans", [
        ("include (default)", default_cfg, default, 1),
        ("exclude the feature", cfg, score_rf(cfg, fit_rows, val_rows), 0)])

    # d) training window (counted back from 2022-23); "last 3" added because CLAUDE.md asks for it
    options = [("all 8 seasons (default)", default_cfg, default, 0)]
    for n in (3, 5):
        cfg = make_rf_config(full, window=n, **hyper)
        note = "added to follow the CLAUDE.md pitfall" if n == 3 else ""
        options.append((f"last {n} seasons", cfg, score_rf(cfg, fit_rows, val_rows), 1, note))
    cfg = make_rf_config(full, recency=True, **hyper)
    options.append(("all 8, recency weights", cfg, score_rf(cfg, fit_rows, val_rows), 1))
    run_experiment("d. training window", options)

    # e) warm-up rows
    cfg = make_rf_config(full, include_warmup=False, **hyper)
    run_experiment("e. warm-up rows", [
        ("include (default)", default_cfg, default, 0),
        ("exclude from training", cfg, score_rf(cfg, fit_rows, val_rows), 1)])
    return records, decisions, tuned


def assemble_final_config(sets, default_cfg, hyper, decisions, tuned):
    """Combine the per-experiment decisions into ONE configuration."""
    set_label = decisions["a. feature set"]["choice"]["label"]
    if set_label.startswith("FULL_v2"):
        features, base = list(sets["FULL_v2"]), hyper
    else:
        features, base = list(sets[set_label]), {k: tuned[set_label][k] for k in
                                                   ("min_samples_leaf", "max_features", "max_depth")}
    if decisions["b. head-to-head"]["choice"]["label"] == "without h2h":
        features = [f for f in features if f not in H2H]
    if decisions["c. season_no_fans"]["choice"]["label"] == "exclude the feature":
        features = [f for f in features if f != "season_no_fans"]
    window_label = decisions["d. training window"]["choice"]["label"]
    window = {"last 3 seasons": 3, "last 5 seasons": 5}.get(window_label)
    return make_rf_config(features, window=window, recency=window_label.endswith("recency weights"),
                          include_warmup=decisions["e. warm-up rows"]["choice"]["label"].startswith("include"),
                          **base)


# ------------------------------------------------------- importance and plots
def partial_dependence_curve(model, features, background, grid):
    """Average Home-win probability when elo_diff is set to each grid value (and abs_elo_diff
    to its absolute value, if the model uses it); all other features keep their real values."""
    curve = []
    for value in grid:
        data = background[features].copy()
        data["elo_diff"] = value
        if "abs_elo_diff" in features:
            data["abs_elo_diff"] = abs(value)
        curve.append(model.predict_proba(data)[:, 0].mean())
    return np.array(curve)


def gini(counts):
    counts = np.asarray(counts, dtype=float)
    return 1 - np.sum((counts / counts.sum()) ** 2)


def gini_example(fit_rows, features):
    """The worked example for the lesson, with real counts from the fit seasons."""
    y = fit_rows["result"].to_numpy()
    total = np.bincount(y, minlength=3)
    section("Worked example: one split of a decision tree (fit seasons)")
    print(f"All {len(y)} matches: Home win {total[0]}, Draw {total[1]}, Away win {total[2]}")
    print(f"Gini impurity before any split = 1 - sum(p^2) = {gini(total):.4f}")
    mask = fit_rows["elo_diff"].to_numpy() > 50
    left, right = np.bincount(y[~mask], minlength=3), np.bincount(y[mask], minlength=3)
    for name, counts in (("elo_diff <= 50", left), ("elo_diff >  50", right)):
        print(f"  {name}: {counts.sum()} matches (H {counts[0]}, D {counts[1]}, A {counts[2]}), "
              f"shares {np.round(counts / counts.sum(), 3)}, Gini {gini(counts):.4f}")
    after = (left.sum() * gini(left) + right.sum() * gini(right)) / len(y)
    print(f"Weighted Gini after the split = ({left.sum()} x {gini(left):.4f} + {right.sum()} x "
          f"{gini(right):.4f}) / {len(y)} = {after:.4f}")
    print(f"Impurity decrease = {gini(total):.4f} - {after:.4f} = {gini(total) - after:.4f}")
    tree = DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(fit_rows[features], y)
    print(f"The tree's own best first split: {features[tree.tree_.feature[0]]} <= {tree.tree_.threshold[0]:.2f}")


def plot_validation_curve(table, hyper, reference_losses):
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for (max_features, depth), group in table.groupby(["max_features", "max_depth"], dropna=False):
        is_best = (max_features == hyper["max_features"]) and (
            (depth is None or pd.isna(depth)) if hyper["max_depth"] is None else depth == hyper["max_depth"])
        if not is_best:
            ax.plot(group["min_samples_leaf"], group["val_log_loss"], "-", color="#BBBBBB", linewidth=1)
    best = table[(table["max_features"] == hyper["max_features"]) & (
        table["max_depth"].isna() if hyper["max_depth"] is None else table["max_depth"] == hyper["max_depth"])]
    ax.plot(best["min_samples_leaf"], best["train_log_loss"], "o--", color=BLUE,
            label="Training seasons (optimistic: the trees saw these rows)")
    ax.plot(best["min_samples_leaf"], best["val_log_loss"], "s-", color=VERMILION, linewidth=2,
            label=f"Validation seasons (max_features={hyper['max_features']}, max_depth={hyper['max_depth']})")
    ax.plot([], [], "-", color="#BBBBBB", label="Validation, other max_features / max_depth combinations")
    for name, value, color, style in reference_losses:
        ax.axhline(value, color=color, linestyle=style, label=f"{name} ({value:.3f})")
    ax.set_xscale("log")
    ax.set_xticks(LEAF_GRID, [str(v) for v in LEAF_GRID])
    ax.xaxis.set_minor_formatter(NullFormatter())   # hide the stray "6 x 10^0" style minor labels
    ax.set_xlabel("min_samples_leaf (larger = smaller, simpler trees), log scale")
    ax.set_ylabel("Log loss (lower is better)")
    ax.set_title("Random Forest: training vs validation log loss by min_samples_leaf")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="x", visible=False)
    save_figure(fig, "19_rf_validation_curve.png")


def plot_confusion(rf_metrics, rf_title, lr_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (title, metrics) in zip(axes, [(rf_title, rf_metrics), ("ELO2 Logistic Regression (reference)", lr_metrics)]):
        counts = metrics["confusion"]
        shares = counts / counts.sum(axis=1, keepdims=True) * 100
        labels = np.array([[f"{counts[i, j]}\n({shares[i, j]:.0f}%)" for j in range(3)] for i in range(3)])
        sns.heatmap(shares, annot=labels, fmt="", cmap="Blues", vmin=0, vmax=100, cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, linewidths=1, ax=ax)
        ax.grid(False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("What really happened")
        ax.set_title(f"{title}\naccuracy {metrics['accuracy']:.1%}, draws predicted: "
                     f"{int(metrics['predicted_counts'][1])}")
    fig.suptitle("Validation matches: confusion matrices (cell = count and % of the true row)", fontweight="bold")
    save_figure(fig, "20_rf_confusion_matrix.png")


def plot_calibration(series, y):
    home_won = (y == 0).astype(float)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Perfect calibration")
    styles = [(BLUE, "o", -0.006), (VERMILION, "s", 0.0), (GREEN, "^", 0.006)]
    for (name, proba), (color, marker, shift) in zip(series.items(), styles):
        table = calibration_points(proba[:, 0], home_won)
        print(f"\nCalibration of home-win probability, {name}:")
        print(table.round(3).to_string())
        ax.errorbar(table["predicted"] + shift, table["actual"], yerr=table["err"], fmt=marker + "-",
                    color=color, capsize=3, label=name)
    ax.set_xlim(0.1, 0.85)
    ax.set_ylim(0.0, 0.9)
    ax.set_xlabel("Predicted probability of a home win")
    ax.set_ylabel("How often the home team really won")
    ax.set_title("Calibration on validation matches (5 equal-sized groups, 95% error bars)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=True)
    save_figure(fig, "21_rf_calibration.png")


def plot_importance(impurity, perm_mean, perm_std):
    order = perm_mean.sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(14, 9), sharey=True)
    axes[0].barh(range(len(order)), impurity[order], color=BLUE)
    axes[0].set_title("Impurity importance\n(from the training data)")
    axes[0].set_xlabel("Share of the total impurity decrease")
    axes[1].barh(range(len(order)), perm_mean[order], xerr=perm_std[order], color=ORANGE, ecolor="black", capsize=2)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Permutation importance on VALIDATION matches\n(shuffle one feature, see how much log loss gets worse)")
    axes[1].set_xlabel("Increase in log loss when the feature is shuffled (10 repeats, +/- 1 std)")
    axes[0].set_yticks(range(len(order)), order, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
    fig.suptitle("Random Forest feature importance (default features, FULL_v2)", fontweight="bold")
    save_figure(fig, "22_rf_importance.png")


def plot_partial_dependence(grid, curves, rug_values):
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    colors = {"Random Forest (FULL_v2, default)": BLUE, "Random Forest (chosen features)": GREEN,
              "ELO2 Logistic Regression": VERMILION}
    for name, curve in curves.items():
        ax.plot(grid, curve, "-", color=colors[name], linewidth=2.5, label=name)
    ax.plot(rug_values, np.full(len(rug_values), 0.015), "|", color=GREY, alpha=0.25)
    ax.set_xlabel("elo_diff (home Elo minus away Elo, points); grey ticks = where matches are")
    ax.set_ylabel("Average predicted probability of a HOME WIN")
    ax.set_ylim(0, 1)
    ax.set_title("Partial dependence of the Home-win probability on elo_diff")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=True)
    save_figure(fig, "23_rf_partial_dependence.png")


def plot_tree_depth3(fit_rows, features):
    tree = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(fit_rows[features], fit_rows["result"])
    fig, ax = plt.subplots(figsize=(21, 9))
    texts = plot_tree(tree, feature_names=features, class_names=CLASS_NAMES, filled=True, rounded=True,
                      impurity=True, fontsize=9, ax=ax)   # filled=True creates the boxes we recolour below
    colors = [BLUE, GREY, VERMILION]
    values = tree.tree_.value[:, 0, :]
    shares = values / values.sum(axis=1, keepdims=True)
    # plot_tree also returns the small "True"/"False" edge labels, which have no box: keep only nodes
    node_texts = [t for t in texts if t.get_bbox_patch() is not None]
    assert len(node_texts) == tree.tree_.node_count
    for i, text in enumerate(node_texts):   # colour each box by its majority class (colour-blind safe palette)
        patch = text.get_bbox_patch()
        major = int(np.argmax(shares[i]))
        # opaque colour: the majority-class colour mixed with white (purer node = stronger colour)
        strength = 0.25 + 0.6 * (shares[i][major] - 1 / 3)
        patch.set_facecolor(tuple(1 - strength * (1 - c) for c in to_rgb(colors[major])))
        patch.set_alpha(1.0)
        patch.set_edgecolor("black")
    ax.set_title("One decision tree, depth 3, fitted on 2015-16 to 2022-23 (box colour = majority outcome: "
                 "blue Home win, grey Draw, orange-red Away win)", fontweight="bold")
    save_figure(fig, "24_rf_tree_depth3.png")
    return tree


def save_rf(cfg, train_all, model_path, config_path, status):
    """Refit on the full training period (window counted back from 2024-25) and save, compressed."""
    model, used = fit_rf(cfg, train_all, FINAL_END)
    joblib.dump(model, model_path, compress=3)
    size_mb = model_path.stat().st_size / 1024 / 1024
    print(f"Saved {model_path.relative_to(ROOT)}: {len(cfg['features'])} features, {size_mb:.1f} MB "
          f"(limit {MAX_MODEL_MB} MB), {len(used)} rows, seasons {used['season'].min()} to {used['season'].max()}")
    if size_mb > MAX_MODEL_MB:
        raise SystemExit("Model file is over 50 MB: raise min_samples_leaf or lower n_estimators, retest on validation.")
    config_path.write_text(json.dumps({
        "status": status, "features": cfg["features"], "n_estimators": N_TREES,
        "min_samples_leaf": cfg["min_samples_leaf"], "max_features": cfg["max_features"],
        "max_depth": cfg["max_depth"], "random_state": SEED,
        "window_seasons": window_seasons(cfg["window"], FINAL_END), "recency_weights": cfg["recency"],
        "recency_half_life_seasons": 2 if cfg["recency"] else None,
        "include_warmup_rows": cfg["include_warmup"], "drop_2020_21_rows": cfg["drop_2020_21"],
        "class_order": dict(zip(CLASS_CODES, CLASS_NAMES)), "n_training_rows": len(used),
        "scaling": "none (trees do not need it)",
        "diff_features": "computed in code from features.csv columns (see modeling_utils.add_diff_features)",
        "chosen_on": f"validation seasons {VAL_SEASONS} with models fitted on {FIT_SEASONS[0]} to {FIT_SEASONS[-1]}",
        "feature_sets_file": "data/processed/feature_sets_v3.json", "file_size_mb": round(size_mb, 1)}, indent=2))


# --------------------------------------------------------------------- main
def main():
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sets = load_feature_sets(version=3)   # frozen v3: never recomputed here
    all_training = add_diff_features(load_features())   # training rows only; DIFF = same-row arithmetic
    assert verify_diff_is_row_local(all_training)
    fit_rows, val_rows = get_slice("fit", all_training), get_slice("validation", all_training)
    y_val = val_rows["result"].to_numpy()
    print(f"Fit rows: {len(fit_rows)}; validation rows: {len(val_rows)}; test rows loaded: 0")
    print("Feature sets: " + ", ".join(f"{k} {len(sets[k])}" for k in ("FULL_v2", "REDUCED", "ELO2", "DIFF")))

    section("Random Forest: tuning on the default features (FULL_v2, 500 trees)")
    grid, hyper = tune_grid(sets["FULL_v2"], fit_rows, val_rows)
    print(grid.round(4).to_string(index=False))
    print(f"Chosen hyperparameters: {hyper}")
    default_cfg = make_rf_config(sets["FULL_v2"], **hyper)
    default = score_rf(default_cfg, fit_rows, val_rows)

    # the other predictors, reloaded from the saved Stage 5a validation probabilities
    saved = pd.read_csv(ROOT / "reports" / "val_predictions_5a_v2.csv")
    assert (saved["match_id"].to_numpy() == val_rows["match_id"].to_numpy()).all()
    others = {name: saved[[f"{prefix}_{c}" for c in PROBA_COLUMNS]].to_numpy()
              for name, prefix in [("ELO2 Logistic Regression", "elo2_primary"),
                                   ("FULL reference Logistic Regression", "full_reference"),
                                   ("Class frequencies", "class_freq"), ("Bookmaker", "bookmaker")]}
    plot_validation_curve(grid, hyper, [("ELO2 LR", per_match_log_loss(y_val, others["ELO2 Logistic Regression"]).mean(), ORANGE, "-."),
                                        ("Bookmaker", per_match_log_loss(y_val, others["Bookmaker"]).mean(), GREEN, "--"),
                                        ("Class frequencies", per_match_log_loss(y_val, others["Class frequencies"]).mean(), GREY, ":")])

    section("Default Random Forest vs the other predictors (validation seasons)")
    metrics = {"Always predict Home win": evaluate(y_val, pred=np.zeros(len(y_val), dtype=int))}
    for name, proba in others.items():
        metrics[name] = evaluate(y_val, proba)
    metrics["Random Forest (default: FULL_v2)"] = default["metrics"]
    print(pd.DataFrame({k: metrics_row(v) for k, v in metrics.items()}).T.round(4).to_string())
    print_metrics_block("Random Forest (default: FULL_v2)", default["metrics"])
    losses = {name: per_match_log_loss(y_val, proba) for name, proba in others.items()}
    for label, key in (("ELO2 Logistic Regression", "ELO2 Logistic Regression"),
                       ("FULL reference Logistic Regression", "FULL reference Logistic Regression"),
                       ("the bookmaker", "Bookmaker")):
        s = paired_bootstrap(losses[key], default["val_losses"])
        print(f"  Random Forest vs {label}: log-loss improvement {s['mean_diff']:+.4f} "
              f"[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}] (positive = Random Forest better)")
    draws = default["metrics"]["per_class"].loc["Draw"]
    print(f"  Draws: the forest predicted {int(default['metrics']['predicted_counts'][1])} draws; "
          f"draw precision {draws['precision']:.3f}, recall {draws['recall']:.3f}, F1 {draws['f1']:.3f}")

    records, decisions, tuned = run_experiments(sets, default_cfg, default, hyper, fit_rows, val_rows)
    final_cfg = assemble_final_config(sets, default_cfg, hyper, decisions, tuned)

    section("Confirmation of the combined configuration (one run, no grid)")
    final = score_rf(final_cfg, fit_rows, val_rows)
    combined = paired_bootstrap(default["val_losses"], final["val_losses"])
    print(f"Chosen: {len(final_cfg['features'])} features, min_samples_leaf={final_cfg['min_samples_leaf']}, "
          f"max_features={final_cfg['max_features']}, max_depth={final_cfg['max_depth']}, "
          f"window={final_cfg['window'] or 'all 8'}, recency={final_cfg['recency']}, "
          f"warmup included={final_cfg['include_warmup']}")
    print(f"  validation log loss {final['metrics']['log_loss']:.4f} (default {default['metrics']['log_loss']:.4f}), "
          f"accuracy {final['metrics']['accuracy']:.4f} (default {default['metrics']['accuracy']:.4f})")
    print(f"  improvement vs default {combined['mean_diff']:+.4f} [{combined['ci_low']:+.4f}, {combined['ci_high']:+.4f}]")
    reverted = combined["ci_high"] < 0
    if reverted:
        print("  The combination is clearly WORSE than the default: reverting to the default.")
        final_cfg, final = default_cfg, default
    records.append(rf_record("final combined", "chosen configuration", final_cfg, final, combined, note="reverted" if reverted else ""))
    results = pd.DataFrame(records)
    results["decision"] = results.apply(lambda r: "CHOSEN" if r["experiment"] in decisions and
                                        decisions[r["experiment"]]["choice"]["label"] == r["option"] else "", axis=1)
    results.drop(columns=["cfg", "rank"]).to_csv(ABLATIONS_CSV, index=False)
    print(f"\nSaved {ABLATIONS_CSV.relative_to(ROOT)}")

    section("Final-vs-default detail: the chosen Random Forest on the validation seasons")
    print_metrics_block("Random Forest (chosen configuration)", final["metrics"])
    for label, key in (("ELO2 Logistic Regression", "ELO2 Logistic Regression"), ("the bookmaker", "Bookmaker")):
        s = paired_bootstrap(losses[key], final["val_losses"])
        print(f"  chosen Random Forest vs {label}: log-loss improvement {s['mean_diff']:+.4f} "
              f"[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}] (positive = Random Forest better)")
    print("\nPer validation season (log loss and average predicted home-win probability):")
    rows_out = []
    for season in VAL_SEASONS:
        mask = (val_rows["season"] == season).to_numpy()
        row = {"season": season, "actual home-win rate": (y_val[mask] == 0).mean()}
        for name, outcome in (("RF default", default), ("RF chosen", final)):
            row[f"{name} log loss"] = outcome["val_losses"][mask].mean()
            row[f"{name} avg P(home)"] = outcome["val_proba"][mask, 0].mean()
        row["ELO2 LR log loss"] = losses["ELO2 Logistic Regression"][mask].mean()
        row["ELO2 LR avg P(home)"] = others["ELO2 Logistic Regression"][mask, 0].mean()
        rows_out.append(row)
    print(pd.DataFrame(rows_out).set_index("season").round(4).T.to_string())
    sets_by_season = []
    for name in ("FULL_v2", "REDUCED", "ELO2", "DIFF"):
        mask_cfg = tuned.get(name) if name != "FULL_v2" else default_cfg
        out = score_rf(mask_cfg, fit_rows, val_rows) if name != "FULL_v2" else default
        sets_by_season.append({"feature set": name, **{s: out["val_losses"][(val_rows["season"] == s).to_numpy()].mean()
                                                       for s in VAL_SEASONS}})
    print("\nValidation log loss per season for each feature set (drift check):")
    print(pd.DataFrame(sets_by_season).set_index("feature set").round(4).to_string())

    plot_confusion(final["metrics"], "Random Forest (chosen configuration)", metrics["ELO2 Logistic Regression"])
    plot_calibration({"Random Forest (chosen)": final["val_proba"], "ELO2 Logistic Regression": others["ELO2 Logistic Regression"],
                      "Bookmaker": others["Bookmaker"]}, y_val)

    section("Importance (default Random Forest, FULL_v2)")
    X_val, _ = make_xy(val_rows, default_cfg["features"])
    started = time.time()
    perm = permutation_importance(default["model"], X_val, y_val, scoring="neg_log_loss", n_repeats=10,
                                  random_state=SEED, n_jobs=1)
    names = default_cfg["features"]
    impurity = pd.Series(default["model"].feature_importances_, index=names)
    perm_mean, perm_std = pd.Series(perm.importances_mean, index=names), pd.Series(perm.importances_std, index=names)
    table = pd.DataFrame({"impurity": impurity, "permutation (log loss increase)": perm_mean, "perm std": perm_std})
    print(f"(permutation importance took {time.time() - started:.0f} s)")
    print("Top 10 by impurity:");  print(table.sort_values("impurity", ascending=False).head(10).round(4).to_string())
    print("Top 10 by permutation:"); print(table.sort_values("permutation (log loss increase)", ascending=False).head(10).round(4).to_string())
    print(f"Features whose shuffling does NOT hurt (permutation <= 0): {int((perm_mean <= 0).sum())} of {len(names)}")
    plot_importance(impurity, perm_mean, perm_std)

    section("Partial dependence on elo_diff")
    grid_values = np.linspace(fit_rows["elo_diff"].quantile(0.05), fit_rows["elo_diff"].quantile(0.95), 25)
    curves = {"Random Forest (FULL_v2, default)": partial_dependence_curve(default["model"], default_cfg["features"], fit_rows, grid_values)}
    if final_cfg["features"] != default_cfg["features"]:
        curves["Random Forest (chosen features)"] = partial_dependence_curve(final["model"], final_cfg["features"], fit_rows, grid_values)
    from logreg import fit_config, make_config
    elo2_cfg = make_config(sets["ELO2"], json.loads((ROOT / "models" / "logreg_config.json").read_text())["C"])
    elo2_model, _ = fit_config(elo2_cfg, fit_rows, FIT_END)
    curves["ELO2 Logistic Regression"] = partial_dependence_curve(elo2_model, sets["ELO2"], fit_rows, grid_values)
    for name, curve in curves.items():
        print(f"  {name}: P(home) at elo_diff {grid_values[0]:.0f}: {curve[0]:.3f}, at 0: "
              f"{np.interp(0, grid_values, curve):.3f}, at {grid_values[-1]:.0f}: {curve[-1]:.3f}")
    plot_partial_dependence(grid_values, curves, fit_rows["elo_diff"].to_numpy())

    section("Tree views")
    gini_example(fit_rows, default_cfg["features"])
    tree = plot_tree_depth3(fit_rows, default_cfg["features"])
    print("Depth-3 tree, root split:", default_cfg["features"][tree.tree_.feature[0]], "<=", round(float(tree.tree_.threshold[0]), 2))

    section("Validation predictions saved for Stage 6")
    predictions = pd.DataFrame({"match_id": val_rows["match_id"], "result": y_val})
    for prefix, proba in [("rf_default", default["val_proba"]), ("rf_chosen", final["val_proba"]),
                          ("elo2_primary", others["ELO2 Logistic Regression"]),
                          ("full_reference", others["FULL reference Logistic Regression"]),
                          ("class_freq", others["Class frequencies"]), ("bookmaker", others["Bookmaker"])]:
        for column, values in zip(PROBA_COLUMNS, proba.T):
            predictions[f"{prefix}_{column}"] = values
    predictions.to_csv(VAL_PREDICTIONS_CSV, index=False)
    print(f"Saved {VAL_PREDICTIONS_CSV.relative_to(ROOT)} ({len(predictions)} matches)")

    section("Final Random Forests: refit on 2015-16 to 2024-25 (NOT scored on the test season)")
    train_all = get_slice("train_all", all_training)
    save_rf(final_cfg, train_all, MODEL_PATH, CONFIG_PATH, "primary: configuration chosen by the noise rule")
    save_rf(default_cfg, train_all, REFERENCE_MODEL_PATH, REFERENCE_CONFIG_PATH,
            "reference row for Stage 6: the default configuration (FULL_v2)")
    print("Neither model was scored on the test season.")


if __name__ == "__main__":
    main()
