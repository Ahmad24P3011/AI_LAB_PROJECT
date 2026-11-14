"""Stage 5a: baselines and Logistic Regression (the ORIGINAL analysis, v1 feature sets).

NOTE: after review, the primary model and the feature sets were amended (post-hoc, driven by a drift
diagnostic). The amendment lives in src/amendment_5a.py; this file keeps the v1 analysis reproducible
(it reads feature_sets_v1.json) and no longer saves a final model.

Every choice is made on the VALIDATION seasons (2023-24, 2024-25) with models fitted on
2015-16 to 2022-23. The 2025-26 test season is never loaded (see modeling_utils.py).
Run:  python src/modeling_utils.py   (once, writes the frozen feature sets)
      python src/logreg.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from modeling_utils import (CLASS_CODES, CLASS_NAMES, FIT_SEASONS, PROBA_COLUMNS, SEED,
                            TRAIN_SEASONS, VAL_SEASONS, bookmaker_probabilities, evaluate,
                            fit_model, get_slice, load_feature_sets, load_features,
                            make_logreg, paired_bootstrap, per_match_log_loss, predict_proba,
                            recency_weights, window_seasons)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
RESULTS_CSV = ROOT / "reports" / "ablations_logreg.csv"
VAL_PREDICTIONS_CSV = ROOT / "reports" / "val_predictions_5a.csv"

C_GRID = [0.001, 0.01, 0.1, 1, 10]
FIT_END, FINAL_END = FIT_SEASONS[-1], TRAIN_SEASONS[-1]
BLUE, ORANGE, GREY, VERMILION, GREEN = "#0072B2", "#E69F00", "#7F7F7F", "#D55E00", "#009E73"


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 150, "axes.grid.axis": "y"})


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print(f"[saved reports/figures/{name}]")


# ------------------------------------------------------- configurations
def make_config(features, C=None, window=None, recency=False, include_warmup=True,
                drop_2020_21=False):
    return {"features": list(features), "C": C, "window": window, "recency": recency,
            "include_warmup": include_warmup, "drop_2020_21": drop_2020_21}


def training_rows(source_rows, cfg, end_season):
    """Apply the window (counted back from end_season) and the row filters of a config."""
    rows = source_rows[source_rows["season"].isin(window_seasons(cfg["window"], end_season))]
    if not cfg["include_warmup"]:
        rows = rows[rows["warmup"] == 0]
    if cfg["drop_2020_21"]:
        rows = rows[rows["season"] != "2020-21"]
    return rows.reset_index(drop=True)


def fit_config(cfg, source_rows, end_season, class_weight=None):
    rows = training_rows(source_rows, cfg, end_season)
    weights = recency_weights(rows, end_season) if cfg["recency"] else None
    model = fit_model(make_logreg(cfg["C"], class_weight), rows, cfg["features"], weights)
    return model, rows


def score_config(cfg, fit_rows, val_rows, class_weight=None):
    """Fit on the fit seasons, score on the validation seasons."""
    model, rows = fit_config(cfg, fit_rows, FIT_END, class_weight)
    y_val = val_rows["result"].to_numpy()
    val_proba = predict_proba(model, val_rows, cfg["features"])
    train_proba = predict_proba(model, rows, cfg["features"])
    return {"model": model, "rows": rows, "val_proba": val_proba,
            "metrics": evaluate(y_val, val_proba),
            "val_losses": per_match_log_loss(y_val, val_proba),
            "train_loss": float(per_match_log_loss(rows["result"], train_proba).mean())}


def tune_C(base_cfg, fit_rows, val_rows):
    """Try each C; pick the one with the lowest VALIDATION log loss."""
    records = []
    for C in C_GRID:
        outcome = score_config({**base_cfg, "C": C}, fit_rows, val_rows)
        records.append({"C": C, "train_log_loss": outcome["train_loss"],
                        "val_log_loss": outcome["metrics"]["log_loss"],
                        "val_accuracy": outcome["metrics"]["accuracy"]})
    table = pd.DataFrame(records)
    return table, float(table.loc[table["val_log_loss"].idxmin(), "C"])


# ---------------------------------------------------------- baselines
def run_baselines(fit_rows, val_rows):
    y = val_rows["result"].to_numpy()
    shares = fit_rows["result"].value_counts(normalize=True).reindex(CLASS_CODES).to_numpy()
    book_proba, margin = bookmaker_probabilities(val_rows["match_id"])
    results = {
        "Always predict Home win": evaluate(y, pred=np.zeros(len(y), dtype=int)),
        "Class frequencies (knows nothing)": evaluate(y, np.tile(shares, (len(y), 1))),
        "Bookmaker (Bet365, margin removed)": evaluate(y, book_proba),
    }
    return results, shares, book_proba, margin


def metrics_row(metrics):
    return {"accuracy": metrics["accuracy"], "log loss": metrics["log_loss"],
            "macro-F1": metrics["macro_f1"], "Brier": metrics["brier"],
            "draws predicted": int(metrics["predicted_counts"][1])}


def print_metrics_block(name, metrics):
    print(f"\n{name}")
    print(f"  accuracy {metrics['accuracy']:.4f} | log loss {metrics['log_loss']:.4f} | "
          f"macro-F1 {metrics['macro_f1']:.4f} | Brier {metrics['brier']:.4f}")
    print("  confusion matrix (rows = what happened, columns = what was predicted):")
    print(pd.DataFrame(metrics["confusion"], index=CLASS_NAMES, columns=CLASS_NAMES).to_string())
    print("  per class:")
    print(metrics["per_class"].round(3).to_string())


# -------------------------------------------------------- experiments
def compare(default_losses, alt_losses):
    return paired_bootstrap(default_losses, alt_losses)   # positive = alternative is better


def decide(options):
    """The noise rule. options: dicts with label, rank (lower = simpler), val_loss,
    ci_low, ci_high; the default has ci_low = ci_high = None.
    1) If some alternative is clearly better (interval of improvement above 0), take the best one.
    2) Otherwise take the SIMPLEST option that is not clearly worse than the default."""
    alternatives = [o for o in options if o["ci_low"] is not None]
    better = [o for o in alternatives if o["ci_low"] > 0]
    if better:
        return min(better, key=lambda o: o["val_loss"]), "clearly better than the default"
    candidates = [o for o in options if o["ci_low"] is None or o["ci_high"] >= 0]
    choice = min(candidates, key=lambda o: (o["rank"], o["val_loss"]))
    if choice["ci_low"] is None:
        return choice, "no alternative is clearly better: keep the default"
    return choice, "cannot tell it from the default, so the simpler option is kept"


def make_record(experiment, label, cfg, outcome, stats, rank=np.nan, note=""):
    """One row of reports/ablations_logreg.csv."""
    return {"experiment": experiment, "option": label, "n_features": len(cfg["features"]),
            "C": cfg["C"], "window_seasons": cfg["window"] or len(FIT_SEASONS),
            "recency_weights": cfg["recency"], "include_warmup": cfg["include_warmup"],
            "drop_2020_21_rows": cfg["drop_2020_21"], "n_train_rows": len(outcome["rows"]),
            "train_log_loss": outcome["train_loss"], "val_log_loss": outcome["metrics"]["log_loss"],
            "val_accuracy": outcome["metrics"]["accuracy"], "val_macro_f1": outcome["metrics"]["macro_f1"],
            "improvement_vs_default": stats["mean_diff"], "ci_low": stats["ci_low"],
            "ci_high": stats["ci_high"], "rank": rank, "note": note, "cfg": cfg}


def run_experiments(sets, default_cfg, default_outcome, fit_rows, val_rows):
    """ONE change at a time from the default. No grid over combinations."""
    section("Experiments (one change at a time, validation seasons)")
    full = sets["FULL"]
    default_C = default_cfg["C"]
    default_losses = default_outcome["val_losses"]
    records, decisions = [], {}

    def add(experiment, label, cfg, outcome, rank, note=""):
        is_default = label.endswith("(default)")
        stats = ({"mean_diff": np.nan, "ci_low": None, "ci_high": None} if is_default
                 else compare(default_losses, outcome["val_losses"]))
        records.append(make_record(experiment, label, cfg, outcome, stats, rank, note))
        return {"label": label, "rank": rank, "val_loss": outcome["metrics"]["log_loss"],
                "ci_low": stats["ci_low"], "ci_high": stats["ci_high"], "cfg": cfg}

    def run_experiment(name, options):
        entries = [add(name, label, cfg, outcome, rank) for label, cfg, outcome, rank in options]
        choice, reason = decide(entries)
        decisions[name] = {"choice": choice, "reason": reason}
        sub = pd.DataFrame([r for r in records if r["experiment"] == name])
        view = sub[["option", "n_features", "C", "n_train_rows", "val_log_loss", "val_accuracy",
                    "improvement_vs_default", "ci_low", "ci_high"]].copy()
        print(f"\n--- {name}")
        print(view.round(4).to_string(index=False))
        print(f"  -> {choice['label']}: {reason}")

    # a) feature set (C tuned for each set)
    options = [("FULL (default)", default_cfg, default_outcome, 2)]
    tuned = {}
    for name in ["REDUCED", "SMALL"]:
        base = make_config(sets[name])
        table, best_C = tune_C(base, fit_rows, val_rows)
        cfg = {**base, "C": best_C}
        tuned[name] = (cfg, table)
        options.append((name, cfg, score_config(cfg, fit_rows, val_rows), 1 if name == "REDUCED" else 0))
    run_experiment("a. feature set", options)

    # b) head-to-head
    cfg = make_config([f for f in full if not f.startswith("h2h_")], default_C)
    run_experiment("b. head-to-head", [
        ("with h2h (default)", default_cfg, default_outcome, 1),
        ("without h2h", cfg, score_config(cfg, fit_rows, val_rows), 0)])

    # c) season_no_fans
    cfg_exclude = make_config([f for f in full if f != "season_no_fans"], default_C)
    cfg_drop = make_config(full, default_C, drop_2020_21=True)
    run_experiment("c. season_no_fans", [
        ("include (default)", default_cfg, default_outcome, 1),
        ("exclude the feature", cfg_exclude, score_config(cfg_exclude, fit_rows, val_rows), 0),
        ("keep it, drop 2020-21 rows", cfg_drop, score_config(cfg_drop, fit_rows, val_rows), 2)])

    # d) training window (counted back from 2022-23)
    options = [("all 8 seasons (default)", default_cfg, default_outcome, 0)]
    for n in (3, 5):
        cfg = make_config(full, default_C, window=n)
        options.append((f"last {n} seasons", cfg, score_config(cfg, fit_rows, val_rows), 1))
    cfg = make_config(full, default_C, recency=True)
    options.append(("all 8, recency weights", cfg, score_config(cfg, fit_rows, val_rows), 1))
    run_experiment("d. training window", options)

    # e) warm-up rows
    cfg = make_config(full, default_C, include_warmup=False)
    run_experiment("e. warm-up rows", [
        ("include (default)", default_cfg, default_outcome, 0),
        ("exclude from training", cfg, score_config(cfg, fit_rows, val_rows), 1)])
    return records, decisions, tuned


def run_diagnostics(sets, default_cfg, default, fit_rows, val_rows):
    """NOT used for any decision. Explains a surprise: league_avg_goals_prev380 is far
    outside its training range in the validation seasons."""
    section("Diagnostic (NOT used for any decision): league_avg_goals_prev380")
    col = "league_avg_goals_prev380"
    z = (val_rows[col] - fit_rows[col].mean()) / fit_rows[col].std()
    print(f"Fit seasons: mean {fit_rows[col].mean():.3f}, std {fit_rows[col].std():.3f} | "
          f"validation: mean {val_rows[col].mean():.3f} ({val_rows.groupby('season')[col].mean().round(3).to_dict()})")
    print(f"Validation values in training standard deviations: min {z.min():.1f}, "
          f"median {z.median():.1f}, max {z.max():.1f}  (a value above 3 is far outside the training range)")
    full = sets["FULL"]
    no_la = [f for f in full if f != col]
    records = []

    def report(label, cfg, outcome, reference, note):
        stats = compare(reference["val_losses"], outcome["val_losses"])
        print(f"  {label}: validation log loss {outcome['metrics']['log_loss']:.4f}, improvement vs "
              f"its reference {stats['mean_diff']:+.4f} [{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]")
        records.append(make_record("diagnostic (not used for selection)", label, cfg, outcome, stats, note=note))

    without = score_config(make_config(no_la, default_cfg["C"]), fit_rows, val_rows)
    report("FULL without league_avg (reference: FULL default)", make_config(no_la, default_cfg["C"]),
           without, default, "reference = default FULL")
    no_warm_cfg = make_config(no_la, default_cfg["C"], include_warmup=False)
    report("FULL without league_avg, warm-up rows excluded (reference: same without warm-up exclusion)",
           no_warm_cfg, score_config(no_warm_cfg, fit_rows, val_rows), without,
           "the warm-up effect of experiment e disappears without league_avg")
    base = make_config([f for f in sets["SMALL"] if f != col])
    table, best_C = tune_C(base, fit_rows, val_rows)
    cfg = {**base, "C": best_C}
    report("SMALL without league_avg: elo_diff + abs_elo_diff only (reference: FULL default)", cfg,
           score_config(cfg, fit_rows, val_rows), default, "reference = default FULL")
    return records


def assemble_final_config(sets, default_cfg, decisions, tuned):
    """Combine the per-experiment decisions into ONE configuration."""
    feature_choice = decisions["a. feature set"]["choice"]["label"]
    features = list(sets["FULL"] if feature_choice.startswith("FULL") else sets[feature_choice])
    C = default_cfg["C"] if feature_choice.startswith("FULL") else tuned[feature_choice][0]["C"]
    if decisions["b. head-to-head"]["choice"]["label"] == "without h2h":
        features = [f for f in features if not f.startswith("h2h_")]
    nofans = decisions["c. season_no_fans"]["choice"]["label"]
    if nofans == "exclude the feature":
        features = [f for f in features if f != "season_no_fans"]
    window_label = decisions["d. training window"]["choice"]["label"]
    window = {"last 3 seasons": 3, "last 5 seasons": 5}.get(window_label)
    return make_config(features, C, window=window, recency=window_label.endswith("recency weights"),
                       include_warmup=decisions["e. warm-up rows"]["choice"]["label"].startswith("include"),
                       drop_2020_21=nofans == "keep it, drop 2020-21 rows")


# ------------------------------------------------------------- figures
def plot_validation_curve(curve, best_C, class_freq_loss, bookmaker_loss):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(curve["C"], curve["train_log_loss"], "o-", color=BLUE, label="Training seasons (2015-16 to 2022-23)")
    ax.plot(curve["C"], curve["val_log_loss"], "s-", color=VERMILION, label="Validation seasons (2023-24, 2024-25)")
    ax.axhline(class_freq_loss, color=GREY, linestyle=":", label=f"Class frequencies ({class_freq_loss:.3f})")
    ax.axhline(bookmaker_loss, color=GREEN, linestyle="--", label=f"Bookmaker ({bookmaker_loss:.3f})")
    ax.axvline(best_C, color="black", linewidth=0.8, alpha=0.5)
    ax.annotate(f"best C = {best_C:g}", xy=(best_C, curve["val_log_loss"].min()),
                xytext=(best_C * 2.5, curve["val_log_loss"].min() + 0.012),
                arrowprops={"arrowstyle": "->", "color": "black"})
    ax.set_xscale("log")
    ax.set_xlabel("C (larger = weaker regularization = more flexible model), log scale")
    ax.set_ylabel("Log loss (lower is better)")
    ax.set_title("Logistic Regression: training vs validation log loss for each C")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(axis="x", visible=False)
    save_figure(fig, "15_logreg_validation_curve.png")


def plot_confusion(unweighted, balanced):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (title, metrics) in zip(axes, [("Main model (unweighted)", unweighted),
                                           ("class_weight = 'balanced'", balanced)]):
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
    fig.suptitle("Validation matches: confusion matrices (cell = count and % of the true row)",
                 fontweight="bold")
    save_figure(fig, "16_confusion_matrix.png")


def calibration_points(p_home, home_won, bins=5):
    """Group matches by predicted home-win probability (equal-sized groups)."""
    groups = pd.qcut(pd.Series(p_home), bins, duplicates="drop")
    table = pd.DataFrame({"p": p_home, "won": home_won, "g": groups}).groupby("g", observed=True).agg(
        predicted=("p", "mean"), actual=("won", "mean"), n=("won", "size"))
    table["err"] = 1.96 * np.sqrt(table["actual"] * (1 - table["actual"]) / table["n"])
    return table


def plot_calibration(lr_proba, book_proba, y):
    home_won = (y == 0).astype(float)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Perfect calibration")
    for name, proba, color, marker, shift in [("Logistic Regression", lr_proba, BLUE, "o", -0.004),
                                              ("Bookmaker", book_proba, VERMILION, "s", 0.004)]:
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
    save_figure(fig, "17_calibration.png")


def bootstrap_coefficients(cfg, fit_rows, n_boot=30):
    """Refit the model on resampled training rows to see how stable each coefficient is."""
    rng = np.random.default_rng(SEED)
    rows = training_rows(fit_rows, cfg, FIT_END)
    coefs = []
    for _ in range(n_boot):
        sample = rows.iloc[rng.integers(0, len(rows), len(rows))]
        model = fit_model(make_logreg(cfg["C"]), sample, cfg["features"])
        coefs.append(model.named_steps["model"].coef_[0])   # the Home win row
    return np.std(coefs, axis=0)


def plot_coefficients(small_model, small_cfg, full_model, full_cfg, full_sd):
    small = pd.DataFrame(small_model.named_steps["model"].coef_.T, index=small_cfg["features"],
                         columns=CLASS_NAMES)
    full_home = pd.Series(full_model.named_steps["model"].coef_[0], index=full_cfg["features"])
    order = full_home.sort_values().index
    sd = pd.Series(full_sd, index=full_cfg["features"])[order]
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.5), gridspec_kw={"width_ratios": [1, 1.7]})
    small.plot.barh(ax=axes[0], color=[BLUE, GREY, VERMILION], width=0.75)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Standardized coefficient (change in the outcome's score per +1 std of the feature)")
    axes[0].set_title("SMALL model (3 features), all three outcomes")
    axes[0].legend(loc="upper right")
    colors = [BLUE if v > 0 else VERMILION for v in full_home[order]]
    axes[1].barh(range(len(order)), full_home[order], xerr=sd, color=colors, ecolor="black", capsize=2)
    axes[1].set_yticks(range(len(order)), order, fontsize=8)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Standardized coefficient for HOME WIN (black bars: +/- 1 std over 30 refits)")
    axes[1].set_title("FULL model (34 features), Home win coefficients")
    for ax in axes:
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
    save_figure(fig, "18_coefficients.png")
    return small, full_home, sd


# ---------------------------------------------------------- worked example
def worked_example(small_model, small_cfg, val_rows, y_book):
    """Show every step for one real validation match using the 3-feature model."""
    row = val_rows[(val_rows["HomeTeam"] == "Man City") & (val_rows["AwayTeam"] == "Liverpool")
                   & (val_rows["season"] == "2023-24")].iloc[[0]]
    scaler, model = small_model.named_steps["scale"], small_model.named_steps["model"]
    raw = row[small_cfg["features"]].to_numpy()[0]
    z = (raw - scaler.mean_) / scaler.scale_
    scores = model.coef_ @ z + model.intercept_
    probabilities = np.exp(scores) / np.exp(scores).sum()
    truth = int(row["result"].iloc[0])
    print(f"\nWorked example: {row['HomeTeam'].iloc[0]} v {row['AwayTeam'].iloc[0]}, "
          f"{row['Date'].iloc[0].date()}, real result: {CLASS_NAMES[truth]} "
          f"({row['home_goals'].iloc[0]}-{row['away_goals'].iloc[0]})")
    print("  feature            raw value   training mean   training std   standardized")
    for name, r, m, s, zz in zip(small_cfg["features"], raw, scaler.mean_, scaler.scale_, z):
        print(f"  {name:24} {r:9.3f} {m:13.3f} {s:14.3f} {zz:14.3f}")
    print("  weights (rows = outcome), intercepts:")
    print(pd.DataFrame(model.coef_, index=CLASS_NAMES, columns=small_cfg["features"])
          .assign(intercept=model.intercept_).round(4).to_string())
    print(f"  scores = intercept + sum(weight x standardized value): "
          + ", ".join(f"{n} {s:.4f}" for n, s in zip(CLASS_NAMES, scores)))
    print(f"  probabilities (softmax) = e^score / sum: "
          + ", ".join(f"{n} {p:.4f}" for n, p in zip(CLASS_NAMES, probabilities)))
    print(f"  log loss of this match = -ln(probability of {CLASS_NAMES[truth]}) = "
          f"-ln({probabilities[truth]:.4f}) = {-np.log(probabilities[truth]):.4f}")
    print(f"  (bookmaker said: " + ", ".join(f"{n} {p:.3f}" for n, p in zip(CLASS_NAMES, y_book)) + ")")
    return row


# -------------------------------------------------------------------- main
def main():
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sets = load_feature_sets(version=1)   # FROZEN v1 sets: never recomputed here
    all_training = load_features()
    fit_rows = get_slice("fit", all_training)
    val_rows = get_slice("validation", all_training)
    y_val = val_rows["result"].to_numpy()
    print(f"Fit rows: {len(fit_rows)} ({FIT_SEASONS[0]} to {FIT_SEASONS[-1]}); validation rows: "
          f"{len(val_rows)} ({', '.join(VAL_SEASONS)}); test rows loaded: 0")
    print(f"Latest fit date {fit_rows['Date'].max().date()}, earliest validation date "
          f"{val_rows['Date'].min().date()}")

    section("Baselines (validation seasons)")
    baselines, class_shares, book_proba, margin = run_baselines(fit_rows, val_rows)
    print(f"Class proportions in the fit seasons: " + ", ".join(
        f"{n} {s:.3f}" for n, s in zip(CLASS_NAMES, class_shares)))
    print(f"Bookmaker margin (overround) in the validation matches: {margin * 100:.1f}%")
    table = pd.DataFrame({k: metrics_row(v) for k, v in baselines.items()}).T
    print(table.round(4).to_string())
    for name, metrics in baselines.items():
        print_metrics_block(name, metrics)

    section("Logistic Regression: tuning C (FULL features, all 8 fit seasons)")
    base = make_config(sets["FULL"])
    curve, best_C = tune_C(base, fit_rows, val_rows)
    print(curve.round(4).to_string(index=False))
    print(f"Chosen C = {best_C:g}")
    default_cfg = {**base, "C": best_C}
    default = score_config(default_cfg, fit_rows, val_rows)
    plot_validation_curve(curve, best_C, baselines["Class frequencies (knows nothing)"]["log_loss"],
                          baselines["Bookmaker (Bet365, margin removed)"]["log_loss"])

    section("Default Logistic Regression: validation results")
    print_metrics_block("Logistic Regression (FULL, default)", default["metrics"])
    balanced = score_config(default_cfg, fit_rows, val_rows, class_weight="balanced")
    print_metrics_block("Same model with class_weight='balanced'", balanced["metrics"])
    stats = paired_bootstrap(default["val_losses"], balanced["val_losses"])
    print(f"  log-loss improvement of 'balanced' vs main (positive = balanced better): "
          f"{stats['mean_diff']:+.4f}, 95% interval [{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]")
    plot_confusion(default["metrics"], balanced["metrics"])
    plot_calibration(default["val_proba"], book_proba, y_val)

    section("Comparison with the baselines")
    book_losses = per_match_log_loss(y_val, book_proba)
    freq_losses = per_match_log_loss(y_val, np.tile(class_shares, (len(y_val), 1)))
    for name, losses in [("class frequencies", freq_losses), ("bookmaker", book_losses)]:
        s = paired_bootstrap(losses, default["val_losses"])
        print(f"  Logistic Regression vs {name}: log-loss improvement {s['mean_diff']:+.4f} "
              f"[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}] (positive = Logistic Regression better)")

    records, decisions, tuned = run_experiments(sets, default_cfg, default, fit_rows, val_rows)
    diagnostic_records = run_diagnostics(sets, default_cfg, default, fit_rows, val_rows)
    final_cfg = assemble_final_config(sets, default_cfg, decisions, tuned)

    section("Confirmation of the combined configuration (one run, no grid)")
    final_outcome = score_config(final_cfg, fit_rows, val_rows)
    combined = paired_bootstrap(default["val_losses"], final_outcome["val_losses"])
    print(f"Chosen: {len(final_cfg['features'])} features, C={final_cfg['C']:g}, "
          f"window={final_cfg['window'] or 'all 8'}, recency={final_cfg['recency']}, "
          f"warmup included={final_cfg['include_warmup']}, drop 2020-21={final_cfg['drop_2020_21']}")
    print(f"  validation log loss {final_outcome['metrics']['log_loss']:.4f} "
          f"(default {default['metrics']['log_loss']:.4f}), accuracy "
          f"{final_outcome['metrics']['accuracy']:.4f} (default {default['metrics']['accuracy']:.4f})")
    print(f"  improvement vs default {combined['mean_diff']:+.4f} "
          f"[{combined['ci_low']:+.4f}, {combined['ci_high']:+.4f}]")
    reverted = combined["ci_high"] < 0
    if reverted:
        print("  The combination is clearly WORSE than the default: reverting to the default.")
        final_cfg, final_outcome = default_cfg, default
    records.append({"experiment": "final combined", "option": "chosen configuration",
                    "n_features": len(final_cfg["features"]), "C": final_cfg["C"],
                    "window_seasons": final_cfg["window"] or len(FIT_SEASONS),
                    "recency_weights": final_cfg["recency"], "include_warmup": final_cfg["include_warmup"],
                    "drop_2020_21_rows": final_cfg["drop_2020_21"], "n_train_rows": len(final_outcome["rows"]),
                    "train_log_loss": final_outcome["train_loss"],
                    "val_log_loss": final_outcome["metrics"]["log_loss"],
                    "val_accuracy": final_outcome["metrics"]["accuracy"],
                    "val_macro_f1": final_outcome["metrics"]["macro_f1"],
                    "improvement_vs_default": combined["mean_diff"], "ci_low": combined["ci_low"],
                    "ci_high": combined["ci_high"], "rank": np.nan, "note": "reverted" if reverted else "", "cfg": final_cfg})
    results = pd.DataFrame(records + diagnostic_records)
    results = results.assign(decision=results.apply(
        lambda r: ("CHOSEN" if r["experiment"] in decisions and decisions[r["experiment"]]["choice"]["label"] == r["option"]
                   else ""), axis=1)).drop(columns=["cfg", "rank"])
    results.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved {RESULTS_CSV.relative_to(ROOT)}")

    section("Coefficients")
    small_cfg = tuned["SMALL"][0]
    small_outcome = score_config(small_cfg, fit_rows, val_rows)
    full_sd = bootstrap_coefficients(default_cfg, fit_rows)
    small, full_home, sd = plot_coefficients(small_outcome["model"], small_cfg, default["model"],
                                             default_cfg, full_sd)
    print("SMALL model, standardized coefficients (rows = feature):")
    print(small.round(3).to_string())
    print("\nFULL model, Home-win coefficients (sorted), with the variation across 30 refits:")
    show = pd.DataFrame({"coef": full_home.reindex(sd.index), "sd_over_refits": sd}).round(3)
    print(pd.concat([show.head(6), show.tail(6)]).to_string())
    worked_example(small_outcome["model"], small_cfg, val_rows, book_proba[
        val_rows[(val_rows["HomeTeam"] == "Man City") & (val_rows["AwayTeam"] == "Liverpool")
                 & (val_rows["season"] == "2023-24")].index[0]])

    section("Validation predictions saved for Stage 6 comparisons")
    predictions = pd.DataFrame({"match_id": val_rows["match_id"], "result": y_val})
    for prefix, proba in [("logreg_default", default["val_proba"]), ("logreg_chosen", final_outcome["val_proba"]),
                          ("class_freq", np.tile(class_shares, (len(y_val), 1))), ("bookmaker", book_proba)]:
        for column, values in zip(PROBA_COLUMNS, proba.T):
            predictions[f"{prefix}_{column}"] = values
    predictions.to_csv(VAL_PREDICTIONS_CSV, index=False)
    print(f"Saved {VAL_PREDICTIONS_CSV.relative_to(ROOT)} ({len(predictions)} validation matches)")

    print("\n(The final model is saved by src/amendment_5a.py, after the feature-set amendment.)")


if __name__ == "__main__":
    main()
