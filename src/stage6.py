"""Stage 6: honest model comparison on the 2025-26 test season.

This is a SCORING stage, not a training stage. The saved models in models/ are used exactly as they are.
The only fitting is the refit of the two DIFF diagnostics (Random Forest and XGBoost), which happens
BEFORE the test set is loaded, with the configurations frozen in docs/decisions.md.

Three commands:
  python src/stage6.py refit-diff   refit and save the two DIFF diagnostics (training data only)
  python src/stage6.py dry-run      run the whole pipeline on the VALIDATION seasons, writing to a scratch
                                    folder (finds bugs without touching the test season)
  python src/stage6.py run          the real thing: the test season is loaded ONCE, scored once, results saved

The test-set rule: Stage 5 code can never see 2025-26 (modeling_utils.get_slice refuses it). Stage 6 has its
own loader, load_test_set(), that can run only once per process and whose single call site is in main().
modeling_utils.py is NOT changed, so the old guards stay exactly as they were.
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from features import FEATURE_COLUMNS
from features_goals import GOAL_FEATURES_CSV, REL_COLUMNS
from logreg import BLUE, GREEN, GREY, ORANGE, VERMILION, calibration_points, setup_style
from modeling_utils import (CLASS_NAMES, FEATURES_CSV, ODDS_CSV, PROBA_COLUMNS, SEED, TRAIN_SEASONS, FEATURE_SETS_V3_JSON,
                            FEATURE_SETS_V3_SHA, TestSetAccessError, add_diff_features, evaluate, get_slice,
                            load_feature_sets, per_match_log_loss, window_seasons)
from poisson_model import PoissonGoalModel, outcome_probs, poisson_deviance, score_grid, top_scorelines
from preprocess import TEST_SEASONS
from rf import fit_rf, make_rf_config, save_rf
from xgb import AUDIT, FIXED, MAX_TREES, STOP_ROUNDS, STOP_SEASONS, fit_xgb, make_xgb_config

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
PREREG_MD = REPORTS / "stage6_preregistration.md"
MODELS_BEFORE_JSON = REPORTS / "stage6_models_before.json"
RUN_LOG_JSON = REPORTS / "stage6_run_log.json"
DIFF_AUDIT_JSON = REPORTS / "xgb_diff_early_stopping_audit.json"
FINAL_END = TRAIN_SEASONS[-1]
N_BOOT, BOOT_SEED = 4000, 42          # one resampling table for every comparison
DRIFT_LIMIT = 3.0
ODDS_COLUMNS = ["B365H", "B365D", "B365A"]

# ------------------------------------------------------------------ the predictors
# kind: how the predictor is produced. file/config: the frozen artefacts (None for baselines).
PREDICTORS = [
    dict(id="lr_elo2", label="Logistic Regression ELO2", family="Logistic Regression", role="primary", kind="sk",
         file="logreg.joblib", config="logreg_config.json", note="PRE-REGISTERED PRIMARY of the project"),
    dict(id="lr_full", label="Logistic Regression FULL (reference)", family="Logistic Regression", role="reference", kind="sk",
         file="logreg_full_reference.joblib", config="logreg_full_reference_config.json", note="33 features, C=0.01"),
    dict(id="rf_elo2", label="Random Forest ELO2", family="Random Forest", role="primary", kind="sk",
         file="random_forest.joblib", config="random_forest_config.json", note="500 trees, leaf 80, depth 8"),
    dict(id="rf_full", label="Random Forest FULL_v2 (reference)", family="Random Forest", role="reference", kind="sk",
         file="random_forest_full_reference.joblib", config="random_forest_full_reference_config.json", note="33 features"),
    dict(id="xgb_elo2", label="XGBoost ELO2", family="XGBoost", role="primary", kind="sk",
         file="xgboost.joblib", config="xgboost_config.json", note="62 rounds, depth 3"),
    dict(id="xgb_full", label="XGBoost FULL_v2 (reference)", family="XGBoost", role="reference", kind="sk",
         file="xgboost_full_reference.joblib", config="xgboost_full_reference_config.json", note="33 features, 30 rounds"),
    dict(id="poisson", label="Poisson (scoreline grid)", family="Poisson", role="primary", kind="poisson",
         file="poisson_stacked.joblib", config="poisson_config.json",
         note="goals model scored as a classifier; stacked, rho 0, lambda cap 3.5"),
    dict(id="diff_rf", label="DIFF Random Forest (diagnostic)", family="Random Forest", role="diagnostic", kind="sk",
         file="diff_rf.joblib", config="diff_rf_config.json", note="post-hoc diagnostic, not a competing primary", diff=True),
    dict(id="diff_xgb", label="DIFF XGBoost (diagnostic)", family="XGBoost", role="diagnostic", kind="sk",
         file="diff_xgb.joblib", config="diff_xgb_config.json", note="post-hoc diagnostic, not a competing primary", diff=True),
    dict(id="always_home", label="Always predict home win", family="Baseline", role="baseline", kind="home",
         file=None, config=None, note="hard predictions only: log loss is undefined"),
    dict(id="class_freq", label="Class frequencies (training)", family="Baseline", role="baseline", kind="freq",
         file=None, config=None, note="2015-16 to 2024-25 class frequencies for every match"),
    dict(id="bookmaker", label="Bookmaker (Bet365)", family="Baseline", role="baseline", kind="book",
         file=None, config=None, note="odds converted to probabilities, margin removed; benchmark, never a feature"),
]
BY_ID = {p["id"]: p for p in PREDICTORS}
PRIMARY_ID = "lr_elo2"
COMPETITORS = ["lr_full", "rf_elo2", "rf_full", "xgb_elo2", "xgb_full", "poisson", "diff_rf", "diff_xgb"]   # pre-registration reading 1
PAIRS_ELO2_FULL = [("lr_elo2", "lr_full"), ("rf_elo2", "rf_full"), ("xgb_elo2", "xgb_full")]
PAIRS_ELO2_DIFF = [("rf_elo2", "diff_rf"), ("xgb_elo2", "diff_xgb")]


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def file_sha(path, n=64):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:n]


# ------------------------------------------------------------------ the test-set loader (only here)
TEST_LOADS = {"count": 0, "seasons": None}      # the counter that check_stage6.py looks at


def load_test_set(seasons):
    """The ONLY place in the project that reads test rows. It works once per process: a second call raises
    before reading anything. `seasons` is TEST_SEASONS for the real run; the dry run passes validation seasons.
    Returns one table: features.csv columns + the level-neutral goal features + the raw Bet365 odds."""
    if TEST_LOADS["count"] >= 1:
        raise TestSetAccessError("the test set may be loaded only once")
    TEST_LOADS["count"] += 1
    TEST_LOADS["seasons"] = list(seasons)
    features = pd.concat([c[c["season"].isin(seasons)] for c in pd.read_csv(FEATURES_CSV, parse_dates=["Date"], chunksize=500)],
                         ignore_index=True)
    goals = pd.concat([c[c["season"].isin(seasons)] for c in pd.read_csv(GOAL_FEATURES_CSV, parse_dates=["Date"], chunksize=500)],
                      ignore_index=True)
    assert (features["match_id"].to_numpy() == goals["match_id"].to_numpy()).all()
    new_goal_columns = [c for c in goals.columns if c not in features.columns]
    table = pd.concat([features, goals[new_goal_columns]], axis=1)
    ids = set(table["match_id"])
    odds = pd.concat([c[c["match_id"].isin(ids)] for c in pd.read_csv(ODDS_CSV, usecols=["match_id"] + ODDS_COLUMNS, chunksize=500)])
    table = table.merge(odds, on="match_id", how="left", validate="one_to_one")
    return table.sort_values(["Date", "match_id"]).reset_index(drop=True)


def get_test_slice(table):
    """The Stage 6 counterpart of get_slice('test'): allowed here only, and only for the seasons loaded by load_test_set."""
    assert TEST_LOADS["count"] == 1 and set(table["season"]) <= set(TEST_LOADS["seasons"]), "not a table from load_test_set"
    return table.reset_index(drop=True)


# ------------------------------------------------------------------ 0. refit the two DIFF diagnostics
def hash_models_dir():
    return {p.name: file_sha(p) for p in sorted(MODELS.iterdir()) if p.is_file()}


def refit_diff():
    """Refit DIFF Random Forest and DIFF XGBoost on all 10 training seasons with the frozen configurations.
    Training rows only: get_slice('train_all') refuses test rows. Nothing else in models/ is touched."""
    before = hash_models_dir()
    assert not any(n.startswith("diff_") for n in before), "DIFF files already exist: refusing to overwrite them"
    MODELS_BEFORE_JSON.write_text(json.dumps(before, indent=2))
    section("Refit the DIFF diagnostics (training rows only; the test set is not loaded)")
    sets = load_feature_sets(3)
    recorded = FEATURE_SETS_V3_SHA.read_text().split()[0]
    assert hashlib.sha256(FEATURE_SETS_V3_JSON.read_bytes()).hexdigest() == recorded, "feature_sets_v3.json changed"
    diff_features = sets["DIFF"]
    assert len(diff_features) == 16
    train_all = add_diff_features(get_slice("train_all"))
    print(f"Training rows: {len(train_all)} ({train_all['season'].min()} to {train_all['season'].max()}); DIFF features: {len(diff_features)}")

    rf_cfg = make_rf_config(diff_features, min_samples_leaf=80, max_features=0.5, max_depth=8)
    save_rf(rf_cfg, train_all, MODELS / "diff_rf.joblib", MODELS / "diff_rf_config.json",
            "DIAGNOSTIC (not a competing primary): DIFF features, configuration frozen in docs/decisions.md before any test result")

    xgb_cfg = make_xgb_config(diff_features, max_depth=3, learning_rate=0.1, min_child_weight=20)
    AUDIT.clear()
    model, rows, n_trees, _ = fit_xgb(xgb_cfg, train_all, FINAL_END, "final_refit_diff")
    joblib.dump(model, MODELS / "diff_xgb.joblib", compress=3)
    model.save_model(MODELS / "diff_xgb.ubj")
    (MODELS / "diff_xgb_config.json").write_text(json.dumps({
        "status": "DIAGNOSTIC (not a competing primary): DIFF features, configuration frozen in docs/decisions.md before any test result",
        "features": diff_features, "n_trees": n_trees, "max_depth": 3, "learning_rate": 0.1, "min_child_weight": 20, **FIXED,
        "objective": "multi:softprob", "num_class": 3, "tree_method": "hist", "random_state": SEED, "n_jobs": 1,
        "monotone_constraint_on_elo_diff": False, "window_seasons": window_seasons(None, FINAL_END), "recency_weights": False,
        "include_warmup_rows": True, "drop_2020_21_rows": False, "class_order": dict(zip([0, 1, 2], CLASS_NAMES)),
        "n_training_rows": len(rows), "scaling": "none (trees do not need it)",
        "early_stopping": f"number of trees chosen by early stopping (rounds {STOP_ROUNDS}, cap {MAX_TREES}), fitting on 2015-16 to 2022-23 "
                          f"and stopping on the last {STOP_SEASONS} training seasons (2023-24, 2024-25); then refit on all 10 seasons",
        "feature_sets_file": "data/processed/feature_sets_v3.json (DIFF)", "xgboost_version": xgboost.__version__}, indent=2))
    DIFF_AUDIT_JSON.write_text(json.dumps(AUDIT, indent=1))
    entry = AUDIT[-1]
    assert entry["purpose"] == "final_refit_diff" and entry["inner_stop_seasons"] == ["2023-24", "2024-25"]
    assert not set(entry["inner_fit_seasons"]) & set(TEST_SEASONS) and not set(entry["inner_stop_seasons"]) & set(TEST_SEASONS)
    print(f"DIFF XGBoost: early stopping fit on {entry['inner_fit_seasons'][0]} to {entry['inner_fit_seasons'][-1]}, stopped on "
          f"{entry['inner_stop_seasons']} (last fit date {entry['inner_fit_last_date']}, first stop date {entry['inner_stop_first_date']}) "
          f"-> {n_trees} rounds; refit on {len(rows)} rows. Audit: {DIFF_AUDIT_JSON.relative_to(ROOT)}")
    after = hash_models_dir()
    assert all(after[n] == h for n, h in before.items()), "an existing model file changed"
    print(f"New model files: {sorted(set(after) - set(before))}; the {len(before)} earlier files are byte-identical.")


# ------------------------------------------------------------------ 1. loading and fingerprints
def load_frozen_models():
    """Load every saved model as it is, print a one-line fingerprint, and verify that the feature list of the
    fitted model equals the one in its config. Any mismatch stops the stage."""
    models, configs = {}, {}
    for p in PREDICTORS:
        if p["file"] is None:
            continue
        path, config_path = MODELS / p["file"], MODELS / p["config"]
        config = json.loads(config_path.read_text())
        if p["kind"] == "poisson":
            model = PoissonGoalModel.load(config_path)
            features = model.required_columns
            assert config["required_columns"] == features and model.lambda_cap == 3.5 and model.rho == 0.0
            summary = f"stacked, level {model.spec['level']}, league {model.spec['league']}, rho {model.rho}, cap {model.lambda_cap}"
        else:
            model = joblib.load(path)
            features = config["features"]
            assert list(model.feature_names_in_) == list(features), f"{p['id']}: feature list differs from its config"
            assert list(model.classes_) == [0, 1, 2]
            summary = {"lr_elo2": f"C={config.get('C')}", "lr_full": f"C={config.get('C')}"}.get(
                p["id"], f"trees/rounds={config.get('n_estimators', config.get('n_trees'))}, depth={config.get('max_depth')}")
        models[p["id"]], configs[p["id"]] = model, config
        print(f"  {p['id']:10} {p['file']:34} {path.stat().st_size / 1024:8.1f} KB  config {file_sha(config_path, 10)}  "
              f"{len(features):2d} features  {summary}")
    return models, configs


# ------------------------------------------------------------------ 2. predictions
def bookmaker_baseline(test):
    """Bet365 odds -> probabilities for the test matches. 1/odds adds up to more than 1 (the margin); dividing by the sum
    removes it. This is the ONLY use of odds in the project besides the Stage 5 validation benchmark."""
    raw = 1 / test[ODDS_COLUMNS].to_numpy(float)
    if np.isnan(raw).any():
        bad = test.loc[np.isnan(raw).any(axis=1), ["match_id", "HomeTeam", "AwayTeam"]]
        raise SystemExit(f"odds are missing for {len(bad)} test matches:\n{bad}")
    return raw / raw.sum(axis=1, keepdims=True), float((raw.sum(axis=1) - 1).mean())


def poisson_outputs(model, test):
    """Goal-level and outcome-level outputs of the Poisson model on the test matches (capped rates, as deployed)."""
    lh, la = model.predict(test)
    raw_h, raw_a = model.predict_uncapped(test)
    grid, _ = score_grid(lh, la, model.rho)
    hg, ag, y = test["home_goals"].to_numpy(), test["away_goals"].to_numpy(), test["result"].to_numpy()
    flat = grid.reshape(len(grid), -1).argmax(axis=1)
    top_h, top_a = flat // grid.shape[2], flat % grid.shape[2]
    top_outcome = np.where(top_h > top_a, 0, np.where(top_h == top_a, 1, 2))
    proba = outcome_probs(grid)
    return {"lh": lh, "la": la, "grid": grid, "proba": proba, "top_h": top_h, "top_a": top_a,
            "n_capped": int(((raw_h > model.lambda_cap) | (raw_a > model.lambda_cap)).sum()),
            "mae_home": float(np.abs(lh - hg).mean()), "mae_away": float(np.abs(la - ag).mean()),
            "mae_total": float(np.abs(lh + la - (hg + ag)).mean()),
            "deviance": float((poisson_deviance(hg, lh) + poisson_deviance(ag, la)).mean() / 2),
            "exact_score_accuracy": float(((top_h == hg) & (top_a == ag)).mean()),
            "rounded_score_accuracy": float(((np.round(lh) == hg) & (np.round(la) == ag)).mean()),
            "outcome_of_top_scoreline_accuracy": float((top_outcome == y).mean()),
            "outcome_of_summed_grid_accuracy": float((proba.argmax(axis=1) == y).mean()),
            "draws_by_top_scoreline": int((top_outcome == 1).sum()),
            "real_mean_total_goals": float((hg + ag).mean()), "pred_mean_total_goals": float((lh + la).mean())}


def predict_all(test, models, train_class_freq):
    """One (n x 3) probability table per predictor (or hard predictions for always-home). No model is refit."""
    with_diff = add_diff_features(test)          # same-row arithmetic: home_X - away_X
    probabilities, extra = {}, {}
    for p in PREDICTORS:
        if p["kind"] == "sk":
            table = with_diff if p.get("diff") else test
            features = list(models[p["id"]].feature_names_in_)
            probabilities[p["id"]] = models[p["id"]].predict_proba(table[features])
        elif p["kind"] == "poisson":
            extra["poisson"] = poisson_outputs(models["poisson"], test)
            probabilities["poisson"] = extra["poisson"]["proba"]
        elif p["kind"] == "freq":
            probabilities["class_freq"] = np.tile(train_class_freq, (len(test), 1))
        elif p["kind"] == "book":
            probabilities["bookmaker"], extra["overround"] = bookmaker_baseline(test)
    return probabilities, extra


def results_table(y, probabilities, n):
    rows = []
    for p in PREDICTORS:
        if p["kind"] == "home":
            m = evaluate(y, pred=np.zeros(len(y), dtype=int))
        else:
            m = evaluate(y, proba=probabilities[p["id"]])
        f1 = m["per_class"]["f1"]
        rows.append({"predictor": p["label"], "family": p["family"], "role": p["role"], "accuracy": m["accuracy"],
                     "log_loss": m["log_loss"], "macro_f1": m["macro_f1"], "home_f1": f1["Home win"], "draw_f1": f1["Draw"],
                     "away_f1": f1["Away win"], "draws_predicted": int(m["predicted_counts"][1]), "n_test_matches": n,
                     "notes": p["note"], "_id": p["id"], "_metrics": m})
    return rows


# ------------------------------------------------------------------ 3. bootstrap
def bootstrap_table(losses, picks):
    """Paired bootstrap on ONE shared resampling table. diff = loss(A) - loss(B): negative means A is better."""
    wanted = []                                   # (group, a, b)
    prob_ids = [i for i in BY_ID if i != "always_home"]
    for m in prob_ids:
        if m != PRIMARY_ID:
            wanted.append(("vs ELO2 LR (primary)", m, PRIMARY_ID))
    for m in prob_ids:
        if m != "bookmaker":
            wanted.append(("vs bookmaker", m, "bookmaker"))
    wanted += [("ELO2 primary vs FULL reference", a, b) for a, b in PAIRS_ELO2_FULL]
    wanted += [("ELO2 primary vs DIFF diagnostic", a, b) for a, b in PAIRS_ELO2_DIFF]
    rows, seen = [], set()
    for group, a, b in wanted:
        diff = losses[a] - losses[b]
        means = diff[picks].mean(axis=1)
        low, high = np.percentile(means, [2.5, 97.5])
        verdict = "A better" if high < 0 else ("A worse" if low > 0 else "cannot tell")
        rows.append({"group": group, "model_a": BY_ID[a]["label"], "model_b": BY_ID[b]["label"], "a_id": a, "b_id": b,
                     "mean_diff_a_minus_b": float(diff.mean()), "ci_low": float(low), "ci_high": float(high),
                     "verdict": verdict, "n_matches": len(diff), "n_boot": len(picks), "seed": BOOT_SEED})
        seen.add((a, b))
    return pd.DataFrame(rows)


def decide_winner(boot):
    """The pre-registered rule: ELO2 LR wins unless a COMPETITOR (reading 1) has a log-loss difference vs ELO2 LR whose
    95% interval is entirely below zero. Anything that does is 'post-hoc best on the test set, not pre-registered'."""
    block = boot[(boot["group"] == "vs ELO2 LR (primary)") & boot["a_id"].isin(COMPETITORS)]
    better = block[block["verdict"] == "A better"]
    worse = block[block["verdict"] == "A worse"]
    return {"primary_won": len(better) == 0,
            "significantly_better_than_primary": list(better["model_a"]),
            "significantly_worse_than_primary": list(worse["model_a"]),
            "cannot_tell": list(block[block["verdict"] == "cannot tell"]["model_a"])}


# ------------------------------------------------------------------ 4. periods, drift, pattern
def add_game_numbers(test):
    """Each team's game number within its season (its matchweek, as far as the data tell). No external data."""
    counts, home_no, away_no = {}, [], []
    for season, home, away in zip(test["season"], test["HomeTeam"], test["AwayTeam"]):
        counts[(season, home)] = counts.get((season, home), 0) + 1
        counts[(season, away)] = counts.get((season, away), 0) + 1
        home_no.append(counts[(season, home)])
        away_no.append(counts[(season, away)])
    out = test[["match_id", "season"]].copy()
    out["game_no"] = np.maximum(home_no, away_no)
    return out["game_no"].to_numpy(), counts


def period_table(test, losses, probabilities, y):
    game_no, counts = add_game_numbers(test)
    per_season_min = {s: min(v for (se, _), v in counts.items() if se == s) for s in set(test["season"])}
    complete = np.array([g <= per_season_min[s] for g, s in zip(game_no, test["season"])])
    blocks = pd.cut(game_no, [0, 10, 20, 30, 38], labels=["matchweeks 1-10", "11-20", "21-30", "31-38"])
    rows = []
    ids = ["lr_elo2", "rf_elo2", "xgb_elo2", "poisson", "bookmaker", "class_freq"]
    for name, mask in [("full available set", np.ones(len(test), bool)), ("completed-matchweeks slice", complete)] + \
                      [(str(b), (blocks == b)) for b in blocks.categories]:
        for i in ids:
            rows.append({"slice": name, "predictor": BY_ID[i]["label"], "n_matches": int(mask.sum()),
                         "log_loss": float(losses[i][mask].mean()), "accuracy": float((probabilities[i][mask].argmax(axis=1) == y[mask]).mean()),
                         "home_win_rate": float((y[mask] == 0).mean())})
    return pd.DataFrame(rows), int((~complete).sum()), per_season_min


def drift_table(fit_rows, test):
    """The Stage 5a drift logic: z-scores in fit-season (2015-16 to 2022-23) standard deviations. A diagnostic only."""
    mean = fit_rows[FEATURE_COLUMNS].mean()
    std = fit_rows[FEATURE_COLUMNS].std(ddof=0).replace(0, np.nan)
    z = (test[FEATURE_COLUMNS] - mean) / std
    table = pd.DataFrame({"train_mean": mean, "train_std": std, "test_mean": test[FEATURE_COLUMNS].mean(), "test_mean_z": z.mean(),
                          "max_abs_z": z.abs().max(), "share_of_rows_above_3": (z.abs() > DRIFT_LIMIT).mean(),
                          "flagged_above_3": z.abs().max() > DRIFT_LIMIT}).sort_values("max_abs_z", ascending=False)
    table.index.name = "feature"
    return table


def pattern_table(test, y, probabilities, losses):
    """Home-win rate of the season, each model's average predicted home-win probability, and log loss vs ELO2 LR."""
    rows = []
    for p in PREDICTORS:
        if p["kind"] == "home":
            continue
        rows.append({"predictor": p["label"], "real_home_win_rate": float((y == 0).mean()),
                     "mean_predicted_home_win": float(probabilities[p["id"]][:, 0].mean()), "log_loss": float(losses[p["id"]].mean()),
                     "log_loss_minus_lr_elo2": float(losses[p["id"]].mean() - losses[PRIMARY_ID].mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 5. figures
def save_fig(fig, directory, name):
    fig.tight_layout()
    fig.savefig(Path(directory) / name)
    plt.close(fig)
    print(f"[saved {Path(directory).name}/{name}]")


def plot_confusions(rows, directory):
    shown = ["lr_elo2", "rf_elo2", "xgb_elo2", "poisson", "bookmaker", "always_home"]
    by_id = {r["_id"]: r for r in rows}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.6))
    for ax, i in zip(axes.ravel(), shown):
        m = by_id[i]["_metrics"]
        sns.heatmap(m["confusion"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 12},
                    xticklabels=["Home", "Draw", "Away"], yticklabels=["Home", "Draw", "Away"])
        ax.set_title(f"{BY_ID[i]['label']}\naccuracy {m['accuracy'] * 100:.1f}%, draws predicted {m['predicted_counts'][1]}", fontsize=10.5)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion matrices on the test season (rows = what really happened)", fontweight="bold")
    save_fig(fig, directory, "37_test_confusion_matrices.png")


def plot_calibration_test(probabilities, y, directory):
    home_won = (y == 0).astype(float)
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Perfect calibration")
    series = [("lr_elo2", BLUE, "o", -0.008), ("rf_elo2", ORANGE, "D", -0.004), ("xgb_elo2", GREEN, "^", 0.0),
              ("poisson", "#CC79A7", "v", 0.004), ("bookmaker", VERMILION, "s", 0.008)]
    rows = []
    for i, color, marker, shift in series:
        table = calibration_points(probabilities[i][:, 0], home_won)
        ax.errorbar(table["predicted"] + shift, table["actual"], yerr=table["err"], fmt=marker + "-", color=color, capsize=3,
                    label=BY_ID[i]["label"])
        rows.append((BY_ID[i]["label"], float(probabilities[i][:, 0].mean()), float(home_won.mean())))
    ax.set_xlabel("Predicted probability of a home win")
    ax.set_ylabel("How often the home team really won")
    ax.set_title("Calibration on the test season (5 equal groups, 95% error bars)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="x", visible=True)
    save_fig(fig, directory, "38_test_calibration.png")
    return pd.DataFrame(rows, columns=["predictor", "mean_predicted_home_win", "actual_home_win_rate"])


def plot_log_loss_bar(rows, losses, picks, directory):
    items = [r for r in rows if not np.isnan(r["log_loss"])]
    items.sort(key=lambda r: r["log_loss"])
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    colors = {"primary": BLUE, "reference": "#9DB8D2", "diagnostic": ORANGE, "baseline": GREEN}
    for k, r in enumerate(items):
        means = losses[r["_id"]][picks].mean(axis=1)
        low, high = np.percentile(means, [2.5, 97.5])
        is_main = r["_id"] == PRIMARY_ID
        ax.barh(k, r["log_loss"], color=VERMILION if is_main else colors[r["role"]], edgecolor="black" if is_main else "none", linewidth=2,
                hatch="//" if r["role"] == "diagnostic" else None, xerr=[[r["log_loss"] - low], [high - r["log_loss"]]],
                error_kw={"capsize": 3, "ecolor": "#333333"})
        ax.text(high + 0.004, k, f"{r['log_loss']:.4f}", va="center", fontsize=9, fontweight="bold" if is_main else "normal")
    book = [r["log_loss"] for r in items if r["_id"] == "bookmaker"][0]
    ax.axvline(book, color="black", linestyle="--", linewidth=1.2, label="Bookmaker log loss")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([("* " if r["_id"] == PRIMARY_ID else "") + r["predictor"] for r in items], fontsize=9.5)
    ax.invert_yaxis()
    lo = min(r["log_loss"] for r in items)
    ax.set_xlim(lo - 0.08, max(r["log_loss"] for r in items) + 0.06)
    ax.set_xlabel("Log loss on the test season (lower is better; bars show 95% bootstrap intervals; x-axis does not start at 0)")
    ax.set_title("Test log loss by predictor (* red = pre-registered primary; hatched = diagnostic)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    save_fig(fig, directory, "39_log_loss_bar.png")


def plot_goals(out, test, directory):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, side, lam, goals, color in ((axes[0], "Home team", out["lh"], test["home_goals"].to_numpy(), BLUE),
                                        (axes[1], "Away team", out["la"], test["away_goals"].to_numpy(), ORANGE)):
        hexes = ax.hexbin(lam, goals, gridsize=(14, 7), cmap="Greys", mincnt=1, extent=(0.3, 3.6, -0.5, 7.5))
        grouped = pd.DataFrame({"lam": lam, "goals": goals, "bin": pd.qcut(lam, 8)}).groupby("bin", observed=True).mean()
        ax.plot(grouped["lam"], grouped["goals"], "o-", color=color, linewidth=2.5, label="Average actual goals in each group of 8")
        ax.plot([0.3, 3.6], [0.3, 3.6], "--", color=VERMILION, label="Perfect: prediction = average actual")
        ax.set_xlabel("Predicted expected goals (lambda)")
        ax.set_title(side)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="x", visible=True)
    axes[0].set_ylabel("Actual goals scored")
    fig.colorbar(hexes, ax=axes.tolist(), label="Number of matches", shrink=0.8)
    fig.suptitle("Poisson: predicted vs actual goals, test season (grey = how many matches)", fontweight="bold")
    fig.savefig(Path(directory) / "40_poisson_predicted_vs_actual_goals_test.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved {Path(directory).name}/40_poisson_predicted_vs_actual_goals_test.png]")


def plot_drift(drift, validation_drift, directory):
    shown = drift.sort_values("max_abs_z")
    fig, ax = plt.subplots(figsize=(9.5, 10.5))
    ax.barh(shown.index, shown["max_abs_z"], color=[VERMILION if v else BLUE for v in shown["flagged_above_3"]])
    if validation_drift is not None:
        ax.plot(validation_drift.reindex(shown.index)["max_abs_z"], shown.index, "k|", markersize=11, markeredgewidth=2)
    ax.axvline(DRIFT_LIMIT, color="black", linestyle="--", linewidth=1)
    ax.text(DRIFT_LIMIT + 0.05, 0.2, "limit 3", fontsize=9)
    ax.set_xlabel("Largest |z| of any row: distance from the fit-season mean in fit-season standard deviations")
    ax.set_title("Drift of the test features (diagnostic only; red = some row beyond 3)")
    handles = [Patch(color=VERMILION, label="Test season, some row beyond 3"), Patch(color=BLUE, label="Test season, all rows within 3")]
    if validation_drift is not None:
        handles.append(Line2D([], [], color="black", marker="|", linestyle="", markersize=11, markeredgewidth=2, label="Validation seasons (Stage 5a)"))
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    save_fig(fig, directory, "41_drift_test_bar.png")


# ------------------------------------------------------------------ main
def main(mode):
    dry = mode == "dry-run"
    out_dir = Path(sys.argv[2]) if dry else REPORTS            # dry run: scratch folder given on the command line
    fig_dir = out_dir / "figures" if dry else REPORTS / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    seasons = ["2023-24", "2024-25"] if dry else TEST_SEASONS
    section(f"Stage 6 {'DRY RUN on the VALIDATION seasons (not a result)' if dry else 'on the test season'}")

    before = json.loads(MODELS_BEFORE_JSON.read_text())
    prereg_hash = file_sha(PREREG_MD)
    print(f"Pre-registration {PREREG_MD.name}: sha256 {prereg_hash[:16]}...")

    section("Frozen models: fingerprints")
    models, _ = load_frozen_models()
    hashes_loaded = hash_models_dir()
    assert all(hashes_loaded[n] == h for n, h in before.items()), "a model file changed since before the DIFF refit"
    print(f"All {len(before)} earlier model files are byte-identical to the recorded hashes; new: {sorted(set(hashes_loaded) - set(before))}")

    section("Training information that is not test data (class frequencies, fit-season statistics)")
    train_all = get_slice("train_all")
    class_freq = np.bincount(train_all["result"].to_numpy(), minlength=3) / len(train_all)
    fit_rows = get_slice("fit", train_all)
    print(f"Class frequencies of the 10 training seasons: Home {class_freq[0]:.4f}, Draw {class_freq[1]:.4f}, Away {class_freq[2]:.4f}")

    section("Loading the test set (this call happens once)")
    test = get_test_slice(load_test_set(seasons))
    y = test["result"].to_numpy()
    n = len(test)
    print(f"Loaded {n} matches of {sorted(set(test['season']))}, {test['Date'].min().date()} to {test['Date'].max().date()}; "
          f"teams {len(set(test['HomeTeam']) | set(test['AwayTeam']))}; odds present for {int(test[ODDS_COLUMNS].notna().all(axis=1).sum())}; "
          f"actual home/draw/away = {(y == 0).sum()}/{(y == 1).sum()}/{(y == 2).sum()}")
    for p in PREDICTORS:                    # every model's feature columns must exist in the test table
        if p["kind"] != "sk":
            continue
        features = list(models[p["id"]].feature_names_in_)
        assert set(features) <= set(add_diff_features(test.head(2)).columns) and test[[f for f in features if f in test]].notna().all().all()
    assert set(models["poisson"].required_columns) <= set(test.columns) and test[models["poisson"].required_columns].notna().all().all()

    probabilities, extra = predict_all(test, models, class_freq)
    losses = {i: per_match_log_loss(y, pr) for i, pr in probabilities.items()}
    rows = results_table(y, probabilities, n)
    table = pd.DataFrame(rows).drop(columns=["_id", "_metrics"])
    table.to_csv(out_dir / "stage6_results.csv", index=False)
    section("Headline table (ordered by log loss; always-home-win has no log loss)")
    view = table.sort_values("log_loss", na_position="last")[["predictor", "role", "accuracy", "log_loss", "macro_f1", "home_f1", "draw_f1",
                                                              "away_f1", "draws_predicted"]]
    print(view.round(4).to_string(index=False))
    print(f"Bookmaker margin (average overround): {extra['overround'] * 100:.2f}%")
    for r in rows:
        if r["_id"] in ("lr_elo2", "poisson", "bookmaker"):
            print(f"\n{r['predictor']}: confusion matrix (rows = actual Home/Draw/Away):\n{r['_metrics']['confusion']}")
            print(r["_metrics"]["per_class"].round(3).to_string())

    section("Paired bootstrap (log loss, 4,000 resamples of the test matches, one shared table, seed 42)")
    picks = np.random.default_rng(BOOT_SEED).integers(0, n, size=(N_BOOT, n))
    boot = bootstrap_table(losses, picks)
    boot.drop(columns=["a_id", "b_id"]).to_csv(out_dir / "stage6_bootstrap.csv", index=False)
    show = boot.drop(columns=["a_id", "b_id", "n_matches", "n_boot", "seed"])
    for group, block in show.groupby("group", sort=False):
        print(f"\n{group}  (diff = loss(A) - loss(B); negative = A better)")
        print(block.drop(columns="group").round(4).to_string(index=False))
    verdict = decide_winner(boot)
    print(f"\nPre-registered rule: primary won = {verdict['primary_won']}; significantly better than ELO2 LR: "
          f"{verdict['significantly_better_than_primary'] or 'none'}; significantly worse: {verdict['significantly_worse_than_primary'] or 'none'}; "
          f"cannot tell: {verdict['cannot_tell']}")

    section("Periods of the season and the completed-matchweeks slice")
    periods, n_incomplete, per_season_min = period_table(test, losses, probabilities, y)
    periods.to_csv(out_dir / "stage6_periods.csv", index=False)
    print(f"Fewest games played by any team in the loaded data: {per_season_min}; matches outside the completed-matchweeks slice: {n_incomplete}")
    print(periods.round(4).to_string(index=False))

    section("Drift of the test features (diagnostic only; no model is changed because of it)")
    drift = drift_table(fit_rows, test)
    validation_drift = pd.read_csv(REPORTS / "drift_validation.csv", index_col="feature")
    drift["validation_mean_z"], drift["validation_max_abs_z"] = validation_drift["val_mean_z"], validation_drift["max_abs_z"]
    drift.to_csv(out_dir / "drift_test.csv")
    print(drift[["train_mean", "test_mean", "test_mean_z", "max_abs_z", "share_of_rows_above_3", "flagged_above_3",
                 "validation_mean_z", "validation_max_abs_z"]].round(3).head(14).to_string())
    print(f"Features with some row beyond {DRIFT_LIMIT:g}: {int(drift['flagged_above_3'].sum())} of {len(drift)}; "
          f"mean |mean z|: {drift['test_mean_z'].abs().mean():.3f}; the largest |mean z|: {drift['test_mean_z'].abs().idxmax()} "
          f"({drift['test_mean_z'].abs().max():.2f})")
    if dry:                                    # the dry run must reproduce the Stage 5a validation drift exactly
        assert np.allclose(drift["max_abs_z"].reindex(validation_drift.index), validation_drift["max_abs_z"], equal_nan=True)
        print("(dry run) the drift code reproduces reports/drift_validation.csv exactly.")

    section("Calibration and the home-win pattern")
    calibration = plot_calibration_test(probabilities, y, fig_dir)
    print(calibration.round(4).to_string(index=False))
    pattern = pattern_table(test, y, probabilities, losses)
    pattern.to_csv(out_dir / "stage6_pattern.csv", index=False)
    print(pattern.round(4).to_string(index=False))

    section("Poisson as a goals model and as a score predictor")
    po = extra["poisson"]
    goals = {k: v for k, v in po.items() if not isinstance(v, np.ndarray)}
    pd.Series(goals).to_csv(out_dir / "stage6_poisson_goals.csv", header=["value"])
    for k, v in goals.items():
        print(f"  {k:36} {v:.4f}" if isinstance(v, float) else f"  {k:36} {v}")
    print(f"  matches where a rate was capped at 3.5: {po['n_capped']}")
    example_idx = 0                              # fixed in advance: the first match of the season, not chosen for how it looks
    r = test.iloc[example_idx]
    probs = po["proba"][example_idx]
    top = top_scorelines(po["grid"][example_idx], 5)
    example = {"match": f"{r['HomeTeam']} v {r['AwayTeam']}", "date": str(r["Date"].date()), "lambda_home": float(po["lh"][example_idx]),
               "lambda_away": float(po["la"][example_idx]), "p_home": float(probs[0]), "p_draw": float(probs[1]), "p_away": float(probs[2]),
               "top5": [(h, a, round(p * 100, 1)) for h, a, p in top], "real_score": f"{int(r['home_goals'])}-{int(r['away_goals'])}",
               "elo_diff": float(r["elo_diff"]), "grid_corner_percent": (po["grid"][example_idx][:6, :6] * 100).round(1).tolist()}
    (out_dir / "stage6_worked_example.json").write_text(json.dumps(example, indent=1))
    print("Worked example (the opening match):", json.dumps({k: v for k, v in example.items() if k != "grid_corner_percent"}))
    plot_goals(po, test, fig_dir)

    section("Figures")
    plot_confusions(rows, fig_dir)
    plot_log_loss_bar(rows, losses, picks, fig_dir)
    plot_drift(drift, validation_drift, fig_dir)

    section("Saving per-match predictions and the run log")
    predictions = test[["match_id", "Date", "HomeTeam", "AwayTeam", "result", "home_goals", "away_goals"] + ODDS_COLUMNS].copy()
    for i, pr in probabilities.items():
        for c, col in zip(PROBA_COLUMNS, pr.T):
            predictions[f"{i}_{c}"] = col
    predictions["poisson_lambda_home"], predictions["poisson_lambda_away"] = po["lh"], po["la"]
    predictions.to_csv(out_dir / "test_predictions_stage6.csv", index=False)
    log = {"mode": mode, "test_set_loads": TEST_LOADS["count"], "seasons_loaded": TEST_LOADS["seasons"], "n_matches": n,
           "preregistration_sha256": prereg_hash, "models_before": before, "models_after": hash_models_dir(),
           "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "n_matches": n}, "verdict": verdict,
           "overround_mean": extra["overround"], "n_predictors": len(rows)}
    (out_dir / "stage6_run_log.json").write_text(json.dumps(log, indent=1))
    print(f"Saved results, bootstrap, periods, drift, pattern, Poisson goals, run log and per-match predictions to {out_dir}")
    assert hash_models_dir() == hashes_loaded, "a model file changed during the stage"


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "refit-diff":
        refit_diff()
    elif command == "replot-drift":      # redraw figure 41 from the SAVED drift table (no test data is read)
        setup_style()
        plot_drift(pd.read_csv(REPORTS / "drift_test.csv", index_col="feature"),
                   pd.read_csv(REPORTS / "drift_validation.csv", index_col="feature"), REPORTS / "figures")
    elif command in ("dry-run", "run"):
        main(command)
    else:
        raise SystemExit("usage: python src/stage6.py refit-diff | dry-run <scratch folder> | run")
