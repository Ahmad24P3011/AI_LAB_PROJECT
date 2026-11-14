"""Stage 5a amendment (post-hoc, diagnostic-driven). Run after src/modeling_utils.py.

What this does, on the VALIDATION seasons only (the 2025-26 test season is never loaded):
  1. Drift table: for all 34 features, how far validation values lie from the training range.
  2. Primary Logistic Regression = ELO2 (elo_diff + abs_elo_diff); FULL (v2, 33 features, keeps
     head-to-head) is kept as a reference row for Stage 6. C is tuned on validation log loss.
  3. Home-win check per validation season: average predicted probability vs real home-win rate.
  4. Refit both on 2015-16 to 2024-25 and save them (NOT scored on the test season).

The choice of ELO2 as primary was made BEFORE looking at the numbers below (see docs/decisions.md);
the comparisons here are reported, not used to choose.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS
from logreg import FIT_END, FINAL_END, fit_config, make_config, score_config, tune_C
from modeling_utils import (CLASS_CODES, CLASS_NAMES, FIT_SEASONS, PROBA_COLUMNS, TRAIN_SEASONS,
                            VAL_SEASONS, bookmaker_probabilities, get_slice, load_feature_sets,
                            load_features, paired_bootstrap, per_match_log_loss, predict_proba,
                            window_seasons)

ROOT = Path(__file__).resolve().parent.parent
DRIFT_CSV = ROOT / "reports" / "drift_validation.csv"
RESULTS_CSV = ROOT / "reports" / "amendment_5a_results.csv"
VAL_PREDICTIONS_CSV = ROOT / "reports" / "val_predictions_5a_v2.csv"
PRIMARY_MODEL = ROOT / "models" / "logreg.joblib"
PRIMARY_CONFIG = ROOT / "models" / "logreg_config.json"
REFERENCE_MODEL = ROOT / "models" / "logreg_full_reference.joblib"
REFERENCE_CONFIG = ROOT / "models" / "logreg_full_reference_config.json"
DRIFT_LIMIT = 3.0   # flag a feature if any validation value is more than 3 training std away


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


# ------------------------------------------------------------ 1. drift table
def drift_table(fit_rows, val_rows, sets):
    """Validation values measured in TRAINING (fit-season) standard deviations."""
    mean = fit_rows[FEATURE_COLUMNS].mean()
    std = fit_rows[FEATURE_COLUMNS].std(ddof=0).replace(0, np.nan)
    z = (val_rows[FEATURE_COLUMNS] - mean) / std
    in_sets = {f: [name for name in ("FULL", "REDUCED", "ELO2") if f in sets[name]] for f in FEATURE_COLUMNS}
    table = pd.DataFrame({
        "train_mean": mean, "train_std": std, "val_mean": val_rows[FEATURE_COLUMNS].mean(),
        "val_mean_z": z.mean(), "max_abs_z": z.abs().max(),
        "share_of_val_rows_above_3": (z.abs() > DRIFT_LIMIT).mean(),
        "flagged_above_3": z.abs().max() > DRIFT_LIMIT,
        "in_v2_classifier_sets": pd.Series({f: "+".join(v) or "none" for f, v in in_sets.items()}),
    }).sort_values("max_abs_z", ascending=False)
    table.index.name = "feature"
    return table


# ------------------------------------------------ 3. home-win check per season
def home_win_check(val_rows, probabilities):
    """Average predicted home-win probability vs the real home-win rate, per season."""
    rows = []
    for season in VAL_SEASONS:
        mask = (val_rows["season"] == season).to_numpy()
        row = {"season": season, "matches": int(mask.sum()),
               "actual home-win rate": float((val_rows.loc[mask, "result"] == 0).mean())}
        for name, proba in probabilities.items():
            row[f"{name}: avg predicted"] = float(proba[mask, 0].mean())
        rows.append(row)
    table = pd.DataFrame(rows).set_index("season")
    for name in probabilities:
        table[f"{name}: gap"] = table[f"{name}: avg predicted"] - table["actual home-win rate"]
    return table


# ---------------------------------------------------------------- saving
def save_model(cfg, source_rows, model_path, config_path, status, set_name):
    """Refit on the full training period (2015-16 to 2024-25) and save. NOT scored on test."""
    model, used = fit_config(cfg, source_rows, FINAL_END)
    joblib.dump(model, model_path)
    config_path.write_text(json.dumps({
        "status": status, "feature_set": set_name, "features": cfg["features"], "C": cfg["C"],
        "window_seasons": window_seasons(cfg["window"], FINAL_END), "recency_weights": cfg["recency"],
        "include_warmup_rows": cfg["include_warmup"], "drop_2020_21_rows": cfg["drop_2020_21"],
        "class_order": dict(zip(CLASS_CODES, CLASS_NAMES)), "n_training_rows": len(used),
        "scaler": "StandardScaler inside the pipeline, fitted on these rows only",
        "chosen_on": f"validation seasons {VAL_SEASONS} with models fitted on "
                     f"{FIT_SEASONS[0]} to {FIT_SEASONS[-1]}",
        "feature_sets_file": "data/processed/feature_sets_v2.json (amendment of feature_sets_v1.json)",
    }, indent=2))
    print(f"Saved {model_path.relative_to(ROOT)} + {config_path.name}: {set_name}, {len(cfg['features'])} "
          f"features, C={cfg['C']:g}, {len(used)} rows ({used['season'].min()} to {used['season'].max()})")


def main():
    sets = load_feature_sets(version=2)
    all_training = load_features()                       # training rows only (test rows never loaded)
    fit_rows = get_slice("fit", all_training)
    val_rows = get_slice("validation", all_training)
    y_val = val_rows["result"].to_numpy()
    print(f"Fit rows {len(fit_rows)}, validation rows {len(val_rows)}, test rows loaded: 0")
    print(f"Amendment of {sets['date']}: ELO2 = {sets['ELO2']}; FULL = {len(sets['FULL'])} features "
          f"(reference); league_avg_goals_prev380 removed from every classifier set")

    section("Drift diagnostic: validation features in training standard deviations (all 34)")
    drift = drift_table(fit_rows, val_rows, sets)
    drift.to_csv(DRIFT_CSV)
    view = drift[["train_mean", "train_std", "val_mean", "val_mean_z", "max_abs_z",
                  "share_of_val_rows_above_3", "in_v2_classifier_sets"]]
    print(view.round(3).to_string())
    flagged = drift[drift["flagged_above_3"]]
    print(f"\nFlagged (max |z| above {DRIFT_LIMIT:g}): {len(flagged)} of {len(drift)}")
    for feature, row in flagged.iterrows():
        print(f"  {feature}: mean z {row['val_mean_z']:+.2f}, max |z| {row['max_abs_z']:.2f}, "
              f"{row['share_of_val_rows_above_3'] * 100:.0f}% of validation rows above 3; "
              f"in v2 sets: {row['in_v2_classifier_sets']}")
    print(f"Saved {DRIFT_CSV.relative_to(ROOT)}. (Diagnostic only. To be repeated for the test "
          f"features in Stage 6, labelled a diagnostic, not a tuning step.)")

    section("Primary (ELO2) and reference (FULL v2): tuning C on validation log loss")
    outcomes, records, tuned = {}, [], {}
    for name in ("ELO2", "FULL"):
        base = make_config(sets[name])
        curve, best_C = tune_C(base, fit_rows, val_rows)
        tuned[name] = {**base, "C": best_C}
        outcomes[name] = score_config(tuned[name], fit_rows, val_rows)
        print(f"\n{name} ({len(sets[name])} features): chosen C = {best_C:g}")
        print(curve.round(4).to_string(index=False))
        for _, r in curve.iterrows():
            records.append({"model": name, "n_features": len(sets[name]), **r.to_dict(),
                            "chosen": r["C"] == best_C})
    pd.DataFrame(records).to_csv(RESULTS_CSV, index=False)

    section("Validation results (fit on 2015-16 to 2022-23)")
    book_proba, _ = bookmaker_probabilities(val_rows["match_id"])
    shares = fit_rows["result"].value_counts(normalize=True).reindex(CLASS_CODES).to_numpy()
    freq_proba = np.tile(shares, (len(y_val), 1))
    losses = {"Class frequencies": per_match_log_loss(y_val, freq_proba),
              "Bookmaker": per_match_log_loss(y_val, book_proba),
              "FULL reference": outcomes["FULL"]["val_losses"], "ELO2 primary": outcomes["ELO2"]["val_losses"]}
    summary = pd.DataFrame({
        "log loss": {k: v.mean() for k, v in losses.items()},
        "accuracy": {"Class frequencies": float((freq_proba.argmax(1) == y_val).mean()),
                     "Bookmaker": float((book_proba.argmax(1) == y_val).mean()),
                     "FULL reference": outcomes["FULL"]["metrics"]["accuracy"],
                     "ELO2 primary": outcomes["ELO2"]["metrics"]["accuracy"]}})
    print(summary.round(4).to_string())
    for label, a, b in [("ELO2 vs FULL reference", "FULL reference", "ELO2 primary"),
                        ("ELO2 vs class frequencies", "Class frequencies", "ELO2 primary"),
                        ("ELO2 vs bookmaker", "Bookmaker", "ELO2 primary")]:
        s = paired_bootstrap(losses[a], losses[b])
        print(f"  {label}: log-loss improvement {s['mean_diff']:+.4f} [{s['ci_low']:+.4f}, "
              f"{s['ci_high']:+.4f}] (positive = ELO2 better). Reported, NOT used to choose.")

    section("Home-win check per validation season")
    probabilities = {"ELO2": outcomes["ELO2"]["val_proba"], "FULL ref": outcomes["FULL"]["val_proba"],
                     "Bookmaker": book_proba, "Class freq": freq_proba}
    check = home_win_check(val_rows, probabilities)
    print((check.T * 1).round(3).to_string())
    for season, row in check.iterrows():
        print(f"  {season}: home teams won {row['actual home-win rate'] * 100:.1f}% of {int(row['matches'])} "
              f"matches; ELO2 predicted {row['ELO2: avg predicted'] * 100:.1f}% "
              f"(gap {row['ELO2: gap'] * 100:+.1f} points), bookmaker {row['Bookmaker: avg predicted'] * 100:.1f}% "
              f"(gap {row['Bookmaker: gap'] * 100:+.1f})")

    section("Saved validation predictions and final models")
    predictions = pd.DataFrame({"match_id": val_rows["match_id"], "result": y_val})
    for prefix, proba in [("elo2_primary", outcomes["ELO2"]["val_proba"]),
                          ("full_reference", outcomes["FULL"]["val_proba"]),
                          ("class_freq", freq_proba), ("bookmaker", book_proba)]:
        for column, values in zip(PROBA_COLUMNS, proba.T):
            predictions[f"{prefix}_{column}"] = values
    predictions.to_csv(VAL_PREDICTIONS_CSV, index=False)
    print(f"Saved {VAL_PREDICTIONS_CSV.relative_to(ROOT)}")
    train_all = get_slice("train_all", all_training)
    save_model(tuned["ELO2"], train_all, PRIMARY_MODEL, PRIMARY_CONFIG, "primary", "ELO2")
    save_model(tuned["FULL"], train_all, REFERENCE_MODEL, REFERENCE_CONFIG,
               "reference row for Stage 6", "FULL (v2, 33 features, keeps head-to-head)")
    print("Neither model was scored on the test season.")


if __name__ == "__main__":
    main()
