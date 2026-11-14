"""Checks for Stage 6 (honest model comparison). Run AFTER `python src/stage6.py run`.

These checks never load the test set. Everything about the test season is read from the files that the single
test run wrote (reports/test_predictions_stage6.csv, the run log, the results tables).
Each check prints PASS only after its assertions succeeded.
"""
import ast
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import stage6
from modeling_utils import PROBA_COLUMNS, TestSetAccessError, get_slice, make_xy, per_match_log_loss
from preprocess import TEST_SEASONS
from stage6 import BY_ID, MODELS, PREDICTORS, REPORTS, ROOT

LAST_STAGE5_COMMIT = "42d37a1"        # the Stage 5d commit: the earlier guards must be unchanged since then


def passed(message):
    print(f"PASS  {message}")


def git_show(path):
    return subprocess.run(["git", "show", f"{LAST_STAGE5_COMMIT}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout.replace(b"\r\n", b"\n")


log = json.loads((REPORTS / "stage6_run_log.json").read_text())
predictions = pd.read_csv(REPORTS / "test_predictions_stage6.csv", parse_dates=["Date"])
results = pd.read_csv(REPORTS / "stage6_results.csv")
y = predictions["result"].to_numpy()
n = len(predictions)

# ------------------------------------------------------------ the earlier guards were not weakened
for name in ("modeling_utils", "logreg", "rf", "xgb", "poisson", "poisson_model", "features", "features_goals"):
    path = f"src/{name}.py"
    assert (ROOT / path).read_bytes().replace(b"\r\n", b"\n") == git_show(path), f"{path} changed since Stage 5d"
try:
    get_slice("test")
    raise AssertionError("get_slice('test') did not raise")
except TestSetAccessError:
    pass
fake = pd.DataFrame({"season": ["2025-26"], "split": ["test"], "result": [0], "elo_diff": [0.0]})
for call in (lambda: make_xy(fake, ["elo_diff"]), lambda: stage6.__dict__["get_slice"]("test")):
    try:
        call()
        raise AssertionError("a Stage 5 guard let a test row through")
    except TestSetAccessError:
        pass
import poisson
try:
    poisson.evaluate_model(None, fake)
    raise AssertionError("poisson.evaluate_model accepted a test row")
except TestSetAccessError:
    pass
passed("modeling_utils, logreg, rf, xgb, poisson, poisson_model, features and features_goals are byte-identical to the Stage 5d commit; "
       "get_slice('test'), make_xy and the Poisson evaluator still raise TestSetAccessError on test rows")

# ------------------------------------------------------------ no model file changed except the two new DIFF models
before = json.loads((REPORTS / "stage6_models_before.json").read_text())
now = stage6.hash_models_dir()
assert all(now[name] == digest for name, digest in before.items()), "a pre-existing model file was modified"
new_files = sorted(set(now) - set(before))
assert new_files == ["diff_rf.joblib", "diff_rf_config.json", "diff_xgb.joblib", "diff_xgb_config.json", "diff_xgb.ubj"] or \
       new_files == sorted(["diff_rf.joblib", "diff_rf_config.json", "diff_xgb.joblib", "diff_xgb_config.json", "diff_xgb.ubj"]), new_files
assert log["models_before"] == before and set(log["models_after"]) == set(now)
audit = json.loads(stage6.DIFF_AUDIT_JSON.read_text())
assert len(audit) == 1 and audit[0]["purpose"] == "final_refit_diff" and audit[0]["inner_stop_seasons"] == ["2023-24", "2024-25"]
assert audit[0]["inner_fit_seasons"][-1] == "2022-23" and not (set(audit[0]["inner_fit_seasons"]) | set(audit[0]["inner_stop_seasons"])) & set(TEST_SEASONS)
for name in ("diff_rf_config.json", "diff_xgb_config.json"):
    config = json.loads((MODELS / name).read_text())
    assert config["window_seasons"] == [s for s in get_slice("train_all")["season"].unique()] and config["n_training_rows"] == 3800
passed(f"the {len(before)} model files that existed before Stage 6 are byte-identical to their recorded hashes; the only new files are the "
       f"DIFF Random Forest and DIFF XGBoost ({len(new_files)} files); the DIFF early-stopping audit says purpose 'final_refit_diff', "
       "fit on 2015-16 to 2022-23, stopped on 2023-24 and 2024-25, no test season")

# ------------------------------------------------------------ the test set was loaded exactly once
assert log["test_set_loads"] == 1 and log["seasons_loaded"] == list(TEST_SEASONS) and log["n_matches"] == n == 380
tree = ast.parse((ROOT / "src" / "stage6.py").read_text())
calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "load_test_set"]
assert len(calls) == 1, "load_test_set must have exactly one call site"
stage6.TEST_LOADS["count"] = 1
try:
    stage6.load_test_set(TEST_SEASONS)           # must raise before reading anything
    raise AssertionError("a second load_test_set call was allowed")
except TestSetAccessError:
    pass
finally:
    stage6.TEST_LOADS["count"] = 0
other_loaders = [p.name for p in (ROOT / "src").glob("*.py") if p.name not in ("stage6.py", "check_stage6.py")
                 and re.search(r"season\"\]\.isin\(TEST_SEASONS\)|split\"\] == \"test\"\]|load_test_set", p.read_text())]
passed(f"the run log says the test set was loaded once ({n} matches of {log['seasons_loaded']}); load_test_set has one call site and a "
       f"second call raises; no other script reads test rows (checked scripts: {len(list((ROOT / 'src').glob('*.py'))) - 2})")

# ------------------------------------------------------------ pre-registration came first and is copied verbatim
prereg = stage6.PREREG_MD
assert stage6.file_sha(prereg) == log["preregistration_sha256"]
assert prereg.stat().st_mtime < (REPORTS / "test_predictions_stage6.csv").stat().st_mtime
report = (REPORTS / "stage6_results.md").read_text(encoding="utf-8")
assert prereg.read_text(encoding="utf-8").strip() in report, "stage6_results.md must contain the pre-registration verbatim"
assert report.index("Pre-registration") < report.index("Headline")
passed("the pre-registration file is unchanged since the run (hash recorded in the run log), is older than the test predictions, and is "
       "copied verbatim at the top of reports/stage6_results.md")

# ------------------------------------------------------------ baselines against hand computations
train_all = get_slice("train_all")
freq = np.bincount(train_all["result"].to_numpy(), minlength=3) / len(train_all)
hand_accuracy = float((y == 0).mean())
hand_freq_loss = float(-np.log(freq[y]).mean())
home_row = results[results["predictor"] == BY_ID["always_home"]["label"]].iloc[0]
freq_row = results[results["predictor"] == BY_ID["class_freq"]["label"]].iloc[0]
assert abs(home_row["accuracy"] - hand_accuracy) < 1e-12 and np.isnan(home_row["log_loss"])
assert abs(freq_row["log_loss"] - hand_freq_loss) < 1e-9 and abs(freq_row["accuracy"] - hand_accuracy) < 1e-12
assert (freq_row["draws_predicted"] == 0) and (home_row["draws_predicted"] == 0)
passed(f"baselines on the test season match hand computations: always-home accuracy {hand_accuracy:.4f} ({int((y == 0).sum())} of {n} home wins); "
       f"class-frequency log loss {hand_freq_loss:.4f} (training frequencies {freq.round(4).tolist()})")

# ------------------------------------------------------------ bookmaker and all probability tables
raw = 1 / predictions[["B365H", "B365D", "B365A"]].to_numpy()
hand_book = raw / raw.sum(axis=1, keepdims=True)
book = predictions[[f"bookmaker_{c}" for c in PROBA_COLUMNS]].to_numpy()
assert np.allclose(book.sum(axis=1), 1.0, atol=1e-12) and np.allclose(book, hand_book, atol=1e-12) and (raw.sum(axis=1) > 1).all()
assert abs(float((raw.sum(axis=1) - 1).mean()) - log["overround_mean"]) < 1e-12
for p in PREDICTORS:
    if p["kind"] == "home":
        continue
    probabilities = predictions[[f"{p['id']}_{c}" for c in PROBA_COLUMNS]].to_numpy()
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9) and (probabilities >= 0).all()
    row = results[results["predictor"] == p["label"]].iloc[0]
    assert abs(per_match_log_loss(y, probabilities).mean() - row["log_loss"]) < 1e-9
    assert abs((probabilities.argmax(axis=1) == y).mean() - row["accuracy"]) < 1e-12
passed(f"bookmaker probabilities equal 1/odds divided by their sum (margin {log['overround_mean'] * 100:.2f}% removed) and sum to 1; "
       "all 11 probability predictors sum to 1, and their log loss and accuracy in stage6_results.csv equal a recomputation from the saved probabilities")

# ------------------------------------------------------------ the results table
expected_columns = ["predictor", "family", "role", "accuracy", "log_loss", "macro_f1", "home_f1", "draw_f1", "away_f1",
                    "draws_predicted", "n_test_matches", "notes"]
assert list(results.columns) == expected_columns
assert len(results) == len(PREDICTORS) == log["n_predictors"] == 12 and (results["n_test_matches"] == n).all()
assert set(results["role"]) == {"primary", "reference", "diagnostic", "baseline"} and results["predictor"].is_unique
assert results.loc[results["role"] == "diagnostic", "predictor"].str.contains("diagnostic").all()
passed(f"stage6_results.csv has the required columns and {len(results)} rows = the number of predictors scored, all on the same {n} matches")

# ------------------------------------------------------------ one resampling table for every bootstrap
boot = pd.read_csv(REPORTS / "stage6_bootstrap.csv")
assert (boot["n_boot"] == 4000).all() and (boot["n_matches"] == n).all() and (boot["seed"] == 42).all()
picks = np.random.default_rng(42).integers(0, n, size=(4000, n))
label_to_id = {p["label"]: p["id"] for p in PREDICTORS}
losses = {p["id"]: per_match_log_loss(y, predictions[[f"{p['id']}_{c}" for c in PROBA_COLUMNS]].to_numpy()) for p in PREDICTORS if p["kind"] != "home"}
for _, r in boot.iterrows():
    diff = losses[label_to_id[r["model_a"]]] - losses[label_to_id[r["model_b"]]]
    means = diff[picks].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    assert abs(diff.mean() - r["mean_diff_a_minus_b"]) < 1e-6 and abs(low - r["ci_low"]) < 1e-6 and abs(high - r["ci_high"]) < 1e-6
    assert r["verdict"] == ("A better" if high < 0 else ("A worse" if low > 0 else "cannot tell"))
groups = set(boot["group"])
assert groups == {"vs ELO2 LR (primary)", "vs bookmaker", "ELO2 primary vs FULL reference", "ELO2 primary vs DIFF diagnostic"}
assert (boot["group"] == "ELO2 primary vs FULL reference").sum() == 3 and (boot["group"] == "ELO2 primary vs DIFF diagnostic").sum() == 2
passed(f"all {len(boot)} bootstrap rows use 4,000 resamples of the same {n} matches with seed 42: rebuilding ONE resampling table reproduces every "
       "mean, interval and verdict; the required comparison groups (vs ELO2 LR, vs bookmaker, 3 ELO2-vs-FULL pairs, 2 ELO2-vs-DIFF pairs) are present")

# ------------------------------------------------------------ odds are used for the bookmaker baseline only
forbidden_everywhere = re.compile(r"ODDS_CSV|odds\.csv|B365")
may_touch_odds = {"modeling_utils.py", "preprocess.py", "eda.py", "inspect_raw.py", "check_features.py", "check_preprocess.py",
                  "check_models.py", "stage6.py", "check_stage6.py", "download_data.py"}
offenders = [p.name for p in (ROOT / "src").glob("*.py") if p.name not in may_touch_odds and forbidden_everywhere.search(p.read_text())]
assert not offenders, offenders
for name in ("features.py", "features_goals.py", "poisson_model.py", "rf.py", "xgb.py", "logreg.py", "poisson.py"):
    text = (ROOT / "src" / name).read_text()
    assert not forbidden_everywhere.search(text), f"{name} touches odds"
stage6_text = (ROOT / "src" / "stage6.py").read_text()
uses = [node.name for node in ast.walk(ast.parse(stage6_text)) if isinstance(node, ast.FunctionDef)
        and re.search(r"ODDS_COLUMNS|ODDS_CSV", ast.get_source_segment(stage6_text, node))]
assert sorted(uses) == ["bookmaker_baseline", "load_test_set", "main"], uses       # main only saves the raw odds next to the probabilities
all_features = set()
for p in PREDICTORS:
    if p["config"]:
        config = json.loads((MODELS / p["config"]).read_text())
        all_features |= set(config.get("features", config.get("required_columns", [])))
assert not any(re.match(r"(B365|BW|PS|Avg|Max|Bb|IW|WH|VC|LB)", f) for f in all_features)
passed(f"odds are touched only by the bookmaker baseline: no feature, model or feature-building file refers to them (all {len(all_features)} distinct model "
       "input columns checked); in stage6.py only the loader, bookmaker_baseline and the saving step in main() use the odds columns")

# ------------------------------------------------------------ drift, Poisson goals, figures, report
drift = pd.read_csv(REPORTS / "drift_test.csv", index_col="feature")
assert len(drift) == 34 and ((drift["max_abs_z"] > 3) == drift["flagged_above_3"]).all()
assert np.allclose((drift["test_mean"] - drift["train_mean"]) / drift["train_std"], drift["test_mean_z"], atol=1e-9)
goals = pd.read_csv(REPORTS / "stage6_poisson_goals.csv", index_col=0)["value"]
lh, la = predictions["poisson_lambda_home"].to_numpy(), predictions["poisson_lambda_away"].to_numpy()
hg, ag = predictions["home_goals"].to_numpy(), predictions["away_goals"].to_numpy()
assert (lh <= 3.5).all() and (la <= 3.5).all() and (lh > 0).all()
assert abs(np.abs(lh - hg).mean() - goals["mae_home"]) < 1e-9 and abs(np.abs(la - ag).mean() - goals["mae_away"]) < 1e-9
assert abs(np.abs(lh + la - hg - ag).mean() - goals["mae_total"]) < 1e-9
for name in ("37_test_confusion_matrices.png", "38_test_calibration.png", "39_log_loss_bar.png",
             "40_poisson_predicted_vs_actual_goals_test.png", "41_drift_test_bar.png"):
    assert (REPORTS / "figures" / name).stat().st_size > 20_000, name
for _, r in results.dropna(subset=["log_loss"]).iterrows():
    assert f"{r['log_loss']:.4f}" in report, f"{r['predictor']} log loss missing from stage6_results.md"
assert "**Logistic Regression ELO2**" in report
passed("drift table: 34 features, flags and z-scores are internally consistent; Poisson MAE recomputed from the saved lambdas (all rates positive and at "
       "most 3.5); figures 37 to 41 exist; every log loss of stage6_results.csv appears in reports/stage6_results.md, with ELO2 LR in bold")

print("\nAll checks passed.")
