"""Checks for Stage 9 (the Streamlit app). Run from the repo root:  python src/check_app.py

The app only SERVES saved artifacts. These checks prove (1) everything it needs exists and loads, (2) its predictions are
sane and use exactly the layout the models were trained on, (3) it cannot read raw 2025-26 scores or train anything,
and (4) it degrades gracefully without crests. Each check prints PASS only after its assertions succeeded.
"""
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
sys.path.insert(0, str(APP))
sys.path.insert(0, str(ROOT / "src"))

import predictor as P                                               # noqa: E402
import team_state_builder as B                                      # noqa: E402
from features import PARAMS_JSON                                    # noqa: E402
from features_goals import GOAL_PARAMS_JSON                         # noqa: E402
from team_colors import CODES, COLORS, SLUG                         # noqa: E402


def passed(message):
    print(f"PASS  {message}")


# ------------------------------------------------ 1. every file the app needs exists and loads
needed = [P.TEAM_STATE_CSV, P.BOOKMAKER_CSV, P.REAL_RESULTS_CSV, APP / ".." / "data" / "app" / "sim_position_probs.csv", APP / ".." / "data" / "app" / "hero_ball.svg",
          ROOT / "reports" / "sim_vs_real_table.csv", ROOT / "reports" / "sim_points_by_game.csv", ROOT / "reports" / "sim_headline_metrics.json",
          ROOT / ".streamlit" / "config.toml", APP / "requirements.txt"] + [P.MODELS_DIR / f"{m}.joblib" for m in P.CLASSIFIERS] + [P.MODELS_DIR / "poisson_config.json"]
missing = [str(p) for p in needed if not Path(p).exists()]
assert not missing, f"missing files: {missing}"
predictor = P.Predictor()
for name in ("sim_vs_real_table.csv", "sim_points_by_game.csv"):
    assert len(pd.read_csv(ROOT / "reports" / name)) > 0
json.loads((ROOT / "reports" / "sim_headline_metrics.json").read_text())
passed(f"all {len(needed)} files exist; models, team state and bookmaker table load cleanly")

# ------------------------------------------------ 2. the served models are byte-identical to the ones scored in Stage 6/7
before = json.loads((ROOT / "reports" / "stage6_models_before.json").read_text())
for name in ["logreg.joblib", "random_forest.joblib", "xgboost.joblib", "poisson_stacked.joblib", "poisson_config.json"]:
    assert hashlib.sha256((P.MODELS_DIR / name).read_bytes()).hexdigest() == before[name], f"{name} changed since Stage 6"
passed("logreg, random_forest, xgboost and poisson files are unchanged since Stage 6 (sha256 equal)")

# ------------------------------------------------ 3. every 2025-26 team is in the team state, with sane values
table = pd.read_csv(ROOT / "reports" / "sim_vs_real_table.csv")
state = predictor.state
assert set(table["team"]) == set(state.index) == set(COLORS) == set(CODES) == set(SLUG), "team lists disagree"
assert len(state) == 20 and state[P.STATE_COLUMNS[1:]].notna().all().all()
assert state["elo"].between(1000, 2300).all() and (state["matches_played_2025_26"] == 38).all()
assert state[["gf_10_rel", "ga_10_rel"]].gt(0).all().all()
passed(f"20 teams in the team state; Elo {state['elo'].min():.0f} to {state['elo'].max():.0f}; every team played 38 matches")

# ------------------------------------------------ 4. smoke test: Liverpool v Bournemouth (the Stage 6 worked example pair)
result = predictor.predict("Liverpool", "Bournemouth")
print("      Liverpool v Bournemouth, end-of-2025-26 state (home / draw / away):")
for name, p in result["probabilities"].items():
    assert p.shape == (3,) and (p > 0).all() and abs(p.sum() - 1) < 1e-6, (name, p)
    print(f"        {name:14s} {np.round(p, 4)}  sum {p.sum():.6f}")
assert result["bookmaker"] is not None and abs(result["bookmaker"].sum() - 1) < 1e-6
print(f"        {'bookmaker':14s} {np.round(result['bookmaker'], 4)}  sum {result['bookmaker'].sum():.6f}  (odds of the real Aug 2025 fixture)")
lam = result["lambda_home"] + result["lambda_away"]
print(f"        Poisson lambdas {result['lambda_home']:.3f} + {result['lambda_away']:.3f} = {lam:.3f}")
assert 1.5 < lam < 5, lam
assert abs(result["grid"].sum() - 1) < 1e-9 and len(result["top_scorelines"]) == 10
passed("four models and the bookmaker return probabilities that sum to 1; Poisson total goals rate is between 1.5 and 5")

# ------------------------------------------------ 5. all 380 pairs: sane, and the app's Poisson inputs equal the training-time feature code
matches = pd.read_csv(B.MATCHES_CSV, parse_dates=["Date"])
params = json.loads(PARAMS_JSON.read_text())
goal_defaults = json.loads(GOAL_PARAMS_JSON.read_text())["goal_defaults"]
pairs, goals, _, final_elo = B.pair_inputs(matches, params, goal_defaults)
worst, totals = 0.0, []
for i, (home, away) in enumerate(zip(pairs["HomeTeam"], pairs["AwayTeam"])):
    ours = P.poisson_inputs(state.loc[home], state.loc[away]).iloc[0]
    theirs = goals.iloc[i]
    worst = max(worst, float((ours - theirs[ours.index]).abs().max()))
    r = predictor.predict(home, away)
    totals.append(r["lambda_home"] + r["lambda_away"])
    assert all(abs(p.sum() - 1) < 1e-6 for p in r["probabilities"].values())
assert worst < 1e-9, f"app inputs differ from the feature pipeline by {worst}"
assert min(totals) > 1.0 and max(totals) < 6.0, (min(totals), max(totals))
assert all(abs(final_elo[t] - state.loc[t, "elo"]) < 1e-9 for t in state.index)
passed(f"all 380 pairs: app inputs equal the training-time feature code (max difference {worst:.1e}); total-goals rate {min(totals):.2f} to {max(totals):.2f}")

# ------------------------------------------------ 5b. the "What really happened" data (display only)
real = pd.read_csv(P.REAL_RESULTS_CSV)
assert len(real) == 380 and not real.duplicated(["HomeTeam", "AwayTeam"]).any() and set(real["HomeTeam"]) == set(state.index)
assert list(real.columns) == ["Date", "HomeTeam", "AwayTeam", "home_score", "away_score"]
worked = json.loads((ROOT / "reports" / "stage6_worked_example.json").read_text())
found = predictor.real_result("Liverpool", "Bournemouth")
assert f"{found['home_score']}-{found['away_score']}" == worked["real_score"] == "4-2" and found["outcome"] == 0
assert predictor.real_result("Liverpool", "Liverpool") is None
# the real score is display-only: predictions must not change with it (the Predictor's models never see the results table)
saved = predictor.real_results
predictor.real_results = saved.assign(home_score=0, away_score=0)
assert all(np.allclose(predictor.predict("Liverpool", "Bournemouth")["probabilities"][m], result["probabilities"][m], rtol=0, atol=1e-12) for m in result["probabilities"])   # tolerance: the forest sums trees in parallel (1e-16 noise)
predictor.real_results = saved
passed(f"real results: 380 fixtures, one per team pair; Liverpool 4-2 Bournemouth equals the Stage 6 worked example; predictions do not depend on them")

# ------------------------------------------------ 6. the app cannot read raw scores or train anything
runtime_files = [p for p in list(APP.glob("*.py")) + list(APP.glob("views/*.py")) if p.name not in ("team_state_builder.py", "fetch_crests.py")]
forbidden = re.compile(r"matches_clean|features\.csv|features_goals|train\.csv|test\.csv|E0_20|odds\.csv|FTHG|FTAG|\bhome_goals\b|\baway_goals\b")
for path in runtime_files:
    text = path.read_text(encoding="utf-8")
    code = re.sub(r'"""[\s\S]*?"""', "", text)                      # docstrings and comments only explain; only real code counts
    code = "\n".join(line for line in code.splitlines() if not line.strip().startswith("#"))
    assert not forbidden.search(code), f"{path.name} mentions a raw data file or score column: {forbidden.search(code).group()}"
    assert ".fit(" not in code and "fit_transform" not in code, f"{path.name} contains training code"
    assert not re.search(r"^\s*(from|import)\s+(simulator|stage6|features|features_goals|modeling_utils|xgb|rf|logreg|poisson)\b", code, re.M), f"{path.name} imports training code"
passed(f"{len(runtime_files)} runtime app files: no raw-data file names, no score columns, no .fit(), no training imports")
heavy = [p for p in APP.rglob("*") if p.is_file() and p.stat().st_size > 200_000 and "__pycache__" not in p.parts]
assert not heavy, heavy
assert not list(APP.rglob("*.csv")) and not list(APP.rglob("*.joblib")) and not list(APP.rglob("*.ipynb"))
passed("app/ holds only small code files (no data, models or notebooks)")

# ------------------------------------------------ 6b. both pages link to each other (the sidebar starts collapsed on narrow screens)
for page, target in (("match_predictor", "season_table"), ("season_table", "match_predictor")):
    assert "styles.nav_links()" in (APP / "views" / f"{page}.py").read_text(encoding="utf-8"), page
assert 'page_link("views/season_table.py"' in (APP / "styles.py").read_text(encoding="utf-8") and 'page_link("views/match_predictor.py"' in (APP / "styles.py").read_text(encoding="utf-8")
passed("each page shows in-page links to both pages (navigation does not depend on the sidebar)")

# ------------------------------------------------ 6c. phone layout: tables scroll instead of being cut off, tiles do not stack, phone CSS exists
styles_text = (APP / "styles.py").read_text(encoding="utf-8")
for needle in ("@media (max-width: 700px)", ".table-wrap", ".tile-row", ".hide-m"):
    assert needle in styles_text, needle
for page in ("match_predictor", "season_table"):
    text = (APP / "views" / f"{page}.py").read_text(encoding="utf-8")
    assert 'class="table-wrap"' in text and "tile_row(" in text and "st.columns(3)" not in text and "st.columns(4)" not in text, page
passed("phone layout: media query, scrollable table wrappers and tile rows are in place on both pages")

# ------------------------------------------------ 7. app/requirements.txt is the minimal list
requirements = (APP / "requirements.txt").read_text().lower()
for package in ("streamlit", "pandas", "numpy", "scikit-learn", "xgboost", "joblib", "statsmodels", "scipy"):
    assert package in requirements, package
for package in ("jupyter", "matplotlib", "seaborn", "shap", "plotly"):
    assert package not in requirements, package
passed("app/requirements.txt lists the 8 needed packages and none of jupyter, matplotlib, seaborn, shap")

# ------------------------------------------------ 8. crests: present and well formed, and the app degrades gracefully without them
import styles                                                       # noqa: E402  (imports streamlit; fine outside a running app)
crest_dir = ROOT / "data" / "app" / "team_crests"
present = [t for t in SLUG if (crest_dir / f"{SLUG[t]}.png").exists()]
for team in present:
    image = Image.open(crest_dir / f"{SLUG[team]}.png")
    assert image.size == (128, 128) and image.mode == "RGBA", (team, image.size, image.mode)
assert "<img" in styles.crest("Arsenal", 40) or len(present) < 20
with tempfile.TemporaryDirectory() as empty:
    original, styles.CREST_DIR = styles.CREST_DIR, Path(empty)
    styles._data_uri.cache_clear()
    fallback = styles.crest("Arsenal", 40)
    styles.CREST_DIR = original
assert "<img" not in fallback and "ARS" in fallback and COLORS["Arsenal"] in fallback
passed(f"{len(present)} of 20 crest PNGs present and 128x128 RGBA; with no crest files a coloured 3-letter badge is shown instead")

# ------------------------------------------------ 9. the team-state guard: a state file with a missing team is refused
bad = state.reset_index().iloc[:19]
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "state.csv"
    bad.to_csv(path, index=False)
    original, P.TEAM_STATE_CSV = P.TEAM_STATE_CSV, path
    try:
        P.Predictor()
        raise AssertionError("an incomplete team state was accepted")
    except P.AppDataError as error:
        message = str(error)
    finally:
        P.TEAM_STATE_CSV = original
assert "out of date" in message
passed("a team state file with 19 teams is refused with the friendly 'app data is out of date' message")

print("\nAll Stage 9 checks passed.")
