# CLAUDE.md: Premier League Prediction Project (AI Lab, Group #8)

## Who I'm working with
A university student who has learned only KNN and K-means so far. They are doing
this project to LEARN, not just to get it built. Treat every stage as a lesson.

Read `template/requirements.md` and `template/project_plan.md` first.
- `template/requirements.md` is the official course checklist. It must be fully satisfied.
- `template/project_plan.md` is the group's plan for meeting it.
Do not change the plan's scope without asking. If the plan and the
requirements ever conflict, tell me and the requirements win.

## Working rules (follow these every session)

1. **One stage at a time.** The stages are listed below. Only work on the stage
   I ask for. Never start the next stage on your own.
2. **Before coding a stage**, tell me in 3 to 6 sentences what you are about to
   build and why. Wait for me only if something is ambiguous. Otherwise proceed.
3. **Build, then test.** Run your code and check it works. Show me the real
   outputs (shapes, numbers, plots saved to `reports/figures/`). Never claim
   something works without running it.
4. **After the stage, write a lesson file** at `docs/learning/NN_stage_name.md`
   using the template at `template/learning_template.md`. Then STOP and tell me
   the stage is done and which lesson file to read.
5. **Do not continue to the next stage** until I say "go" or "next stage".
   If I ask questions about the lesson, answer them first.
6. **Explain for a beginner.**
   - Assume I know only KNN and K-means.
   - Define every new term the first time you use it, in plain words.
   - Use analogies and tiny examples with real numbers.
   - When introducing a new model, compare it to KNN or K-means.
   - Explain how to read each output and each plot.
   - Do not use jargon without explaining it.
7. **Keep code simple and readable.** Prefer clear code over clever code.
   Add short comments saying WHY, not just what. Use small functions.
   No unnecessary abstractions or extra libraries.
8. **No data leakage.** Features for a match may only use information from
   matches played BEFORE it. Split by time, never randomly. Flag any risk.
9. **Be honest about results.** Realistic accuracy is about 50 to 55%. Never
   tune on the test set. Never fake or inflate numbers. If something fails,
   say so and explain why.
10. **Never download or scrape data without checking.** If the sandbox cannot
    reach a data source, tell me which file to download manually and where to
    put it. Do not invent or fabricate data. Respect each site's terms.
11. **Ask before big decisions** (changing models, dropping a feature,
    switching a data source). Give me 2 to 3 options with a recommendation.
12. 12. **Keep the repo tidy and push every stage to GitHub.**
    - Follow the folder layout below. Keep `requirements.txt` updated.
    - At the end of each stage, commit with a clear message, e.g.
      "Stage 1: data collection". Commit the lesson file in the same stage.
    - Right after committing, run `git push` to `origin` and confirm it
      succeeded. Tell me the result in the stage summary.
    - Before pushing, check that no secrets (keys, tokens, passwords) and no
      large raw data files are being committed.
    - Never use `git push --force`, never rewrite history, and never change my
      git or GitHub credentials or settings. If a push fails, show me the
      error and stop instead of working around it.
    - Do not commit large raw data files, `.venv/`, or files over about 50 MB.
13. **Never edit the `template/` folder.** It holds reference files only.

## Folder layout

```
CLAUDE.md
README.md
requirements.txt
template/           reference files (requirements, plan, lesson template), do not edit
data/raw/           original downloaded files, never edited
data/processed/     cleaned data and features
notebooks/          optional exploration notebooks
src/                reusable Python code
models/             saved trained models
reports/figures/    saved plots
app/                Streamlit app
docs/learning/      lesson files (one per stage)
```

## Stages

| Stage | Name | Goal |
|---|---|---|
| 0 | Setup | Folder layout, environment, git, data access check |
| 1 | Data collection | Get match data (and later player data) into `data/raw/` |
| 2 | Preprocessing | Missing values, duplicates, encoding, scaling, time-based split |
| 3 | EDA | Plots and patterns, with an explanation of each |
| 4 | Feature engineering | Rolling form, home/away strength, Elo, head-to-head, shots |
| 5a | Baseline + Logistic Regression | Always-home-win baseline, then Logistic Regression |
| 5b | Random Forest | Train, evaluate, compare to 5a |
| 5c | XGBoost | Train, evaluate, feature importance |
| 5d | Poisson Regression | Expected goals per team, MAE |
| 6 | Model comparison | One table, same test set, choose the best with reasons |
| 7 | Season simulation | Predict 2025-26 fixtures, Monte Carlo (about 1,000 runs) |
| 8 | Player leaders | Simple top scorers and assisters estimates |
| 9 | Streamlit app | Match Predictor, Season Table, Player Leaders pages |
| 10 | Report and submission | README, report material, final checklist |

## Evaluation rules
- Metrics for classifiers: accuracy, log loss, F1, confusion matrix, plus the
  always-home-win baseline. Poisson: MAE on goals.
- All models are compared on the SAME test set (season 2025-26).
- Hyperparameter tuning uses only training data (for example time-based
  validation on the last training seasons).
- Draws are hard to predict. Expect lower accuracy on that class and explain it.

## When I ask for "the lesson"
Rewrite or extend the lesson file for the current stage. Do not move on.

## Final stage check
In Stage 10, go through `template/requirements.md` item by item and show me
where each requirement is satisfied in the project.