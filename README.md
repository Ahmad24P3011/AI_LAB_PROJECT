# Premier League Prediction (AI Lab, Group #8)

We build a machine learning system that predicts Premier League match results
and simulates a full season table. We train on past seasons (up to 2024-25) and
test on the 2025-26 season, which has already been played, so we can compare our
predictions with what really happened. The final model is deployed in a simple
Streamlit web app with a match predictor, a simulated season table and player
leaderboards. Everything uses free tools and free data.

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

## How to set up

Windows (PowerShell):
```
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/check_setup.py
```

Mac / Linux:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/check_setup.py
```

The last command should end with `Setup OK`.

## Run the app (later, Stage 9)

```
streamlit run app/app.py
```
