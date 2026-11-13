# Premier League Prediction: Project Plan

AI Lab group project (Group #8). Prepared for team discussion.

## 1. Overview
We will build a machine learning system that predicts Premier League match
results and simulates a full season table. We train on past seasons (up to
2024-25) and then test the system on the 2025-26 season, which has already been
played. This lets us compare our predictions with what actually happened. The
final model is deployed in a simple Streamlit web app.

Everything uses free tools and free data, so the cost is $0.

## 2. Problem Statement
Football results are hard to predict because of the many factors involved: team
form, home advantage, attacking and defensive strength, and chance. We want to
answer two questions using historical data:
- Given two teams and a home/away setting, what is the likely result and score?
- If we predict every fixture of a season, what would the final league table
  look like, and who would be the top scorers and assisters?

## 3. What the Project Will Do

### Core features
- **Match predictor:** the user picks a home team and an away team (for example
  Chelsea vs Arsenal). The app shows the probability of Home win / Draw / Away
  win and the expected goals for each side.
- **Season simulator:** predict all 380 fixtures of 2025-26, run a Monte Carlo
  simulation (about 1,000 runs, goals sampled from a Poisson distribution) and
  produce the final points table with each team's chance of winning the title,
  finishing in the top 4 and being relegated.
- **Player leaderboards (simple version):** estimated top scorers and top
  assisters, based on each player's per-90-minute rates and expected minutes,
  or on his share of his team's goals.

### Deliberately limited or removed (too unreliable or too expensive)
- Predicting exactly which player scores in one match: shown only as
  probabilities (team expected goals multiplied by the player's goal share),
  never as a certain prediction.
- Player records against a specific opponent: too few samples to be trustworthy.
- Most saves: dropped, or shown as a rough estimate only.
- Deep learning: not needed. Tree-based models usually work best on tabular
  data like this.

## 4. Dataset

### Sources (all free)
- football-data.co.uk: match results, shots, cards and betting odds for every season.
- Fantasy Premier League (FPL) API: player goals, assists, minutes, saves and
  clean sheets.
- Kaggle Premier League datasets: backup and extra data.
- Understat: expected goals (xG), optional. We will check each site's terms
  before scraping.

### Seasons
Training data is several seasons up to 2024-25 (for example 2015-16 to
2024-25). The test set is 2025-26.

### Target (output)
- Match outcome: Home win / Draw / Away win (classification).
- Goals scored by home and away team (used for the Poisson model and the
  simulation).

### Input features
All features are built using only matches played before the fixture, so no
future information leaks in.
- Rolling form: points, goals scored and conceded over the last 5 matches.
- Home and away strength: average goals for and against at home and away.
- Elo rating for each team.
- Head-to-head history between the two teams.
- Shots and shots on target averages.
- Optional: team-level player aggregates such as squad scoring strength.

## 5. Data Preprocessing
1. **Missing values:** identify them and fill (mean/median for numbers) or drop
   where appropriate.
2. **Duplicates:** remove duplicate matches and player rows.
3. **Encoding:** one-hot encode team names (or use team strength features);
   encode the target labels.
4. **Feature scaling:** standardise numeric features, required for Logistic
   Regression.
5. **Train/test split:** time-based. Train on earlier seasons and test on
   2025-26. We will not use a random split, because it would leak future
   matches into training.

## 6. Exploratory Data Analysis
Plots and analysis to include:
- Distribution of match outcomes (home advantage: how often home teams win).
- Distribution of goals per match.
- Correlation heatmap of features.
- Team strength over seasons (points and goal difference).
- Home vs away performance per team.
- Top scorers and assisters across seasons.

For each plot we will write the pattern it shows, for example that home teams
win about 45% of matches.

## 7. Models (minimum 2, maximum 4)

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline classifier for Home/Draw/Away |
| Random Forest | Non-linear classifier, handles feature interactions |
| XGBoost | Gradient boosting, likely the strongest on tabular data |
| Poisson Regression | Predicts expected goals, drives the score prediction and season simulation |

Evaluation metrics: accuracy, log loss, F1-score, confusion matrix, and a
comparison against a simple baseline such as always predicting a home win. For
Poisson: mean absolute error on goals.

A realistic accuracy is around 50-55%, similar to bookmakers. We should not
expect 80%. Showing a result that beats the baseline and explaining why is what
matters.

## 8. Model Comparison
- Put all models side by side in one table using the same metrics on the same
  test set.
- Choose the best model and justify it with numbers.
- Explain how the chosen model works. For XGBoost this means boosting (trees
  built one after another, each correcting the previous errors) and feature
  importance.
- Everyone in the group must be able to explain this, since understanding the
  selected model is mandatory.

## 9. Web App (Streamlit)
Streamlit is free, written in Python, and can be hosted free on Streamlit
Community Cloud.
- **Page 1, Match Predictor:** select home and away teams and see win/draw/loss
  probabilities and the expected score.
- **Page 2, Season Table:** the simulated 2025-26 table next to the real one,
  with title, top-4 and relegation probabilities.
- **Page 3, Player Leaders:** estimated top scorers and assisters.

## 10. Tools (all free)
Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib/seaborn, Streamlit,
Google Colab or Jupyter, GitHub.

## 11. Submission Checklist
- GitHub repository link with the source code
- Project report
- Dataset and source links
- Demo of the web app

## 12. Suggested Work Split
- Member 1: data collection, preprocessing and EDA.
- Member 2: feature engineering, model training and comparison.
- Member 3: Streamlit app, season simulation and report.

All three members should understand the whole pipeline, especially the chosen
model, because any member may be asked about it.

## 13. Risks
- Data leakage: features must only use past matches.
- Player-level predictions are weak, so they are presented as estimates.
- Scraping limits: use downloadable CSVs and the FPL API first.
- Draws are hard to predict: expect lower accuracy on that class.