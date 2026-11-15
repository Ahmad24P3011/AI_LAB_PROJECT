# Lesson 09: The Streamlit web app

**Stage goal in one sentence:** turn the saved models and the saved simulation into a web app where anyone can pick two teams and see the predictions, and browse the simulated 2025-26 table next to the real one.

**Files created or changed:**
- `app/streamlit_app.py`: entry point (page settings, theme, sidebar, the two pages)
- `app/views/match_predictor.py`: Page 1, pick two teams, see probabilities, expected score, comparison table, scoreline chart
- `app/views/season_table.py`: Page 2, simulated v real table, points-fan chart, position heatmap
- `app/predictor.py`: loads the saved models and answers "home team v away team" (no Streamlit code, so it can be tested)
- `app/styles.py`: colours, CSS and small HTML builders (banner, tiles, crests)
- `app/team_colors.py`: each team's colour, 3-letter code and crest file name
- `app/team_state_builder.py`: run once; writes the end-of-2025-26 team state, the bookmaker table and the position-probability table
- `app/fetch_crests.py`: run once; downloads the 20 crests
- `app/requirements.txt`: the minimal, pinned package list for deployment
- `.streamlit/config.toml`: the dark purple theme
- `data/processed/team_state_end_of_2025-26.csv`: one row per team (Elo and last-10 form after its last match)
- `data/app/bookmaker_2025-26.csv`, `data/app/real_results_2025-26.csv`, `data/app/sim_position_probs.csv`, `data/app/hero_ball.svg`: small files the app reads
- `app/make_hero_ball.py`: draws the banner football
- `src/check_app.py`: the checks; `src/screenshot_app.py`: takes the two screenshots
- `reports/figures/46_app_screenshot_match.png`, `47_app_screenshot_season.png`
- `README.md` ("How to run the app"), `docs/decisions.md` (Stage 9 section), `requirements.txt` (added `pillow`), `.gitignore` (crest PNGs)

---

## 1. The big picture
Until now everything lived in code files and CSV tables that only we can run. A **web app** turns that work into a service: someone with a browser, who has never seen Python, can use the models. It is like the difference between a recipe in a notebook and a restaurant: the recipe (training) was finished in Stages 4 to 7, and the app is the counter where customers order. That is also course requirement 7: "deploy the model in a web app and demonstrate how users provide input and receive results". Stage 10 (report and final checklist) comes after this.

## 2. What we built
1. **The team state** (`team_state_builder.py`). A model needs numbers describing the two teams "before the match". For a match that has not happened we need those numbers as they stand today. We took each team's Elo and last-10 goal form as of its last 2025-26 match (24 May 2026). Example: Arsenal Elo 1728, Man City 1697, Burnley 1339.
2. **Page 1, Match Predictor.** You pick a home and an away team. The app shows the three probabilities from the primary model (ELO2 Logistic Regression), a football-style scoreboard with the Poisson expected goals, a table comparing all four model families and the bookmaker, and a bar chart of the ten likeliest exact scores. Example (Liverpool v Bournemouth): Logistic Regression 40% / 28% / 32%, expected goals 1.4 : 1.3, most likely exact score 1-1 (12%).
3. **Page 2, Season Table.** The Stage 7 results: simulated and real table side by side, a chart of how a chosen team's points could have gone over 1,000 simulated seasons, and a heatmap of where every team finished.
4. **"What really happened" (Page 1).** Below the scoreline chart the app shows the real 2025-26 result of the chosen fixture (for example Liverpool 4-2 Bournemouth, 15 August 2025) and the chance each model gave to the outcome that really happened (40%, 47%, 44%, 41% and the bookmaker 73% for a home win). **Important:** this is a sanity check, not a test. The app describes teams as of the END of the season, which already includes this very match, so it is not a fair out-of-sample prediction; the fair test is Stage 6. The scores come from one small precomputed file (`data/app/real_results_2025-26.csv`) and are never a model input (the check proves that predictions do not change if the scores are zeroed).
5. **Honest framing.** A sidebar says what the project is and lists three limitations; every page ends with "For educational purposes only."

## 3. New concepts (explained simply)
- **Web app:** a program that runs on a server and shows its results in your browser. *Analogy:* a vending machine; you press a button, the machine (server) does the work, you only see the result.
- **Streamlit:** a Python tool that builds a web page from an ordinary script. Every time you click something, Streamlit runs the script again from top to bottom and redraws the page. *Analogy:* a whiteboard that is wiped and redrawn from the same instructions after every click. (KNN and K-means have nothing like this; they are algorithms, not delivery tools.)
- **Serving a model:** loading a model that was already trained and asking it for predictions. It is cheap and instant. **Training** is the slow learning step. The app never trains.
- **Cache (`st.cache_resource`):** Streamlit remembers a result between re-runs so the models load once, not on every click.
- **Hypothetical match:** a fixture that is not in the data. We make up "Liverpool v Bournemouth, today" and ask the model what it thinks.
- **Pinned versions:** writing `numpy==2.5.4` instead of `numpy`. A saved model is like a file written in a particular program version; the same version is needed to open it safely.
- **Deployment:** putting the app on a public server (Streamlit Community Cloud, free) so it has a web address.

## 4. How the code works
- **`predictor.py` → `Predictor.predict(home, away)`.** In: two team names. It looks up their rows in the team state, computes the Elo gap (`home elo - away elo`), gives it to the three classifiers, builds the 12 inputs the Poisson model needs, converts the two expected goals into a grid of scorelines and sums the grid into Home/Draw/Away. Out: a dictionary of plain numbers.
- **The "end of 2025-26 state" trick** (`team_state_builder.py`). Instead of re-writing the feature code for a future match, we add 380 pretend fixtures (every home/away pair) to the real match list on a single day after the season, and run the project's own feature code. That code only uses matches on EARLIER dates, so every pretend fixture receives exactly the state of each team after its last real match. This way the app's numbers are produced by the same code the models were trained with. `check_app.py` proves it: for all 380 pairs the app's inputs equal the pipeline's, with maximum difference 0.
- **Why Elo is computed separately.** The Elo walk updates row by row, so the pretend fixtures would disturb one another. Elo comes from the real matches only.
- **The label swap.** Stage 7 documented that two Poisson columns were stored under swapped labels in `features_goals.py`. The saved model learned from that layout, so `poisson_inputs()` reproduces it (and says so in a comment). The check above covers it.
- **Pages and styling.** `streamlit_app.py` calls `st.navigation` with the two pages. `styles.py` holds the CSS and returns small HTML strings (banner, tiles, scoreboard, crests); the standings table is plain HTML so the colour shading is exact.
- **No raw data in the app.** The pages read only `models/`, the small files above and `reports/sim_*`. `check_app.py` scans the runtime code for raw data file names, score columns and `.fit(`.

## 5. Results and how to read them
**Checks (`python src/check_app.py`, all passed):**
```
PASS  all 13 files exist; models, team state and bookmaker table load cleanly
PASS  logreg, random_forest, xgboost and poisson files are unchanged since Stage 6 (sha256 equal)
PASS  20 teams in the team state; Elo 1339 to 1728; every team played 38 matches
        logreg         [0.3985 0.2778 0.3237]  sum 1.000000
        random_forest  [0.4682 0.2524 0.2794]  sum 1.000000
        xgboost        [0.4382 0.2657 0.2961]  sum 1.000000
        poisson        [0.4054 0.2583 0.3363]  sum 1.000000
        bookmaker      [0.7301 0.1582 0.1117]  sum 1.000000
        Poisson lambdas 1.416 + 1.267 = 2.683
PASS  all 380 pairs: app inputs equal the training-time feature code (max difference 0.0e+00)
...  (and: no raw data / training code in the app, minimal requirements, crests, friendly error for a bad team state)
```
**Figure 46** (`reports/figures/46_app_screenshot_match.png`) is Page 1; **figure 47** is Page 2.

How to read Page 1: the three coloured tiles are the primary model's probabilities (green home, grey draw, cyan away). The scoreboard shows the Poisson expected goals (a mean, so 1.4, not a whole number). In the table, the green bar marks the primary model. A "good" result is not a confident one: all models here give the draw 25 to 28%, and they disagree by a few points, which is the honest picture from Stage 6 (they tied).

**Surprise worth understanding.** The bookmaker row (73% home win) is far from the models (40 to 47%). It is not that the bookmaker is "right": it is the price of the REAL fixture in August 2025, when the Elo gap was +155 points for Liverpool (the Stage 6 worked example); the models describe the teams at the END of the season, when Bournemouth's Elo (1606) was above Liverpool's (1584). Different dates, so the page says "not like-for-like".

How to read Page 2: green shading is the chance of the title, magenta the chance of relegation, "Position error" is simulated mean position minus real position. The fan chart shows the median (line) and 25-75 / 5-95 percent bands of cumulative points over 1,000 simulated seasons; the green dot is the real final total (Arsenal: 85 against a median near 70, near the top of the 5-95 band, as Stage 7 found).

## 6. Common mistakes and gotchas
- **Training inside the app.** It would make the app slow and risk leakage; ours loads saved files only (`check_app.py` forbids `.fit(`).
- **Reading raw 2025-26 scores in the app.** Forbidden by the check. The Season Table only shows the Stage 7 comparison files.
- **Different package versions on the server.** Pickled models can fail or change silently. We pinned the versions, which forced Python 3.12 instead of 3.11.
- **Inventing a team state.** We built it with the project's feature code, not by hand, and tested all 380 pairs.
- **Comparing things from different dates.** The bookmaker row is labelled, not hidden.
- **Dark colours on a dark page.** Fulham's black disappeared; we blend dark team colours toward white.
- **Copyright.** Club crests are trademarked, non-free logos. They are included here for this non-commercial university project only (sources in `data/app/team_crests/SOURCES.md`); if the files are missing, the app shows coloured 3-letter badges instead. No player photos are used.

## 7. Check yourself
1. What is the difference between training a model and serving it? Which one does the app do?
2. Why does the app need a "team state" file, and why did we build it with 380 pretend fixtures instead of writing new code?
3. The bookmaker gives Liverpool 73% against Bournemouth and the models give 40 to 47%. Is the bookmaker simply better? Explain.
4. Why did we pin exact package versions in `app/requirements.txt`?
5. On the fan chart, is the shaded band a prediction interval for the real 2025-26 season? What is it?

## 8. Try it yourself
Open Page 1 and compare the pairs Arsenal v Burnley and Burnley v Arsenal. Predict first: will the home-win probability of the first be larger than the away-win probability of the second, and by how much? Then run it. (Hint: the models learned a home advantage, so the two are not mirror images.)

## 9. How this connects to the report
Feeds the "Deployment / web application" section and the demo required by the submission checklist.

> *Draft paragraph:* The selected models were deployed in a Streamlit web app with two pages. The Match Predictor lets the user choose a home and an away team and returns the Home/Draw/Away probabilities of the primary Logistic Regression, the Poisson expected score, and a comparison with the Random Forest, XGBoost and the bookmaker. The Season Table page shows the simulated 2025-26 table against the real one. The app loads saved models only; predictions for a hypothetical match use each team's Elo and form as of its last 2025-26 match, built with the same feature code used in training. The app states its limitations (draws are not predicted well; the bookmaker is a strong benchmark) and is for educational purposes only.

**Deployment, in 5 steps (Streamlit Community Cloud, free):** (1) push the repo to GitHub; (2) sign in at share.streamlit.io with GitHub and choose "New app"; (3) pick the repository, branch `master` and main file `app/streamlit_app.py`; (4) in "Advanced settings" choose Python 3.12; (5) click Deploy.

**What a real prediction service would need** (and this project does not have): live injury and suspension news, confirmed team sheets, transfers, fixtures that are really upcoming, odds that update, and regular retraining. It is a teaching demo, not for betting.

---

## Answers to "Check yourself"
1. Training = the model learns from past data (slow, done once, in Stages 5 to 7). Serving = loading the saved model and asking it for a prediction (instant). The app only serves.
2. The models need numbers describing each team before the match. For a match that has not been played, those numbers do not exist in the data, so we compute them as of the last 2025-26 match. We used the project's own feature code (via 380 pretend fixtures) so the numbers are made exactly the way the models expect; a hand-written copy could differ subtly.
3. No. The bookmaker's odds are for the real August 2025 fixture, before the season; the models use end-of-season ratings, when Bournemouth had overtaken Liverpool in Elo. They describe different moments. On the test season (Stage 6) the bookmaker and the models were statistically tied.
4. A saved model file is tied to the library versions that wrote it. Pinning the versions makes the server load it identically; the pinned numpy, scipy and xgboost versions are why Python 3.12 is needed.
5. No. It is the spread of results over 1,000 simulated seasons of the same model (percentiles 5, 25, 50, 75, 95). The real season is one more draw from reality, which can fall anywhere, as Arsenal's 85 points did near the top of the band.
