# Lesson 00: Setup

**Stage goal in one sentence:** Prepare a tidy, reproducible workspace (folders, Python environment, git) and check that we can get match data from the internet.

**Files created or changed:**
- `README.md`: project description, folder layout and setup instructions
- `requirements.txt`: the list of Python libraries the project needs
- `.gitignore`: tells git which files NOT to track
- `src/check_setup.py`: small script that checks every library is installed
- `.venv/`: the virtual environment (not tracked by git)
- `data/raw/`, `data/processed/`, `notebooks/`, `models/`, `reports/figures/`, `app/`, `docs/learning/`: empty folders, each holding a `.gitkeep` file so git keeps them
- `docs/learning/00_setup.md`: this lesson

---

## 1. The big picture
Stage 0 is like setting up a kitchen before cooking. You do not cook a meal yet, but you put the knives, pans and ingredients in sensible places, and check the stove works. Every later stage (collecting data, cleaning it, training models, building the app) needs the same workspace, so doing it once, carefully, saves a lot of confusion. The last thing we check is the "supermarket": can we actually fetch data from football-data.co.uk automatically? The answer decides how Stage 1 will work.

## 2. What we built
1. **Folder layout.** Each kind of file has one home, so nothing gets lost.
   - `data/raw/`: the original files exactly as downloaded. We never edit them.
   - `data/processed/`: cleaned data and the features we create from it.
   - `notebooks/`: optional experiments in Jupyter.
   - `src/`: reusable Python code (functions we write once and use many times).
   - `models/`: trained models saved to disk.
   - `reports/figures/`: plots saved as images.
   - `app/`: the Streamlit web app.
   - `docs/learning/`: lesson files like this one, one per stage.
   - `template/`: reference files from the course and the plan. We do not edit these.
2. **Virtual environment `.venv`.** A private box of Python libraries just for this project.
3. **`requirements.txt`.** The shopping list of libraries, so a teammate can rebuild the same environment with one command.
4. **`.gitignore`.** A list of things git should skip (the `.venv`, cache files, big raw data).
5. **`src/check_setup.py`.** Imports every library and prints its version.
6. **Git repository and first commit.** A saved snapshot of the project.
7. **Data access test.** We downloaded (in memory only) the 2023-24 Premier League file to see if the internet and the site work.

**Why keep raw and processed data separate?** Imagine you photocopy a document before highlighting it. The photocopy (raw) is your safety copy. If you make a mistake while cleaning, you simply start again from the raw file. It also makes our work honest and repeatable: anyone can see exactly what changed between raw and processed.

*Tiny example with our data:* the 2023-24 file has 380 matches, because 20 teams each play 38 games and every game involves two teams: 20 x 38 / 2 = 380. The first row is Burnley 0-3 Man City (result `A`, meaning away win). That row stays untouched in `data/raw/`. Later, our cleaned version goes in `data/processed/`.

## 3. New concepts (explained simply)
- **Virtual environment:** a separate folder holding the exact Python libraries for one project.
  Analogy: a lunchbox per project. Project A can use pandas 2.x and project B pandas 1.x without them fighting. It also keeps your computer's main Python clean.
- **Library (package):** code written by other people that we reuse instead of writing it ourselves. You install it once with `pip install`, then `import` it.
- **pandas:** tables of data (like Excel in code). Used to load the match CSV files, clean them and build features (Stages 1 to 4).
- **numpy:** fast maths on lists of numbers (arrays). pandas and scikit-learn are built on top of it. Used everywhere, especially in the Monte Carlo season simulation.
- **scikit-learn:** ready-made machine learning tools. You already know KNN and K-means, and both live in this library. We use it for the train/test split, scaling, Logistic Regression, Random Forest and the evaluation metrics (Stages 2 to 6).
- **xgboost:** a strong model that builds many small decision trees one after another, each fixing the mistakes of the last. Used in Stage 5c.
- **matplotlib:** draws plots (bars, lines, histograms). Used to save pictures in `reports/figures/`.
- **seaborn:** nicer-looking plots made on top of matplotlib, for example the correlation heatmap in the EDA stage.
- **scipy:** scientific maths, including probability distributions such as the Poisson distribution we use to sample goals in the season simulation.
- **statsmodels:** classical statistical models. We use it for Poisson Regression, which predicts the expected number of goals (Stage 5d).
- **streamlit:** turns a Python script into a web app with almost no web knowledge. Used for the final app (Stage 9).
- **requests:** downloads things from the internet, such as CSV files or the Fantasy Premier League data.
- **jupyter:** notebooks, where you run code in small chunks and see results right below. Optional, for exploration.
- **Git:** a tool that saves the history of your project, like "save points" in a video game.
- **Commit:** one save point, with a message describing what changed. We commit at the end of every stage so we can go back if something breaks and so the group can see the progress. The course also requires a GitHub link, and a clean commit history helps.
- **Relation to KNN/K-means:** none of today's work is a model yet. But KNN and K-means come with scikit-learn, so you will meet familiar tools in unfamiliar clothes later.

## 4. How the code works
`src/check_setup.py` has one small function, `main()`:
- It holds a list of (pip name, import name) pairs. They differ sometimes: you `pip install scikit-learn` but write `import sklearn`.
- For each pair it tries `importlib.import_module(...)`. If this works, it prints the version. If Python cannot find it, an `ImportError` is raised, which we catch and report as `MISSING`.
- At the end it prints `Setup OK` only if nothing is missing. This way the script cannot "pretend" to pass.

## 5. Results and how to read them
**check_setup.py output:** see the real output in the Stage 0 summary in chat. What to look for: every line says `OK` with a version number, and the last line is `Setup OK`. A line saying `MISSING` means that library is not installed in the active environment (a common cause is forgetting to activate `.venv`).

**Data access test:**
- URL: `https://www.football-data.co.uk/mmz4281/2324/E0.csv`
- Result: **success**. HTTP status 200, about 172 KB, 380 matches plus a header row.
- Columns include the date, teams, full-time goals (`FTHG`, `FTAG`), result (`FTR`), shots (`HS`, `AS`, `HST`, `AST`) and betting odds.
- What it tells us: Stage 1 can download the season files automatically with `requests` or pandas. The other sources (FPL API) will be tested in Stage 1 before we depend on them.
- A bad result would be a status like 403 (blocked) or 404 (wrong file name), or a timeout. Then we would download the files by hand and put them in `data/raw/`.

## 6. Common mistakes and gotchas
- **Forgetting to activate the virtual environment.** Then `pip install` and `python` use the wrong Python. Look for `(.venv)` at the start of your terminal line.
- **Committing the `.venv` or huge data files.** The `.gitignore` prevents this.
- **Editing raw data by hand.** Never do it; always produce cleaned copies in `data/processed/`.
- **Empty folders vanish in git.** Git does not track empty folders, so we add a hidden `.gitkeep` file to each.
- **Forgetting `requirements.txt`.** Without it, a teammate cannot rebuild your setup.

## 7. Check yourself
1. Why do we keep `data/raw/` and `data/processed/` separate?
2. What problem does a virtual environment solve?
3. What is a commit, and why do we make one at the end of every stage?
4. Which library would you use to draw a plot, and which to download a file?
5. Why did the 2023-24 file contain exactly 380 matches?

## 8. Try it yourself
Open a terminal in the project folder with the virtual environment active and run:
```
python -c "import pandas as pd; df = pd.read_csv('https://www.football-data.co.uk/mmz4281/2324/E0.csv'); print(df.shape)"
```
Before running it, predict how many rows and roughly how many columns you will see. Then change `2324` to `2223` (the 2022-23 season) and predict again. Did the row count change? (Hint: both seasons have 20 teams.) Do not save the file anywhere; this is just a peek.

## 9. How this connects to the report
This stage feeds the **Dataset** and **Tools/Methodology** sections of the report, and the GitHub link requirement in the submission.

*Draft paragraph:* "The project is organised in a reproducible repository with separate folders for raw data, processed data, source code, models, figures and the web app. We use a Python virtual environment and a `requirements.txt` file so any team member can recreate the same setup. Match data comes from football-data.co.uk, which provides free CSV files per season with results, shots and betting odds. We verified that these files can be downloaded automatically. All work is version controlled with git, with one commit per project stage."

---

## Answers to "Check yourself"
1. The raw folder is the untouched original, our safety copy. If cleaning goes wrong we start over from raw. It also makes the work repeatable and honest.
2. Different projects need different library versions. A virtual environment gives each project its own set, so they do not clash and your main Python stays clean.
3. A commit is a saved snapshot of the project with a message. Committing after each stage lets us go back if something breaks and shows clear progress to the group and the course.
4. matplotlib (or seaborn) for plots; requests (or pandas' `read_csv` with a URL) to download.
5. 20 teams each play 38 matches, but every match involves 2 teams, so 20 x 38 / 2 = 380.
