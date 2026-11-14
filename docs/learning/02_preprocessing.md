# Lesson 02: Preprocessing

**Stage goal in one sentence:** Turn the 11 raw season files into one clean, date-sorted match table with a time-based train/test split, and learn why each cleaning step exists.

**Files created or changed:**
- `src/preprocess.py`: the whole cleaning pipeline (run `python src/preprocess.py`)
- `src/check_preprocess.py`: assertions that verify the outputs
- `data/processed/matches_clean.csv`: all 4,180 matches, sorted by date, with `season`, `split`, targets and match stats
- `data/processed/train.csv`: 3,800 matches (2015-16 to 2024-25)
- `data/processed/test.csv`: 380 matches (2025-26)
- `data/processed/odds.csv`: the betting columns, kept apart (not model features)
- `models/team_encoder.joblib`: the fitted one-hot encoder (demo)
- `models/stats_scaler.joblib`: the fitted scaler (demo)
- `docs/learning/02_preprocessing.md`: this lesson

---

## 1. The big picture
Stage 1 bought the ingredients. Stage 2 is washing and chopping them so they can go into the pot. Models are fussy: they need numbers, a sensible order, and no rotten values. Right now the data is spread over 11 files with different date formats and different columns. After this stage there is one tidy table, plus a fair way to test a model on a season it has never seen. Stage 3 will draw plots of it, and Stage 4 will build the features the models actually learn from.

## 2. What we built
`preprocess.py` runs these steps in order. Each step is a small function.

1. **Load and combine.** 11 files become one table of 4,180 rows, with a `season` column. *Why:* one table is easier to sort and check.
2. **Parse dates.** The 2016-17 file writes `13/08/16`; the others write `08/08/2015`. We try the 4-digit pattern first, then the 2-digit one, always day first.
   - Before -> after: `'13/08/16'` -> `2016-08-13`, `'08/08/2015'` -> `2015-08-08`.
   - *Why:* if `13/08/16` were read as a text string or as month 13, sorting by date would break. The check script also confirms every date falls inside its own season.
3. **Sort by date** (then home team, then away team for matches on the same day). *Why:* later stages compute "form over the last 5 matches", which only makes sense if the table is in time order. Each row gets a `match_id` (0 to 4179).
4. **Team names.** We built a `TEAM_NAME_FIXES` dictionary, but it is **empty**, because Stage 1 found no club spelled two ways. We printed all 34 club names to confirm. *Why keep it:* the day we add another data source (for example player data), spellings will differ, and the fix goes in one clearly named place.
5. **Duplicates.** Rule: two rows with the same Date + HomeTeam + AwayTeam are the same match. Found: **0**.
6. **Missing values.** Core columns (date, teams, score, result) had **0** missing, so no row was dropped. The match stats (shots, corners, fouls, cards) also had **0** missing. We still wrote the median-fill code and applied it (0 cells filled), because the next dataset might not be so clean.
7. **Betting odds** go to `odds.csv`. There are 156 betting columns, but each season has a different set (39 columns with data in 2018-19, 108 in 2025-26), and 9.7% of the cells that exist in 2025-26 are empty. We do not use odds as features and do not fill them in.
8. **Targets.** `result` (0 = Home win, 1 = Draw, 2 = Away win), `home_goals` and `away_goals` (whole numbers). We also checked that the result agrees with the score in all 4,180 rows (0 disagreements).
9. **Split by time.** Train = 2015-16 to 2024-25 (3,800 matches). Test = 2025-26 (380 matches).
10. **Two demos** that show encoding and scaling (see sections 3 and 5).

## 3. New concepts (explained simply)
- **Cleaning / preprocessing:** fixing and reshaping data so models can use it. Garbage in, garbage out.
- **Missing value:** an empty cell. Options: drop the row (lose data), fill with the **mean** (the average, pulled around by extreme values), or fill with the **median** (the middle value when sorted, ignores extremes). Tiny made-up example: shots 8, 10, 11, 12, 40. The mean is 16.2, but the median is 11, which is more typical. In our training data the median home shots is 13 and the mean 13.9, so they are close, and the median is the safer choice.
- **Data leakage:** letting the model see information it would not have in real life. It makes the score look better than the truth. Two examples from this stage:
  1. *Median on test data.* If we computed the fill value from all matches including 2025-26, the test season would quietly influence training. So the median comes from training rows only, and that same number is used for test.
  2. *Same-match stats.* The shots, corners and cards in a row describe THAT match. They are known only after it ends. A model that uses them to "predict" that same match is reading the answer sheet. So these columns are not model inputs as they stand. In Stage 4 we turn them into averages over each team's **previous** matches.
- **Duplicate:** the same record twice. It would count one match double and distort training.
- **Encoding:** converting categories into numbers.
  - *Labels:* the result letters H, D, A become 0, 1, 2. The order is fixed everywhere: 0 = Home win, 1 = Draw, 2 = Away win.
  - *One-hot encoding:* for team names, we make one 0/1 column per team. First match of the dataset: Bournemouth v Aston Villa becomes `HomeTeam_Bournemouth = 1`, `AwayTeam_Aston Villa = 1`, and the other 66 columns are 0. We do NOT number the teams 1 to 34, because the model would think team 30 is "bigger" than team 3.
  - **Unseen team:** a team the encoder never saw while being fitted. With `handle_unknown="ignore"` it gives all zeros for that team instead of crashing. Real example: if we fit on 2015-16 to 2022-23 only, then Luton and Ipswich (promoted in 2023-24 and 2024-25) are unseen, and a row like Luton v West Ham gets an all-zero home-team block with no error. The model then knows nothing about that club. (For 2025-26 itself there is no problem: every 2025-26 team also appeared in training at some point.)
  - **Honest note:** our main plan is to describe teams by strength features (form, goals scored and conceded, Elo) in Stage 4. One-hot teams are just one possible option, not a decision.
- **Scaling (standardising):** rewriting each number as "how many standard deviations from the average". The result has mean 0 and standard deviation 1.
  - *Tiny example:* home shots have mean 13.9 and standard deviation 5.7 in training. A match with 20 shots becomes (20 - 13.9) / 5.7 = 1.07, meaning "a bit above normal".
  - **Why KNN needs it (you know this one):** KNN uses distances. Compare match A (20 home shots, 0 red cards) and match B (10 shots, 1 red card). Unscaled distance = sqrt(10² + 1²) = 10.05, so the red card barely counts: shots dominate only because their numbers are bigger. After scaling, the shots gap is 10/5.7 = 1.74 and the red-card gap is 1/0.24 = 4.18 standard deviations, so the distance is sqrt(1.74² + 4.18²) = 4.53, and now a red card counts properly. K-means has the same issue because it also uses distances.
  - **Which of our models need it:** Logistic Regression does (it is sensitive to the size of numbers and trains better when they are comparable). Random Forest and XGBoost do not: trees only ask "is this value above or below a cutoff?", so stretching a column changes nothing.
- **Time-based split:** train on the past, test on the future. Analogy: study old exam papers, then sit the new exam. A random mix would let the model learn from matches played after the ones it is tested on.
- **Validation set:** a part of the training data used to compare settings, so we never tune on the test set. Our fixed scheme (`VALIDATION_SCHEME` in `preprocess.py`): fit on 2015-16 to 2022-23 (3,040 matches), validate on 2023-24 and 2024-25 (760 matches).

## 4. How the code works
All in `src/preprocess.py`; `main()` calls these in order and prints a report.
- `load_raw()`: reads each CSV (tries UTF-8, then latin-1), and stacks them with a season label.
- `parse_dates(df)`: converts with `%d/%m/%Y`, and wherever that fails (NaT) falls back to `%d/%m/%y`. It raises an error if any date is still unreadable, rather than guessing.
- `standardize_teams(df)`: strips spaces and applies `TEAM_NAME_FIXES`.
- `drop_duplicates(df)` and `drop_rows_missing_core(df)`: remove bad rows and print exactly what was removed.
- `add_targets(df)`: makes `result`, `home_goals`, `away_goals` and checks them against each other.
- `split_by_time(df)` and `validation_split(train)`: choose rows by season, never at random. `main()` asserts that the last training date is before the first test date.
- `fill_stats_with_train_median(df)`: *why it takes the whole table*: it needs the `split` column, so it can compute medians from the training rows only and then fill both parts.
- `demo_encoding(train)` and `demo_scaling(train, test)`: call `.fit()` on training data only, then `.transform()` on other data. This `fit` on train / `transform` on both pattern is the core rule you will reuse in every stage.
- `src/check_preprocess.py`: 13 checks using `assert`. The final "All checks passed" prints only if every assertion held.

The constants at the top (`TRAIN_SEASONS`, `TEST_SEASONS`, `VALIDATION_SCHEME`, `RESULT_CODES`, `STAT_COLUMNS`) are imported by later stages, so the split is defined in exactly one place.

## 5. Results and how to read them
**Cleaning report:** 0 duplicates, 0 rows dropped, 0 missing values in core columns or match stats, 0 spelling fixes needed, 0 disagreements between result and score. *Reading it:* a clean report is the good outcome. It means there is nothing hidden that would later distort a model. A bad result would be, say, 30 duplicates or a season with many empty shot columns.

**Result counts (0 Home win, 1 Draw, 2 Away win):**

| Part | Matches | Home win | Draw | Away win |
|---|---|---|---|---|
| Whole table | 4,180 | 1,853 (44.3%) | 990 (23.7%) | 1,337 (32.0%) |
| Train | 3,800 | 1,691 (44.5%) | 886 (23.3%) | 1,223 (32.2%) |
| Test (2025-26) | 380 | 162 (42.6%) | 104 (27.4%) | 114 (30.0%) |

*What it tells us:* always guessing "home win" would be right about 44% of the time in training and 42.6% in the test season. That is our baseline to beat. Notice that 2025-26 had more draws (27.4%) than the training average (23.3%): seasons differ, and one test season of 380 matches is small, so results will be noisy.

**Boundary dates:** latest training match 2025-05-25, earliest test match 2025-08-15. Training ends before testing starts. For tuning: the latest "fit" match is 2023-05-28 and the earliest validation match 2023-08-11.

**Scaling table (training data):** the "before" columns show the real scale: home shots 13.9 on average with spread 5.7, red cards 0.06 with spread 0.24. After scaling every training column has mean 0 and standard deviation 1. The test columns are *close to* 0 but not exactly: for example away yellow cards are +0.21 in test, meaning 2025-26 had somewhat more away yellows than the training average. That difference is real information and it is correct that it is not hidden: if we scaled test with its own mean it would be hidden.

**check_preprocess.py:** all 13 checks print `PASS` and the last line is `All checks passed.` (shown in the chat summary).

## 6. Common mistakes and gotchas
- **Fitting on all the data.** Computing a median, mean, encoder or scaler on train plus test leaks test information. We call `.fit()` on training rows only.
- **Random split.** Would put 2025 matches into training and then "predict" 2024 ones. Not allowed.
- **Mixed date formats.** `13/08/16` and `08/08/2015` in one column. We parse both and check dates against their season.
- **Same-match stats as inputs.** Shots, corners and cards of match M are known only after M. They are targets-in-disguise if used for M.
- **Numbering teams 1 to 34.** The model would assume an order that does not exist. Use one-hot instead.
- **Forgetting the unseen-team case.** A newly promoted club has no history. Our encoder returns zeros instead of crashing, but the model will know nothing about that club. This is one reason Stage 4 describes teams by strength numbers.
- **Editing raw files.** Everything is written to `data/processed/`; `data/raw/` was not touched.

## 7. Check yourself
1. Why do we fill missing shots with the median from the training data and not from the whole dataset?
2. The shots in a match row describe that match. Why can't we use them as inputs to predict that same match, and what will we do with them in Stage 4?
3. Why is "H, D, A -> 0, 1, 2" fine for the target, but numbering teams 1 to 34 is a bad idea for team names?
4. Why would KNN give a poor result if we did not scale shots and red cards, and why do Random Forest and XGBoost not care?
5. Why do we split by season instead of picking 10% of the matches at random?

## 8. Try it yourself
In the project folder with the environment active, run:
```
python -c "import pandas as pd; tr = pd.read_csv('data/processed/train.csv'); te = pd.read_csv('data/processed/test.csv'); print(tr['AY'].mean(), te['AY'].mean())"
```
`AY` is away yellow cards per match. Before running, predict: is the 2025-26 average higher or lower than the training average? Then change `AY` to `HS` (home shots) and `HR` (home red cards) and predict again. Which column differs most between train and test, measured against its spread (standard deviation 1.3 for `AY`, 5.7 for `HS`, 0.24 for `HR`)? Finally, think: if we had scaled the test data using its own mean, what would we have hidden?

## 9. How this connects to the report
This stage feeds the **Data Preprocessing** section of the report: missing values, duplicates, encoding, scaling, and the train/test split. These are required items on the course checklist.

*Draft paragraph:* "Eleven seasons of match data (4,180 matches) were combined into one table, with dates parsed from two different formats and sorted chronologically. We checked for duplicate matches (none), missing values in the core columns (none) and inconsistent team names (none). Match outcomes were encoded as Home win, Draw and Away win, and team names can be one-hot encoded. Numeric features are standardised using statistics from training data only. To avoid data leakage the split is by time: seasons 2015-16 to 2024-25 are used for training and tuning, and the complete 2025-26 season is kept as an unseen test set."

---

## Answers to "Check yourself"
1. If the median came from the whole dataset, the test matches would influence the numbers the model trains with. That is leakage and makes the test result look better than it really is. Test data should only be filled using numbers learned from training.
2. Because those stats are only known after the match is over, so using them to predict it is like reading the answer sheet. In Stage 4 we replace them with averages over each team's earlier matches (for example shots per game over the last 5 matches), which are known before kick-off.
3. The result classes are the answer the model must output, and they are just names, so 0, 1, 2 are fine labels. For inputs, numbers imply order and size, so team 30 would look "bigger" than team 3. One-hot gives each team its own 0/1 column with no fake order.
4. KNN uses distances, so a column with big numbers (shots, spread about 5.7) dominates a column with small numbers (red cards, spread about 0.24), and red cards would be almost ignored. Scaling puts them on the same footing. Trees only compare a value against a cutoff, so stretching or shrinking a column does not change their splits.
5. A random split would put future matches in the training set and test on past ones, which cannot happen in real life and leaks information. Splitting by season mimics the real task: learn from the past, predict the next season.
