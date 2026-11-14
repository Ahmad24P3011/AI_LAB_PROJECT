# Lesson 01: Data collection

**Stage goal in one sentence:** Download 11 seasons of raw Premier League match data (2015-16 to 2025-26) in a repeatable way, and check what is really inside the files.

**Files created or changed:**
- `src/download_data.py`: downloads one CSV per season into `data/raw/`
- `src/inspect_raw.py`: read-only checks (row counts, columns, duplicates, team names)
- `data/raw/E0_2015-16.csv` ... `E0_2025-26.csv`: the 11 raw season files (never edited)
- `data/raw/SOURCES.md`: where the data came from, terms found, and a data dictionary
- `.gitignore`: changed so the small raw CSVs and SOURCES.md are tracked by git
- `docs/learning/01_data_collection.md`: this lesson

---

## 1. The big picture
Every model learns from examples, so before anything else we need the examples. Think of cooking: Stage 0 built the kitchen, and Stage 1 is buying the ingredients and checking them before we cook. If an ingredient is rotten (a missing column, a misspelled team name, a half-finished season), we want to find out now, not after a model quietly learns from bad data. Stage 2 will clean what we found, Stage 3 will draw plots, and Stage 4 will turn the rows into features.

## 2. What we built
1. **`download_data.py`.** It loops over 11 season codes and downloads each file.
   - *Why:* a script means anyone in the group can rebuild the exact same raw folder with one command, instead of clicking in a browser and maybe picking the wrong file.
   - It skips files that already exist, waits 1 second between requests (politeness), and rejects anything that is not a real CSV (some sites send an HTML error page with a "200 OK" status).
   - *Real result:* 11 of 11 downloaded, 94 KB to 199 KB each.
2. **`inspect_raw.py`.** It only reads, never writes.
   - *Why:* we must know what is in the data before we trust it.
   - *Real result:* every season has exactly 380 rows (20 teams x 38 games / 2 = 380), 0 duplicate rows, and 0 empty cells in the key columns (teams, goals, result, shots, Bet365 odds).
3. **`SOURCES.md`.** A record of the source, the date downloaded, the terms found and what each column means. The report needs this ("describe the dataset and its source").

**One real row, explained.** The first match of 2025-26 (15 Aug 2025):

| Column | Value | Meaning |
|---|---|---|
| HomeTeam / AwayTeam | Liverpool / Bournemouth | who played, and who was at home |
| FTHG / FTAG | 4 / 2 | full-time goals: Liverpool 4, Bournemouth 2 |
| FTR | H | full-time result: home win |
| HTHG / HTAG | 1 / 0 | at half time Liverpool led 1-0 |
| HS / AS | 19 / 10 | shots: Liverpool 19, Bournemouth 10 |
| HST / AST | 10 / 3 | shots on target (a shot that would score if not saved) |
| HC / AC | 6 / 7 | corners won |
| HF / AF | 7 / 10 | fouls committed |
| HY / AY, HR / AR | 1 / 2, 0 / 0 | yellow and red cards |
| B365H / B365D / B365A | 1.30 / 6.00 / 8.50 | Bet365 odds for home win / draw / away win |

## 3. New concepts (explained simply)
- **Raw dataset:** the data exactly as it came from the source.
  Analogy: the original photo. You edit a copy, never the original, so a mistake can always be undone and anyone can see what you changed. That is why raw files are never edited and cleaned data goes to `data/processed/`.
- **CSV:** a plain text table, one row per line, columns separated by commas.
- **Row (observation):** one match here. In KNN terms, one point in space.
- **Column (feature or target):** one measured thing about a match. In KNN, one axis of that space.
- **Target:** the thing we want to predict. Features are the clues we use.
- **Decimal odds:** what a bet pays back per 1 unit staked, stake included. Odds of 1.30 means you win only 0.30 extra, so the bookmaker thinks the event is very likely. Rough chance = 1 / odds. Liverpool: 1/1.30 = 77%, draw 1/6 = 17%, Bournemouth 1/8.5 = 12%. These sum to 105.4%, not 100%. The extra 5.4% is the bookmaker's built-in profit margin.
- **Train vs test seasons:** train means the model learns from past seasons (2015-16 to 2024-25, 3,800 matches). Test means we then check it on a season it has never seen (2025-26, 380 matches).
  Analogy: study from past exam papers, then sit a new exam. If you saw the new exam's answers while studying, your mark would mean nothing. Same idea as keeping a test point out of KNN's neighbours.
- **Classification and regression:** classification predicts a category, regression predicts a number. We have both in one dataset, because it contains both kinds of answers (see below).

## 4. How the code works
- `download_data.py`
  - `SEASONS`: a list pairing the site's code (`2324`) with our file label (`2023-24`).
  - `looks_like_csv(response)`: checks the first 2,000 characters contain the header word `HomeTeam` and no `<html`. *Why:* a status 200 alone does not prove the content is a CSV.
  - `download_season(code, label)`: skips if the file exists; otherwise requests, checks the status and content, then saves `response.content` as raw bytes. *Why bytes:* so the file is stored unchanged, with no re-encoding.
  - `main()`: loops, sleeps 1 second after each real request, and prints OK / SKIP / FAIL per season.
- `inspect_raw.py`
  - `load_file(path)`: tries UTF-8, then latin-1, and tells you if it needed the fallback. (It never did: all 11 files opened as UTF-8.)
  - Separate small functions print each check: basic counts, column overlap, empty cells, team names and 2025-26 progress.

## 5. Results and how to read them
**Rows and columns per season** (all 380 rows, 0 duplicates):

| Seasons | Columns |
|---|---|
| 2015-16 to 2017-18 | 65 |
| 2018-19 | 62 |
| 2019-20 to 2023-24 | 106 |
| 2024-25 | 120 |
| 2025-26 | 132 |

- *What it tells us:* the **match information is the same** in every season, but the **betting-odds columns change**. Only 35 columns appear in all 11 seasons. Those 35 include the basics (teams, goals, result, shots, corners, fouls, cards) plus Bet365, Bet&Win and Pinnacle odds.
- *Good vs bad:* good is 380 rows and no duplicates. A season with fewer rows would mean an unfinished season or missing games.

**2025-26 is complete:** 380 of 380 matches, dated 15 Aug 2025 to 24 May 2026, and every team has played 38 games. So it is a valid test season.

**Team names:** 34 distinct clubs appear across 11 seasons, and **no club is spelled two different ways**. Names that appear in only some seasons are just promoted or relegated clubs (for example Cardiff only in 2018-19, Ipswich only in 2024-25). The site's own short forms are `Man City`, `Man United`, `Nott'm Forest` and `Wolves`. We must keep that exact spelling when we join data from other sources (for example FPL uses different names), otherwise Man United and "Manchester Utd" would look like two clubs.

**Other things found:**
- **Dates:** 2016-17 uses a 2-digit year (`13/08/16`); the others use 4 digits. Stage 2 must parse dates carefully (day first).
- **Missing odds:** Pinnacle pre-match odds (`PSH`) are empty for 170 of 380 matches in 2025-26. We should not rely on that column. Bet365 and market-average columns are complete.
- **Old seasons lack some columns:** the market average and maximum odds are called `BbAvH`, `BbMxH` in 2015-16 to 2018-19 and `AvgH`, `MaxH` from 2019-20. A model that needs odds would have to use the shared columns, or we skip odds as features.
- **Why this would hurt if ignored:** a model or script that expects a column in every season would crash or silently use empty values for old seasons. A team name spelled differently would be treated as a new team with no history.

**Class balance across all 4,180 matches:** home win 44.3%, draw 23.7%, away win 32.0%. Average goals: home 1.55, away 1.27. Home advantage is real. The 2025-26 test season has 42.6% home wins, and the training seasons 44.5%.

**Why odds exist, and the 50-55% benchmark:** odds are the bookmakers' own predictions. If we always back the bookmaker's favourite (lowest Bet365 odds), the favourite won **54.9%** of all 4,180 matches (47% in the worst season, 61% in the best). That is where "bookmakers get about 50-55%" comes from, and it is a fair target. Do not expect 80%.

**The target(s):**
- `FTR` (H / D / A) is a **classification** target: a category.
- `FTHG` and `FTAG` (goals) are **regression** targets: numbers. The Poisson model predicts these.

## 6. Common mistakes and gotchas
- **Editing a raw file by hand (even "just one typo").** We never do; the script only writes new files, never overwrites.
- **Downloading without checking.** An HTML error page saved as `.csv` would load but contain garbage. We check content.
- **Hammering the site.** We wait 1 second between requests and skip files we already have.
- **Using odds as features carelessly.** Closing odds (`...C...`) come from just before kick-off. They are fine for an honest comparison, but they already contain the bookmaker's opinion, so a model using them is partly copying the bookmakers. We decide this in a later stage.
- **Data leakage warning for later:** columns like `FTHG`, `HS` or `HY` describe the match itself. They exist only AFTER the match, so they must never be used as features for that same match. Only past matches may feed features.
- **Spelling differences between sources.** Watch team names when joining other data.

## 7. Check yourself
1. Why do we never edit files in `data/raw/`, and where do cleaned files go instead?
2. In the Liverpool vs Bournemouth row, what do `FTHG`, `FTAG` and `FTR` say, and which of them is a classification target and which are regression targets?
3. Why does a season have exactly 380 matches?
4. Odds of 2.00 on a team: roughly what chance does the bookmaker give it, and why do the three outcome chances add up to more than 100%?
5. Why do we train on 2015-16 to 2024-25 and test on 2025-26, instead of mixing all matches randomly?

## 8. Try it yourself
In the project folder with the environment active, run:
```
python -c "import pandas as pd; d = pd.read_csv('data/raw/E0_2015-16.csv'); print(d['FTR'].value_counts(normalize=True))"
```
Before running it, predict: what share of 2015-16 matches ended in a home win? Is it higher or lower than the 44.3% average across all seasons? Then change the file name to `E0_2020-21.csv` (the season played without crowds because of Covid) and predict again. Did home advantage change?

## 9. How this connects to the report
This stage feeds the **Dataset** section of the report (source, size, features, target) and the submission requirement to give the dataset link.

*Draft paragraph:* "We use match-level data from football-data.co.uk, which publishes free CSV files for every Premier League season. Our dataset covers eleven seasons (2015-16 to 2025-26), 4,180 matches in total, each with the final and half-time score, shots, corners, fouls, cards and bookmaker odds. The result (home win, draw or away win) is the classification target, and goals scored by each team are the regression targets. The ten seasons up to 2024-25 are used for training and the complete 2025-26 season for testing. A check of the raw files showed no duplicates, consistent team names, and different numbers of odds columns in older seasons, which we handle during preprocessing."

---

## Answers to "Check yourself"
1. Raw files are our untouched originals. If cleaning goes wrong we can start again from them, and anyone can see exactly what we changed. Cleaned data goes in `data/processed/`.
2. Liverpool scored 4 and Bournemouth 2, so the result is H (home win). `FTR` is the classification target. `FTHG` and `FTAG` are numbers, so they are regression targets.
3. 20 teams each play every other team twice (home and away), which is 38 games each. Each match has two teams, so 20 x 38 / 2 = 380.
4. About 1 / 2.00 = 50%. The three chances add up to more than 100% because the bookmaker adds a profit margin (in our example 105.4%).
5. Because the future must not leak into training. A random mix would let the model learn from matches played after the ones it is tested on, which is cheating and gives an unrealistic score. Testing on a full unseen season is how the model would really be used.
