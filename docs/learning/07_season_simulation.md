# Lesson 07: Season simulation (matchweek by matchweek)

**Stage goal in one sentence:** Play the 2025-26 season 1,000 times on the computer, starting from pre-season knowledge only, and read off each team's chances of the title, the top 4 and relegation, then compare with what really happened.

**Files created or changed:**
- `src/simulator.py`: the simulator (`python src/simulator.py simulate | compare | all`)
- `src/check_sim.py`: 8 checks, including a "no information from the future" test
- `src/diag_poisson_layout.py`: validation-only diagnostic of a label bug found at the start of this stage
- `src/try_sim_size.py`: the "Try it yourself" script of this lesson
- `reports/sim_preregistration.md`: what we promised to report, written before the simulation ran
- `reports/sim_initial_state.json`: the shared starting state of all 1,000 seasons (every team's start values and sources)
- `reports/sim_tables.csv` (20,000 rows: one per simulation and team), `reports/sim_scorelines.csv` (380,000 rows: every simulated match; 5.8 MB), `reports/sim_run_info.json`
- `reports/sim_summary_by_team.csv`, `sim_expected_table.csv`, `sim_points_by_game.csv`: the summaries
- `reports/sim_vs_real_table.csv`, `sim_headline_metrics.json`, `sim_ceiling_comparison.csv`: the comparison step (real results read once)
- `reports/poisson_layout_diagnostic.csv`: the diagnostic output
- `reports/figures/42_sim_points_fan.png`, `43_sim_position_histograms.png`, `44_sim_vs_real_scatter.png`, `45_sim_position_heatmap.png`
- `docs/decisions.md`, `README.md` (updated) and this lesson

---

## 1. The big picture
In Stage 5d we built a model that turns two teams into two numbers, the expected goals of each side. In Stage 6 we asked it to predict one match at a time, *knowing everything that had really happened before that match*. A fan, however, asks a bigger question before the season starts: "who will win the league?" Nobody knows the results yet, so the model has to forecast the whole season from the starting position. Think of a weather forecast: tomorrow's forecast uses today's real weather, but a seasonal forecast ("will this winter be cold?") has to play the weather forward many times and count how often it is cold. This stage is the seasonal forecast: we **play the season forward 1,000 times** and count.

## 2. What we built
1. **The starting state** (`sim_initial_state.json`). Each team's Elo rating at the start of the season (from `elo_start_of_2025-26.csv`, built from matches before 2025-26 only; Liverpool 1680, Man City 1672, Arsenal 1671 at the top; the three promoted teams Burnley, Leeds and Sunderland at 1400). Every team's recent-form numbers start "neutral" (equal to the league average) because nobody has played yet. League scoring level: 2.83 goals per match (the average of the 10 training seasons).
2. **One simulated match.** Features come from the simulated history so far, the Poisson model turns them into two expected-goal rates, and we draw the goals at random. Example (the opening match, Liverpool v Bournemouth): Liverpool's Elo edge is +155, so the rates are about 2.0 and 1.0 goals; one simulation might draw 2-0, another 1-1, another 3-2.
3. **Updating after every match.** The simulated score changes both teams' Elo (K = 30; a win against a stronger team moves it more) and their recent-goals windows, so the next match of each team is predicted from *simulated* form, not real form. That is "matchweek by matchweek" updating.
4. **One simulated season:** 380 matches in date order, then the table (points; ties broken by goal difference, then goals scored, then alphabetical order).
5. **1,000 seasons** (seed 0 to 999), then the counts: how many seasons each team won the title, finished in the top 4, or finished in the bottom 3.
6. **The comparison step**, done once at the very end: the real 2025-26 table, and the Stage 6 one-match-ahead numbers on the same 380 matches.

## 3. New concepts (explained simply)

- **Monte Carlo simulation:** answering a question about a random process by running the process many times and counting. *Analogy:* to learn the chance of rolling a total of 7 with two dice you could do the algebra, or roll the dice 1,000 times and count the sevens. Here the "dice" are the goal counts. **Connection to Stage 5d:** a Poisson rate (lambda) is the *average* number of goals; the actual number in one match is random around it. A simulated season is one random realisation of 380 such matches, and a Monte Carlo run is 1,000 realisations. Only the averages and counts across the 1,000 are the forecast, never one run.
- **Reading a probability from the simulation:** "Liverpool wins the title with probability 0.214" means **214 of the 1,000 simulated seasons had Liverpool first**. It does *not* mean "Liverpool will win", and it does not mean the model is 21% sure; it means that in the world the model describes, about one season in five ends that way. If you replayed the real 2025-26 season again and again, you would expect Liverpool champions about one time in five, and a different champion the other four.
- **Monte Carlo standard error:** because we use a finite number of seasons, our estimate of a probability wobbles. For a probability p estimated from n seasons the wobble is about sqrt(p x (1-p) / n). For Liverpool, p = 0.21 and n = 1,000: sqrt(0.21 x 0.79 / 1000) = 0.013, so 0.214 could really be anywhere from about 0.19 to 0.24. For small probabilities (title chances of 0.3%) the absolute wobble is small but the relative wobble is large.
- **Inverse-CDF sampling (how the random goals are drawn):** each simulation has its own list of random numbers between 0 and 1, fixed by its seed (simulation 7 always uses the stream `default_rng(7)`). A random number u turns into a goal count by asking "how many goals is the smallest number whose cumulative probability reaches u?" For a rate of 1.55, a u below 0.21 gives 0 goals, a u between 0.21 and 0.54 gives 1, and so on (the P(0) = 0.212 and P(1) = 0.329 of the Lesson 05d walk-through). This is a standard way to sample, and it is why simulation i gives *exactly the same season* whether it runs alone or inside the batch of 1,000 (a check proves it).
- **Matchweek-by-matchweek updating vs frozen pre-season:** a frozen simulation computes every team's rates once, before the season, and keeps them for all 38 games. Matchweek-by-matchweek updating lets the season "develop": a team that gets lucky early rises in Elo and is slightly stronger afterwards. Real seasons behave like the second (form and results feed back), and it produces a wider spread of final tables.
- **The leakage trap this opens:** to update after each matchweek we need results. If we used the *real* 2025-26 results to update, the "forecast" would know what happened (by matchweek 20, the real table), which would be an illusion of skill, not a simulation. So every update uses the simulated scores only, and the real results are read exactly once, after everything is saved. Three guards enforce it: the loader reads only fixture columns (a whitelist), the simulation core reads no file, and a test changes one simulated score and proves that nothing earlier is affected.
- **Independence assumption:** the home goals and the away goals of a match are drawn independently. This is the standard assumption of simple Poisson models. In our training seasons the real home and away goals of a match have a small negative correlation of -0.14 (not exactly zero), so the independent draw is a simplification, not the truth.
- **The ceiling:** the best the model can do when it knows everything up to the previous match (Stage 6's one-match-ahead setup). A full-season forecast must do worse, because it does not know the past. The gap between them is "the price of not knowing the past".

## 4. How the code works
- `load_fixtures()` / `read_features_columns(usecols)`: the only reading of 2025-26 data before the comparison. It reads `features.csv` with a whitelist of five columns (`match_id`, `Date`, `season`, `HomeTeam`, `AwayTeam`) and **raises an error for any other column**, so the real results, and the real features that were built from them, cannot be loaded by accident.
- `build_schedule(...)`: everything that does not depend on a score: the 380 fixtures in date order, the team list (alphabetical, for the last tie-break), each team's game numbers, and the groups of matches on the same date (treated as simultaneous, as in Stage 4).
- `side_plan` / `relative_goal_features`: for a match and a team, which *earlier* matches are in its last-10 window and last-10 venue window, and the average goals scored and conceded divided by the league average over the same period. This is the Stage 5d formula, computed on simulated goals. If a team has no history the features are 1.0 (league average); if fewer than 20 league matches exist the neutral league default is used.
- `run_simulation(seeds, model, schedule, state)`: the loop. It handles one date at a time, **for all 1,000 simulations at once** (arrays with one entry per simulation). That works because the schedule is identical in every simulation: only the scores differ. In order: build inputs from history before this date, call `model.predict` (the cap of 3.5 goals is inside), draw the goals, and only then update Elo and the league totals.
- `league_table(...)` and `rank_teams(...)`: points, goals and final positions, vectorised, with the tie-break points, goal difference, goals scored, alphabetical.
- `compare()`: reads the real results (once, through `load_real_results_once`), builds the real table the same way, computes the metrics, and compares with the Stage 6 predictions saved in `test_predictions_stage6.csv`.
- **The one decision to review, shown in the code:** `LEGACY_SWAPPED_LABELS = True`. While reading `features_goals.py` we found that two of its eight relative goal columns were stored under swapped labels (the column `home_ga_10_rel` holds the away team's goals scored, and `away_gf_10_rel` holds the home team's goals conceded). The saved Poisson model was trained and scored with that layout, so the simulator reproduces it to keep the model exactly as it was tested; `check_sim.py` proves the simulator's features equal `features_goals.py`'s on a real season. On the validation seasons the bug is immaterial: refitting with the columns swapped back changes the deviance by +0.0006 [-0.0012, +0.0024] and the log loss by +0.0003 [-0.0003, +0.0009] (both inside the noise). Whether to fix it and refit is for you to decide.

## 5. Results and how to read them

**What we got right and wrong this year (the real table is Arsenal 85, Man City 78, Man United 71, Aston Villa 65, Liverpool 60, ... West Ham 39, Burnley 22, Wolves 20).**

Simulated top 6 by mean points:

| Team | Mean points | P(title) | P(top 4) | Real finish |
|---|---|---|---|---|
| Liverpool | 71.0 | 0.214 | 0.607 | 5th (60 pts) |
| Man City | 70.2 | 0.187 | 0.566 | 2nd (78) |
| Arsenal | 70.0 | 0.195 | 0.561 | **1st (85)** |
| Chelsea | 61.9 | 0.075 | 0.340 | 10th (52) |
| Aston Villa | 61.5 | 0.073 | 0.330 | 4th (65) |
| Newcastle | 58.7 | 0.048 | 0.274 | 12th (49) |

Simulated bottom 3 by mean points: Leeds 36.2 (P(relegation) 0.452), Burnley 36.5 (0.441), Sunderland 36.9 (0.432). *How to read this:* the three favourites (Liverpool, Man City, Arsenal) are almost level, each with a title chance of about one in five, and together they take 60% of the titles. Nobody below them has more than 8%.

**Simulated vs real, in numbers** (`reports/sim_vs_real_table.csv`, figure 44):

| Metric | Result |
|---|---|
| Mean absolute position error (simulated mean position vs real position) | 3.83 places |
| Spearman rank correlation (1 = identical order) | 0.564 |
| Mean absolute points error | 9.8 points (average signed error +0.8) |
| Real top 4 captured by the simulated top 4 (by mean points) | 2 of 4 (Arsenal, Man City) |
| Real relegated captured by the simulated bottom 3 | 1 of 3 (Burnley) |
| Real total points inside the simulated 5th to 95th percentile range | 18 of 20 teams |

- **The picture is "right idea, big misses".** The simulator put Arsenal and Man City in the top 3 and gave Burnley a high relegation chance. It treated the three favourites as about equal, so it had no way to say Arsenal would win by 7 points. Spearman 0.56 means the order is clearly related to the real one, far from random (0) and far from perfect (1).
- **The big misses.** Man United finished 3rd with 71 points (the simulator expected 44.3, gave them a 5% top-4 chance and a 25% relegation chance; 71 is above even the simulator's 95th percentile, 69). Liverpool, the simulator's favourite (21% for the title), finished 5th with 60. Wolves finished last with 20 points (expected 45.7, relegation chance 18%; 20 is below the simulator's 5th percentile, 25). Sunderland and Leeds, given a 43% and 45% relegation chance, finished 7th and 14th. None of this was foreseeable from pre-season ratings alone: a real season contains managerial changes, injuries and transfers that no starting Elo knows.
- **18 of 20 inside the 90% range is about what a well-calibrated forecast should give** (90% of 20 teams is 18). That is the good news: the simulator's *uncertainty* is honest even though its single best guesses are often wrong. The 5th to 95th percentile ranges are wide (for Arsenal, 47 to 92 points), and the real results mostly fell inside them.
- **Why the champion's points look under-predicted.** The "expected table" gives Arsenal 70.0 points on average, 15 below the real 85. But a *simulated champion* typically scores **86.1** points (the champion of each simulated season, whichever team it is, averaged over 1,000 seasons), and 56% of simulated seasons have a champion with 85 points or more. So 85 is a completely ordinary champion score; the expected table is lower because it averages each team over all the seasons in which it did *not* win. A similar effect shows at the bottom (the simulated last-placed team averages 21.6 points; Wolves had 20). A single simulated season is as spread out as a real one (average standard deviation of the 20 teams' points 17.7, against 15.8 in the real table); it is only the average table that looks compressed. The model also gives the champion a small extra handicap: Elo ratings pull strong teams toward the average between seasons (`keep` = 0.9), and the stacked Poisson model gives a modest value to a huge Elo gap, so the true best team is somewhat under-rated. We did not measure how much each of these contributes.

**The ceiling comparison** (same 380 matches; `reports/sim_ceiling_comparison.csv`):

| | Log loss | Accuracy | Macro-F1 | Deviance on total goals |
|---|---|---|---|---|
| One-match-ahead Poisson (real previous matches, Stage 6) | 1.0303 | 48.4% | 0.365 | 1.0417 |
| Simulation, pre-season only (share of 1,000 seasons per outcome) | 1.0500 | 46.1% | 0.346 | 1.0345 |

*In plain words:* knowing the real past makes the match predictions better by 0.0197 in log loss and 2.4 points of accuracy. The 95% interval of that gap is [-0.0005, +0.0395], so by our noise rule this is **at the edge** (the interval just touches zero): the direction is what we expected, but we cannot call it clear on 380 matches. The pre-season-only simulation is as good as the one-step model at predicting how many goals a match has (deviance 1.0345 against 1.0417): the 2025-26 goal level (2.75 per match) happened to be close to the 2.83 we assumed. The price of not knowing the past is paid in *who wins*, not in *how many goals*. (The simulator's match probabilities are the share of the 1,000 seasons with each outcome, smoothed by half a count so that no outcome has exactly zero; none of the real outcomes had a zero share.)

### Figures
- **Figure 42** (`42_sim_points_fan.png`): for the top 6 and the bottom 3, the median (line), the middle 50% (dark band) and the 5th to 95th percentile (light band) of points after each game. *How to read it:* the bands widen as the season goes on because luck accumulates; at the end, the bottom teams' bands are as wide as the top teams'. A good result is a band that contains where the real team ended up.
- **Figure 43** (`43_sim_position_histograms.png`): where each top-6 team finishes, over the 1,000 seasons (red = champion, orange = top 4). Liverpool, Man City and Arsenal have a tall bar at position 1 and a long tail; Chelsea, Villa and Newcastle are spread almost evenly from 1 to 10.
- **Figure 44** (`44_sim_vs_real_scatter.png`): each team is a dot; x = simulated mean points, y = real points; the dashed line is "simulated = real". Dots above the line over-performed (Arsenal, Man United, Sunderland); dots below under-performed (Liverpool, Crystal Palace, Chelsea). The cloud follows the line loosely, with a slope steeper than the line at the top (the compression we explained).
- **Figure 45** (`45_sim_position_heatmap.png`): the 20 x 20 table of "how often does team X finish in position Y". Dark cells along the diagonal mean "we know where this team finishes"; our heat map is spread out, especially in the middle of the table, which is the honest message: apart from the top three and the bottom three, almost any order is plausible.

## 6. Common mistakes and gotchas
- **Using real results to update (leakage).** The oldest trap of a "forecast". Avoided by whitelisted loading, a simulation core that reads no files, a one-time real read, and a perturbation test.
- **Reading one simulated season as the forecast.** Only the counts over many seasons mean anything.
- **Reading a probability as a prediction.** 0.214 is not "Liverpool will win"; it is also not "Liverpool should have won".
- **Too few simulations.** With 100 seasons the wobble of a title probability is about 0.04 (4 points); with 1,000 it is about 0.013 (see "Try it yourself").
- **Assuming the simulation can beat the one-step model.** It cannot: it has less information. The Stage 6 numbers are a ceiling, not a target.
- **Blaming the model for every miss.** Man United's 71 points and Wolves' 20 are outside the simulator's 90% range; with 20 teams, one or two such results per season are expected by chance (2 of 20 is exactly what a calibrated 90% range predicts).
- **Averaging when you should not.** The expected table is a table of averages, so it is more compressed than any real season. Use the per-season distribution (the champion's points in each simulated season) to judge extreme results.
- **Known pitfalls from `CLAUDE.md` addressed:**
  - *Season simulation:* real 2025-26 results are not used to build features; updating uses simulated results (mode (b), decided upstream).
  - *Newly promoted teams:* Elo 1400 (the Stage 4 rule) and neutral form; the three promoted teams get the default.
  - *Start-of-season cold start:* the Elo snapshot is the Stage 4 carry-over with shrinkage (keep 0.9); form starts neutral and fills in as simulated matches accumulate, with the Stage 4 shrinkage of venue averages.
  - *Small samples and noise:* the title probabilities carry a Monte Carlo error of about 0.013; the 380-match comparison with the ceiling is at the edge of the noise.
  - *Honest results:* the misses (Man United, Liverpool, Wolves) are reported, nothing was tuned after seeing the real table, and the metrics were fixed in `reports/sim_preregistration.md` before the run.

## 7. Check yourself
1. What does "P(title) = 0.214 for Liverpool" mean in terms of the 1,000 simulated seasons? What does it not mean?
2. Why is a matchweek-by-matchweek simulation not allowed to use the real 2025-26 results, even though they are in our data?
3. The expected table gives Arsenal 70 points and the real Arsenal got 85. Does this prove the simulator is broken? Use the numbers.
4. Why is the one-match-ahead model better than the full-season simulation, and why is the gap on the total number of goals different from the gap on who wins?
5. If we ran 10,000 seasons instead of 1,000, would the probability for Liverpool change by a lot? Would the *real* chance of Liverpool winning change?

## 8. Try it yourself
Open `src/try_sim_size.py` and run `python src/try_sim_size.py` (a few minutes). It repeats the whole simulation four times at 100, 1,000 and 5,000 seasons (new blocks of seeds each time) and prints Liverpool's title probability in each repeat, the spread between repeats, and the theoretical Monte Carlo error sqrt(p(1-p)/n). **Before you run it, predict:** by what factor should the wobble shrink when you go from 100 to 5,000 seasons (hint: it depends on the square root of 50)? Then compare. Our run gave, for Liverpool: 100 seasons: 0.19, 0.21, 0.20, 0.17 (spread 0.017, theory 0.039); 1,000 seasons: 0.214, 0.202, 0.225, 0.214 (spread 0.009, theory 0.013); 5,000 seasons: 0.211, 0.213, 0.219, 0.213 (spread 0.003, theory 0.006). With only four repeats the measured spread is itself noisy (it should be near the theory but can be lower), yet the pattern is clear: the wobble shrinks roughly as 1/sqrt(n) predicts (a 10-fold increase, 100 to 1,000 seasons, cuts it by about a factor of 3), and it never reaches zero. Extension: change the team to "Wolves" and compare the *relative* wobble of a small probability.

## 9. How this connects to the report
This stage feeds the **Season Simulation** section (and the Season Table page of the app in Stage 9). *Draft paragraph:* "We simulated the 2025-26 season 1,000 times from a pre-season snapshot. In each simulated match the Poisson goal model gave the two expected-goal rates from the Elo ratings and the recent scoring of the two teams, goals were drawn at random, and the ratings and recent form were updated with the simulated score before the next match, so no real 2025-26 result was used. The simulation gave Liverpool, Manchester City and Arsenal about a one-in-five title chance each (0.214, 0.187 and 0.195) and the three promoted teams a relegation chance of about 44%. Against the real table the average position error was 3.8 places, the rank correlation 0.56, and 2 of the real top four and 1 of the three relegated teams were identified; 18 of 20 teams finished inside the simulated 90% range. The one-match-ahead model with real history had a log loss of 1.030 against 1.050 for the pre-season simulation; the gap is the price of not knowing the past. Large misses (Manchester United 3rd, Liverpool 5th, Wolves last) show how much of a real season no pre-season rating can foresee."

---

## Answers to "Check yourself"
1. It means that 214 of the 1,000 simulated seasons ended with Liverpool in first place. It does not mean Liverpool *will* win (the real champion was Arsenal, who had a 0.195 chance and won), and it does not mean we are 21% confident in some single forecast; it describes how often the title goes to Liverpool in the random world the model describes (with a Monte Carlo error of about 0.013).
2. Because a forecast may only know what was known before the season. Updating with real results would make the "forecast" of matchweek 20 know the first 19 real matchweeks, which looks like skill but is just copying. Simulated results keep the whole thing a genuine forecast. The real results are used only at the end, to compare.
3. No. Arsenal's real 85 points is inside the simulated 5th to 95th percentile range (47 to 92), and a simulated champion averages 86.1 points: 85 is an ordinary champion's total. The expected table averages Arsenal over all seasons, including the many where Arsenal did not win, so it is lower than any typical champion's total.
4. Because it knows the real past (form, injuries and results as they happened), while the simulation uses only pre-season information and its own simulated results. The gap is clear in "who wins" (log loss 1.0500 against 1.0303, accuracy 46.1% against 48.4%) but absent in "how many goals": the total-goals deviance is as good for the simulation (1.0345 against 1.0417), because the league's goal level (2.83 assumed, 2.75 real) is almost the same in both. Knowing the past mostly tells you *who is strong*, not *how many goals a match has*.
5. The estimate for Liverpool would move only slightly (the Monte Carlo error would fall from about 0.013 to about 0.004), toward whatever the model's true value is. The real chance of Liverpool winning would not change at all: more simulations only measure the model's number more precisely. They cannot repair what the model gets wrong.
