# Stage 7 pre-registration (written before the simulation is run and before any real 2025-26 result is read)

Binding for the report. Nothing below is changed after the comparison with the real table.

**What is simulated.** The 380 fixtures of 2025-26, in date order, 1,000 times (seeds 0 to 999), matchweek by matchweek: each match is
predicted by `models/poisson_stacked.joblib` (stacked, rho 0, rate cap 3.5, exactly as saved) from features built ONLY from the simulated
earlier matches; goals are drawn from independent Poisson distributions; Elo (K = 30, H = 50) and the rolling goal windows are updated with
the simulated score. Start: `elo_start_of_2025-26.csv` (keep 0.9); neutral form (every relative goal feature 1.0 until a team has history);
league goal level held at 2.83 (the 10-season training mean). No real 2025-26 score is read by the simulator.

**Headline metrics (fixed now).** Against the real 2025-26 final table, read once at the end: mean absolute position error (simulated mean
position vs real position), Spearman rank correlation, mean absolute points error, how many of the simulator's top 4 (by mean points) are in the
real top 4, and how many of its bottom 3 are in the real bottom 3. Title, top-4 and relegation probabilities are reported as shares of the
1,000 simulated seasons.

**Match-level comparison (fixed now).** The simulator's probability for a match is the share of the 1,000 sims with each outcome. Because a
share can be exactly 0, the log loss uses (count + 0.5) / (1000 + 1.5) (the raw shares are also reported). It is compared with the
Stage 6 one-match-ahead Poisson probabilities (real previous matches), recomputed from the saved test predictions, on the same 380 fixtures.

**Expectation (stated in advance).** The one-match-ahead "ceiling" should beat the simulation clearly; the gap is the price of not knowing the
past. We expect the simulation to compress the table (the champion's points underestimated, the bottom teams' overestimated).

**Known limitation recorded in advance.** The saved Poisson model was trained with the away-side attack/defence inputs in swapped columns
(a labelling bug in `src/features_goals.py` found at the start of Stage 7). The simulator reproduces the layout the model was trained and
scored on, so the pre-registered model is unchanged; the effect is measured on validation seasons only in `reports/poisson_layout_diagnostic.csv`.
