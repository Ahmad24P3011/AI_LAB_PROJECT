# EDA findings (Stage 3)

Data used: training seasons only, 2015-16 to 2024-25, 3,800 matches. The test season (2025-26) was not used. Re-create every figure with `python src/eda.py`.

## 01. Outcome distribution: `reports/figures/01_outcome_distribution.png`
- **Pattern:** home win 1,691 (44.5%), draw 886 (23.3%), away win 1,223 (32.2%). The plan's "home teams win about 45%" is right (44.5%).
- **Implication:** always predicting "home win" is correct 44.5% of the time. This is the baseline every model must beat. Draws are the smallest class.

## 02. Home advantage by season: `02_home_advantage_by_season.png`
- **Pattern:** the home-win share moves between 37.9% and 49.2% from season to season. Random luck alone moves a 380-match season by about 2.5 points, so gaps under about 5 points are not meaningful. One season stands out: **2020-21** (mostly played without fans) had only 37.9% home wins (6.6 points below average) and 40.3% away wins, more than home wins. 2024-25 is also low (40.8%) but inside the noise range.
- **Implication:** home advantage is real but not constant, and fans seem to matter. We keep 2020-21 for now. Later we can test whether dropping it or flagging it changes validation results (a decision to make together, not yet).

## 03. Goals per match and Poisson: `03_goals_per_match.png`
- **Pattern:** average goals are 1.55 for the home team and 1.28 for the away team (2.83 total). A Poisson curve with the same mean follows the real bars closely: every bar is within 3 percentage points. But the fit is not perfect: real matches have more 0-goal and more 4+ goal results than Poisson predicts (home 0 goals: 23.2% real vs 21.2% Poisson; away 0 goals: 30.7% vs 27.9%). Variance is about 1.13 to 1.15 times the mean (Poisson says 1.0), and the chi-square test says the small misfit is detectable with 3,800 matches.
- **Implication:** Poisson is a sensible model for goals and justifies our Poisson regression. A single average is too crude because teams differ in strength; the regression should let each match have its own expected goals.

## 04. Goals per season: `04_goals_per_season.png`
- **Pattern:** total goals per match stay between 2.68 and 2.85 in eight of ten seasons, but **2023-24 had 3.28** (home 1.80, away 1.48), the highest. 2024-25 came back to 2.93.
- **Implication:** the league's scoring level can shift from one season to the next by more than a tenth of a goal per team. The model should not assume a fixed average, and the 2025-26 level may differ from what we trained on. We cannot check that now (test data is off limits).

## 05 and 06. Team strength: `05_points_heatmap.png`, `06_goal_difference_heatmap.png`
- **Pattern:** a small group (Man City, Liverpool, Arsenal, Chelsea, Tottenham, Man United) is usually near the top, but individual teams swing a lot: Leicester 81 points (2015-16 champions) then 44; Chelsea 50 then 93; Tottenham 66 then 38 in 2024-25. Champions needed 81 to 100 points.
- **Implication:** team strength is real but changes, so we should describe teams by recent form and recent strength, not a single fixed rating.

## 07. How fast strength changes: `07_strength_persistence.png`
- **Pattern:** a team's points correlate 0.70 with its points one season later, 0.69 two seasons later, 0.63 three, 0.55 four and 0.62 five seasons later (fewer teams survive that long, so the later bars are noisy). A typical team changes by 10.5 points from one season to the next (median 9, 90th percentile 21). Teams drift toward the middle: next-season points are about 13 + 0.76 x this-season points.
- **Implication:** old seasons still carry information, but recent matches matter more. We keep all 10 training seasons, and Stage 4 should weight recent matches more (rolling windows, an Elo rating that adapts after each match). Newly promoted teams have no previous Premier League season at all.

## 08. Home vs away by team: `08_home_vs_away_by_team.png`
- **Pattern:** across all teams, 1.57 points per home match vs 1.20 away. Biggest home advantage: Swansea (+0.61 points per game), Arsenal (+0.60), Newcastle (+0.54), Tottenham (+0.52), Everton (+0.46). Smallest: Crystal Palace (+0.15), Southampton (+0.16), Leeds (+0.18). But the noise is large: about +/- 0.33 points per game per team, and only 15 of 27 teams have a home advantage clearly above zero.
- **Implication:** most of the differences between teams are probably luck. A league-wide home advantage is reliable; a separate home advantage for each team is not. Per-team home/away averages in Stage 4 need enough matches (or should be blended with the team's overall strength).

## 09. Same-match stat correlations: `09_same_match_correlation.png`
- **Pattern:** shots on target are strongly linked to goals (HST with home goals +0.60; AST with away goals +0.60) and moderately to the result (HST with home points +0.43; AST -0.41). Fouls and cards are almost unrelated to the result (fouls -0.03, home yellow cards -0.09). Shots, shots on target and corners are also correlated with each other (HS with HST +0.69).
- **Implication:** these numbers describe the match that was just played, so they can NOT be used to predict it (leakage). Only their averages over each team's earlier matches may be used (Stage 4). Fouls and cards look weak even as history. Where features overlap (HS, HST, HC) we should not expect each to add new information.

## 10. Bookmaker odds as a reference: `10_bookmaker_odds.png`
- **Pattern:** Bet365 odds are available for every training match. The bookmaker margin (overround) averages 4.4% (2.8% to 5.5% by season). The favourite won **55.5%** of matches vs 44.5% for always-home-win; by season it ranges from 47.4% to 61.1%. Odds are well calibrated: when the bookmaker says 25% it happens 25% of the time, when it says 45% it happens 44%, at 84% it happens 89%.
- **Implication:** about 55% is a realistic ceiling for a model built from public information, so it is our target. Our plan (50-55%) is sensible. Odds are used for reference only, not as features.

## 11. Draws: `11_draws.png`
- **Pattern:** 73% of draws are 0-0 (27%) or 1-1 (46%). Draws are more common in evenly matched games: 30% when the bookmaker sees the teams as nearly equal, falling to 16% when one team is a clear favourite. But even in the closest games only 30% end in a draw, and the bookmaker never made the draw its favourite outcome in 3,800 matches (highest draw chance seen: 34%).
- **Implication:** draws will be the hardest class: a model will almost never pick "draw" as the most likely result. We should report the draw class separately (F1 per class, confusion matrix) and judge models with log loss (probabilities) as well as accuracy. Expect low accuracy on draws and say so in the report.
