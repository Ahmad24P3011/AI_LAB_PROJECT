"""Try it yourself (Lesson 07): how much do the title probabilities wobble when we run 100, 1,000 or 5,000 seasons?
Run from the project folder:  python src/try_sim_size.py
For each size we repeat the whole simulation 4 times with different seeds and look at how much Liverpool's P(title) changes
from one repeat to the next. The Monte Carlo standard error predicts it: sqrt(p x (1 - p) / n_sims).
"""
import numpy as np

import simulator as S
from poisson_model import PoissonGoalModel

schedule = S.build_schedule(S.load_fixtures())          # fixtures only: no real result is read
state = S.load_initial_state(schedule)
model = PoissonGoalModel.load(S.POISSON_CONFIG)
team = schedule["index"]["Liverpool"]

for n_sims in (100, 1000, 5000):
    estimates = []
    for repeat in range(4):
        seeds = [repeat * 100_000 + i for i in range(n_sims)]               # a different block of seeds each repeat
        result = S.run_simulation(seeds, model, schedule, state)
        position = S.league_table(schedule, result["home_goals"], result["away_goals"])["position"]
        estimates.append(float((position[:, team] == 1).mean()))
    p = float(np.mean(estimates))
    print(f"{n_sims:5d} seasons: Liverpool P(title) in 4 repeats = {[round(e, 3) for e in estimates]}; "
          f"spread (std) {np.std(estimates, ddof=1):.4f}; theory sqrt(p(1-p)/n) = {np.sqrt(p * (1 - p) / n_sims):.4f}")
