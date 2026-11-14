"""Try it yourself (Lesson 06): a paired bootstrap by hand, about 100 resamples.
Question: is ELO2 Logistic Regression really different from the FULL Logistic Regression on the 2025-26 test season?
Run from the project folder:  python try_bootstrap.py
"""
import numpy as np
import pandas as pd

p = pd.read_csv("reports/test_predictions_stage6.csv")          # the saved test predictions (380 matches)
y = p["result"].to_numpy()                                      # 0 = home win, 1 = draw, 2 = away win

def match_losses(name):
    """-log(probability the model gave to what really happened), one number per match."""
    probs = p[[f"{name}_p_home", f"{name}_p_draw", f"{name}_p_away"]].to_numpy()
    return -np.log(probs[np.arange(len(y)), y])

diff = match_losses("lr_elo2") - match_losses("lr_full")        # per match: negative means ELO2 did better there
print(f"average difference on the real 380 matches: {diff.mean():+.4f}")

rng = np.random.default_rng(1)
means = []
for _ in range(100):                                            # 100 resamples (the real stage used 4,000)
    picks = rng.integers(0, len(diff), size=len(diff))          # draw 380 matches WITH replacement
    means.append(diff[picks].mean())                            # the average difference in this pretend season
low, high = np.percentile(means, [2.5, 97.5])
print(f"95% interval from 100 resamples: [{low:+.4f}, {high:+.4f}]")
print("Does it contain 0?", "yes: cannot tell them apart" if low < 0 < high else "no: a real difference")
