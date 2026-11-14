"""Stage 7 diagnostic (validation seasons only, nothing saved over): how much does the swapped-label bug in
src/features_goals.py matter for the Poisson model?

The two columns home_ga_10_rel and away_gf_10_rel hold each other's values (see the simulator's module docstring).
We refit the deployed configuration (stacked, level-neutral features, league offset, all fit seasons, rho 0) twice on
2015-16 to 2022-23, once with the saved (legacy) layout and once with the two columns swapped back, and score both on
the validation seasons 2023-24 and 2024-25. The test season is never loaded. No saved model or feature file is changed.
"""
import numpy as np
import pandas as pd

from modeling_utils import get_slice, paired_bootstrap
from poisson import load_goal_rows, make_cfg, score_cfg
from poisson_model import make_spec

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def swap_back(table):
    fixed = table.copy()
    fixed["home_ga_10_rel"], fixed["away_gf_10_rel"] = table["away_gf_10_rel"].to_numpy(), table["home_ga_10_rel"].to_numpy()
    return fixed


def main():
    rows = load_goal_rows()
    fit, val = get_slice("fit", rows), get_slice("validation", rows)
    cfg = make_cfg(make_spec("stacked", "rel", "offset"), window=None, recency=False, rho_mode="zero")
    legacy = score_cfg(cfg, fit, val)
    fixed = score_cfg(cfg, swap_back(fit), swap_back(val))
    dev = paired_bootstrap(legacy["dev_match"], fixed["dev_match"])          # positive: the corrected layout has the lower deviance
    out = paired_bootstrap(legacy["losses"], fixed["losses"])
    table = pd.DataFrame([{"layout": name, "val_deviance": r["dev_mean"], "val_outcome_log_loss": r["metrics"]["log_loss"],
                           "val_accuracy": r["metrics"]["accuracy"], "val_mae_total": r["mae_total"],
                           "att_weight": r["model"].coefficients()["stacked"]["att"], "opp_def_weight": r["model"].coefficients()["stacked"]["opp_def"],
                           "att_venue_weight": r["model"].coefficients()["stacked"]["att_venue"], "opp_def_venue_weight": r["model"].coefficients()["stacked"]["opp_def_venue"]}
                          for name, r in (("legacy (as saved and scored)", legacy), ("corrected (columns swapped back)", fixed))])
    table.to_csv(ROOT / "reports" / "poisson_layout_diagnostic.csv", index=False)
    print(table.round(4).to_string(index=False))
    print(f"\nDeviance, corrected better by {dev['mean_diff']:+.4f} [{dev['ci_low']:+.4f}, {dev['ci_high']:+.4f}]; "
          f"outcome log loss, corrected better by {out['mean_diff']:+.4f} [{out['ci_low']:+.4f}, {out['ci_high']:+.4f}] (positive = corrected is better)")


if __name__ == "__main__":
    main()
