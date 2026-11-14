"""Stage 4 sanity plots (training rows only). Run after src/features.py.

Run:  python src/feature_plots.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from features import FEATURE_COLUMNS, FEATURES_CSV
from preprocess import RESULT_LABELS

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
BLUE, ORANGE, GREY, VERMILION, GREEN = "#0072B2", "#E69F00", "#7F7F7F", "#D55E00", "#009E73"
SKY, PINK = "#56B4E9", "#CC79A7"
OUTCOME_COLORS = {"Home win": BLUE, "Draw": GREY, "Away win": VERMILION}


def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 150,
        "axes.grid.axis": "y",
    })


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print(f"[saved reports/figures/{name}]")


def load_training_rows():
    """Training rows only, and without warm-up rows (their features are mostly defaults)."""
    data = pd.read_csv(FEATURES_CSV, parse_dates=["Date"])
    return data[(data["split"] == "train") & (data["warmup"] == 0)].reset_index(drop=True)


# -------------------------------------------------- 12. Elo difference vs outcome
def plot_elo_vs_outcome(train):
    print("\n--- 12: outcome shares by Elo difference (5 equal-sized buckets)")
    train = train.copy()
    train["bucket"] = pd.qcut(train["elo_diff"], 5)
    shares = pd.crosstab(train["bucket"], train["result"], normalize="index") * 100
    shares.columns = [RESULT_LABELS[c] for c in shares.columns]
    sizes = train["bucket"].value_counts().sort_index()
    labels = [f"{iv.left:.0f} to {iv.right:.0f}\n(n={sizes[iv]})" for iv in shares.index]
    print(shares.round(1).assign(matches=sizes.values).set_index(shares.index.astype(str)).to_string())

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name in ["Home win", "Draw", "Away win"]:
        ax.plot(range(5), shares[name], marker="o", color=OUTCOME_COLORS[name], label=name, linewidth=2)
    for x in range(5):
        lowest = shares.iloc[x].idxmin()  # label the lowest line below its point so labels never collide
        for name in ["Home win", "Draw", "Away win"]:
            y = shares[name].iloc[x]
            ax.text(x, y - 4.2 if name == lowest else y + 1.6, f"{y:.0f}%", ha="center",
                    fontsize=9, color=OUTCOME_COLORS[name])
    ax.set_xticks(range(5), labels)
    ax.set_ylim(0, 80)
    ax.set_xlabel("Elo difference = home Elo minus away Elo (points), lowest to highest bucket")
    ax.set_ylabel("Share of matches (%)")
    ax.set_title("Outcomes by pre-match Elo difference (training matches)")
    ax.legend(loc="upper center", ncol=3)
    save(fig, "12_elo_diff_vs_outcome.png")


# ------------------------------------------------------- 13. Elo over time
def team_elo_series(data, team):
    """Pre-match Elo of one team, in date order. A break is inserted when the team
    was absent from the league for a season, so the line does not join the gap."""
    home = data[data["HomeTeam"] == team][["Date", "home_elo"]].rename(columns={"home_elo": "elo"})
    away = data[data["AwayTeam"] == team][["Date", "away_elo"]].rename(columns={"away_elo": "elo"})
    series = pd.concat([home, away]).sort_values("Date").reset_index(drop=True)
    gap = series["Date"].diff().dt.days > 100
    breaks = series[gap].copy()
    breaks["Date"] -= pd.Timedelta(days=1)
    breaks["elo"] = np.nan
    return pd.concat([series, breaks]).sort_values("Date")


def plot_elo_over_time():
    print("\n--- 13: Elo of 6 teams over time")
    data = pd.read_csv(FEATURES_CSV, parse_dates=["Date"])
    data = data[data["split"] == "train"]
    teams = {"Man City": BLUE, "Liverpool": VERMILION, "Arsenal": ORANGE,
             "Man United": GREY, "Leicester": GREEN, "Brentford": PINK}
    fig, ax = plt.subplots(figsize=(12, 6))
    for team, color in teams.items():
        series = team_elo_series(data, team)
        ax.plot(series["Date"], series["elo"], color=color, label=team, linewidth=1.8)
        valid = series.dropna()
        print(f"  {team:11} first {valid['elo'].iloc[0]:.0f}, min {valid['elo'].min():.0f}, "
              f"max {valid['elo'].max():.0f}, last {valid['elo'].iloc[-1]:.0f}")
    for season_start in data.groupby("season")["Date"].min():
        ax.axvline(season_start, color="#DDDDDD", linewidth=0.8, zorder=0)
    brentford_first = data.loc[(data["HomeTeam"] == "Brentford") | (data["AwayTeam"] == "Brentford"), "Date"].min()
    ax.annotate("Brentford promoted\n(starts at 1400)", xy=(brentford_first, 1400),
                xytext=(pd.Timestamp("2019-09-01"), 1330), fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "black"})
    ax.set_xlabel("Date (thin grey lines = start of each season)")
    ax.set_ylabel("Elo rating before the match")
    ax.set_title("Elo ratings over time for six teams (gaps = not in the Premier League)")
    ax.legend(ncol=3, loc="upper left")
    save(fig, "13_elo_over_time.png")


# ---------------------------------------------------- 14. feature correlations
def plot_feature_correlation(train):
    print("\n--- 14: correlation of the model features")
    corr = train[FEATURE_COLUMNS].corr()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().rename("corr").reset_index()
    pairs.columns = ["feature_a", "feature_b", "corr"]
    strong = pairs[pairs["corr"].abs() > 0.8].sort_values("corr", key=abs, ascending=False)
    print(f"Pairs with |correlation| above 0.8: {len(strong)}")
    print(strong.round(2).to_string(index=False))

    labels = corr.round(2).astype(str).where(corr.abs() >= 0.6, "")  # annotate only strong cells
    fig, ax = plt.subplots(figsize=(16, 13.5))
    sns.heatmap(corr, cmap="PuOr", vmin=-1, vmax=1, center=0, annot=labels, fmt="",
                annot_kws={"size": 6.5}, linewidths=0.3, cbar_kws={"label": "Correlation (-1 to +1)"}, ax=ax)
    ax.grid(False)
    ax.set_title("Correlation between the 34 model features (training matches; numbers shown when |r| >= 0.6)")
    save(fig, "14_feature_correlation.png")
    return strong


def main():
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    train = load_training_rows()
    print(f"Training rows without warm-up: {len(train)}")
    plot_elo_vs_outcome(train)
    plot_elo_over_time()
    plot_feature_correlation(train)


if __name__ == "__main__":
    main()
