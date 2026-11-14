"""Stage 3: exploratory data analysis (EDA) on the TRAINING data only.

Run:  python src/eda.py
It prints the key numbers and saves every figure to reports/figures/.
The test season (2025-26) is never used here: looking at test outcomes would let
the test influence our choices, which is a quiet form of leakage.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files only, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from preprocess import RESULT_LABELS, TRAIN_SEASONS

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
ODDS_CSV = ROOT / "data" / "processed" / "odds.csv"
FIG_DIR = ROOT / "reports" / "figures"

# Colour-blind-safe colours (Okabe-Ito palette)
BLUE, ORANGE, GREY, VERMILION, GREEN = "#0072B2", "#E69F00", "#7F7F7F", "#D55E00", "#009E73"
OUTCOME_COLORS = {"Home win": BLUE, "Draw": GREY, "Away win": VERMILION}


def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 150,
        "axes.grid.axis": "y",  # horizontal guide lines only, so bar labels stay clean
    })


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print(f"[saved reports/figures/{name}]")


# --------------------------------------------------------------- loading
def load_train():
    """Load training matches only and add a few helper columns."""
    df = pd.read_csv(TRAIN_CSV, parse_dates=["Date"])
    assert (df["split"] == "train").all() and set(df["season"]) == set(TRAIN_SEASONS)
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["home_points"] = df["result"].map({0: 3, 1: 1, 2: 0})  # league points
    df["away_points"] = df["result"].map({0: 0, 1: 1, 2: 3})
    return df


def load_training_odds(train):
    """odds.csv also contains 2025-26 rows. We read only the 3 Bet365 columns and
    immediately keep the training match_ids, so no test row is ever used."""
    odds = pd.read_csv(ODDS_CSV, usecols=["match_id", "B365H", "B365D", "B365A"])
    odds = odds[odds["match_id"].isin(train["match_id"])]
    return train.merge(odds, on="match_id", how="left")


# ------------------------------------------------- a. outcome distribution
def outcome_distribution(df):
    section("a. Outcome distribution (training, 2015-16 to 2024-25)")
    counts = df["result"].value_counts().sort_index()
    percent = counts / len(df) * 100
    for code in counts.index:
        print(f"{RESULT_LABELS[code]:9} {counts[code]:5} matches  {percent[code]:5.1f}%")
    print(f"Plan said 'home teams win about 45%': real value {percent[0]:.1f}% "
          f"(difference {percent[0] - 45:+.1f} points). Draw rate: {percent[1]:.1f}%.")

    fig, ax = plt.subplots(figsize=(7, 5))
    names = [RESULT_LABELS[c] for c in counts.index]
    ax.bar(names, counts.values, color=[OUTCOME_COLORS[n] for n in names])
    for i, code in enumerate(counts.index):
        ax.text(i, counts[code] + 20, f"{counts[code]}\n({percent[code]:.1f}%)",
                ha="center", va="bottom")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_ylabel("Number of matches")
    ax.set_title(f"Match outcomes, {len(df)} matches in 10 seasons (2015-16 to 2024-25)")
    save(fig, "01_outcome_distribution.png")


# ------------------------------------------------ b. home advantage by season
def home_advantage_by_season(df):
    section("b. Outcome shares per season")
    shares = pd.crosstab(df["season"], df["result"], normalize="index") * 100
    shares.columns = [RESULT_LABELS[c] for c in shares.columns]
    print(shares.round(1).to_string())

    mean_home = shares["Home win"].mean()
    # Random luck alone moves a 380-match season's home-win share by about this much:
    p = df["result"].eq(0).mean()
    se = np.sqrt(p * (1 - p) / 380) * 100
    print(f"\nAverage home win {mean_home:.1f}%. One standard error for a 380-match season "
          f"is about {se:.1f} points, so gaps under ~{2 * se:.0f} points can be luck.")
    for season, value in shares["Home win"].items():
        gap = value - mean_home
        if abs(gap) > 2 * se:
            print(f"Stands out: {season} home win {value:.1f}% ({gap:+.1f} points from average)")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name in ["Home win", "Draw", "Away win"]:
        ax.plot(shares.index, shares[name], marker="o", color=OUTCOME_COLORS[name], label=name)
        for x, y in zip(shares.index, shares[name]):
            # labels go above the point, except the draw line (below) and the one home-win
            # point that sits right under the away-win line
            below = name == "Draw" or (name == "Home win" and x == "2020-21")
            ax.text(x, y - 2.6 if below else y + 1.0, f"{y:.0f}", ha="center", fontsize=8,
                    color=OUTCOME_COLORS[name])
    # shaded band around the 2020-21 season (categories sit at x = 0, 1, 2, ...)
    ax.axvspan(4.55, 5.45, color=ORANGE, alpha=0.15)
    ax.text(5, 55, "2020-21:\nmostly without fans", ha="center", va="center", fontsize=9)
    ax.set_ylim(15, 58)
    ax.set_xlabel("Season")
    ax.set_ylabel("Share of matches (%)")
    ax.set_title("Home win, draw and away win share per season")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45)
    save(fig, "02_home_advantage_by_season.png")


# ------------------------------------------------------ c. goals per match
def goals_per_match(df):
    section("c. Goals per match and the Poisson fit")
    print(f"Average goals: home {df['home_goals'].mean():.2f}, away {df['away_goals'].mean():.2f}, "
          f"total per match {df['total_goals'].mean():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, column, name, color in [(axes[0], "home_goals", "Home team", BLUE),
                                    (axes[1], "away_goals", "Away team", ORANGE)]:
        goals = df[column]
        mean = goals.mean()
        capped = goals.clip(upper=5)  # the last bar means "5 or more"
        observed = capped.value_counts(normalize=True).reindex(range(6), fill_value=0)
        expected = pd.Series(stats.poisson.pmf(range(5), mean), index=range(5))
        expected[5] = stats.poisson.sf(4, mean)  # chance of 5 or more

        # Chi-square goodness of fit: 6 groups, 1 estimated parameter (the mean) -> 4 df
        observed_counts = observed * len(goals)
        expected_counts = expected * len(goals)
        chi2 = ((observed_counts - expected_counts) ** 2 / expected_counts).sum()
        p_value = stats.chi2.sf(chi2, df=4)
        print(f"\n{name}: mean {mean:.3f}, variance {goals.var():.3f} "
              f"(Poisson needs variance = mean; ratio {goals.var() / mean:.2f})")
        table = pd.DataFrame({"observed %": observed * 100, "Poisson %": expected * 100}).round(1)
        table.index = ["0", "1", "2", "3", "4", "5+"]
        table["gap (points)"] = (table["observed %"] - table["Poisson %"]).round(1)
        print(table.to_string())
        print(f"Chi-square = {chi2:.1f}, p-value = {p_value:.2f} "
              f"({'no clear misfit' if p_value > 0.05 else 'small but detectable misfit'})")

        ax.bar(range(6), observed * 100, color=color, label="Real matches")
        ax.plot(range(6), expected * 100, "o-", color="black", label=f"Poisson (mean {mean:.2f})")
        ax.set_xticks(range(6), ["0", "1", "2", "3", "4", "5+"])
        ax.set_xlabel(f"Goals scored by the {name.lower()} in one match")
        ax.set_title(name)
        ax.legend()
    axes[0].set_ylabel("Share of matches (%)")
    fig.suptitle("Goals per match: real data (bars) vs Poisson curve (line)", fontweight="bold")
    save(fig, "03_goals_per_match.png")


# ---------------------------------------------------- d. goals per season
def goals_per_season(df):
    section("d. Average goals per season")
    per_season = df.groupby("season")[["home_goals", "away_goals", "total_goals"]].mean()
    print(per_season.round(2).to_string())
    print(f"Lowest total: {per_season['total_goals'].idxmin()} "
          f"({per_season['total_goals'].min():.2f}); highest: {per_season['total_goals'].idxmax()} "
          f"({per_season['total_goals'].max():.2f})")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(per_season.index, per_season["total_goals"], marker="o", color="black",
            linewidth=2.5, label="Total (home + away)")
    ax.plot(per_season.index, per_season["home_goals"], marker="s", color=BLUE, label="Home team")
    ax.plot(per_season.index, per_season["away_goals"], marker="^", color=ORANGE, label="Away team")
    for x, y in zip(per_season.index, per_season["total_goals"]):
        ax.text(x, y + 0.07, f"{y:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, 3.4)
    ax.set_xlabel("Season")
    ax.set_ylabel("Average goals per match")
    ax.set_title("Average goals per match in each season")
    ax.legend(loc="lower right")
    plt.setp(ax.get_xticklabels(), rotation=45)
    save(fig, "04_goals_per_season.png")


# ----------------------------------------------- e. team strength over time
def build_team_tables(df):
    """One row per team per match (a 'long' table), and one row per team-season."""
    home = pd.DataFrame({"season": df["season"], "team": df["HomeTeam"], "venue": "home",
                         "points": df["home_points"], "gf": df["home_goals"], "ga": df["away_goals"]})
    away = pd.DataFrame({"season": df["season"], "team": df["AwayTeam"], "venue": "away",
                         "points": df["away_points"], "gf": df["away_goals"], "ga": df["home_goals"]})
    long = pd.concat([home, away], ignore_index=True)
    table = long.groupby(["season", "team"]).agg(points=("points", "sum"), gf=("gf", "sum"),
                                                 ga=("ga", "sum"), games=("points", "size")).reset_index()
    table["gd"] = table["gf"] - table["ga"]
    assert (table["games"] == 38).all(), "every team should play 38 games per season"
    return long, table


def heatmap(pivot, name, title, cmap, label, center=None):
    order = pivot.mean(axis=1).sort_values(ascending=False).index  # strongest team on top
    fig, ax = plt.subplots(figsize=(11, 12))
    ax.set_facecolor("#E8E8E8")  # grey = team not in the league that season
    sns.heatmap(pivot.loc[order], annot=True, fmt=".0f", cmap=cmap, center=center, linewidths=0.5,
                cbar_kws={"label": label}, annot_kws={"size": 8}, ax=ax)
    ax.grid(False)  # otherwise grid lines show through the grey cells
    ax.set_xlabel("Season (grey = not in the Premier League)")
    ax.set_ylabel("Team")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45)
    save(fig, name)


def team_strength(table):
    section("e. Team strength over seasons")
    points = table.pivot(index="team", columns="season", values="points")
    gd = table.pivot(index="team", columns="season", values="gd")
    heatmap(points, "05_points_heatmap.png", "League points per team per season (38 games)",
            "YlGnBu", "Points")
    heatmap(gd, "06_goal_difference_heatmap.png",
            "Goal difference per team per season (goals for minus goals against)",
            "PuOr", "Goal difference", center=0)

    seasons = list(points.columns)
    print("Champions by points:")
    for season in seasons:
        top = table[table["season"] == season].sort_values("points", ascending=False).iloc[0]
        print(f"  {season}: {top['team']} {top['points']:.0f} pts")

    # How well does a team's points in one season predict its points k seasons later?
    results = []
    for lag in range(1, 6):
        xs, ys = [], []
        for i in range(len(seasons) - lag):
            a, b = points[seasons[i]], points[seasons[i + lag]]
            both = a.notna() & b.notna()  # only teams that played in both seasons
            xs += a[both].tolist()
            ys += b[both].tolist()
        results.append((lag, np.corrcoef(xs, ys)[0, 1], len(xs), xs, ys))
    print("\nCorrelation of a team's points with its points k seasons later:")
    for lag, corr, n, _, _ in results:
        print(f"  k={lag}: correlation {corr:.2f} ({n} team pairs)")
    lag1_x, lag1_y = results[0][3], results[0][4]
    slope, intercept = np.polyfit(lag1_x, lag1_y, 1)
    change = np.abs(np.array(lag1_y) - np.array(lag1_x))
    print(f"Next-season points ~ {intercept:.1f} + {slope:.2f} x this-season points "
          f"(slope under 1 = teams drift back toward the middle)")
    print(f"Typical change in points from one season to the next: mean {change.mean():.1f}, "
          f"median {np.median(change):.1f}, 90th percentile {np.percentile(change, 90):.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].scatter(lag1_x, lag1_y, color=BLUE, alpha=0.6)
    line_x = np.array([15, 100])
    axes[0].plot(line_x, line_x, "--", color=GREY, label="No change")
    axes[0].plot(line_x, intercept + slope * line_x, color=VERMILION, label="Best-fit line")
    axes[0].set_xlabel("Points in season t")
    axes[0].set_ylabel("Points in season t+1")
    axes[0].set_title("Points this season vs next (same team)")
    axes[0].legend()
    axes[1].bar([r[0] for r in results], [r[1] for r in results], color=GREEN)
    for lag, corr, n, _, _ in results:
        axes[1].text(lag, corr + 0.01, f"{corr:.2f}\n(n={n})", ha="center", fontsize=9)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Gap between seasons (k seasons)")
    axes[1].set_ylabel("Correlation of points")
    axes[1].set_title("How fast does team strength fade?")
    save(fig, "07_strength_persistence.png")


# ----------------------------------------------- f. home vs away per team
def home_vs_away(long):
    section("f. Home vs away performance per team")
    agg = long.groupby(["team", "venue"]).agg(ppg=("points", "mean"), sd=("points", "std"),
                                              gf=("gf", "mean"), ga=("ga", "mean"),
                                              n=("points", "size")).unstack("venue")
    result = pd.DataFrame({
        "home_ppg": agg[("ppg", "home")], "away_ppg": agg[("ppg", "away")],
        "home_gf": agg[("gf", "home")], "home_ga": agg[("ga", "home")],
        "away_gf": agg[("gf", "away")], "away_ga": agg[("ga", "away")],
        "n_home": agg[("n", "home")],
        # statistical noise of the home-minus-away gap
        "se": np.sqrt(agg[("sd", "home")] ** 2 / agg[("n", "home")] + agg[("sd", "away")] ** 2 / agg[("n", "away")]),
    })
    result["advantage"] = result["home_ppg"] - result["away_ppg"]
    print(f"All teams: {long[long.venue == 'home']['points'].mean():.2f} points per home game vs "
          f"{long[long.venue == 'away']['points'].mean():.2f} per away game")
    keep = result[result["n_home"] >= 57].sort_values("advantage")  # at least 3 seasons
    print(f"Teams with at least 3 seasons in the data: {len(keep)} of {len(result)}")
    shown = keep.sort_values("advantage", ascending=False).round(2)
    print("\nBiggest home advantage (points per game, home minus away):")
    print(shown.head(5)[["home_ppg", "away_ppg", "advantage", "se", "n_home"]].to_string())
    print("\nSmallest home advantage:")
    print(shown.tail(5)[["home_ppg", "away_ppg", "advantage", "se", "n_home"]].to_string())
    clear = int((keep["advantage"] > 2 * keep["se"]).sum())
    print(f"\n{clear} of {len(keep)} teams have a home advantage bigger than 2 standard errors "
          f"(clearly above zero). Typical noise per team: +/- {2 * keep['se'].mean():.2f} points per game.")

    fig, ax = plt.subplots(figsize=(9, 9))
    y = np.arange(len(keep))
    ax.hlines(y, keep["away_ppg"], keep["home_ppg"], color=GREY, linewidth=2)
    ax.scatter(keep["home_ppg"], y, color=BLUE, s=60, label="At home", zorder=3)
    ax.scatter(keep["away_ppg"], y, color=ORANGE, s=60, label="Away", zorder=3)
    ax.set_yticks(y, keep.index)
    ax.set_xlabel("Average league points per match (3 = win, 1 = draw, 0 = loss)")
    ax.set_title("Home vs away points per match (teams with at least 3 seasons)\n"
                 "sorted by home advantage, biggest at the top")
    ax.legend(loc="lower right")
    save(fig, "08_home_vs_away_by_team.png")


# ------------------------------------------- g. same-match stat correlations
def stat_correlations(df):
    section("g. Same-match stats vs goals and result")
    columns = ["HS", "HST", "HC", "HF", "HY", "HR", "AS", "AST", "AC", "AF", "AY", "AR",
               "home_goals", "away_goals", "home_points"]
    corr = df[columns].corr()
    stat_names = columns[:12]  # the 12 shot/corner/foul/card columns
    for target in ["home_goals", "away_goals", "home_points"]:
        strongest = corr.loc[stat_names, target]
        order = strongest.abs().sort_values(ascending=False).head(4).index
        print(f"Strongest stat correlations with {target}: "
              + ", ".join(f"{c} {strongest[c]:+.2f}" for c in order))
    print(f"Weak ones for comparison: fouls HF vs home_points {corr.loc['HF', 'home_points']:+.2f}, "
          f"home yellow cards HY vs home_points {corr.loc['HY', 'home_points']:+.2f}")

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="PuOr", vmin=-1, vmax=1, center=0,
                annot_kws={"size": 7.5}, linewidths=0.5, cbar_kws={"label": "Correlation (-1 to +1)"}, ax=ax)
    ax.set_title("Correlation of same-match stats with goals and result\n"
                 "(these are known only AFTER the match: not usable as predictors)")
    save(fig, "09_same_match_correlation.png")


# ----------------------------------------------------- h. bookmaker odds
def add_implied_probabilities(df):
    """Odds -> probabilities. 1/odds is the 'implied probability'. The three add up
    to MORE than 100% (the overround = bookmaker's margin), so we divide by the sum."""
    raw = 1 / df[["B365H", "B365D", "B365A"]].to_numpy()
    total = raw.sum(axis=1)
    df = df.copy()
    df["overround"] = total - 1
    df[["p_home", "p_draw", "p_away"]] = raw / total[:, None]
    df["favourite"] = np.argmax(df[["p_home", "p_draw", "p_away"]].to_numpy(), axis=1)  # 0 H, 1 D, 2 A
    return df


def bookmaker_reference(df):
    section("h. Bookmaker odds as a reference (Bet365, training seasons only)")
    missing = df.groupby("season")[["B365H", "B365D", "B365A"]].apply(lambda g: g.isna().any(axis=1).sum())
    print("Matches with missing Bet365 odds per season:", missing.to_dict())
    df = add_implied_probabilities(df.dropna(subset=["B365H", "B365D", "B365A"]))

    print(f"Average overround (bookmaker margin): {df['overround'].mean() * 100:.1f}% "
          f"(per season: {df.groupby('season')['overround'].mean().mul(100).round(1).min():.1f}% "
          f"to {df.groupby('season')['overround'].mean().mul(100).round(1).max():.1f}%)")
    df["fav_correct"] = df["favourite"] == df["result"]
    df["home_correct"] = df["result"] == 0
    per_season = df.groupby("season")[["fav_correct", "home_correct"]].mean() * 100
    print("\nAccuracy per season (%): bookmaker favourite vs always-home-win")
    print(per_season.round(1).to_string())
    print(f"All training seasons: favourite {df['fav_correct'].mean() * 100:.1f}%, "
          f"always-home-win {df['home_correct'].mean() * 100:.1f}%")
    print(f"Matches where the draw was the bookmaker's favourite outcome: "
          f"{int((df['favourite'] == 1).sum())} of {len(df)}; highest draw probability seen "
          f"{df['p_draw'].max() * 100:.0f}%, average {df['p_draw'].mean() * 100:.0f}%")

    # Calibration: when the bookmaker says X%, how often does it really happen?
    probs = df[["p_home", "p_draw", "p_away"]].to_numpy().ravel()
    happened = np.eye(3)[df["result"].to_numpy()].ravel()
    bins = np.arange(0, 1.01, 0.1)
    which = np.digitize(probs, bins) - 1
    rows = [(probs[which == b].mean() * 100, happened[which == b].mean() * 100, (which == b).sum())
            for b in range(10) if (which == b).sum() >= 30]
    print("\nCalibration (bookmaker chance % -> how often it really happened %):")
    for said, real, n in rows:
        print(f"  said {said:4.0f}% -> happened {real:4.0f}%   ({n} cases)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    width = 0.4
    x = np.arange(len(per_season))
    axes[0].bar(x - width / 2, per_season["fav_correct"], width, color=BLUE, label="Bookmaker favourite")
    axes[0].bar(x + width / 2, per_season["home_correct"], width, color=GREY, label="Always home win")
    axes[0].set_xticks(x, per_season.index, rotation=45)
    axes[0].set_ylim(0, 75)
    axes[0].set_xlabel("Season")
    axes[0].set_ylabel("Matches predicted correctly (%)")
    axes[0].set_title("How often the bookmaker favourite won")
    axes[0].legend(loc="upper left")
    said, real, sizes = zip(*rows)
    axes[1].plot([0, 80], [0, 80], "--", color=GREY, label="Perfect")
    axes[1].scatter(said, real, s=[max(n / 15, 30) for n in sizes], color=VERMILION, label="Real (size = cases)")
    axes[1].set_xlim(-2, 92)
    axes[1].set_ylim(-2, 95)
    axes[1].set_xlabel("Chance given by the bookmaker (%, margin removed)")
    axes[1].set_ylabel("How often it really happened (%)")
    axes[1].set_title("Are the odds honest? (calibration)")
    axes[1].legend(loc="upper left")
    save(fig, "10_bookmaker_odds.png")
    return df


# ----------------------------------------------------------------- i. draws
def draws(df):
    section("i. Draws")
    df = df.copy()
    df["is_draw"] = df["result"] == 1
    # (Draw rate by total goals would be pointless: an odd total can never be a draw.)
    drawn = df[df["is_draw"]]
    score = drawn["home_goals"].clip(upper=3).astype(str)  # in a draw both teams score the same
    score = (score + "-" + score).replace({"3-3": "3-3 or more"})
    by_score = score.value_counts().reindex(["0-0", "1-1", "2-2", "3-3 or more"])
    print(f"Of {len(drawn)} draws, by scoreline:")
    for name, n in by_score.items():
        print(f"  {name:12} {n:4}  ({n / len(drawn) * 100:.0f}% of draws)")
    print(f"-> {(by_score['0-0'] + by_score['1-1']) / len(drawn) * 100:.0f}% of draws are 0-0 or 1-1")

    # Before the match: how evenly matched do the odds say the teams are?
    df["gap"] = (df["p_home"] - df["p_away"]).abs()
    edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50+"]
    df["closeness"] = pd.cut(df["gap"], bins=edges, labels=labels, include_lowest=True)
    by_gap = df.groupby("closeness", observed=True)["is_draw"].agg(["mean", "size"])
    by_gap["mean"] *= 100
    print("\nDraw rate by how close the match looks beforehand (|home chance - away chance|, points):")
    print(by_gap.rename(columns={"mean": "draw %", "size": "matches"}).round(1).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].bar(by_score.index, by_score.values / len(drawn) * 100, color=GREY)
    for i, n in enumerate(by_score.values):
        axes[0].text(i, n / len(drawn) * 100 + 1, f"{n / len(drawn) * 100:.0f}%\n(n={n})",
                     ha="center", fontsize=9)
    axes[0].set_ylim(0, 60)
    axes[0].set_xlabel("Final score of the drawn match")
    axes[0].set_ylabel("Share of all draws (%)")
    axes[0].set_title("What draws look like: mostly 0-0 and 1-1")
    axes[1].bar(by_gap.index.astype(str), by_gap["mean"], color=BLUE)
    for i, (rate, n) in enumerate(zip(by_gap["mean"], by_gap["size"])):
        axes[1].text(i, rate + 0.7, f"{rate:.0f}%\n(n={n})", ha="center", fontsize=9)
    axes[1].set_ylim(0, 42)
    axes[1].set_xlabel("Gap between bookmaker's home and away win chance (percentage points)")
    axes[1].set_ylabel("Share of matches that were draws (%)")
    axes[1].set_title("Draws by pre-match closeness (usable before kick-off)")
    save(fig, "11_draws.png")


def main():
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_train()
    print(f"Training data: {len(df)} matches, seasons {df['season'].min()} to {df['season'].max()}, "
          f"dates {df['Date'].min().date()} to {df['Date'].max().date()}")

    outcome_distribution(df)
    home_advantage_by_season(df)
    goals_per_match(df)
    goals_per_season(df)
    long, table = build_team_tables(df)
    team_strength(table)
    home_vs_away(long)
    stat_correlations(df)
    with_odds = bookmaker_reference(load_training_odds(df))
    draws(with_odds)

    section("Figures")
    for path in sorted(FIG_DIR.glob("*.png")):
        print(f"{path.name}  {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
