"""Stage 2: turn the raw season files into one clean, time-ordered match table.

Run:  python src/preprocess.py
Raw files in data/raw/ are only read, never changed.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# ---------------------------------------------------------------- constants
# Team spelling fixes: {"spelling found in some file": "the one name we keep"}.
# The Stage 1 inspection found NO club spelled two ways in these files, so this
# is empty. If a new data source spells a club differently, add the fix here.
TEAM_NAME_FIXES = {}

# Fixed order for the target. The number is the class code the models will use.
RESULT_CODES = {"H": 0, "D": 1, "A": 2}
RESULT_LABELS = {0: "Home win", 1: "Draw", 2: "Away win"}

CORE_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
HALF_TIME_COLUMNS = ["HTHG", "HTAG", "HTR"]
# Per-match stats: shots, shots on target, fouls, corners, yellow and red cards.
STAT_COLUMNS = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
# Columns we do not need at all: constant, or not part of the plan.
UNUSED_COLUMNS = ["Div", "Referee", "Time"]

# Time-based split. Later stages should import these constants.
TRAIN_SEASONS = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
                 "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
TEST_SEASONS = ["2025-26"]
# For tuning later: use ONLY training seasons. Fit on the early ones and
# check on the two most recent training seasons. The test season is never used.
VALIDATION_SCHEME = {
    "fit": TRAIN_SEASONS[:8],       # 2015-16 ... 2022-23
    "validate": TRAIN_SEASONS[8:],  # 2023-24, 2024-25
}


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


# ------------------------------------------------------------------ loading
def load_raw():
    """Read every season file into one table and add a `season` column."""
    frames = {}  # season label -> table
    for path in sorted(RAW_DIR.glob("E0_*.csv")):
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            print(f"note: {path.name} is not UTF-8, reading as latin-1")
            df = pd.read_csv(path, encoding="latin-1")
        frames[path.stem.replace("E0_", "")] = df
    # keys= puts the season label in front of every row, then we turn it into a column
    combined = pd.concat(frames, names=["season", None]).copy()  # copy() tidies memory layout
    return combined.reset_index(level="season").reset_index(drop=True)


def standardize_columns(df):
    """Tidy column names and drop columns we never use."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df.drop(columns=UNUSED_COLUMNS, errors="ignore")


# -------------------------------------------------------------------- dates
def parse_dates(df):
    """Older files use '13/08/16', newer ones '13/08/2016'. Try the 4-digit
    pattern first, then the 2-digit one. Always day first."""
    df = df.copy()
    four_digit = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    two_digit = pd.to_datetime(df["Date"], format="%d/%m/%y", errors="coerce")
    df["Date"] = four_digit.fillna(two_digit)
    if df["Date"].isna().any():
        raise ValueError("Some dates could not be parsed")
    return df


# -------------------------------------------------------------------- teams
def standardize_teams(df):
    """Remove stray spaces and apply TEAM_NAME_FIXES so each club has one name."""
    df = df.copy()
    for column in ["HomeTeam", "AwayTeam"]:
        df[column] = df[column].str.strip().replace(TEAM_NAME_FIXES)
    return df


# --------------------------------------------------------------- duplicates
def drop_duplicates(df):
    """Two rows with the same date and the same two teams are the same match."""
    keys = ["Date", "HomeTeam", "AwayTeam"]
    count = int(df.duplicated(subset=keys).sum())
    print(f"Duplicate matches (same Date + HomeTeam + AwayTeam): {count}")
    return df.drop_duplicates(subset=keys, keep="first")


# ----------------------------------------------------------- missing values
def drop_rows_missing_core(df):
    """A match without teams, score or result is useless: drop it and say so."""
    bad = df[df[CORE_COLUMNS].isna().any(axis=1)]
    if bad.empty:
        print("Rows missing a core column: 0 (nothing dropped)")
        return df
    for index, row in bad.iterrows():
        missing = [c for c in CORE_COLUMNS if pd.isna(row[c])]
        print(f"Dropping row {index} ({row['season']}): missing {missing}")
    return df.drop(index=bad.index)


def report_missing(df, columns):
    """Count and percentage of empty cells per column, and per season if any."""
    counts = df[columns].isna().sum()
    table = pd.DataFrame({"missing": counts, "percent": (counts / len(df) * 100).round(2)})
    print(table.to_string())
    problem_columns = counts[counts > 0].index.tolist()
    if not problem_columns:
        print("-> No missing values in these columns.")
    for column in problem_columns:
        print(f"\nMissing '{column}' per season:")
        print(df.groupby("season")[column].apply(lambda s: s.isna().sum()).to_string())


def report_odds(df, odds_columns):
    """Odds are patchy: each season has a different set of bookmaker columns."""
    print(f"{'season':8} {'columns with data':>18} {'empty cells in them':>20}")
    for season, group in df.groupby("season"):
        has_data = [c for c in odds_columns if group[c].notna().any()]
        empty = group[has_data].isna().mean().mean() * 100
        print(f"{season:8} {len(has_data):>18} {empty:>19.1f}%")


def fill_stats_with_train_median(df):
    """Fill missing match stats with the MEDIAN of the TRAINING rows only.
    Using the test rows to compute the median would leak test information."""
    medians = df.loc[df["split"] == "train", STAT_COLUMNS].median()
    filled = int(df[STAT_COLUMNS].isna().sum().sum())
    df = df.copy()
    df[STAT_COLUMNS] = df[STAT_COLUMNS].fillna(medians)
    print(f"Cells filled with the training median: {filled}")
    print("Training medians (these would be used for train AND test):")
    print(medians.to_string())
    return df


# ------------------------------------------------------------------ targets
def add_targets(df):
    """result: 0 Home win, 1 Draw, 2 Away win. Also integer goals per team."""
    df = df.copy()
    df["result"] = df["FTR"].map(RESULT_CODES)
    if df["result"].isna().any():
        raise ValueError("FTR contains a value other than H, D, A")
    df["result"] = df["result"].astype(int)
    df["home_goals"] = df["FTHG"].astype(int)
    df["away_goals"] = df["FTAG"].astype(int)
    # Sanity check: the result must agree with the score.
    expected = (df["home_goals"] - df["away_goals"]).apply(
        lambda d: 0 if d > 0 else (2 if d < 0 else 1))
    print(f"Rows where FTR disagrees with the score: {int((expected != df['result']).sum())}")
    return df


def print_class_counts(df, name):
    counts = df["result"].value_counts().sort_index()
    percent = (counts / len(df) * 100).round(1)
    summary = pd.DataFrame({"label": [RESULT_LABELS[c] for c in counts.index],
                            "matches": counts.values, "percent": percent.values},
                           index=counts.index)
    print(f"\n{name} ({len(df)} matches):")
    print(summary.to_string())


# -------------------------------------------------------------------- split
def sort_matches(df):
    """Chronological order, with team names as the tie-breaker for same-day games."""
    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    df.insert(0, "match_id", range(len(df)))
    return df


def split_by_time(df):
    """Whole seasons go to train or test. Never a random split."""
    df = df.copy()
    df["split"] = df["season"].isin(TEST_SEASONS).map({True: "test", False: "train"})
    return df, df[df["split"] == "train"], df[df["split"] == "test"]


def validation_split(train):
    """Tuning split that only touches training data (see VALIDATION_SCHEME)."""
    fit = train[train["season"].isin(VALIDATION_SCHEME["fit"])]
    validate = train[train["season"].isin(VALIDATION_SCHEME["validate"])]
    return fit, validate


def separate_odds(df):
    """Betting columns go to their own table. They are not model features."""
    keep = (["match_id", "season", "split"] + CORE_COLUMNS + ["result", "home_goals",
            "away_goals"] + HALF_TIME_COLUMNS + STAT_COLUMNS)
    odds_columns = [c for c in df.columns if c not in keep]
    matches = df[keep]
    odds = df[["match_id", "season", "Date", "HomeTeam", "AwayTeam"] + odds_columns]
    return matches, odds, odds_columns


# ------------------------------------------------------------------- demos
def demo_encoding(train):
    """One-hot encode team names. Fit on training data only."""
    team_columns = ["HomeTeam", "AwayTeam"]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train[team_columns])
    print(f"Teams seen in training: {len(encoder.categories_[0])} -> "
          f"{len(encoder.get_feature_names_out())} new 0/1 columns "
          f"(one per team for home, one per team for away)")

    row = train.iloc[[0]]
    encoded = encoder.transform(row[team_columns])[0]
    active = encoder.get_feature_names_out()[encoded == 1]
    print(f"\nFirst match: {row.iloc[0]['HomeTeam']} v {row.iloc[0]['AwayTeam']}")
    print(f"Columns that are 1: {list(active)}  (all {len(encoded) - 2} others are 0)")

    # Real unseen-team example: fit only on the early training seasons.
    fit, validate = validation_split(train)
    early = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(fit[team_columns])
    known = set(early.categories_[0])
    unseen = sorted((set(validate["HomeTeam"]) | set(validate["AwayTeam"])) - known)
    print(f"\nUnseen-team demo: encoder fitted on {VALIDATION_SCHEME['fit'][0]} to "
          f"{VALIDATION_SCHEME['fit'][-1]} only.")
    print(f"Teams in 2023-24/2024-25 that it has never seen: {unseen}")
    sample = validate[validate["HomeTeam"].isin(unseen)].iloc[[0]]
    vector = early.transform(sample[team_columns])[0]
    print(f"Row: {sample.iloc[0]['HomeTeam']} v {sample.iloc[0]['AwayTeam']} -> "
          f"home-team block all zeros: {vector[:len(early.categories_[0])].sum() == 0}, "
          f"no error raised")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(encoder, MODELS_DIR / "team_encoder.joblib")
    print("\nSaved encoder (fitted on all training data): models/team_encoder.joblib")


def demo_scaling(train, test):
    """Standardise the match stats. Fit on training data only."""
    scaler = StandardScaler().fit(train[STAT_COLUMNS])
    train_scaled = pd.DataFrame(scaler.transform(train[STAT_COLUMNS]), columns=STAT_COLUMNS)
    test_scaled = pd.DataFrame(scaler.transform(test[STAT_COLUMNS]), columns=STAT_COLUMNS)

    table = pd.DataFrame({
        "train mean before": train[STAT_COLUMNS].mean(),
        "train std before": train[STAT_COLUMNS].std(ddof=0),
        "train mean after": train_scaled.mean(),
        "train std after": train_scaled.std(ddof=0),
        "test mean after": test_scaled.mean(),
    }).round(3) + 0.0  # "+ 0.0" turns a printed -0.0 into 0.0
    print(table.to_string())

    joblib.dump(scaler, MODELS_DIR / "stats_scaler.joblib")
    print("\nSaved scaler (fitted on training data): models/stats_scaler.joblib")


# --------------------------------------------------------------------- main
def main():
    section("1. Load raw files")
    df = load_raw()
    print(f"Loaded {df['season'].nunique()} seasons, {len(df)} rows, {df.shape[1]} columns")
    df = standardize_columns(df)

    section("2. Dates")
    before = df.groupby("season")["Date"].first()[["2015-16", "2016-17", "2025-26"]]
    df = parse_dates(df)
    after = df.groupby("season")["Date"].first()[["2015-16", "2016-17", "2025-26"]]
    print("First row of some seasons, before -> after:")
    for season in before.index:
        print(f"  {season}: '{before[season]}' -> {after[season].date()}")

    section("3. Team names")
    df = standardize_teams(df)
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    print(f"Spelling fixes applied: {len(TEAM_NAME_FIXES)}")
    print(f"{len(teams)} unique teams: {', '.join(teams)}")

    section("4. Duplicates")
    df = drop_duplicates(df)

    section("5. Core columns")
    df = drop_rows_missing_core(df)
    df = sort_matches(df)
    print(f"Sorted by date. First: {df['Date'].iloc[0].date()}, last: {df['Date'].iloc[-1].date()}")

    section("6. Targets")
    df = add_targets(df)
    print("Result codes:", {RESULT_LABELS[v]: v for v in RESULT_CODES.values()})

    section("7. Time-based split")
    df, train, test = split_by_time(df)
    fit, validate = validation_split(train)
    print(f"Train: {len(train)} matches, seasons {TRAIN_SEASONS[0]} to {TRAIN_SEASONS[-1]}")
    print(f"  of which tuning-fit: {len(fit)} ({VALIDATION_SCHEME['fit'][0]} to "
          f"{VALIDATION_SCHEME['fit'][-1]}), tuning-validate: {len(validate)} "
          f"({', '.join(VALIDATION_SCHEME['validate'])})")
    print(f"Test:  {len(test)} matches, season {TEST_SEASONS[0]}")
    print(f"Latest training date: {train['Date'].max().date()}")
    print(f"Earliest test date:   {test['Date'].min().date()}")
    assert train["Date"].max() < test["Date"].min(), "Train overlaps test!"
    print("Check passed: every training match is before every test match.")
    print(f"Latest tuning-fit date: {fit['Date'].max().date()} < earliest "
          f"validation date: {validate['Date'].min().date()}")
    for name, part in [("Whole table", df), ("Train", train), ("Test", test)]:
        print_class_counts(part, f"Result counts: {name}")

    section("8. Missing values")
    report_missing(df, CORE_COLUMNS + HALF_TIME_COLUMNS + STAT_COLUMNS)
    odds_columns = [c for c in df.columns if c not in CORE_COLUMNS + HALF_TIME_COLUMNS
                    + STAT_COLUMNS + ["season", "match_id", "split", "result",
                                      "home_goals", "away_goals"]]
    print(f"\nBetting columns ({len(odds_columns)}): different columns exist in "
          f"different seasons, so many cells are empty by design:")
    report_odds(df, odds_columns)
    print("Decision: odds go to data/processed/odds.csv, are NOT model features, and "
          "are NOT filled in.")
    df = fill_stats_with_train_median(df)
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    section("9. One-hot encoding demo (team names)")
    demo_encoding(train)

    section("10. Scaling demo (match stats)")
    demo_scaling(train, test)

    section("11. Save outputs")
    matches, odds, _ = separate_odds(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_csv(PROCESSED_DIR / "matches_clean.csv", index=False)
    matches[matches["split"] == "train"].to_csv(PROCESSED_DIR / "train.csv", index=False)
    matches[matches["split"] == "test"].to_csv(PROCESSED_DIR / "test.csv", index=False)
    odds.to_csv(PROCESSED_DIR / "odds.csv", index=False)
    for name in ["matches_clean", "train", "test", "odds"]:
        path = PROCESSED_DIR / f"{name}.csv"
        rows = len(pd.read_csv(path))
        print(f"{path.relative_to(ROOT)}: {rows} rows, {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
