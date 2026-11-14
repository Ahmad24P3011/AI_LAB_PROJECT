"""Inspect the raw season files (read-only: nothing is modified or saved)."""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_file(path):
    """Try UTF-8 first, then latin-1. Tell the user if we had to fall back."""
    try:
        return pd.read_csv(path, encoding="utf-8"), "utf-8"
    except UnicodeDecodeError:
        print(f"  note: {path.name} is not UTF-8, reading it as latin-1")
        return pd.read_csv(path, encoding="latin-1"), "latin-1"


def load_all():
    data = {}
    for path in sorted(RAW_DIR.glob("E0_*.csv")):
        season = path.stem.replace("E0_", "")
        data[season], _ = load_file(path)
    return data


def print_basic_checks(data):
    print("=== Rows, columns, duplicates per season ===")
    print(f"{'season':8} {'rows':>5} {'cols':>5} {'dup rows':>9} {'complete?':>10}")
    for season, df in data.items():
        # Real matches have a result; empty rows (blank lines) are not matches.
        rows = len(df)
        dups = int(df.duplicated().sum())
        flag = "yes" if rows == 380 else f"NO ({rows})"
        print(f"{season:8} {rows:>5} {df.shape[1]:>5} {dups:>9} {flag:>10}")


def print_column_overlap(data):
    print("\n=== Columns in ALL seasons vs only some ===")
    sets = {s: set(df.columns) for s, df in data.items()}
    common = set.intersection(*sets.values())
    everything = set.union(*sets.values())
    print(f"In all {len(data)} seasons ({len(common)}): {sorted(common)}")
    partial = sorted(everything - common)
    print(f"\nOnly in some seasons ({len(partial)}):")
    for col in partial:
        present = [s for s in data if col in sets[s]]
        print(f"  {col:10} -> {present[0]} ... {present[-1]} ({len(present)} seasons)")


def print_missing_in_data(data):
    print("\n=== Empty (NaN) cells in key columns, per season ===")
    key = ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HS", "HST", "B365H"]
    print(f"{'season':8} " + " ".join(f"{c:>8}" for c in key))
    for season, df in data.items():
        counts = [str(int(df[c].isna().sum())) if c in df else "n/a" for c in key]
        print(f"{season:8} " + " ".join(f"{c:>8}" for c in counts))


def print_teams(data):
    print("\n=== Teams per season ===")
    teams = {}
    for season, df in data.items():
        teams[season] = sorted(set(df["HomeTeam"].dropna()) | set(df["AwayTeam"].dropna()))
        print(f"{season}: {len(teams[season])} teams: {', '.join(teams[season])}")

    # Look for names that might be the same club spelled differently.
    all_names = sorted(set().union(*teams.values()))
    print(f"\nAll {len(all_names)} distinct names across seasons:")
    print(", ".join(all_names))
    print("\nNames appearing in only some seasons (spot spelling variants here):")
    for name in all_names:
        seasons = [s for s in teams if name in teams[s]]
        if len(seasons) < len(teams):
            print(f"  {name:16} {len(seasons):>2} seasons: {seasons[0]} ... {seasons[-1]}")


def print_latest_season_progress(data):
    print("\n=== 2025-26 completeness ===")
    df = data["2025-26"]
    dates = pd.to_datetime(df["Date"], dayfirst=True)
    print(f"Rows: {len(df)} of 380. Dates: {dates.min().date()} to {dates.max().date()}")
    played = df["FTR"].notna().sum()
    print(f"Matches with a result: {played}")
    games = pd.concat([df["HomeTeam"], df["AwayTeam"]]).value_counts()
    print(f"Games per team: min {games.min()}, max {games.max()}")


def main():
    data = load_all()
    print_basic_checks(data)
    print_column_overlap(data)
    print_missing_in_data(data)
    print_teams(data)
    print_latest_season_progress(data)


if __name__ == "__main__":
    main()
