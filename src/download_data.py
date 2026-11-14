"""Download one Premier League CSV per season from football-data.co.uk.

Safe to re-run: files that already exist are skipped, never overwritten,
because raw data must stay exactly as downloaded.
"""
import time
from pathlib import Path

import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# (site code, label used in our file name)
SEASONS = [
    ("1516", "2015-16"), ("1617", "2016-17"), ("1718", "2017-18"),
    ("1819", "2018-19"), ("1920", "2019-20"), ("2021", "2020-21"),
    ("2122", "2021-22"), ("2223", "2022-23"), ("2324", "2023-24"),
    ("2425", "2024-25"), ("2526", "2025-26"),
]


def looks_like_csv(response):
    """A real file starts with a header row containing 'HomeTeam'.
    Some sites return an HTML error page with status 200, so check the content."""
    start = response.content[:2000].decode("latin-1", errors="replace")
    return "HomeTeam" in start and "<html" not in start.lower()


def download_season(code, label):
    """Download one season. Returns True on success or skip, False on failure."""
    target = RAW_DIR / f"E0_{label}.csv"
    if target.exists():
        print(f"SKIP  {label}: already exists ({target.name})")
        return True, False  # (ok, made_request)

    url = BASE_URL.format(code=code)
    try:
        response = requests.get(url, timeout=30)
    except requests.RequestException as error:
        print(f"FAIL  {label}: could not reach {url} ({error})")
        return False, True

    if response.status_code != 200:
        print(f"FAIL  {label}: HTTP {response.status_code} for {url}")
        return False, True
    if not looks_like_csv(response):
        print(f"FAIL  {label}: response is not a valid CSV ({url})")
        return False, True

    target.write_bytes(response.content)  # bytes, so the file is stored unchanged
    print(f"OK    {label}: saved {target.name} ({len(response.content) / 1024:.0f} KB)")
    return True, True


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for code, label in SEASONS:
        ok, made_request = download_season(code, label)
        if not ok:
            failed.append(label)
        if made_request:
            time.sleep(1)  # be polite to the website

    print()
    if failed:
        print("Failed seasons:", ", ".join(failed))
    else:
        print("All seasons are in data/raw/.")


if __name__ == "__main__":
    main()
