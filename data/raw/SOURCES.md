# Data sources

Files in this folder are raw: they are saved exactly as downloaded and are never edited.
Cleaned versions go in `data/processed/`. Re-create them with `python src/download_data.py`.

## 1. football-data.co.uk: Premier League match results

- **Name:** Football-Data.co.uk, English Premier League (division code E0)
- **URL pattern:** `https://www.football-data.co.uk/mmz4281/SSSS/E0.csv`
  (SSSS = 1516, 1617, 1718, 1819, 1920, 2021, 2122, 2223, 2324, 2425, 2526)
- **Date downloaded:** 2026-11-14
- **Saved as:** `E0_2015-16.csv` ... `E0_2025-26.csv` (11 files, 380 matches each, 4,180 matches in total)
- **What it contains:** one row per match: date, teams, full-time and half-time score,
  shots, corners, fouls, cards, referee and pre-match (and, from 2019-20, closing)
  betting odds from several bookmakers (closing odds from most bookmakers only from
  2019-20; Pinnacle closing odds are in every season). Older seasons have fewer columns
  (65 columns in 2015-16 up to 132 in 2025-26).
- **Column notes:** https://www.football-data.co.uk/notes.txt
- **Terms of use:** I found no licence or terms-of-use statement on the data pages or in
  `notes.txt`. The site offers the files as free downloads. We use them for a university
  project, cite the source in the report, download politely (1 request per second, once)
  and do not redistribute them as a product. If the course needs a formal statement,
  contact the site owner.

## 2. Player data (not downloaded yet)
Investigated only, see the Stage 1 summary. Nothing from player sources is stored here.

## Data dictionary (only columns that exist in all 11 files unless noted)

| Column | Meaning |
|---|---|
| Div | Division code (E0 = Premier League) |
| Date | Match date, `dd/mm/yyyy` (older files may use 2-digit years) |
| Time | Kick-off time (only 2019-20 onward) |
| HomeTeam / AwayTeam | Team names (as spelled by this site, e.g. "Man United", "Nott'm Forest") |
| FTHG / FTAG | Full-time goals scored by home / away team |
| FTR | Full-time result: H = home win, D = draw, A = away win (our classification target) |
| HTHG / HTAG / HTR | Same three things at half time |
| Referee | Match referee |
| HS / AS | Total shots by home / away team |
| HST / AST | Shots on target by home / away team |
| HF / AF | Fouls committed by home / away team |
| HC / AC | Corners won by home / away team |
| HY / AY | Yellow cards for home / away team |
| HR / AR | Red cards for home / away team |
| B365H / B365D / B365A | Bet365 decimal odds for home win / draw / away win (before the match) |
| BWH / BWD / BWA | Bet&Win odds for home / draw / away |
| PSH / PSD / PSA | Pinnacle odds for home / draw / away (in 2025-26 about 170 of 380 are empty) |
| PSCH / PSCD / PSCA | Pinnacle closing odds (odds just before kick-off) |
| AvgH / AvgD / AvgA | Market average odds (only 2019-20 onward; older files use `BbAvH` etc.) |
| MaxH / MaxD / MaxA | Market maximum odds (only 2019-20 onward; older files use `BbMxH` etc.) |
| Other bookmaker columns | `WH*`, `IW*`, `VC*`, `LB*`, `1XB*`, `BF*`, `BMGM*`, `BV*`, `CL*` and others appear only in some seasons. Over/under 2.5 goals and Asian handicap odds also exist in some seasons. |

Odds are "decimal odds": odds of 1.30 means a 1 unit bet returns 1.30 in total if it wins.
The chance implied by odds is roughly 1 / odds (1 / 1.30 = 77%).
