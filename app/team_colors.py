"""Team branding for the app: primary colour, 3-letter code, file slug. Names match the project's data (Stage 2 cleaning)."""

COLORS = {
    "Arsenal": "#EF0107", "Aston Villa": "#95BFE5", "Bournemouth": "#DA291C", "Brentford": "#E30613",
    "Brighton": "#0057B8", "Burnley": "#6C1D45", "Chelsea": "#034694", "Crystal Palace": "#1B458F",
    "Everton": "#003399", "Fulham": "#000000", "Leeds": "#FFCD00", "Liverpool": "#C8102E",
    "Man City": "#6CABDD", "Man United": "#DA291C", "Newcastle": "#241F20", "Nott'm Forest": "#DD0000",
    "Sunderland": "#FF0000", "Tottenham": "#132257", "West Ham": "#7A263A", "Wolves": "#FDB913",
}

CODES = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE", "Brighton": "BHA",
    "Burnley": "BUR", "Chelsea": "CHE", "Crystal Palace": "CRY", "Everton": "EVE", "Fulham": "FUL",
    "Leeds": "LEE", "Liverpool": "LIV", "Man City": "MCI", "Man United": "MUN", "Newcastle": "NEW",
    "Nott'm Forest": "NFO", "Sunderland": "SUN", "Tottenham": "TOT", "West Ham": "WHU", "Wolves": "WOL",
}

# file name of the crest PNG (data/app/team_crests/{slug}.png)
SLUG = {team: team.lower().replace("'", "").replace(" ", "_") for team in COLORS}

# only used by the one-off fetch_crests.py
WIKI_ARTICLE = {
    "Arsenal": "Arsenal F.C.", "Aston Villa": "Aston Villa F.C.", "Bournemouth": "AFC Bournemouth",
    "Brentford": "Brentford F.C.", "Brighton": "Brighton & Hove Albion F.C.", "Burnley": "Burnley F.C.",
    "Chelsea": "Chelsea F.C.", "Crystal Palace": "Crystal Palace F.C.", "Everton": "Everton F.C.",
    "Fulham": "Fulham F.C.", "Leeds": "Leeds United F.C.", "Liverpool": "Liverpool F.C.",
    "Man City": "Manchester City F.C.", "Man United": "Manchester United F.C.", "Newcastle": "Newcastle United F.C.",
    "Nott'm Forest": "Nottingham Forest F.C.", "Sunderland": "Sunderland A.F.C.", "Tottenham": "Tottenham Hotspur F.C.",
    "West Ham": "West Ham United F.C.", "Wolves": "Wolverhampton Wanderers F.C.",
}
