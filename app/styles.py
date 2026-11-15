"""Look and feel of the app: palette, CSS and small HTML builders. Pure helpers (they return strings); the pages call them.

Palette (the only colours used): PL purple background, PL green = home win / call to action, grey = draw,
PL cyan = away win, PL magenta = danger / relegation. The banner gradient is hard-coded below.
"""
import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st

from team_colors import CODES, COLORS, SLUG

ROOT = Path(__file__).resolve().parent.parent
CREST_DIR = ROOT / "data" / "app" / "team_crests"
HERO_SVG = ROOT / "data" / "app" / "hero_ball.svg"

GREEN, GREY, CYAN, MAGENTA = "#00FF87", "#9CA3AF", "#04F5FF", "#E90052"
PURPLE, CARD, VIOLET = "#37003C", "#240029", "#963CFF"
GRADIENT = f"linear-gradient(90deg, {PURPLE} 0%, {VIOLET} 55%, {CYAN} 100%)"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stSelectbox, .stRadio {{ font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }}
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"], [data-testid="stSidebarCollapseButton"] {{ color: #FFFFFF; }}
[data-testid="stPageLink"] a {{ background: {CARD}; border-radius: 999px; padding: 6px 16px; }}
[data-testid="stPageLink"] a:hover {{ background: #3a0a42; }}   /* keeps the small sidebar toggle usable on phones */
.block-container {{ padding: 0 2rem 2rem 2rem; max-width: 1180px; }}
h1 {{ font-size: 48px; font-weight: 800; }} h2 {{ font-size: 32px; font-weight: 800; }} h3 {{ font-size: 20px; font-weight: 600; }}
p, li, label {{ font-size: 16px; }}
.hero {{ background: {GRADIENT}; height: 140px; border-radius: 0 0 28px 28px; padding: 26px 36px; margin: 0 0 22px 0;
         display: flex; align-items: center; justify-content: space-between; }}
.hero h1 {{ margin: 0; padding: 0; font-size: 48px; line-height: 1.05; color: #fff; text-shadow: none; }}
.hero p {{ margin: 6px 0 0 0; font-size: 18px; color: rgba(255,255,255,.8); }}
.hero img {{ height: 92px; opacity: .92; }}
.card {{ background: {CARD}; border-radius: 16px; padding: 18px 20px; }}
.tile {{ background: {CARD}; border-radius: 16px; padding: 18px 12px; text-align: center; min-height: 128px; }}
.tile .big {{ font-size: 56px; font-weight: 800; line-height: 1.05; }}
.tile .label {{ font-size: 14px; color: {GREY}; margin-top: 4px; }}
.tile .big.sm {{ font-size: 34px; }}
.matchup {{ display: flex; align-items: center; justify-content: space-between; border-radius: 16px; padding: 22px 16px; background: {CARD}; overflow: hidden; }}
.side {{ flex: 1 1 0; min-width: 0; text-align: center; padding: 18px 8px; border-radius: 14px; }}
.side .name {{ font-size: clamp(16px, 2.2vw, 28px); font-weight: 800; margin-top: 8px; overflow-wrap: anywhere; }}
.vs {{ width: 80px; height: 80px; border-radius: 50%; background: {GRADIENT}; display: flex; align-items: center; justify-content: center;
       font-weight: 800; font-size: 24px; margin: 0 18px; flex: none; }}
.score {{ background: #000; border-radius: 14px; padding: 14px 10px; text-align: center; font-family: 'Courier New', ui-monospace, monospace;
          font-size: 48px; font-weight: 800; letter-spacing: 2px; margin: 18px 0 4px 0; white-space: nowrap; overflow: hidden; }}
.score small {{ display: block; font-family: 'Inter', sans-serif; font-size: 14px; letter-spacing: 0; font-weight: 400; color: {GREY}; white-space: normal; }}
.crest {{ display: inline-block; vertical-align: middle; border-radius: 50%; }}
.badge {{ display: inline-flex; align-items: center; justify-content: center; border-radius: 50%; color: #fff; font-weight: 800; vertical-align: middle; }}
.pl-table {{ width: 100%; border-collapse: collapse; font-size: 15px; }}
.pl-table th {{ text-align: right; color: {GREY}; font-weight: 600; font-size: 13px; padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,.15); }}
.pl-table td {{ text-align: right; padding: 7px 10px; background: {CARD}; border-bottom: 2px solid {PURPLE}; }}
.pl-table th:first-child, .pl-table td:first-child {{ text-align: left; }}
.pl-table td:first-child {{ font-weight: 700; }}
.pl-table tr:hover td {{ background: #3a0a42; }}
.pl-table .primary td:first-child {{ border-left: 4px solid {GREEN}; }}
.table-wrap {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
.tile-row {{ display: flex; gap: 12px; flex-wrap: nowrap; margin-bottom: 4px; }}
.tile-row .tile {{ flex: 1 1 0; min-width: 0; }}
.score.sm {{ font-size: 40px; }}
.note {{ color: {GREY}; font-size: 13px; }}
.disclaimer {{ text-align: center; color: {GREY}; font-size: 13px; margin-top: 36px; }}
[data-testid="stSidebar"] {{ background: {CARD}; }}
@media (max-width: 700px) {{
  .block-container {{ padding: 0 1rem 2rem 1rem; }}
  .hero {{ height: auto; padding: 18px 20px; }} .hero h1 {{ font-size: 32px; }} .hero p {{ font-size: 15px; }} .hero img {{ height: 56px; }}
  .tile {{ padding: 14px 6px; min-height: 100px; }} .tile .big {{ font-size: 36px; }} .tile .big.sm {{ font-size: 22px; }} .tile .label {{ font-size: 12px; }}
  .tile-row.wrap {{ flex-wrap: wrap; }} .tile-row.wrap .tile {{ flex: 1 1 calc(50% - 6px); }}
  .score {{ font-size: 22px; letter-spacing: 0; padding: 12px 6px; }} .score.sm {{ font-size: 22px; }}
  .vs {{ width: 52px; height: 52px; font-size: 16px; margin: 0 8px; }}
  .matchup {{ padding: 14px 8px; }} .side .name {{ font-size: 16px; }}
  .pl-table {{ font-size: 13px; }} .pl-table th, .pl-table td {{ padding: 6px 6px; }} .standings td:first-child {{ white-space: nowrap; }}
  .hide-m {{ display: none; }}
  h2, h3 {{ font-size: 22px; }}
}}
</style>
"""


def html(markup):
    """Render HTML through st.markdown. Whitespace is collapsed so Markdown never mistakes indented HTML for a code block."""
    st.markdown(" ".join(markup.split()), unsafe_allow_html=True)


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


@lru_cache(maxsize=None)
def _data_uri(path):
    return "data:{};base64,{}".format("image/svg+xml" if path.endswith(".svg") else "image/png", base64.b64encode(Path(path).read_bytes()).decode())


def hero(title, subtitle):
    ball = f'<img src="{_data_uri(str(HERO_SVG))}" alt="">' if HERO_SVG.exists() else ""
    html(f'<div class="hero"><div><h1>{title}</h1><p>{subtitle}</p></div>{ball}</div>')


def visible_color(hex_color, minimum=0.30):
    """Team colours such as Fulham black vanish on a dark purple page: blend toward white until the colour is bright enough."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    luminance = lambda r, g, b: (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
    while luminance(r, g, b) < minimum:
        r, g, b = (round(c + (255 - c) * 0.15) for c in (r, g, b))
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def crest(team, size=28):
    """The crest image if the PNG exists, otherwise a coloured badge with the 3-letter code (the app works without crests)."""
    path = CREST_DIR / f"{SLUG[team]}.png"
    if path.exists():
        return f'<img class="crest" src="{_data_uri(str(path))}" width="{size}" height="{size}" alt="{team}">'
    return (f'<span class="badge" style="width:{size}px;height:{size}px;font-size:{max(size // 3, 9)}px;'
            f'background:{COLORS[team]}">{CODES[team]}</span>')


def tile(value, label, color="#FFFFFF", small=False):
    return f'<div class="tile"><div class="big{" sm" if small else ""}" style="color:{color}">{value}</div><div class="label">{label}</div></div>'


def nav_links():
    """Links to both pages, right under the banner, so the user never depends on the sidebar (it starts collapsed on narrow screens)."""
    left, right = st.columns(2)
    left.page_link("views/match_predictor.py", label="Match Predictor", icon="⚽")
    right.page_link("views/season_table.py", label="Season Table", icon="🏆")


def tile_row(tiles, wrap=False):
    """Tiles side by side in one flex row (st.columns would stack them one under another on a phone). wrap=True gives 2 per row on a phone."""
    return f'<div class="tile-row{" wrap" if wrap else ""}">{"".join(tiles)}</div>'


def disclaimer():
    html('<div class="disclaimer">For educational purposes only.</div>')
