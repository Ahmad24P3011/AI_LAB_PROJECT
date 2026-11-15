"""Page 2: Season Table. Shows the saved Stage 7 simulation (1,000 seasons) next to the real 2025-26 table. Nothing is re-simulated."""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import styles
from styles import CYAN, GREEN, GREY, MAGENTA, PURPLE, crest, html, tile, tile_row, visible_color
from team_colors import COLORS

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS, APP_DATA = ROOT / "reports", ROOT / "data" / "app"
AXIS = dict(labelColor="#FFFFFF", titleColor=GREY, gridColor="#4a1450", domainColor="#4a1450", tickColor="#4a1450")


@st.cache_data
def load_data():
    table = pd.read_csv(REPORTS / "sim_vs_real_table.csv")
    by_game = pd.read_csv(REPORTS / "sim_points_by_game.csv")
    metrics = json.loads((REPORTS / "sim_headline_metrics.json").read_text())
    positions = pd.read_csv(APP_DATA / "sim_position_probs.csv", index_col=0)
    positions.columns = positions.columns.astype(int)
    return table, by_game, metrics, positions


def shade(value, top, color):
    """Background tint: the larger the share, the deeper the colour (alpha 0 to 0.8)."""
    alpha = 0 if top == 0 else 0.8 * value / top
    return f"background:rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},{alpha:.2f})"


def standings_table(table, column, ascending):
    shown = table.sort_values(column, ascending=ascending).reset_index(drop=True)
    top_title, top_rel = table["sim_P_title"].max(), table["sim_P_relegation"].max()
    body = ""
    for _, r in shown.iterrows():
        body += (f'<tr><td>{crest(r["team"], 24)} &nbsp;{r["team"]}</td><td>{r["real_pos"]}</td><td>{r["real_pts"]}</td>'
                 f'<td class="hide-m">{r["sim_mean_pos"]:.1f}</td><td class="hide-m">{r["sim_mean_pts"]:.1f}</td>'
                 f'<td style="{shade(r["sim_P_title"], top_title, GREEN)}">{r["sim_P_title"] * 100:.1f}%</td>'
                 f'<td>{r["sim_P_top4"] * 100:.1f}%</td>'
                 f'<td style="{shade(r["sim_P_relegation"], top_rel, MAGENTA)}">{r["sim_P_relegation"] * 100:.1f}%</td>'
                 f'<td class="hide-m">{r["position_error"]:+.1f}</td></tr>')
    html('<div class="table-wrap"><table class="pl-table standings"><tr><th>Team</th><th>Real pos</th><th>Real pts</th><th class="hide-m">Sim mean pos</th><th class="hide-m">Sim mean pts</th>'
         '<th>P(title)</th><th>P(top 4)</th><th>P(relegation)</th><th class="hide-m">Position error (sim - real)</th></tr>' + body + '</table></div>')


def fan_chart(by_game, table, team):
    data = by_game[by_game["team"] == team]
    color = visible_color(COLORS[team])
    x = alt.X("game_no:Q", title="Matches played", scale=alt.Scale(domain=[1, 38], nice=False), axis=alt.Axis(**AXIS))
    y = alt.Y(title="Points", axis=alt.Axis(**AXIS))
    base = alt.Chart(data)
    outer = base.mark_area(opacity=0.18, color=color).encode(x=x, y=alt.Y("p5:Q", title="Points", axis=alt.Axis(**AXIS)), y2="p95:Q")
    inner = base.mark_area(opacity=0.35, color=color).encode(x=x, y="p25:Q", y2="p75:Q")
    median = base.mark_line(color=color, strokeWidth=3).encode(x=x, y="p50:Q")
    real = pd.DataFrame({"game_no": [38], "points": [int(table.loc[table["team"] == team, "real_pts"].iloc[0])]})
    dot = alt.Chart(real).mark_point(color=GREEN, size=140, filled=True, opacity=1).encode(x="game_no:Q", y="points:Q",
                                                                                        tooltip=[alt.Tooltip("points:Q", title="Real final points")])
    chart = (outer + inner + median + dot).properties(height=340, background="transparent").configure_view(strokeWidth=0)
    st.altair_chart(chart, width="stretch")


def heatmap(table, positions):
    order = list(table.sort_values("real_pos")["team"])
    long = positions.loc[order].reset_index().melt(id_vars="team", var_name="position", value_name="probability")
    real = table[["team", "real_pos"]].rename(columns={"real_pos": "position"})
    team_axis = alt.Y("team:N", sort=order, title=None, axis=alt.Axis(labelColor="#FFFFFF", domain=False, ticks=False))
    cells = (alt.Chart(long).mark_rect(stroke=PURPLE, strokeWidth=1)
             .encode(x=alt.X("position:O", title="Final position", axis=alt.Axis(labelColor="#FFFFFF", titleColor=GREY, labelAngle=0)),
                     y=team_axis,
                     color=alt.Color("probability:Q", scale=alt.Scale(range=["#240029", CYAN], domain=[0, 0.35], clamp=True),
                                     legend=alt.Legend(title="P(finish here)", format=".0%", labelColor="#FFFFFF", titleColor=GREY)),
                     tooltip=["team", "position", alt.Tooltip("probability:Q", format=".1%")]))
    marks = (alt.Chart(real).mark_point(shape="circle", color=GREEN, filled=True, size=90, stroke="#000", strokeWidth=1)
             .encode(x="position:O", y=alt.Y("team:N", sort=order), tooltip=["team", alt.Tooltip("position:O", title="Real position")]))
    st.altair_chart((cells + marks).properties(height=560, background="transparent").configure_view(strokeWidth=0), width="stretch")


styles.hero("Season Table", "Simulated 2025-26, 1,000 Monte Carlo runs")
styles.nav_links()
table, by_game, metrics, positions = load_data()

champion, favourite = metrics["real_champion"], metrics["sim_favourite"]
html(tile_row([
    tile(f'{crest(champion, 44)}<br><span style="font-size:22px">{champion}</span>', f'Real champion ({metrics["real_champion_pts"]} pts)', "#FFFFFF", small=True),
    tile(f'{crest(favourite, 44)}<br><span style="font-size:22px">{favourite}</span>', f'Simulated favourite: P(title) {metrics["sim_P_title_of_favourite"] * 100:.1f}%', GREEN, small=True),
    tile(f'{metrics["mean_abs_position_error"]:.2f}', "Mean absolute position error", CYAN),
    tile(f'{metrics["spearman_rank_correlation"]:.2f}', "Spearman correlation (1 = same order)", CYAN)], wrap=True))

st.markdown("### Simulated table next to the real one")
SORTS = {"Real position": ("real_pos", True), "Simulated mean position": ("sim_mean_pos", True), "P(title)": ("sim_P_title", False)}
sort_choice = st.radio("Sort by", list(SORTS), horizontal=True, label_visibility="collapsed")
standings_table(table, *SORTS[sort_choice])
st.caption("Green shading: chance of winning the title. Magenta shading: chance of relegation. Position error = simulated mean position minus real position. (On a phone, the simulated-position and error columns are hidden; use a wider screen to see them.)")

st.markdown("### How a team's points could have gone")
team = st.selectbox("Team", sorted(table["team"]), index=sorted(table["team"]).index("Arsenal"))
left, right = st.columns([1, 4])
left.markdown(f'<div style="padding-top:8px">{crest(team, 72)}</div>', unsafe_allow_html=True)
real_row = table[table["team"] == team].iloc[0]
left.caption(f'Real: position {int(real_row["real_pos"])}, {int(real_row["real_pts"])} pts. Simulated mean: {real_row["sim_mean_pts"]:.0f} pts.')
with right:
    fan_chart(by_game, table, team)
st.caption("Line = median; dark band = 25th to 75th percentile; light band = 5th to 95th percentile of cumulative points over 1,000 simulated "
           "seasons. These are percentile bands across simulations, not a prediction interval for the one real season. "
           "The green dot is the real final total.")

st.markdown("### Where each team finished, over 1,000 simulated seasons")
heatmap(table, positions)
st.caption("Brighter cyan = more of the 1,000 simulated seasons ended with that team in that position. The green circle marks the real 2025-26 position.")

with st.expander("How to read these numbers"):
    st.write("P(title) is the share of the 1,000 simulated seasons in which that team finished first; P(top 4) and P(relegation) work the same way. "
             "Each simulated season plays all 380 fixtures one matchweek at a time, drawing goals from the Poisson model and updating each "
             "team's rating with the simulated results. Of the 20 real points totals, 18 fell inside the simulated 5th to 95th percentile band. "
             "The simulation still ranked the real champion third (Arsenal's title chance was 19.5%), because it spreads probability across all "
             "plausible seasons instead of picking one.")
styles.disclaimer()
