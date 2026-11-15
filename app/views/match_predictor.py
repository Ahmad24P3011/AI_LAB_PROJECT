"""Page 1: Match Predictor. The user picks two teams; the saved models answer from each team's end-of-2025-26 state."""
import altair as alt
import pandas as pd
import streamlit as st

import styles
from predictor import CLASSIFIERS, AppDataError, Predictor
from styles import CYAN, GREEN, GREY, crest, html, tile
from team_colors import CODES, COLORS


@st.cache_resource(show_spinner="Loading the saved models...")
def load_predictor():
    return Predictor()


def pct(p):
    return f"{p * 100:.0f}%"


def matchup_card(home, away):
    def side(team, align):
        return (f'<div class="side" style="background:linear-gradient(135deg, {COLORS[team]}33, transparent)">'
                f'{crest(team, 96)}<div class="name">{team}</div><div class="note">{align}</div></div>')
    html(f'<div class="matchup">{side(home, "HOME")}<div class="vs">VS</div>{side(away, "AWAY")}</div>')


def probability_tiles(p):
    columns = st.columns(3)
    for column, (value, label, color) in zip(columns, [(p[0], "Home win", GREEN), (p[1], "Draw", GREY), (p[2], "Away win", CYAN)]):
        column.markdown(tile(pct(value), label, color), unsafe_allow_html=True)


def scoreboard(r):
    html(f'<div class="score">{CODES[r["home"]]} {r["lambda_home"]:.1f} : {r["lambda_away"]:.1f} {CODES[r["away"]]}'
         f'<small>Poisson expected goals. The single most likely exact score is {r["top_scorelines"][0][0]}-{r["top_scorelines"][0][1]} '
         f'({r["top_scorelines"][0][2] * 100:.0f}%)</small></div>')


def comparison_table(r):
    rows = [(CLASSIFIERS["logreg"], r["probabilities"]["logreg"], "n/a", True),
            (CLASSIFIERS["random_forest"], r["probabilities"]["random_forest"], "n/a", False),
            (CLASSIFIERS["xgboost"], r["probabilities"]["xgboost"], "n/a", False),
            ("Poisson (scoreline grid)", r["probabilities"]["poisson"], f'{r["lambda_home"]:.1f} - {r["lambda_away"]:.1f}', False)]
    if r["bookmaker"] is not None:
        rows.append(("Bookmaker reference (see note)", r["bookmaker"], "n/a", False))
    else:
        rows.append(("Bookmaker reference", None, "n/a", False))
    body = ""
    for name, p, score, primary in rows:
        cells = ["-", "-", "-"] if p is None else [pct(v) for v in p]
        body += (f'<tr class="{"primary" if primary else ""}"><td>{name}</td><td style="color:{GREEN}">{cells[0]}</td>'
                 f'<td style="color:{GREY}">{cells[1]}</td><td style="color:{CYAN}">{cells[2]}</td><td>{score}</td></tr>')
    html('<table class="pl-table"><tr><th>Model</th><th>P(Home)</th><th>P(Draw)</th><th>P(Away)</th><th>Expected score</th></tr>' + body + '</table>')
    if r["bookmaker"] is None:
        st.caption("No bookmaker odds exist for this pairing.")
    else:
        st.caption("Bookmaker note: these are the odds for this fixture as it was really played in 2025-26, set before that match "
                   "(margin removed). The models use each team's form at the END of the season, so the two are not like-for-like.")


def real_result_section(predictor, r):
    """What really happened in this fixture in 2025-26, next to what each model gave to that outcome."""
    real = predictor.real_result(r["home"], r["away"])
    st.markdown("### What really happened")
    if real is None:
        st.caption("This pairing was not played in 2025-26.")
        return
    names = ["Home win", "Draw", "Away win"]
    colors = [GREEN, GREY, CYAN]
    date = pd.to_datetime(real["date"]).strftime("%d %B %Y")
    html(f'<div class="score" style="font-size:40px">{CODES[r["home"]]} {real["home_score"]} : {real["away_score"]} {CODES[r["away"]]}'
         f'<small>Real full-time result, {date}: <b style="color:{colors[real["outcome"]]}">{names[real["outcome"]]}</b></small></div>')
    given = {"Logistic Regression (primary)": r["probabilities"]["logreg"], "Random Forest": r["probabilities"]["random_forest"],
             "XGBoost": r["probabilities"]["xgboost"], "Poisson": r["probabilities"]["poisson"]}
    if r["bookmaker"] is not None:
        given["Bookmaker"] = r["bookmaker"]
    st.caption("Chance each model gave to the outcome that really happened:")
    columns = st.columns(len(given))
    for column, (name, p) in zip(columns, given.items()):
        picked = names[int(p.argmax())]
        column.markdown(tile(pct(p[real["outcome"]]), f"{name}<br>top pick: {picked}",
                             colors[real["outcome"]], small=True), unsafe_allow_html=True)
    exact = r["grid"][real["home_score"], real["away_score"]] if max(real["home_score"], real["away_score"]) <= 10 else 0.0
    st.caption(f"The Poisson model gave the exact score {real['home_score']}-{real['away_score']} a {exact * 100:.1f}% chance. "
               "Read this as a sanity check, not a test: the models here use each team's form at the END of 2025-26, which already includes "
               "this very match, and the real score is never an input to any model. The fair test of the models on 2025-26 is Stage 6, "
               "which predicted each match using only earlier ones.")


def scoreline_chart(r):
    rows = [{"score": f"{h}-{a}", "prob": p * 100, "outcome": "Home win" if h > a else "Draw" if h == a else "Away win"}
            for h, a, p in r["top_scorelines"]]
    data = pd.DataFrame(rows)
    chart = (alt.Chart(data).mark_bar(cornerRadiusEnd=4)
             .encode(y=alt.Y("score:N", sort=None, title=None, axis=alt.Axis(labelColor="#FFF", labelFontSize=14, labelOverlap=False)),
                     x=alt.X("prob:Q", title="Probability (%)", axis=alt.Axis(labelColor="#FFF", titleColor=GREY, gridColor="#4a1450")),
                     color=alt.Color("outcome:N", legend=alt.Legend(title=None, orient="bottom", labelColor="#FFF"),
                                     scale=alt.Scale(domain=["Home win", "Draw", "Away win"], range=[GREEN, GREY, CYAN])),
                     tooltip=["score", alt.Tooltip("prob:Q", format=".1f")])
             .properties(height=400, background="transparent").configure_view(strokeWidth=0))
    st.altair_chart(chart, width="stretch")


styles.hero("Match Predictor", "Premier League 2025-26 - end-of-season form")
try:
    predictor = load_predictor()
except AppDataError as error:
    st.error(f"{error}. Please re-run app/team_state_builder.py.")
    st.stop()

st.caption("This page shows what the saved models would predict for a hypothetical match RIGHT NOW, using each team's Elo rating "
           "and goal form as of its last 2025-26 match. It is not a forecast of a real upcoming fixture.")
teams = predictor.teams
left, middle, right = st.columns([3, 1, 3])
home_team = left.selectbox("Home team", teams, index=teams.index("Liverpool"), key="home_team")
middle.markdown("<div style='text-align:center;font-weight:800;font-size:24px;padding-top:34px;color:#04F5FF'>VS</div>", unsafe_allow_html=True)
away_options = [t for t in teams if t != home_team]
away_team = right.selectbox("Away team", away_options, index=away_options.index("Bournemouth") if "Bournemouth" in away_options else 0, key="away_team")

result = predictor.predict(home_team, away_team)
matchup_card(home_team, away_team)
st.write("")
st.markdown("### Win / draw / loss (ELO2 Logistic Regression, the primary model)")
probability_tiles(result["probabilities"]["logreg"])
scoreboard(result)
st.markdown("### All models side by side")
comparison_table(result)
st.markdown("### Most likely scorelines (Poisson)")
scoreline_chart(result)

real_result_section(predictor, result)

with st.expander("Match context"):
    favourite = home_team if result["elo_diff"] > 0 else away_team
    gap = abs(result["elo_diff"])
    st.write(f"Elo: {home_team} {result['elo_home']:.0f}, {away_team} {result['elo_away']:.0f}. "
             f"**{favourite}** is rated higher by {gap:.0f} Elo points. The classifiers use this gap as their input "
             f"(home advantage is already learned in their intercept; it is not added to the gap).")
    if result["bookmaker"] is not None:
        model_p, book_p = result["probabilities"]["logreg"], result["bookmaker"]
        st.write(f"Model v bookmaker (home / draw / away): {pct(model_p[0])} / {pct(model_p[1])} / {pct(model_p[2])} against "
                 f"{pct(book_p[0])} / {pct(book_p[1])} / {pct(book_p[2])}.")
with st.expander("How this works"):
    st.write("These are the models the project built: three classifiers (Logistic Regression, Random Forest, XGBoost) that look only at "
             "the gap in Elo ratings, and a Poisson model that predicts how many goals each side will score. On the 2025-26 test "
             "season they tied the bookmaker within the noise (about 1.03 log loss each, accuracy about 47 to 50%), and no model "
             "predicts draws well. Treat every number here as a rough probability, not a certainty.")
styles.disclaimer()
