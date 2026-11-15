"""Premier League Prediction: the web app (Stage 9). Run from the repo root:  streamlit run app/streamlit_app.py

This file is the entry point: page settings, the shared look, the sidebar and the two pages (app/views/).
The app only SERVES things saved by earlier stages (models, team state, simulation outputs); it trains nothing.
"""
import streamlit as st

import styles

st.set_page_config(page_title="Premier League Prediction", page_icon="⚽", layout="wide", initial_sidebar_state="auto")
styles.inject_css()

with st.sidebar:
    st.markdown("### About")
    st.markdown("A university project (AI Lab, Group 8). Four model families (Logistic Regression, Random Forest, XGBoost and "
                "Poisson regression) were trained on ten Premier League seasons (2015-16 to 2024-25) and tested on 2025-26. "
                "On that test season they tied the bookmaker within the noise: no model was clearly better than another.")
    st.markdown("### Limitations")
    st.markdown("- **No model predicts draws well.** Draws are about a quarter of matches and look like coin flips to every model.\n"
                "- **The bookmaker is a strong benchmark.** It was the best predictor on the test season, though not significantly.\n"
                "- **The simulation compresses the real table.** It averages over 1,000 seasons, so champions are underpredicted "
                "and weak teams overpredicted.")
    st.markdown("[Source code on GitHub](https://github.com/Ahmad24P3011/AI_LAB_PROJECT)")

pages = [st.Page("views/match_predictor.py", title="Match Predictor", icon="⚽", default=True),
         st.Page("views/season_table.py", title="Season Table", icon="🏆")]
st.navigation(pages).run()
