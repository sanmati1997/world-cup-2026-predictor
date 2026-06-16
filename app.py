"""
app.py - Streamlit front end for the World Cup 2026 predictor.

    streamlit run app.py

Deploy free on Streamlit Community Cloud: push this repo to GitHub, then point
share.streamlit.io at it (main file: app.py). Anyone gets a public URL.
"""

import warnings

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
import model
from predict import get_params

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="centered")


@st.cache_resource(show_spinner="Fitting the model (one time)...")
def load_model():
    return get_params()


@st.cache_data(show_spinner=False)
def all_fixtures(_params):
    _, fixtures = model.load_data()
    wc = fixtures[fixtures["tournament"].str.contains("World Cup", case=False, na=False)]
    known = set(_params["teams"])
    rows = []
    for _, m in wc.sort_values("date").iterrows():
        if m.home_team not in known or m.away_team not in known:
            continue
        r = model.predict(_params, m.home_team, m.away_team, neutral=bool(m.neutral))
        pick = max([(m.home_team, r["p_home"]), ("Draw", r["p_draw"]),
                    (m.away_team, r["p_away"])], key=lambda kv: kv[1])
        rows.append({"date": str(m.date.date()), "match": f"{m.home_team} vs {m.away_team}",
                     "P(A)": f"{r['p_home']:.0%}", "Draw": f"{r['p_draw']:.0%}",
                     "P(B)": f"{r['p_away']:.0%}", "pick": pick[0], "score": r["top_scores"][0][0]})
    return pd.DataFrame(rows)


params = load_model()
teams = params["teams"]

st.title("⚽ World Cup 2026 Predictor")
st.caption("Dixon-Coles Poisson model | honest, calibrated probabilities | "
           "inspired by [@mar-antaya](https://github.com/mar-antaya)")

tab1, tab2, tab3 = st.tabs(["Match predictor", "All WC fixtures", "How it works"])

with tab1:
    c1, c2 = st.columns(2)
    t1 = c1.selectbox("Team A", teams, index=teams.index("Spain") if "Spain" in teams else 0)
    t2 = c2.selectbox("Team B", teams, index=teams.index("Brazil") if "Brazil" in teams else 1)

    venue = st.radio(
        "Venue",
        [f"Neutral (most World Cup games)", f"{t1} at home", f"{t2} at home"],
        horizontal=True,
        help="World Cup matches are neutral except for the host nations (USA, Canada, Mexico) "
             "playing at home. 'At home' adds the home-advantage boost to that team.",
    )

    with st.expander("Injuries / missing players (the before-vs-after knob)"):
        st.caption("Drop a team's strength when key players are out, then watch the odds shift.")
        adj1 = st.slider(f"{t1} available strength", 0.70, 1.00, 1.00, 0.01)
        adj2 = st.slider(f"{t2} available strength", 0.70, 1.00, 1.00, 0.01)

    if t1 == t2:
        st.warning("Pick two different teams.")
    else:
        # order teams so the home side (if any) is passed as 'home' to the model
        if venue == f"{t2} at home":
            H, A, neutral, hA, aA = t2, t1, False, adj2, adj1
        else:
            H, A, neutral, hA, aA = t1, t2, venue.startswith("Neutral"), adj1, adj2

        r = model.predict(params, H, A, neutral=neutral, home_adj=hA, away_adj=aA)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{H} win", f"{r['p_home']:.0%}")
        m2.metric("Draw", f"{r['p_draw']:.0%}")
        m3.metric(f"{A} win", f"{r['p_away']:.0%}")
        st.bar_chart(pd.DataFrame(
            {"probability": {f"{H} win": r["p_home"], "Draw": r["p_draw"], f"{A} win": r["p_away"]}}))
        st.write(f"**Expected goals:** {H} {r['exp_home_goals']:.2f} - {r['exp_away_goals']:.2f} {A}")
        st.write("**Likely scores:** " + ",  ".join(f"`{s}` {p:.0%}" for s, p in r["top_scores"][:4]))

        ex = model.explain(params, H, A, neutral=neutral, home_adj=hA, away_adj=aA)
        st.info("**Why:** " + ex["why"])
        with st.expander("The drivers behind this (no black box)"):
            for d in ex["drivers"]:
                st.write("- " + d)
        if not neutral:
            st.caption(f"Home advantage applied to {H}. Most World Cup matches are neutral; "
                       f"this matters mainly for host nations.")

with tab2:
    st.caption("Every upcoming World Cup 2026 fixture in the dataset, predicted.")
    st.dataframe(all_fixtures(params), use_container_width=True, hide_index=True)

with tab3:
    st.markdown(
        "**Model:** Dixon-Coles bivariate Poisson. Each team has an attack and a defense "
        "rating learned from international results since 2014 (recent matches weigh more). "
        "For a match it builds a full scoreline matrix, so win/draw/loss, expected goals, "
        "and likely scores all come from one model, and **draws are handled structurally**.\n\n"
        "**Honesty:** on a leakage-free backtest (3,266 matches) it scores **log-loss 0.865 vs "
        "1.056** for a no-skill baseline and is **well-calibrated** (when it says 70%, it happens "
        "~70%). Outcome accuracy is ~59% - capped by football's randomness, like any model or "
        "bookmaker. The value is trustworthy probabilities over many games, **not a crystal ball "
        "for any single match. Not betting advice.**")

st.divider()
st.caption("Built by Sanmati Sawalwade · sanmati1997.github.io · inspired by @mar-antaya")
