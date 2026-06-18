"""
app.py - World Cup 2026 predictor, product UI.

    streamlit run app.py

Presentation layer only. All model logic stays in model.py. The app loads showing
a finished prediction for the next real upcoming fixture; the sidebar team picker
and the injury toggles refine a prediction that is already on screen. Backtest
metrics in the header are read from the real backtest output
(data_cache/backtest_metrics.json), never hardcoded.

Deploy free on Streamlit Community Cloud: push to GitHub, point share.streamlit.io
at this repo (main file: app.py).
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
import model
from predict import get_params

# ----------------------------------------------------------------------------
# palette (two team identity colors plus a neutral gray for the draw)
# ----------------------------------------------------------------------------
BLUE = "#2563eb"; BLUE_SOFT = "#dbeafe"
RED = "#dc2626"; RED_SOFT = "#fee2e2"
GRAY = "#94a3b8"; GRAY_SOFT = "#e2e8f0"
INK = "#0f172a"; MUTE = "#64748b"; LINE = "#e5e9f0"
RAMP = [[0.0, "#f8fafc"], [0.2, "#dbeafe"], [0.45, "#93c5fd"],
        [0.7, "#3b82f6"], [1.0, "#1e3a8a"]]
INJURY_MULT = 0.88  # documented what-if: a key attacker out trims that side's attack to 88%
K = 6               # scoreline heatmap is shown for 0..K goals per side

_toggle = getattr(st, "toggle", st.checkbox)

st.set_page_config(page_title="World Cup 2026 Predictor", layout="wide")

# ----------------------------------------------------------------------------
# all custom styling lives here, in one consolidated block
# ----------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none !important; }}
.block-container {{ max-width: 980px; padding-top: 1.1rem; padding-bottom: 3rem; }}
.stApp {{ background: #ffffff; }}

/* top bar */
.topbar {{ display:flex; align-items:center; justify-content:space-between;
  border-bottom:1px solid {LINE}; padding:0 0 14px 0; margin-bottom:22px; flex-wrap:wrap; gap:8px; }}
.brand {{ font-weight:700; font-size:18px; color:{INK}; letter-spacing:-0.01em; }}
.topright {{ display:flex; align-items:center; gap:16px; }}
.live {{ display:inline-flex; align-items:center; gap:6px; font-size:12px; color:{MUTE}; }}
.live .dot {{ width:8px; height:8px; border-radius:50%; background:#22c55e; box-shadow:0 0 0 0 rgba(34,197,94,.6);
  animation:pulse 1.8s infinite; }}
@keyframes pulse {{ 0%{{box-shadow:0 0 0 0 rgba(34,197,94,.5)}} 70%{{box-shadow:0 0 0 7px rgba(34,197,94,0)}} 100%{{box-shadow:0 0 0 0 rgba(34,197,94,0)}} }}
.trust {{ font-size:12px; color:{MUTE}; }}
.trust b {{ color:{INK}; font-weight:700; }}

/* section label */
.seclabel {{ font-size:11px; font-weight:700; letter-spacing:0.09em; text-transform:uppercase;
  color:{MUTE}; margin:26px 0 10px 0; }}

/* matchup */
.matchup {{ display:flex; align-items:center; justify-content:space-between; gap:14px; }}
.team {{ font-weight:700; font-size:30px; line-height:1.1; flex:1; letter-spacing:-0.02em; }}
.teamA {{ color:{BLUE}; text-align:left; }}
.teamB {{ color:{RED}; text-align:right; }}
.vs {{ text-align:center; min-width:150px; }}
.vs .m1 {{ font-size:12px; font-weight:700; color:{INK}; }}
.vs .m2 {{ font-size:12px; color:{MUTE}; margin-top:2px; }}
.vs .m3 {{ font-size:12px; color:{MUTE}; }}

/* outcome bar legend */
.olegend {{ display:flex; justify-content:space-between; font-size:13px; color:{INK}; margin-bottom:6px; }}
.olegend .sw {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:middle; }}

/* hero score */
.hero {{ text-align:center; padding:6px 0 2px 0; }}
.hero .num {{ font-weight:700; font-size:64px; line-height:1; color:{INK}; letter-spacing:-0.03em; }}
.hero .sub {{ font-size:13px; color:{MUTE}; margin-top:6px; }}
.sidescore {{ text-align:center; padding:10px 0; }}
.sidescore .s {{ font-weight:700; font-size:22px; }}
.sidescore .l {{ font-size:12px; color:{MUTE}; margin-top:2px; }}

/* xg */
.xgrow {{ display:flex; gap:14px; }}
.xg {{ flex:1; border:1px solid {LINE}; border-radius:12px; padding:16px; text-align:center; }}
.xg .v {{ font-weight:700; font-size:30px; letter-spacing:-0.02em; }}
.xg .l {{ font-size:12px; color:{MUTE}; margin-top:3px; }}
.xgA .v {{ color:{BLUE}; }}
.xgB .v {{ color:{RED}; }}

/* why */
.whyline {{ font-size:14px; color:{INK}; line-height:1.5; background:{GRAY_SOFT}33;
  border-left:3px solid {GRAY}; padding:11px 14px; border-radius:0 8px 8px 0; }}
.divhead {{ display:flex; justify-content:space-between; font-size:12px; font-weight:700; margin-bottom:2px; }}
.divhead .a {{ color:{BLUE}; }} .divhead .b {{ color:{RED}; }}

/* injury */
.delta {{ font-size:13px; color:{INK}; }}
.delta .arrow {{ color:{MUTE}; }}

@media (max-width: 640px) {{
  .xgrow {{ flex-direction:column; }}
  .team {{ font-size:22px; }}
  .hero .num {{ font-size:48px; }}
  .vs {{ min-width:110px; }}
}}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# cached data and model
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Fitting the model (one time)...")
def load_params():
    return get_params()


@st.cache_data(show_spinner=False)
def load_fixtures():
    _, fixtures = model.load_data()
    wc = fixtures[fixtures["tournament"].str.contains("World Cup", case=False, na=False)].copy()
    return wc


@st.cache_data(show_spinner=False)
def load_metrics():
    path = "data_cache/backtest_metrics.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    try:
        import evaluate
        return evaluate.run_backtest()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def fixtures_table(_params):
    known = set(_params["teams"])
    rows = []
    for _, m in load_fixtures().sort_values("date").iterrows():
        if m.home_team not in known or m.away_team not in known:
            continue
        r = model.predict(_params, m.home_team, m.away_team, neutral=bool(m.neutral))
        pick = max([(m.home_team, r["p_home"]), ("Draw", r["p_draw"]),
                    (m.away_team, r["p_away"])], key=lambda kv: kv[1])
        rows.append({"Date": str(m.date.date()), "Match": f"{m.home_team} v {m.away_team}",
                     "Team A win": f"{r['p_home']:.0%}", "Draw": f"{r['p_draw']:.0%}",
                     "Team B win": f"{r['p_away']:.0%}", "Pick": pick[0],
                     "Likely score": r["top_scores"][0][0]})
    return pd.DataFrame(rows)


params = load_params()
teams = params["teams"]
known = set(teams)
sched = load_fixtures()
sched = sched[sched["home_team"].isin(known) & sched["away_team"].isin(known)].sort_values("date")
metrics = load_metrics()


def next_fixture():
    today = pd.Timestamp.now().normalize()
    up = sched[sched["date"] >= today]
    if len(up):
        return up.iloc[0]
    if len(sched):
        return sched.iloc[0]
    return None


def fixture_meta(a, b):
    """Real schedule metadata for A vs B (either orientation), or None if this is
    not a scheduled fixture. Never invents a date or venue."""
    m = sched[((sched.home_team == a) & (sched.away_team == b)) |
              ((sched.home_team == b) & (sched.away_team == a))]
    if not len(m):
        return None
    r = m.iloc[0]
    city = r["city"] if pd.notna(r["city"]) else None
    country = r["country"] if pd.notna(r["country"]) else None
    return {"date": r["date"], "city": city, "country": country,
            "tournament": r["tournament"], "neutral": bool(r["neutral"])}


def predict_AB(A, B, adjA, adjB):
    """Predict A vs B at a neutral venue (every World Cup 2026 fixture is neutral).
    A is blue/left, B is red/right. Returns probabilities, expected goals, the joint
    scoreline matrix, and signed factor contributions, all in A/B terms."""
    sm = model.score_matrix(params, A, B, neutral=True, home_adj=adjA, away_adj=adjB)
    M = sm["matrix"]
    pA = float(np.tril(M, -1).sum())
    pB = float(np.triu(M, 1).sum())
    pD = float(np.trace(M))
    contribs = model.contributions(params, A, B, neutral=True, home_adj=adjA, away_adj=adjB)
    ex = model.explain(params, A, B, neutral=True, home_adj=adjA, away_adj=adjB)
    return {"M": M, "pA": pA, "pB": pB, "pD": pD,
            "xgA": sm["exp_home_goals"], "xgB": sm["exp_away_goals"],
            "contribs": contribs, "why": ex["why"], "drivers": ex["drivers"]}


def best_score(M, side):
    """Most likely exact scoreline within a region. side 'A' = A wins (x>y),
    'B' = B wins (x<y), 'any' = overall."""
    best = None
    for x in range(M.shape[0]):
        for y in range(M.shape[1]):
            if side == "A" and x <= y:
                continue
            if side == "B" and x >= y:
                continue
            if best is None or M[x, y] > best[1]:
                best = ((x, y), float(M[x, y]))
    return best


# ----------------------------------------------------------------------------
# plotly pieces, themed to match
# ----------------------------------------------------------------------------
def outcome_bar(pA, pD, pB, A, B):
    fig = go.Figure()
    for label, p, c in [(A, pA, BLUE), ("Draw", pD, GRAY), (B, pB, RED)]:
        fig.add_bar(x=[p], y=[""], orientation="h", marker=dict(color=c, line=dict(width=0)),
                    text=[f"{p:.0%}"], textposition="inside", insidetextanchor="middle",
                    textfont=dict(color="white", size=15, family="Inter"),
                    hovertemplate=f"{label}: {p:.1%}<extra></extra>", name=label)
    fig.update_layout(barmode="stack", height=60, showlegend=False, bargap=0,
                      margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def heatmap(M_AB, A, B):
    Z = M_AB[:K + 1, :K + 1]
    axis = list(range(K + 1))
    fig = go.Figure(go.Heatmap(
        z=Z, x=axis, y=axis, colorscale=RAMP, zmin=0, xgap=3, ygap=3,
        colorbar=dict(title=dict(text="probability", side="right", font=dict(size=11)),
                      thickness=10, len=0.85, outlinewidth=0, tickformat=".0%",
                      tickfont=dict(size=10)),
        hovertemplate=f"{A} %{{y}} - %{{x}} {B}<br>%{{z:.1%}}<extra></extra>"))

    mx = float(Z.max())
    for i in axis:  # outline the draw diagonal
        fig.add_shape(type="rect", x0=i - 0.5, x1=i + 0.5, y0=i - 0.5, y1=i + 0.5,
                      line=dict(color=INK, width=1.6), fillcolor="rgba(0,0,0,0)", layer="above")
    for yi in axis:  # label the meaningful cells
        for xi in axis:
            if Z[yi, xi] >= max(0.03, mx * 0.4):
                fig.add_annotation(x=xi, y=yi, text=f"{Z[yi, xi]:.0%}", showarrow=False,
                                   font=dict(size=11, family="Inter",
                                             color="white" if Z[yi, xi] > mx * 0.6 else INK))
    fig.update_layout(height=430, margin=dict(l=6, r=6, t=6, b=6),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter", color=INK),
                      xaxis=dict(title=dict(text=f"{B} goals", font=dict(size=12)), dtick=1,
                                 tickfont=dict(size=11), showgrid=False, zeroline=False),
                      yaxis=dict(title=dict(text=f"{A} goals", font=dict(size=12)), dtick=1,
                                 tickfont=dict(size=11), showgrid=False, zeroline=False,
                                 autorange="reversed"))
    return fig


def diverging(contribs, A, B):
    cs = sorted(contribs, key=lambda c: abs(c["value"]))  # largest ends up on top
    labels = [c["factor"] for c in cs]
    xs = [-c["value"] for c in cs]               # A (positive) -> left, B (negative) -> right
    colors = [BLUE if c["value"] > 0 else RED for c in cs]
    txt = [("favors " + (A if c["value"] > 0 else B)) for c in cs]
    fig = go.Figure(go.Bar(x=xs, y=labels, orientation="h", text=txt, hoverinfo="text",
                           marker=dict(color=colors, line=dict(width=0))))
    m = max((abs(x) for x in xs), default=1.0) * 1.18
    fig.update_layout(height=58 + 44 * len(labels), showlegend=False,
                      margin=dict(l=6, r=6, t=4, b=6),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      bargap=0.45,
                      xaxis=dict(visible=False, range=[-m, m], zeroline=True,
                                 zerolinecolor=LINE, zerolinewidth=2),
                      yaxis=dict(tickfont=dict(family="Inter", size=13, color=INK)))
    return fig


def sec(label):
    st.markdown(f'<div class="seclabel">{label}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# header bar (section 1)
# ----------------------------------------------------------------------------
if metrics:
    trust = (f"<b>{metrics['model_logloss']:.3f}</b> log-loss vs "
             f"{metrics['baseline_logloss']:.3f} baseline &nbsp;&middot;&nbsp; "
             f"{metrics['n']:,} matches backtested")
else:
    trust = "backtest metrics unavailable"
st.markdown(f"""
<div class="topbar">
  <div class="brand">World Cup 2026 Predictor</div>
  <div class="topright">
    <span class="live"><span class="dot"></span>live</span>
    <span class="trust">{trust}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# controls (sidebar): the team picker refines what is already on screen
# ----------------------------------------------------------------------------
nf = next_fixture()
default_A = nf["home_team"] if nf is not None else ("Spain" if "Spain" in known else teams[0])
default_B = nf["away_team"] if nf is not None else ("Brazil" if "Brazil" in known else teams[1])
st.session_state.setdefault("A", default_A)
st.session_state.setdefault("B", default_B)


def _reset_to_fixture(a, b):
    # runs as an on_click callback, before the widgets are re-instantiated, which
    # is the only point where a widget-keyed session value may be set.
    st.session_state["A"] = a
    st.session_state["B"] = b


tab_pred, tab_fix, tab_method = st.tabs(["Prediction", "All fixtures", "How it works"])

with tab_pred:
    cA, cB = st.columns(2)
    A = cA.selectbox("Team A", teams, key="A")
    B = cB.selectbox("Team B", teams, key="B")
    st.button("Reset to next fixture", on_click=_reset_to_fixture, args=(default_A, default_B))
    st.caption("Pick any two teams. Every World Cup 2026 fixture is played at a neutral site.")
    if A == B:
        st.warning("Pick two different teams.")
    else:
        # injury state is read at the top so the whole view reflects it on rerun;
        # the toggles themselves are rendered in section 8 below.
        adjA = INJURY_MULT if st.session_state.get("outA", False) else 1.0
        adjB = INJURY_MULT if st.session_state.get("outB", False) else 1.0
        cur = predict_AB(A, B, adjA, adjB)
        base = predict_AB(A, B, 1.0, 1.0)
        M = cur["M"]

        # 2. matchup header
        meta = fixture_meta(A, B)
        if meta:
            m1 = meta["tournament"]
            m2 = meta["date"].strftime("%d %b %Y")
            loc = ", ".join([x for x in (meta["city"], meta["country"]) if x])
            m3 = (loc + " (neutral)") if loc else "Neutral venue"
        else:
            m1 = "Custom matchup"
            m2 = "not a scheduled fixture"
            m3 = "Neutral venue"
        st.markdown(f"""
        <div class="matchup">
          <div class="team teamA">{A}</div>
          <div class="vs"><div class="m1">{m1}</div><div class="m2">{m2}</div><div class="m3">{m3}</div></div>
          <div class="team teamB">{B}</div>
        </div>
        """, unsafe_allow_html=True)

        # 3. outcome bar
        sec("Win, draw, loss")
        st.markdown(f"""
        <div class="olegend">
          <span><span class="sw" style="background:{BLUE}"></span>{A} win</span>
          <span><span class="sw" style="background:{GRAY}"></span>Draw</span>
          <span>{B} win<span class="sw" style="background:{RED};margin:0 0 0 6px"></span></span>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(outcome_bar(cur["pA"], cur["pD"], cur["pB"], A, B),
                        width='stretch', config={"displayModeBar": False})

        # 4. hero scoreline plus each side's most likely winning score
        top = best_score(M, "any")
        bestA = best_score(M, "A")
        bestB = best_score(M, "B")
        sec("Most likely scoreline")
        st.markdown(f"""
        <div class="hero"><div class="num">{top[0][0]} - {top[0][1]}</div>
        <div class="sub">most likely exact score &middot; {top[1]:.0%} probability</div></div>
        """, unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        s1.markdown(f'<div class="sidescore"><div class="s" style="color:{BLUE}">'
                    f'{bestA[0][0]} - {bestA[0][1]}</div><div class="l">{A} most likely win '
                    f'&middot; {bestA[1]:.0%}</div></div>', unsafe_allow_html=True)
        s2.markdown(f'<div class="sidescore"><div class="s" style="color:{RED}">'
                    f'{bestB[0][0]} - {bestB[0][1]}</div><div class="l">{B} most likely win '
                    f'&middot; {bestB[1]:.0%}</div></div>', unsafe_allow_html=True)

        # 5. expected goals
        sec("Expected goals")
        st.markdown(f"""
        <div class="xgrow">
          <div class="xg xgA"><div class="v">{cur['xgA']:.2f}</div><div class="l">{A} expected goals</div></div>
          <div class="xg xgB"><div class="v">{cur['xgB']:.2f}</div><div class="l">{B} expected goals</div></div>
        </div>""", unsafe_allow_html=True)

        # 6. scoreline heatmap (the centerpiece)
        sec("Scoreline probability map")
        st.plotly_chart(heatmap(M, A, B), width='stretch',
                        config={"displayModeBar": False})
        st.caption("Each cell is the joint probability of that exact scoreline from the "
                   "bivariate Poisson model. The outlined diagonal is the set of draws.")

        # 7. why this prediction
        sec("Why this prediction")
        st.markdown(f'<div class="whyline">{cur["why"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="divhead"><span class="a">&larr; favors {A}</span>'
                    f'<span class="b">favors {B} &rarr;</span></div>', unsafe_allow_html=True)
        st.plotly_chart(diverging(cur["contribs"], A, B), width='stretch',
                        config={"displayModeBar": False})
        with st.expander("Rating details"):
            for d in cur["drivers"]:
                st.write("- " + d)

        # 8. injury before and after
        sec("Injury what-if")
        st.caption("Simulate a key attacker missing. This trims that team's attack rate to "
                   f"{INJURY_MULT:.0%} (a manual what-if; the model has no player-level data). "
                   "Flipping a toggle recomputes everything above.")
        i1, i2 = st.columns(2)
        i1.markdown(f"<span style='color:{BLUE};font-weight:700'>{A}</span>", unsafe_allow_html=True)
        _toggle(f"{A}: key attacker out", key="outA")
        i2.markdown(f"<span style='color:{RED};font-weight:700'>{B}</span>", unsafe_allow_html=True)
        _toggle(f"{B}: key attacker out", key="outB")

        if adjA != 1.0 or adjB != 1.0:
            st.markdown(
                f"""<div class="delta">
                {A} win &nbsp;{base['pA']:.0%} <span class="arrow">&rarr;</span> <b>{cur['pA']:.0%}</b>
                &nbsp;&nbsp;|&nbsp;&nbsp; Draw &nbsp;{base['pD']:.0%} <span class="arrow">&rarr;</span> <b>{cur['pD']:.0%}</b>
                &nbsp;&nbsp;|&nbsp;&nbsp; {B} win &nbsp;{base['pB']:.0%} <span class="arrow">&rarr;</span> <b>{cur['pB']:.0%}</b>
                <br>Expected goals &nbsp;{A} {base['xgA']:.2f} <span class="arrow">&rarr;</span> <b>{cur['xgA']:.2f}</b>,
                &nbsp;{B} {base['xgB']:.2f} <span class="arrow">&rarr;</span> <b>{cur['xgB']:.2f}</b>
                </div>""", unsafe_allow_html=True)
        else:
            st.caption("Both teams at full strength. Toggle a key attacker out to see the swing.")

with tab_fix:
    sec("Every upcoming World Cup 2026 fixture, predicted")
    st.dataframe(fixtures_table(params), width='stretch', hide_index=True)
    st.caption("Probabilities and the single most likely scoreline for each scheduled fixture, "
               "at the neutral tournament venue.")

with tab_method:
    sec("The model")
    st.markdown(
        "Dixon-Coles bivariate Poisson. Each team has an attack and a defense rating learned "
        "from international results since 2018, with recent matches weighted more. For a match "
        "the model builds the full scoreline matrix, so win, draw, loss, expected goals, and "
        "likely scores all come from one model, and draws are handled structurally rather than "
        "bolted on.")
    if metrics:
        sec("Backtest (leakage-free)")
        st.markdown(f"Trained only on matches before {metrics['cutoff']}, tested on "
                    f"{metrics['n']:,} played matches after it.")
        dfm = pd.DataFrame({
            "metric": ["log-loss", "Brier", "outcome accuracy"],
            "model": [f"{metrics['model_logloss']:.3f}", f"{metrics['model_brier']:.3f}",
                      f"{metrics['accuracy']:.1%}"],
            "no-skill baseline": [f"{metrics['baseline_logloss']:.3f}",
                                  f"{metrics['baseline_brier']:.3f}", "n/a"]})
        st.dataframe(dfm, width='stretch', hide_index=True)
        sec("Calibration (home-win probability)")
        dfc = pd.DataFrame([{"predicted": f"{c['pred']:.0%}", "observed": f"{c['obs']:.0%}",
                             "matches": c["n"]} for c in metrics["calibration"]])
        st.dataframe(dfc, width='stretch', hide_index=True)
        st.caption("When the model says a number, it happens about that often. Outcome accuracy "
                   "is capped by football's randomness, so the edge shows up in log-loss, Brier, "
                   "and calibration, not in winner-picking. Not betting advice.")
    else:
        st.info("Backtest metrics are not available. Run python evaluate.py to generate them.")

st.divider()
st.caption("Built by Sanmati Sawalwade. Inspired by @mar-antaya. Not betting advice.")
