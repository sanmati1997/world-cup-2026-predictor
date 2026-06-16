"""
predict_all.py - predict every upcoming World Cup 2026 fixture in the dataset
(matches with no score yet), and save them to predictions/wc2026.csv.

    python predict_all.py
"""

import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
import model
from predict import get_params


def main():
    params = get_params()
    _, fixtures = model.load_data()
    wc = fixtures[fixtures["tournament"].str.contains("World Cup", case=False, na=False)]
    known = set(params["teams"])

    rows = []
    for _, m in wc.sort_values("date").iterrows():
        if m.home_team not in known or m.away_team not in known:
            continue
        r = model.predict(params, m.home_team, m.away_team, neutral=bool(m.neutral))
        pick = max([(m.home_team, r["p_home"]), ("Draw", r["p_draw"]),
                    (m.away_team, r["p_away"])], key=lambda kv: kv[1])
        rows.append({
            "date": m.date.date(), "home": m.home_team, "away": m.away_team,
            "P(home)": round(r["p_home"], 3), "P(draw)": round(r["p_draw"], 3),
            "P(away)": round(r["p_away"], 3),
            "pick": pick[0], "conf": round(pick[1], 3),
            "score": r["top_scores"][0][0],
        })

    out = pd.DataFrame(rows)
    os.makedirs("predictions", exist_ok=True)
    out.to_csv("predictions/wc2026.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nsaved {len(out)} predictions -> predictions/wc2026.csv")


if __name__ == "__main__":
    main()
