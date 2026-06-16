"""
predict.py - predict a single match.

    python predict.py "Spain" "Brazil"
    python predict.py "Brazil" "Croatia" --home-adj 0.88   # Brazil missing key attackers
    python predict.py "United States" "Mexico" --host USA  # non-neutral (host at home)

The model is fitted once and cached to data_cache/model.pkl. Use --refit to rebuild.

Injuries / lineups (manual for now): --home-adj / --away-adj scale a team's attack
rate. 1.0 = full strength; drop toward ~0.85 when key players are out. This is the
"before vs after" hook - rerun with an updated adjustment when the lineup is announced.
(Player-level auto-weighting from squad data is the next layer; see README.)
"""

import argparse
import os
import pickle
import warnings

warnings.filterwarnings("ignore")
import model

CACHE = "data_cache/model.pkl"


def get_params(refit=False, since="2014-01-01"):
    if not refit and os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    played, _ = model.load_data()
    params = model.fit(played, since=since)
    with open(CACHE, "wb") as f:
        pickle.dump(params, f)
    return params


def bar(p, width=24):
    return "#" * int(round(p * width))


def show(r):
    h, a = r["home"], r["away"]
    venue = "neutral" if r["neutral"] else f"{h} at home"
    print("=" * 60)
    print(f"  {h}  vs  {a}    ({venue})")
    print("=" * 60)
    rows = [(h + " win", r["p_home"]), ("Draw", r["p_draw"]), (a + " win", r["p_away"])]
    for label, p in rows:
        print(f"  {label:<24} {p:5.1%}  {bar(p)}")
    print("-" * 60)
    pick = max(rows, key=lambda kv: kv[1])
    print(f"  Most likely: {pick[0]} ({pick[1]:.0%})")
    print(f"  Expected goals: {h} {r['exp_home_goals']:.2f} - {r['exp_away_goals']:.2f} {a}")
    print(f"  Likely scores: " + ", ".join(f"{s} ({p:.0%})" for s, p in r["top_scores"][:3]))
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("home")
    ap.add_argument("away")
    ap.add_argument("--host", help="a team name => that team plays at home (non-neutral)")
    ap.add_argument("--home-adj", type=float, default=1.0, help="home attack multiplier (injuries)")
    ap.add_argument("--away-adj", type=float, default=1.0, help="away attack multiplier (injuries)")
    ap.add_argument("--refit", action="store_true")
    args = ap.parse_args()

    params = get_params(refit=args.refit)
    neutral = not (args.host in (args.home, args.away))
    r = model.predict(params, args.home, args.away, neutral=neutral,
                      home_adj=args.home_adj, away_adj=args.away_adj)
    show(r)
    ex = model.explain(params, args.home, args.away, neutral=neutral,
                       home_adj=args.home_adj, away_adj=args.away_adj)
    print(f"  WHY: {ex['why']}")
    for d in ex["drivers"]:
        print(f"    - {d}")
    print("=" * 60)


if __name__ == "__main__":
    main()
