"""
evaluate.py - honest, leakage-free backtest.

Fits the model ONLY on matches before a cutoff date, then predicts every played
match after it. Reports the metrics that actually matter for a probabilistic
model - log-loss and Brier vs a no-skill baseline - plus a calibration table
(when the model says 60%, does it happen ~60% of the time?).

Outcome accuracy is reported too, but it is the least informative number: it is
capped by football's inherent randomness, so a good and a mediocre model look
similar on it. The probabilistic metrics are where model quality shows up.

    python evaluate.py --cutoff 2023-06-01
"""

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import model

EPS = 1e-15


def outcome(hs, as_):
    return "H" if hs > as_ else ("A" if as_ > hs else "D")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2023-06-01", help="train < cutoff, test >= cutoff")
    args = ap.parse_args()

    played, _ = model.load_data()
    cut = pd.Timestamp(args.cutoff)
    train = played[played["date"] < cut]
    test = played[played["date"] >= cut]

    params = model.fit(train, since="2018-01-01")
    known = set(params["teams"])

    # no-skill baseline = base rates of H/D/A in the training set
    base = train.apply(lambda r: outcome(r.home_score, r.away_score), axis=1).value_counts(normalize=True)
    base = {k: float(base.get(k, 0)) for k in "HDA"}

    rows = []
    for _, m in test.iterrows():
        if m.home_team not in known or m.away_team not in known:
            continue
        r = model.predict(params, m.home_team, m.away_team, neutral=bool(m.neutral))
        p = {"H": r["p_home"], "D": r["p_draw"], "A": r["p_away"]}
        rows.append({"actual": outcome(m.home_score, m.away_score),
                     "pH": p["H"], "pD": p["D"], "pA": p["A"]})
    ev = pd.DataFrame(rows)
    n = len(ev)

    def logloss(pcols):
        pa = np.array([ev.iloc[i][pcols[ev.iloc[i]["actual"]]] for i in range(n)])
        return float(-np.mean(np.log(np.clip(pa, EPS, 1))))

    def brier(getp):
        s = 0.0
        for _, row in ev.iterrows():
            for c in "HDA":
                y = 1.0 if row["actual"] == c else 0.0
                s += (getp(row, c) - y) ** 2
        return float(s / n)

    model_ll = logloss({"H": "pH", "D": "pD", "A": "pA"})
    model_br = brier(lambda row, c: row["p" + c])
    base_ll = float(-np.mean([np.log(max(base[a], EPS)) for a in ev["actual"]]))
    base_br = float(np.mean([sum((base[c] - (1.0 if a == c else 0.0)) ** 2 for c in "HDA")
                             for a in ev["actual"]]))
    acc = float(np.mean([max("HDA", key=lambda c: ev.iloc[i]["p" + c]) == ev.iloc[i]["actual"]
                         for i in range(n)]))

    print("=" * 56)
    print(f"  Backtest: train < {args.cutoff} | test n = {n}")
    print("=" * 56)
    print(f"  {'metric':<18}{'model':>10}{'baseline':>12}")
    print(f"  {'log-loss':<18}{model_ll:>10.3f}{base_ll:>12.3f}   (lower better)")
    print(f"  {'Brier':<18}{model_br:>10.3f}{base_br:>12.3f}   (lower better)")
    print(f"  {'outcome accuracy':<18}{acc:>9.1%}{'':>12}   (ceiling-bound)")
    print("=" * 56)

    # calibration on home-win probability
    print("  Calibration (home-win prob):")
    ev["bin"] = (ev["pH"] * 10).clip(0, 9).astype(int)
    for b in range(10):
        g = ev[ev["bin"] == b]
        if len(g) == 0:
            continue
        pred = g["pH"].mean()
        obs = (g["actual"] == "H").mean()
        print(f"    pred {pred:4.0%}  observed {obs:4.0%}  (n={len(g)})")
    print("=" * 56)
    print("  Read log-loss/Brier vs baseline (the edge), and whether predicted")
    print("  ~ observed in the calibration table (whether the probs are honest).")


if __name__ == "__main__":
    main()
