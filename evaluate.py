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

run_backtest() returns the same numbers as a dict and caches them to
data_cache/backtest_metrics.json so the app can show real metrics, never
hardcoded ones. The backtest math is unchanged; it just returns now too.
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import model

EPS = 1e-15
METRICS_PATH = "data_cache/backtest_metrics.json"


def outcome(hs, as_):
    return "H" if hs > as_ else ("A" if as_ > hs else "D")


def run_backtest(cutoff="2023-06-01", save=True):
    """Leakage-free backtest. Train strictly before cutoff, test at or after it.
    Returns a metrics dict and (by default) caches it to JSON."""
    played, _ = model.load_data()
    cut = pd.Timestamp(cutoff)
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
        rows.append({"actual": outcome(m.home_score, m.away_score),
                     "pH": r["p_home"], "pD": r["p_draw"], "pA": r["p_away"]})
    ev = pd.DataFrame(rows)
    n = len(ev)

    col = {"H": "pH", "D": "pD", "A": "pA"}
    pa = np.array([ev.iloc[i][col[ev.iloc[i]["actual"]]] for i in range(n)])
    model_ll = float(-np.mean(np.log(np.clip(pa, EPS, 1))))
    model_br = float(np.mean([sum((ev.iloc[i]["p" + c] - (1.0 if ev.iloc[i]["actual"] == c else 0.0)) ** 2
                                   for c in "HDA") for i in range(n)]))
    base_ll = float(-np.mean([np.log(max(base[a], EPS)) for a in ev["actual"]]))
    base_br = float(np.mean([sum((base[c] - (1.0 if a == c else 0.0)) ** 2 for c in "HDA")
                             for a in ev["actual"]]))
    acc = float(np.mean([max("HDA", key=lambda c: ev.iloc[i]["p" + c]) == ev.iloc[i]["actual"]
                         for i in range(n)]))

    # calibration on home-win probability
    ev["bin"] = (ev["pH"] * 10).clip(0, 9).astype(int)
    calibration = []
    for b in range(10):
        g = ev[ev["bin"] == b]
        if len(g) == 0:
            continue
        calibration.append({"pred": float(g["pH"].mean()),
                            "obs": float((g["actual"] == "H").mean()), "n": int(len(g))})

    result = {"cutoff": cutoff, "n": n,
              "model_logloss": model_ll, "baseline_logloss": base_ll,
              "model_brier": model_br, "baseline_brier": base_br,
              "accuracy": acc, "calibration": calibration}

    if save:
        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        with open(METRICS_PATH, "w") as f:
            json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2023-06-01", help="train < cutoff, test >= cutoff")
    args = ap.parse_args()

    res = run_backtest(args.cutoff)
    n = res["n"]

    print("=" * 56)
    print(f"  Backtest: train < {args.cutoff} | test n = {n}")
    print("=" * 56)
    print(f"  {'metric':<18}{'model':>10}{'baseline':>12}")
    print(f"  {'log-loss':<18}{res['model_logloss']:>10.3f}{res['baseline_logloss']:>12.3f}   (lower better)")
    print(f"  {'Brier':<18}{res['model_brier']:>10.3f}{res['baseline_brier']:>12.3f}   (lower better)")
    print(f"  {'outcome accuracy':<18}{res['accuracy']:>9.1%}{'':>12}   (ceiling-bound)")
    print("=" * 56)
    print("  Calibration (home-win prob):")
    for c in res["calibration"]:
        print(f"    pred {c['pred']:4.0%}  observed {c['obs']:4.0%}  (n={c['n']})")
    print("=" * 56)
    print("  Read log-loss/Brier vs baseline (the edge), and whether predicted")
    print("  ~ observed in the calibration table (whether the probs are honest).")


if __name__ == "__main__":
    main()
