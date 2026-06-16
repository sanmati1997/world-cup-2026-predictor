"""
model.py - Dixon-Coles bivariate Poisson model for international football.

Fits per-team attack and defense strengths, a global home advantage, and the
Dixon-Coles low-score correction (rho), with exponential time decay so recent
matches matter more. From the fitted rates it builds a full scoreline matrix,
so win / draw / loss and expected goals all fall out of one model - and draws
are handled structurally, not as an afterthought.

Trained only on matches that have actually been played; World Cup 2026 fixtures
(blank scores in the dataset) are the prediction targets.
"""

import os
import urllib.request

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def load_data(path="data_cache/results.csv"):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("downloading results dataset (one time)...")
        urllib.request.urlretrieve(DATA_URL, path)
    df = pd.read_csv(path, parse_dates=["date"])
    df["played"] = df["home_score"].notna() & df["away_score"].notna()
    played = df[df["played"]].copy()
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)
    fixtures = df[~df["played"]].copy()
    return played, fixtures


def _tau(hs, as_, lh, la, rho):
    """Dixon-Coles low-score dependence correction (vectorized)."""
    t = np.ones_like(lh)
    m00 = (hs == 0) & (as_ == 0); t[m00] = 1.0 - lh[m00] * la[m00] * rho
    m01 = (hs == 0) & (as_ == 1); t[m01] = 1.0 + lh[m01] * rho
    m10 = (hs == 1) & (as_ == 0); t[m10] = 1.0 + la[m10] * rho
    m11 = (hs == 1) & (as_ == 1); t[m11] = 1.0 - rho
    return np.clip(t, 1e-9, None)


def fit(played, since="2014-01-01", half_life_years=2.0, min_matches=15):
    """Fit the Dixon-Coles model. Returns a params dict."""
    d = played[played["date"] >= pd.Timestamp(since)].copy()

    # keep teams with enough recent matches (others get unreliable ratings)
    counts = pd.concat([d["home_team"], d["away_team"]]).value_counts()
    teams = sorted(counts[counts >= min_matches].index)
    tset = set(teams)
    d = d[d["home_team"].isin(tset) & d["away_team"].isin(tset)]
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    hi = d["home_team"].map(idx).to_numpy()
    ai = d["away_team"].map(idx).to_numpy()
    hs = d["home_score"].to_numpy()
    as_ = d["away_score"].to_numpy()
    neutral = d["neutral"].to_numpy().astype(float)

    # exponential time decay weight (recent matches weigh more)
    age_days = (d["date"].max() - d["date"]).dt.days.to_numpy()
    xi = np.log(2) / (half_life_years * 365.0)
    w = np.exp(-xi * age_days)

    lg_hs = gammaln(hs + 1.0)
    lg_as = gammaln(as_ + 1.0)

    def unpack(theta):
        attack = theta[:n]
        attack = attack - attack.mean()        # identifiability: mean attack = 0
        defense = theta[n:2 * n]
        home_adv = theta[2 * n]
        rho = theta[2 * n + 1]
        return attack, defense, home_adv, rho

    def nll(theta):
        attack, defense, home_adv, rho = unpack(theta)
        lh = np.exp(attack[hi] - defense[ai] + home_adv * (1.0 - neutral))
        la = np.exp(attack[ai] - defense[hi])
        ll = (hs * np.log(lh) - lh - lg_hs) + (as_ * np.log(la) - la - lg_as)
        ll = ll + np.log(_tau(hs, as_, lh, la, rho))
        return -np.sum(w * ll)

    theta0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])
    bounds = [(-3, 3)] * (2 * n) + [(-1.0, 1.0), (-0.2, 0.2)]
    res = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 400, "maxfun": 100000})
    attack, defense, home_adv, rho = unpack(res.x)
    return {
        "teams": teams, "idx": idx,
        "attack": dict(zip(teams, attack)),
        "defense": dict(zip(teams, defense)),
        "home_adv": float(home_adv), "rho": float(rho),
        "n_matches": int(len(d)), "since": since,
    }


def predict(params, home, away, neutral=True, max_goals=10,
            home_adj=1.0, away_adj=1.0):
    """
    Predict one match. home_adj / away_adj scale a team's attack rate to reflect
    missing players (1.0 = full strength, e.g. 0.88 if key attackers are out).
    Returns win/draw/loss probabilities, expected goals, and top scorelines.
    """
    a, dfn = params["attack"], params["defense"]
    for t in (home, away):
        if t not in a:
            raise KeyError(f"'{t}' not in model (too few recent matches or name "
                           f"mismatch). Known example: {params['teams'][:3]} ...")
    ha = params["home_adv"] * (0.0 if neutral else 1.0)
    lh = np.exp(a[home] - dfn[away] + ha) * home_adj
    la = np.exp(a[away] - dfn[home]) * away_adj

    gx = np.arange(0, max_goals + 1)
    ph = poisson.pmf(gx, lh)
    pa = poisson.pmf(gx, la)
    M = np.outer(ph, pa)  # M[x, y] = P(home x, away y)

    rho = params["rho"]
    M[0, 0] *= 1.0 - lh * la * rho
    M[0, 1] *= 1.0 + lh * rho
    M[1, 0] *= 1.0 + la * rho
    M[1, 1] *= 1.0 - rho
    M = np.clip(M, 0, None)
    M /= M.sum()

    p_home = np.tril(M, -1).sum()   # home goals > away goals
    p_away = np.triu(M, 1).sum()
    p_draw = np.trace(M)
    top = sorted([((x, y), M[x, y]) for x in gx for y in gx],
                 key=lambda kv: -kv[1])[:5]
    return {
        "home": home, "away": away, "neutral": neutral,
        "p_home": float(p_home), "p_draw": float(p_draw), "p_away": float(p_away),
        "exp_home_goals": float(lh), "exp_away_goals": float(la),
        "top_scores": [(f"{x}-{y}", float(p)) for (x, y), p in top],
    }
