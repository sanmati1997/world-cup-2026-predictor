# World Cup 2026 Predictor (Dixon-Coles + honest calibration)

A football match predictor built on a **Dixon-Coles bivariate Poisson** model. It rates every national team's attack and defense from decades of results, then builds a full **scoreline distribution** for any matchup, so win / draw / loss, expected goals, and likely scores all come from one model. Draws are handled structurally, not bolted on.

> Inspired by [@mar-antaya](https://github.com/mar-antaya/world_cup_predictions)'s World Cup predictor. I took a different approach on purpose: a goals-based Poisson model instead of a win/draw/loss classifier, plus calibration and an injury hook. Credit to her for the idea and the open-data pointer.

```
python predict.py "Spain" "Brazil"
```
```
============================================================
  Spain  vs  Brazil    (neutral)
============================================================
  Spain win                41.8%  ##########
  Draw                     27.6%  #######
  Brazil win               30.6%  #######
------------------------------------------------------------
  Most likely: Spain win (42%)
  Expected goals: Spain 1.43 - 1.19 Brazil
  Likely scores: 1-1 (13%), 1-0 (10%), 2-1 (9%)
============================================================
```

## What's different here

1. **Goals model, not a classifier.** Dixon-Coles models goals, so it produces full scorelines and expected goals, and **draws fall out of the score matrix** (the diagonal) instead of being under-predicted by a 3-class classifier.
2. **Honest calibration.** The headline isn't accuracy, it's whether the probabilities are *trustworthy*. Measured below.
3. **No black box.** Every team has an interpretable attack and defense rating, and the prediction shows *why* the favorite is favored (attack/defense percentiles, expected goals). The only "hyperparameters" are home advantage, the low-score correction, and a time-decay.

## Results (leakage-free backtest)

Trained only on matches **before** the cutoff, tested on **3,103** played matches after it (`python evaluate.py --cutoff 2023-06-01`):

| metric | model | no-skill baseline |
|---|---|---|
| **log-loss** | **0.874** | 1.058 |
| **Brier** | **0.514** | 0.639 |
| outcome accuracy | 58.7% | (ceiling-bound) |

**Calibration** (when it says X%, does it happen X%?):

| predicted home-win | observed |
|---|---|
| 25% | 26% |
| 45% | 44% |
| 55% | 51% |
| 75% | 75% |
| 95% | 93% |

The probabilities track reality closely across the whole range. That is the point: a calibrated model you can trust over many games, not a crystal ball for any single one. **Outcome accuracy (~59%) is comparable to a classifier and to bookmakers** because football has an information ceiling - the edge shows up in log-loss, Brier, and calibration, not in winner-picking.

## Usage

```
# one match
python predict.py "Argentina" "Saudi Arabia"

# every upcoming World Cup 2026 fixture -> predictions/wc2026.csv
python predict_all.py

# reproduce the backtest + calibration
python evaluate.py --cutoff 2023-06-01
```

## How it works

For a match, with team attack `a`, defense `d`, home advantage `h` (zero at neutral venues):

```
home goals ~ Poisson(exp(a_home - d_away + h))
away goals ~ Poisson(exp(a_away - d_home))
```

Parameters are fit by maximum likelihood over international results since 2018, with **exponential time decay** (recent matches weigh more) and the **Dixon-Coles low-score correction** for the 0-0/1-0/0-1/1-1 dependence. The scoreline matrix is summed to get win/draw/loss.

## Honest limitations

- **Outcome accuracy is capped** (~55-60%) by football's randomness. No model beats that by much; treat single-match calls with humility.
- **Beating the betting market is the real bar**, and this does not claim to. It claims to be well-calibrated and clearly better than no-skill.
- **Team-level only.** It rates teams, not the specific players on the pitch, so it has no notion of injuries, suspensions, or lineups. Player-level / injury adjustment is the natural next layer (it needs live squad data).
- **Low-data teams** (debutants, small nations) have less reliable ratings; a minimum-match filter and time-decay mitigate but do not remove this.
- Slice/team-level only; no xG, lineups, or tactics yet.

## Data

Historical results: the open [martj42/international_results](https://github.com/martj42/international_results) dataset (auto-downloads on first run). 2026 fixtures are the unplayed rows in that same file.

## License
MIT.

Built by [Sanmati Sawalwade](https://sanmati1997.github.io). Inspiration: [@mar-antaya](https://github.com/mar-antaya).
