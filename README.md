# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/angyi-cmyk/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/angyi-cmyk/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles per trophy, from the house's recording of the trophy page up to the case checkpoint. One program, `main.py` (Python 3, standard library only), reads each case at `/input/<id>.in` and writes `/output/<id>.out`.

## Model

Earners over players tracked, fitted on the refresh steps:

- Exophase refreshes a game's counts on its own schedule, so consecutive reads are often the same page. The history is first collapsed into distinct page states (one per refresh, taken at the first read that shows it); fitting on raw reads would let one stale page outweigh every refresh.
- Players tracked N(t) and, per trophy, earners E(t) = rate x N / 100 are each fitted as log-linear in log(1 + hours since release) by least squares over the distinct states, and read at the target. Earners are floored at their last observed count and capped at 99% of the forecast players tracked. The median rate is 100 x E(target) / N(target).
- The spread (logit sd) combines the fit residuals, a random-walk term 0.5 x sqrt(horizon in days), and a floor, so the five quantiles are honest about a multi-day extrapolation of a launch curve.
- When the history holds a single distinct state (no refresh seen yet), the median is shifted down one logit unit and the spread widened: the first refresh after launch dilutes every rate as a wave of new players is tracked.

No data beyond the case files is used; no trophy names, rates, or player counts are invented.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Practice cases in the kit's `input/`: `p00` (checkpoint 2026-11-19T03:00Z, target 04:30Z) and `p01` (03:30Z, target 05:00Z). Both produce valid 55-trophy forecasts in ~0.1 s, byte-identical across runs.

---

<sub>Entered on [make.fun](https://make.fun/?ref=m2756q7x&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
