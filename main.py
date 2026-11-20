"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

Model: earners over players tracked, fitted on the refresh steps.
  - The house reads Exophase's trophy page on a clock, but Exophase refreshes
    a game's counts on its own schedule, so consecutive reads are often the
    same page. The history is first collapsed into distinct page states
    (one per refresh, taken at the first read that shows it): fitting on
    reads would let one stale page outweigh every refresh.
  - Players tracked N(t) and, per trophy, earners E(t) = rate * N / 100 are
    each fitted as log-linear in log(1 + hours since release) by least
    squares over the distinct states, and read at the target. Earners are
    floored at their last observed count (a count cannot fall) and capped
    at 99% of the forecast players tracked. The median rate is
    100 * E(target) / N(target).
  - Spread (logit sd) combines the fit residuals, a random-walk term
    0.5 * sqrt(horizon in days), and a floor, so the five quantiles are
    honest about a multi-day extrapolation of a launch curve.
  - When the history holds a single distinct state (no refresh seen yet),
    the median is shifted down one logit unit and the spread widened: the
    first refresh after launch dilutes every rate, as a wave of new players
    is tracked while few of them have earned anything yet.

Standard library only, deterministic: the same case always writes the same
bytes. Reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR
(else /output), one <id>.out per <id>.in.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Spread of a rate's move, in logit units per square-root day (the house
# baseline's random walk); the fitted trend's own residual is added to it.
SIGMA_RW = 0.5
# Floor / cap on the total logit spread, and the spread for a trophy nobody
# has earned yet (no trend to fit: only a wide, mostly upward band).
SPREAD_MIN = 0.6
SPREAD_MAX = 2.2
SPREAD_ZERO = 2.0
# Default residual when a fit has too few points to measure one.
RESID_DEFAULT = 0.4
# Downward shift (logit) and spread floor when the history shows a single
# distinct page state: the first refresh dilutes the rates.
DILUTION_SHIFT = 1.0
DILUTION_SPREAD = 1.2


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def round4(x):
    """x rounded to 4 decimals, half up (as the starter rounds it)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def fit_line(xs, ys):
    """Least-squares y ~ a + b*x. Returns (a, b, residual_sd)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    if n >= 3:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return a, b, resid


def distinct_states(case):
    """Collapse consecutive identical reads into one state per refresh.

    A state is (time, players_tracked, rates); it starts at the first read
    that shows those values.
    """
    states = []
    prev_key = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != prev_key:
            states.append((parse_time(read["at"]), read["players_tracked"], list(read["rates"])))
            prev_key = key
    return states


def forecast_case(case, name):
    if not case["history"]:
        # No reads at all: no basis for a rate; stay valid with a flat,
        # very wide forecast rather than crashing the case.
        return {"trophies": [
            {"id": t["id"], "q": [round4(inv_logit(logit(1.0) + norm_ppf(lv) * SPREAD_ZERO))
                                  for lv in LEVELS]}
            for t in case["trophies"]]}
    release = case.get("game", {}).get("release_date", "2026-11-19")
    origin = datetime.fromisoformat(release).replace(tzinfo=timezone.utc)
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = math.log1p(max(0.0, (target - origin).total_seconds() / 3600.0))

    states = distinct_states(case)
    xs = [math.log1p(max(0.0, (t - origin).total_seconds() / 3600.0)) for t, _, _ in states]
    ns = [n for _, n, _ in states]

    # Players tracked at the target.
    if len(states) >= 2:
        a_n, b_n, resid_n = fit_line(xs, [math.log(n) for n in ns])
        n_target = math.exp(a_n + b_n * x_target)
    else:
        resid_n = 0.0
        n_target = float(ns[-1])
    n_target = max(n_target, float(ns[-1]))

    spread_rw = SIGMA_RW * math.sqrt(horizon_days)
    # Exactly one distinct state: no refresh seen yet, so no trend to fit
    # and the next refresh's dilution is unobserved. With two or more
    # states the fitted trend already carries the refreshes.
    flat_history = len(states) == 1

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        # Last observed earners for this trophy (skip reads that lack it).
        e_last = None
        for _, n, rates in reversed(states):
            r = rates[i]
            if r is not None:
                e_last = r * n / 100.0
                break
        if e_last is None or e_last <= 0:
            # Nobody tracked has earned it yet: median stays at the floor,
            # with a wide, mostly upward band.
            med = 0.01
            spread = SPREAD_ZERO
            rows.append({"id": trophy["id"], "q": [
                round4(min(99.9999, max(0.0001, inv_logit(logit(med) + norm_ppf(lv) * spread))))
                for lv in LEVELS]})
            continue

        pts = [(x, math.log(rates[i] * n / 100.0))
               for x, (_, n, rates) in zip(xs, states)
               if rates[i] is not None and rates[i] > 0]
        if len(pts) >= 3:
            a_e, b_e, resid_e = fit_line([p[0] for p in pts], [p[1] for p in pts])
            e_target = math.exp(a_e + b_e * x_target)
            spread = math.sqrt(resid_e ** 2 + resid_n ** 2 + spread_rw ** 2)
        else:
            # Too few positive reads to fit a trend: the earner count
            # persists, with a default residual for its unknown growth.
            e_target = e_last
            spread = math.sqrt(RESID_DEFAULT ** 2 + resid_n ** 2 + spread_rw ** 2)
        e_target = max(e_target, e_last)
        e_target = min(e_target, 0.99 * n_target)
        med = clamp_rate(100.0 * e_target / n_target)

        lg = logit(med)
        if flat_history:
            # No refresh seen yet: the next one dilutes the rates.
            lg -= DILUTION_SHIFT
            spread = max(spread, DILUTION_SPREAD)
        spread = min(SPREAD_MAX, max(SPREAD_MIN, spread))
        rows.append({"id": trophy["id"], "q": [
            round4(min(99.9999, max(0.0001, inv_logit(lg + norm_ppf(lv) * spread))))
            for lv in LEVELS]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
