"""Audit the small-bump Q2 put gamma without subtracting three option prices.

Run ``python audit_q2.py`` after the assignment calculation. The independent
recursion below reproduces the N=500 cash-dividend American put lattice at
S=48.50 and h=1e-5. It propagates a baseline value and two endpoint first
divided differences through every linear operation and exercise maximum:

    d_minus = (V(S-h) - V(S)) / h
    d_plus  = (V(S+h) - V(S)) / h
    gamma   = (d_plus + d_minus) / h

The endpoint differences are propagated directly, rather than obtained by
subtracting independently rounded root prices. This reduces cancellation in
the tiny central second difference while preserving payoff/exercise branch
changes across the bump. It audits the finite lattice's central difference;
it does not estimate a smooth continuous-model gamma or remove lattice kinks.

Only the command-line entry point writes q2_gamma_audit.json. Importing this
module and calling ``audit`` or ``stable_put_triplet`` do not write files.
"""
import argparse
import csv
from functools import lru_cache
import json
import math
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
SPOT, STRIKE, MATURITY = 48.50, 50.0, 0.5
RATE, VOLATILITY = 0.045, 0.30
STEPS, BUMP = 500, 1e-5
DIVIDENDS = ((0.10, 0.25), (0.40, 1.50))
PRICE_ABSOLUTE_TOLERANCE = 1e-10
GAMMA_RELATIVE_TOLERANCE = 0.005
MINIMUM_SUBSTANTIVE_GAMMA = 0.01
AUDIT_JSON = "q2_gamma_audit.json"


def stable_put_triplet():
    """Return (V(S), d_minus, d_plus) from an independent fixed-input tree.

    A triplet contains the unbumped node value and its endpoint differences
    with respect to the *initial* stock-price bump. Cash dividends are capped
    by available equity, and American exercise is allowed before dividends,
    after dividends, and at ordinary tree nodes, matching the saved model.
    No production pricing function or production parameter module is used.
    """
    dt = MATURITY / STEPS
    log_up = VOLATILITY * np.sqrt(dt)
    up, down = np.exp(log_up), np.exp(-log_up)
    probability = (np.exp(RATE * dt) - down) / (up - down)
    discount = np.exp(-RATE * dt)
    events = tuple((int(round(when / dt)), cash) for when, cash in DIVIDENDS)

    @lru_cache(maxsize=None)
    def multipliers(steps):
        return np.exp((2 * np.arange(steps + 1) - steps) * log_up)

    def positive_part(value, minus, plus):
        """Propagate max(value, 0) without forming bumped node values."""
        nonnegative = value >= 0

        def endpoint(difference):
            answer = np.empty_like(value)
            answer[nonnegative] = np.maximum(
                difference[nonnegative], -value[nonnegative] / BUMP
            )
            answer[~nonnegative] = np.maximum(
                value[~nonnegative] / BUMP + difference[~nonnegative], 0.0
            )
            return answer

        return np.maximum(value, 0.0), endpoint(minus), endpoint(plus)

    def payoff(stock, minus, plus):
        return positive_part(STRIKE - stock, -minus, -plus)

    def maximum(left, right):
        """Propagate max(left, right), including endpoint branch switches."""
        left_value, left_minus, left_plus = left
        right_value, right_minus, right_plus = right
        choose_left = left_value >= right_value
        # A baseline branch gap enters only as an offset to the competing
        # branch's endpoint difference, never as a root-price difference.
        gap = (right_value - left_value) / BUMP

        def endpoint(left_difference, right_difference):
            return np.where(
                choose_left,
                np.maximum(left_difference, gap + right_difference),
                np.maximum(right_difference, -gap + left_difference),
            )

        return (
            np.maximum(left_value, right_value),
            endpoint(left_minus, right_minus),
            endpoint(left_plus, right_plus),
        )

    def node_stocks(stock, minus, plus, steps):
        factors = multipliers(steps)
        return tuple(values[:, None] * factors for values in (stock, minus, plus))

    def backward(values, stock, minus, plus, steps):
        for step in range(steps - 1, -1, -1):
            values = tuple(
                discount * ((1 - probability) * value[:, :-1]
                            + probability * value[:, 1:])
                for value in values
            )
            values = maximum(values, payoff(*node_stocks(stock, minus, plus, step)))
        return tuple(value[:, 0] for value in values)

    def recurse(stock, minus, plus, steps, remaining_events):
        if not remaining_events:
            terminal = payoff(*node_stocks(stock, minus, plus, steps))
            return backward(terminal, stock, minus, plus, steps)
        first_step, cash = remaining_events[0]
        before = node_stocks(stock, minus, plus, first_step)
        after = positive_part(before[0] - cash, before[1], before[2])
        later = tuple((when - first_step, amount)
                      for when, amount in remaining_events[1:])
        shape = before[0].shape
        continuation = recurse(*(value.ravel() for value in after),
                               steps - first_step, later)
        values = tuple(value.reshape(shape) for value in continuation)
        values = maximum(values, payoff(*before))
        return backward(values, stock, minus, plus, first_step)

    result = recurse(np.array([SPOT]), np.array([-1.0]), np.array([1.0]),
                     STEPS, events)
    return tuple(float(value[0]) for value in result)


def _saved_row(path, *, bump=None):
    with path.open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source)
                if row["option"] == "AmericanPut" and int(row["n"]) == STEPS
                and (bump is None or math.isclose(float(row["bump"]), bump,
                                                 rel_tol=1e-12, abs_tol=0.0))]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one matching AmericanPut row in {path}")
    return rows[0]


def audit(results=ROOT / "results"):
    """Read saved outputs, recompute the stable triplet, and return evidence."""
    results = Path(results)
    greek_row = _saved_row(results / "q2_finite_difference_greeks.csv", bump=BUMP)
    price_row = _saved_row(results / "q2_prices.csv")
    saved_gamma = float(greek_row["gamma"])
    saved_price = float(price_row["with_dividend_value"])
    if not all(math.isfinite(value) for value in (saved_price, saved_gamma)):
        raise ValueError("Saved price and gamma must be finite")

    started = time.perf_counter()
    stable_price, d_minus, d_plus = stable_put_triplet()
    runtime = time.perf_counter() - started
    stable_gamma = (d_plus + d_minus) / BUMP
    stable_delta = (d_plus - d_minus) / 2
    gamma_difference = stable_gamma - saved_gamma
    relative_difference = (abs(gamma_difference) / abs(saved_gamma)
                           if saved_gamma != 0 else math.inf)
    price_error = abs(stable_price - saved_price)
    checks = dict(
        stable_values_finite=all(math.isfinite(value)
                                 for value in (stable_price, stable_gamma, stable_delta)),
        baseline_price_matches=price_error <= PRICE_ABSOLUTE_TOLERANCE,
        gamma_agrees=relative_difference < GAMMA_RELATIVE_TOLERANCE,
        gamma_positive_and_substantive=stable_gamma > MINIMUM_SUBSTANTIVE_GAMMA,
    )
    return dict(
        inputs=dict(option="AmericanPut", spot=SPOT, strike=STRIKE,
                    maturity=MATURITY, risk_free_rate=RATE, volatility=VOLATILITY,
                    n=STEPS, bump=BUMP,
                    dividends=[dict(time=when, amount=cash) for when, cash in DIVIDENDS]),
        method="Independent CRR recursion of baseline values and endpoint first divided differences",
        scope="Same finite cash-dividend lattice and central bump as the saved Q2 result; not a smooth-model gamma",
        saved_price=saved_price, stable_price=stable_price,
        price_absolute_error=price_error,
        price_absolute_tolerance=PRICE_ABSOLUTE_TOLERANCE,
        saved_gamma=saved_gamma, stable_gamma=stable_gamma,
        stable_delta=stable_delta, d_minus=d_minus, d_plus=d_plus,
        gamma_difference=gamma_difference,
        gamma_relative_difference=relative_difference,
        gamma_relative_tolerance=GAMMA_RELATIVE_TOLERANCE,
        minimum_substantive_gamma=MINIMUM_SUBSTANTIVE_GAMMA,
        runtime_seconds=runtime, checks=checks, passed=all(checks.values()),
        interpretation=(
            "A substantive positive gamma survives direct propagation of endpoint "
            "differences, supporting finite-lattice kink crossings as the predominant "
            "source of the small-bump gamma. Ordinary root-price subtraction adds "
            "a smaller floating-point error; this audit also uses floating point."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results",
                        help="Directory containing the saved Q2 CSV files (default: %(default)s)")
    arguments = parser.parse_args()
    proof = audit(arguments.results)
    destination = arguments.results / AUDIT_JSON
    destination.write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    status = "PASS" if proof["passed"] else "FAIL"
    print(f"{status}: stable gamma={proof['stable_gamma']:.10g}; "
          f"saved gamma={proof['saved_gamma']:.10g}; "
          f"difference={proof['gamma_difference']:.6g} "
          f"({proof['gamma_relative_difference']:.4%}); "
          f"runtime={proof['runtime_seconds']:.2f}s")
    if not proof["passed"]:
        failed = ", ".join(name for name, passed in proof["checks"].items() if not passed)
        raise AssertionError(f"Q2 gamma audit failed: {failed}; see {destination}")


if __name__ == "__main__":
    main()
