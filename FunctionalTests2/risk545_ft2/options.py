"""Generalized Black--Scholes--Merton and CRR option valuation.

Adapted from this project's Assignment3 pricing/american implementations into
an independent module. Time is in years; rates are continuously compounded.
Sensitivities are per absolute unit (vega is not divided by 100).

The teacher's Greek conventions intentionally differ. ``greeks`` reports
calendar theta, -dV/dT, and stock-option rho holding q=r-b fixed. In contrast,
``tree_greeks`` reports maturity sensitivity, +dV/dT, and rho holding cost of
carry b fixed. ``build_cases`` preserves and documents both conventions.
Tree gamma uses a 1.5-dollar spot bump.

Cash-dividend trees recombine only between payment dates. Every distinct
pre-dividend node starts its own post-dividend subtree. Exercise at each cash
dividend is considered immediately BEFORE payment.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import special, stats

__all__ = ["price", "greeks", "tree_price", "tree_greeks", "cash_dividend_price", "build_cases"]


def _call_flag(is_call):
    if not isinstance(is_call, (bool, np.bool_)):
        raise ValueError("is_call must be a boolean")
    return bool(is_call)


def _inputs(s, k, t, r, b, sigma):
    arrays = np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in (s, k, t, r, b, sigma)])
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("Pricing inputs must be finite")
    s, k, t, r, b, sigma = arrays
    if (s < 0).any() or (k <= 0).any() or (t < 0).any() or (sigma < 0).any():
        raise ValueError("Require spot >= 0, strike > 0, maturity >= 0, volatility >= 0")
    return arrays


def _result(value):
    return float(value) if np.ndim(value) == 0 else value


def price(s, k, t, r, b, sigma, is_call=True):
    """Broadcastable European GBSM price, with expiry and zero-volatility limits.

    b is cost of carry: b=r-q for a stock with dividend yield q. Spot may be
    zero. Strike must be positive; time and volatility must be nonnegative.
    """
    sign = 1 if _call_flag(is_call) else -1
    s, k, t, r, b, sigma = _inputs(s, k, t, r, b, sigma)
    active = (s > 0) & (t > 0) & (sigma > 0)
    vol_time = np.where(active, sigma*np.sqrt(t), 1.)
    d1 = (np.log(np.maximum(s, np.finfo(float).tiny)/k)+(b+.5*sigma*sigma)*t)/vol_time
    d2 = d1-vol_time
    regular = sign*(s*np.exp((b-r)*t)*special.ndtr(sign*d1)
                    - k*np.exp(-r*t)*special.ndtr(sign*d2))
    deterministic = np.exp(-r*t)*np.maximum(sign*(s*np.exp(b*t)-k), 0.)
    return _result(np.where(active, regular, deterministic))


def greeks(s, k, t, r, b, sigma, is_call=True):
    """European value and analytical Greeks for strictly positive S, T, sigma.

    rho varies r and b together (fixed dividend yield); carry_rho differentiates
    b alone. rho_fixed_carry differentiates r alone. Theta is calendar decay,
    -dV/dT, per year. Inputs broadcast as in price.
    """
    sign = 1 if _call_flag(is_call) else -1
    s, k, t, r, b, sigma = _inputs(s, k, t, r, b, sigma)
    if (s <= 0).any() or (t <= 0).any() or (sigma <= 0).any():
        raise ValueError("Greeks require positive spot, maturity, and volatility")
    root_t = np.sqrt(t)
    d1 = (np.log(s/k)+(b+.5*sigma*sigma)*t)/(sigma*root_t)
    d2 = d1-sigma*root_t
    carry, discount = np.exp((b-r)*t), np.exp(-r*t)
    density = np.exp(-.5*d1*d1)/np.sqrt(2*np.pi)
    value = price(s, k, t, r, b, sigma, is_call)
    delta = sign*carry*special.ndtr(sign*d1)
    gamma = carry*density/(s*sigma*root_t)
    vega = s*carry*density*root_t
    theta = (-s*carry*density*sigma/(2*root_t)
             -sign*(b-r)*s*carry*special.ndtr(sign*d1)
             -sign*r*k*discount*special.ndtr(sign*d2))
    rho = sign*t*k*discount*special.ndtr(sign*d2)
    carry_rho = sign*t*s*carry*special.ndtr(sign*d1)
    return {name: _result(x) for name, x in dict(
        value=value, delta=delta, gamma=gamma, vega=vega, rho=rho,
        theta=theta, carry_rho=carry_rho, rho_fixed_carry=-t*value).items()}


@lru_cache(maxsize=1024)
def _multipliers(steps, log_up):
    return np.exp((2*np.arange(steps+1)-steps)*log_up)


def _plain_subtrees(spots, steps, strike, log_up, p, discount, sign, american,
                    no_early_call=False):
    """Batch distinct CRR origins without merging their descendant nodes."""
    spots = np.asarray(spots, float).reshape(-1)
    # No-dividend calls cannot benefit from early exercise when b=r and r>=0.
    early_exercise = american and not (sign == 1 and no_early_call)
    probabilities = None if early_exercise else stats.binom.pmf(np.arange(steps+1), steps, p)
    result = np.empty(len(spots))
    for start in range(0, len(spots), 2048):
        batch = spots[start:start+2048]
        values = np.maximum(sign*(batch[:, None]*_multipliers(steps, log_up)-strike), 0.)
        if not early_exercise:
            result[start:start+len(batch)] = discount**steps*(values @ probabilities)
            continue
        for step in range(steps-1, -1, -1):
            values = discount*((1-p)*values[:, :-1]+p*values[:, 1:])
            intrinsic = np.maximum(sign*(batch[:, None]*_multipliers(step, log_up)-strike), 0.)
            np.maximum(values, intrinsic, out=values)
        result[start:start+len(batch)] = values[:, 0]
    return result


def _tree_inputs(s, k, t, r, b, sigma, n, is_call, american):
    _call_flag(is_call)
    if not isinstance(american, (bool, np.bool_)):
        raise ValueError("american must be a boolean")
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 1:
        raise ValueError("Tree steps must be a positive integer")
    if any(np.ndim(x) != 0 for x in (k, t, r, b, sigma)):
        raise ValueError("Only spot may be an array for tree valuation")
    spots, _, _, _, _, _ = _inputs(s, k, t, r, b, sigma)
    if sigma <= 0:
        raise ValueError("Tree volatility must be positive")
    return spots


def _tree_parameters(t, r, b, sigma, n):
    dt = t/n
    log_up = sigma*np.sqrt(dt)
    up, down = np.exp(log_up), np.exp(-log_up)
    p = (np.exp(b*dt)-down)/(up-down)
    if not 0 < p < 1:
        raise ValueError("Risk-neutral probability must be in (0,1); refine the tree grid")
    return log_up, p, np.exp(-r*dt)


def tree_price(s, k, t, r, b, sigma, n=500, is_call=True, american=True):
    """CRR price with continuous cost of carry and optional early exercise.

    Spot can be a scalar or an array; other financial inputs are scalar.
    Volatility and strike must be positive. At zero maturity return intrinsic.
    """
    spots = _tree_inputs(s, k, t, r, b, sigma, n, is_call, american)
    sign = 1 if is_call else -1
    if t == 0:
        return _result(np.maximum(sign*(spots-k), 0.))
    log_up, p, discount = _tree_parameters(t, r, b, sigma, n)
    answer = _plain_subtrees(spots.reshape(-1), n, k, log_up, p, discount,
                             sign, bool(american),
                             no_early_call=(b == r and r >= 0)).reshape(spots.shape)
    return _result(answer)


def tree_greeks(s, k, t, r, b, sigma, n=500, is_call=True, gamma_bump=1.5):
    """American CRR Greeks using the teacher's finite-difference convention.

    Delta, vega, rho, and maturity sensitivity use central differences with
    eps**(1/3)*max(abs(parameter), 1). Gamma uses gamma_bump dollars (1.5 for
    group 12.2). Rho holds b fixed; theta means +dV/dT. Tree payoff kinks make
    these local numerical derivatives sensitive to the grid.
    """
    if any(np.ndim(x) != 0 for x in (s, k, t, r, b, sigma)):
        raise ValueError("Tree Greeks require scalar inputs")
    _tree_inputs(s, k, t, r, b, sigma, n, is_call, True)
    if not np.isfinite(gamma_bump) or gamma_bump <= 0 or s <= gamma_bump:
        raise ValueError("Gamma bump must be positive and less than spot")
    if t <= 0:
        raise ValueError("Tree Greeks require positive maturity")
    parameters = np.array([s, k, t, r, b, sigma], dtype=float)
    value = tree_price(*parameters, n=n, is_call=is_call)
    result = {"value": value}
    for name, index in (("delta", 0), ("vega", 5), ("rho", 3), ("theta", 2)):
        h = np.cbrt(np.finfo(float).eps)*max(abs(parameters[index]), 1.)
        if index in (2, 5) and parameters[index] <= h:
            raise ValueError("Maturity and volatility must exceed the finite-difference step")
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += h
        minus[index] -= h
        result[name] = (tree_price(*plus, n=n, is_call=is_call)
                        - tree_price(*minus, n=n, is_call=is_call))/(2*h)
    up, down = parameters.copy(), parameters.copy()
    up[0] += gamma_bump
    down[0] -= gamma_bump
    result["gamma"] = (tree_price(*up, n=n, is_call=is_call)
                       + tree_price(*down, n=n, is_call=is_call)-2*value)/gamma_bump**2
    return result


def _event_subtrees(spots, steps, events, strike, log_up, p, discount, sign, american):
    if not events:
        return _plain_subtrees(spots, steps, strike, log_up, p, discount, sign,
                               american, no_early_call=(discount <= 1))
    first_step, cash = events[0]
    node_prices = spots[:, None]*_multipliers(first_step, log_up)
    # Limited liability prevents negative equity for large cash payments.
    # This floor is inactive for the provided group-12 test inputs.
    after_payment = np.maximum(node_prices-cash, 0.)
    later = tuple((when-first_step, amount) for when, amount in events[1:])
    values = _event_subtrees(after_payment.ravel(), steps-first_step, later,
                             strike, log_up, p, discount, sign, american).reshape(node_prices.shape)
    if american:
        np.maximum(values, np.maximum(sign*(node_prices-strike), 0.), out=values)
    for step in range(first_step-1, -1, -1):
        values = discount*((1-p)*values[:, :-1]+p*values[:, 1:])
        if american:
            intrinsic = np.maximum(sign*(spots[:, None]*_multipliers(step, log_up)-strike), 0.)
            np.maximum(values, intrinsic, out=values)
    return values[:, 0]


def cash_dividend_price(s, k, t, r, sigma, n, dividends: Iterable[tuple[int, float]] = (),
                        is_call=True, american=True):
    """CRR price with discrete cash dividends on exact integer tree steps.

    Dividends contains (grid_step, cash_amount) pairs strictly inside the tree,
    not calendar times. Unsorted payments are sorted, and simultaneous payments
    combined. Cash must be nonnegative. Post-dividend equity is max(S-cash, 0).
    Between payments cost of carry equals r. This nonrecombining model can
    grow rapidly with many payments; batching limits terminal temporary memory.
    """
    spots = _tree_inputs(s, k, t, r, r, sigma, n, is_call, american)
    events = []
    for step, cash in dividends:
        if (not isinstance(step, (int, np.integer)) or isinstance(step, (bool, np.bool_))
                or not 0 < step < n or not np.isfinite(cash) or cash < 0):
            raise ValueError("Dividends require integer steps in (0,n) and finite nonnegative cash")
        if cash:
            events.append((int(step), float(cash)))
    events.sort()
    combined = []
    for step, cash in events:
        if combined and combined[-1][0] == step:
            combined[-1] = (step, combined[-1][1]+cash)
        else:
            combined.append((step, cash))
    sign = 1 if is_call else -1
    if t == 0:
        if combined:
            raise ValueError("A zero-maturity option cannot have future dividends")
        return _result(np.maximum(sign*(spots-k), 0.))
    log_up, p, discount = _tree_parameters(t, r, r, sigma, n)
    answer = _event_subtrees(spots.reshape(-1), n, tuple(combined), k, log_up, p,
                             discount, sign, bool(american)).reshape(spots.shape)
    return _result(answer)


def _option_rows(data_dir, filename):
    frame = pd.read_csv(Path(data_dir)/filename).dropna(subset=["ID"])
    if not frame["Option Type"].isin(["Call", "Put"]).all():
        raise ValueError("Option Type must be Call or Put")
    return frame


def build_cases(data_dir, *, simulations=100000, seed=545):
    """Compute deterministic cases 12.1--12.3 from input CSVs only.

    Simulations/seed are accepted for the common runner interface; option
    results are deterministic and use neither argument.
    """
    continuous = _option_rows(data_dir, "test12_1.csv")
    columns = ["ID", "Value", "Delta", "Gamma", "Vega", "Rho", "Theta"]
    european, american = [], []
    for _, row in continuous.iterrows():
        if row.DayPerYear <= 0:
            raise ValueError("DayPerYear must be positive")
        parameters = (row.Underlying, row.Strike, row.DaysToMaturity/row.DayPerYear,
                      row.RiskFreeRate, row.RiskFreeRate-row.DividendRate, row.ImpliedVol)
        is_call = row["Option Type"] == "Call"
        for records, computed in ((european, greeks(*parameters, is_call=is_call)),
                                   (american, tree_greeks(*parameters, n=500, is_call=is_call))):
            records.append([int(row.ID)]+[computed[name.lower()] for name in columns[1:]])
    cash = _option_rows(data_dir, "test12_3.csv")
    discrete = []
    for _, row in cash.iterrows():
        if row.DayPerYear <= 0 or row.DaysToMaturity != int(row.DaysToMaturity):
            raise ValueError("Cash-dividend cases require positive DayPerYear and integer maturity days")
        n = 2*int(row.DaysToMaturity)
        dates = [int(part.strip()) for part in str(row.DividendDates).split(",")]
        amounts = [float(part.strip()) for part in str(row.DividendAmts).split(",")]
        if len(dates) != len(amounts):
            raise ValueError("Dividend dates and amounts must have matching lengths")
        value = cash_dividend_price(
            row.Underlying, row.Strike, row.DaysToMaturity/row.DayPerYear,
            row.RiskFreeRate, row.ImpliedVol, n,
            [(2*day, amount) for day, amount in zip(dates, amounts)],
            is_call=row["Option Type"] == "Call")
        discrete.append([int(row.ID), value])
    cases = {"12.1": pd.DataFrame(european, columns=columns),
             "12.2": pd.DataFrame(american, columns=columns),
             "12.3": pd.DataFrame(discrete, columns=["ID", "Value"])}
    diagnostics = {
        "european_greeks": {"rho": "dV/dr with dividend yield q=r-b fixed",
                            "theta": "calendar decay -dV/dT per year",
                            "vega": "dV/dsigma per absolute volatility unit"},
        "american_greeks": {"steps": 500, "rho": "dV/dr holding cost of carry b fixed",
                            "theta": "maturity sensitivity +dV/dT per year",
                            "gamma_spot_bump": 1.5,
                            "first_derivative_method": "central, eps^(1/3)*max(abs(parameter),1)"},
        "cash_dividends": {"steps_per_day": 2, "dates": "exact integer grid steps = 2*dividend day",
                           "geometry": "nonrecombining across cash payments; exercise before payment",
                           "equity_floor": "max(S-cash,0); inactive for supplied inputs"},
        "deterministic": True,
    }
    return cases, diagnostics
