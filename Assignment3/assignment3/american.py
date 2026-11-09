"""Self-written CRR trees with exercise and discrete cash dividends.

The tree recombines only between cash-dividend dates. Each distinct node at
a dividend creates a distinct post-dividend subtree. Batching those origins
changes storage and execution, not the non-recombining model geometry.
"""
from functools import lru_cache

import numpy as np
from scipy import optimize, stats


@lru_cache(maxsize=128)
def _terminal_multipliers(steps, log_up):
    return np.exp((2*np.arange(steps+1)-steps)*log_up)


def _plain_subtrees(spots, steps, strike, log_up, p, discount, sign, american):
    """Simultaneously value ordinary CRR subtrees with distinct origins."""
    spots = np.asarray(spots, float).reshape(-1)
    terminal = spots[:, None]*_terminal_multipliers(steps, log_up)
    values = np.maximum(sign*(terminal-strike), 0.)
    if not american:
        probabilities = stats.binom.pmf(np.arange(steps+1), steps, p)
        return discount**steps*(values @ probabilities)
    for step in range(steps-1, -1, -1):
        values = discount*((1-p)*values[:, :-1]+p*values[:, 1:])
        intrinsic = np.maximum(sign*(spots[:, None]*_terminal_multipliers(step, log_up)-strike), 0.)
        np.maximum(values, intrinsic, out=values)
    return values[:, 0]


def _event_subtrees(spots, steps, events, strike, log_up, p, discount, sign, american):
    if not events:
        return _plain_subtrees(spots, steps, strike, log_up, p, discount, sign, american)
    first_step, amount = events[0]
    node_prices = spots[:, None]*_terminal_multipliers(first_step, log_up)
    # Limited liability at negligible-probability extreme low-price nodes.
    post_dividend = np.maximum(node_prices-amount, 0.)
    later = tuple((when-first_step, cash) for when, cash in events[1:])
    values = _event_subtrees(post_dividend.ravel(), steps-first_step, later,
                             strike, log_up, p, discount, sign, american).reshape(node_prices.shape)
    if american:
        np.maximum(values, np.maximum(sign*(node_prices-strike), 0.), out=values)
    for step in range(first_step-1, -1, -1):
        values = discount*((1-p)*values[:, :-1]+p*values[:, 1:])
        if american:
            intrinsic = np.maximum(sign*(spots[:, None]*_terminal_multipliers(step, log_up)-strike), 0.)
            np.maximum(values, intrinsic, out=values)
    return values[:, 0]


def tree_price(s, k, t, r, sigma, n, is_call=True, american=True, dividends=()):
    """CRR value with discrete cash dividends on exact grid dates.

    ``dividends`` is a sequence of ``(time_in_years, cash_amount)`` pairs.
    Exercise on dividend dates occurs just BEFORE the dividend is paid.
    Cash dividends are capped by the available stock equity at extremely low
    nodes (the stock is absorbing at zero); this is irrelevant to quoted
    precision for the assignment's inputs but keeps stock prices nonnegative.
    Scalar or array starting spots are supported. No financial pricing package
    is called.
    """
    spots = np.asarray(s, float)
    scalar = spots.ndim == 0
    if (not np.isfinite(spots).all() or np.any(spots < 0)
            or not all(np.isfinite(v) for v in (k, t, r, sigma))
            or k <= 0 or t < 0 or sigma <= 0
            or not isinstance(is_call, (bool, np.bool_)) or not isinstance(american, (bool, np.bool_))
            or not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1):
        raise ValueError("Finite nonnegative spots/time, positive strike/volatility and integer steps required")
    if t == 0:
        answer = np.maximum((1 if is_call else -1)*(spots-k), 0.)
        return float(answer) if scalar else answer
    dt = t/n
    log_up = sigma*np.sqrt(dt)
    up, down = np.exp(log_up), np.exp(-log_up)
    p = (np.exp(r*dt)-down)/(up-down)
    if not 0 < p < 1:
        raise ValueError("Risk-neutral probability must be in (0,1); refine the grid")
    events = []
    for when, cash in dividends:
        if not np.isfinite(when) or not np.isfinite(cash) or cash < 0 or not 0 < when < t:
            raise ValueError("Dividends need finite nonnegative cash and dates strictly before expiry")
        grid_step = int(round(when/dt))
        if not 0 < grid_step < n or not np.isclose(grid_step*dt, when, rtol=0, atol=1e-11):
            raise ValueError("A dividend date does not land exactly on the tree grid")
        if cash:
            events.append((grid_step, float(cash)))
    events.sort()
    combined = []
    for step, cash in events:
        if combined and combined[-1][0] == step:
            combined[-1] = (step, combined[-1][1]+cash)
        else:
            combined.append((step, cash))
    answer = _event_subtrees(spots.reshape(-1), n, tuple(combined), k, log_up, p,
                             np.exp(-r*dt), 1 if is_call else -1, bool(american)).reshape(spots.shape)
    return float(answer) if scalar else answer


def exercise_boundary(k, remaining_t, r, sigma, remaining_steps, dividend):
    """Call's pre-dividend exercise/continuation indifference stock price.

    The dividend under consideration is the last dividend. Continuation is a
    no-dividend American call after payment, evaluated on the same CRR grid.
    """
    if dividend <= k*(1-np.exp(-r*remaining_t)):
        return None

    def difference(spot):
        continuation = tree_price(max(spot-dividend, 0.), k, remaining_t, r, sigma,
                                  remaining_steps, True, True)
        return max(spot-k, 0.)-continuation

    lower, upper = k, 2*k
    while difference(upper) <= 0:
        upper *= 2
        if upper > 1e6*k:
            return None
    return float(optimize.brentq(difference, lower, upper, xtol=1e-11))
