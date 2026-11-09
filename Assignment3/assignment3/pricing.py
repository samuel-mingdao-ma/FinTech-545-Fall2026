"""Own generalized Black-Scholes-Merton prices and analytical Greeks.

Time is in years. ``b`` is cost of carry: r-q for a dividend-paying stock.
Vega is dollars per absolute volatility unit, not per percentage point.
Theta is calendar-time decay, -dV/dT, in dollars per model year.
"""
import numpy as np
from scipy import optimize, special


def _inputs(s, k, t, r, b, sigma):
    arrays = np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in (s, k, t, r, b, sigma)])
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("Finite pricing inputs required")
    s, k, t, r, b, sigma = arrays
    if (s < 0).any() or (k <= 0).any() or (t < 0).any() or (sigma < 0).any():
        raise ValueError("Require S >= 0, K > 0, T >= 0 and sigma >= 0")
    return arrays


def _result(x):
    return float(x) if np.ndim(x) == 0 else x


def price(s, k, t, r, b, sigma, is_call=True):
    """Problem 1/3: vectorized European GBSM, with exact expiry/zero-vol limits."""
    s, k, t, r, b, sigma = _inputs(s, k, t, r, b, sigma)
    sign = np.where(np.asarray(is_call, dtype=bool), 1., -1.)
    carry, discount = np.exp((b-r)*t), np.exp(-r*t)
    active = (t > 0) & (sigma > 0) & (s > 0)
    vol_time = np.where(active, sigma*np.sqrt(t), 1.)
    d1 = (np.log(np.maximum(s, 1e-300)/k)+(b+.5*sigma*sigma)*t)/vol_time
    d2 = d1-vol_time
    regular = sign*(s*carry*special.ndtr(sign*d1)-k*discount*special.ndtr(sign*d2))
    deterministic = discount*np.maximum(sign*(s*np.exp(b*t)-k), 0.)
    return _result(np.where(active, regular, deterministic))


def greeks(s, k, t, r, b, sigma, is_call=True):
    """Own closed-form delta, gamma, theta and vega, holding b/r independently fixed."""
    s, k, t, r, b, sigma = _inputs(s, k, t, r, b, sigma)
    if (s <= 0).any() or (t <= 0).any() or (sigma <= 0).any():
        raise ValueError("Greeks require positive S, T, and sigma")
    sign = np.where(np.asarray(is_call, dtype=bool), 1., -1.)
    root_t = np.sqrt(t)
    d1 = (np.log(s/k)+(b+.5*sigma*sigma)*t)/(sigma*root_t)
    d2 = d1-sigma*root_t
    carry, discount = np.exp((b-r)*t), np.exp(-r*t)
    density = np.exp(-.5*d1*d1)/np.sqrt(2*np.pi)
    delta = sign*carry*special.ndtr(sign*d1)
    gamma = carry*density/(s*sigma*root_t)
    vega = s*carry*density*root_t
    theta = (-s*carry*density*sigma/(2*root_t)
             -sign*(b-r)*s*carry*special.ndtr(sign*d1)
             -sign*r*k*discount*special.ndtr(sign*d2))
    return {name: _result(value) for name, value in dict(
        price=price(s, k, t, r, b, sigma, is_call), delta=delta, gamma=gamma,
        theta=theta, vega=vega,
    ).items()}


def price_bounds(s, k, t, r, b, is_call=True):
    """European no-arbitrage price bounds under continuous carry."""
    carry_s, discounted_k = s*np.exp((b-r)*t), k*np.exp(-r*t)
    if is_call:
        return max(carry_s-discounted_k, 0.), carry_s
    return max(discounted_k-carry_s, 0.), discounted_k


def implied_volatility(quote, s, k, t, r, b, is_call=True):
    """Invert a scalar quote; preserve infeasible quotes as missing with a reason."""
    if not np.isfinite(quote) or t <= 0 or quote < 0:
        return None, "invalid_quote_or_maturity"
    lower, upper = price_bounds(s, k, t, r, b, is_call)
    tolerance = 1e-10*max(1., s, k)
    if quote < lower-tolerance:
        return None, "below_no_arbitrage_lower_bound"
    if quote > upper+tolerance:
        return None, "above_no_arbitrage_upper_bound"
    if abs(quote-lower) <= tolerance:
        return 0., "zero_volatility_boundary"
    if quote >= upper-tolerance:
        return None, "no_finite_volatility_at_upper_bound"
    objective = lambda volatility: price(s, k, t, r, b, volatility, is_call)-quote
    high = .5
    while objective(high) < 0 and high < 128:
        high *= 2
    if objective(high) < 0:
        return None, "root_bracket_not_found"
    root = optimize.brentq(objective, 0., high, xtol=1e-13, rtol=1e-13, maxiter=200)
    return float(root), "ok"
