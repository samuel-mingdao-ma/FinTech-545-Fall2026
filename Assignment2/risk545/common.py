"""Shared statistical conventions for FinTech 545 Assignment 2."""

from __future__ import annotations

import numpy as np
from scipy import stats


def corrected_aic(log_likelihood: float, n_parameters: int, n_obs: int) -> float:
    """Return AICc with every estimated model parameter included in ``k``."""

    aic = -2.0 * log_likelihood + 2.0 * n_parameters
    return float(
        aic
        + 2.0 * n_parameters * (n_parameters + 1)
        / (n_obs - n_parameters - 1)
    )


def sample_moments(values: np.ndarray) -> dict[str, float]:
    """Return mean, unbiased variance, corrected skewness, and excess kurtosis."""

    x = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "variance": float(np.var(x, ddof=1)),
        "standard_deviation": float(np.std(x, ddof=1)),
        "skewness": float(stats.skew(x, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(x, fisher=True, bias=False)),
    }


def arithmetic_returns(prices: np.ndarray) -> np.ndarray:
    """Compute simple returns with observations along the first axis."""

    p = np.asarray(prices, dtype=float)
    return p[1:] / p[:-1] - 1.0


def historical_var_es(
    pnl: np.ndarray, alpha: float = 0.05
) -> tuple[float, float]:
    """Return positive-loss historical VaR and ES using the course convention.

    This reproduces ``library/RiskStats.jl``: sort the sample, average the
    observations at ``floor(n*alpha)`` and ``ceil(n*alpha)`` (Julia's 1-based
    indices), and negate that threshold. ES is the negative mean of every P&L
    observation at or below the same threshold, including ties.
    """

    values = np.asarray(pnl, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("pnl must be a non-empty one-dimensional array")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    ordered = np.sort(values)
    lower = max(1, int(np.floor(values.size * alpha))) - 1
    upper = max(1, int(np.ceil(values.size * alpha))) - 1
    threshold = float(0.5 * (ordered[lower] + ordered[upper]))
    var = -threshold
    es = -float(np.mean(values[values <= threshold]))
    return float(var), float(es)


def normal_var(
    mean_pnl: float, standard_deviation_pnl: float, alpha: float = 0.05
) -> float:
    """Return positive-loss Normal VaR."""

    return float(
        -(mean_pnl + stats.norm.ppf(alpha) * standard_deviation_pnl)
    )


def exponential_weights(n_obs: int, decay: float) -> np.ndarray:
    """Return normalized EW weights, oldest to newest."""

    powers = np.arange(n_obs - 1, -1, -1)
    weights = (1.0 - decay) * decay**powers
    return weights / weights.sum()


def effective_sample_size(weights: np.ndarray) -> float:
    """Return 1/sum(w^2) for normalized weights."""

    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return float(1.0 / np.sum(w**2))
