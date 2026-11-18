"""Loss VaR/ES and Monte Carlo uncertainty using the instructor's conventions."""

import numpy as np
from scipy import stats


def _alpha(alpha):
    if np.ndim(alpha) != 0 or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("The lower-tail probability must be between zero and one")


def historical_risk(pnl, alpha=0.05):
    """Course empirical VaR and threshold-tail ES, returned as positive losses.

    The threshold averages the floor(n*alpha) and ceil(n*alpha) order
    statistics using one-based indexing. Below the first order statistic,
    both indices are clamped to one. ES includes observations equal to the
    threshold. The course convention can give negative VaR for a gain-only
    sample, so risk values are not artificially floored at zero.
    """
    _alpha(alpha)
    sample = np.asarray(pnl, dtype=float)
    if sample.ndim != 1 or sample.size == 0 or not np.isfinite(sample).all():
        raise ValueError("A nonempty finite one-dimensional P&L sample is required")
    ordered = np.sort(sample)
    mass = sample.size * alpha
    low = max(1, int(np.floor(mass)))
    high = max(1, int(np.ceil(mass)))
    left, right = ordered[low - 1], ordered[high - 1]
    threshold = (left + 0.5 * (right - left) if np.signbit(left) == np.signbit(right)
                 else 0.5 * left + 0.5 * right)
    tail = ordered[ordered <= threshold]
    magnitude = np.max(np.abs(tail))
    tail_mean = 0.0 if magnitude == 0 else float((tail / magnitude).mean() * magnitude)
    return {"var": float(-threshold), "es": -tail_mean}


def parametric_risk(distribution, alpha=0.05):
    """Closed-form Normal or location-scale t VaR and lower-tail ES."""
    _alpha(alpha)
    family = distribution.dist.name
    if family not in {"norm", "t"}:
        raise ValueError("Closed-form risk supports Normal and Student t distributions")
    args, kwargs = distribution.args, distribution.kwds
    offset = 1 if family == "t" else 0
    mean = kwargs.get("loc", args[offset] if len(args) > offset else 0.0)
    scale = kwargs.get("scale", args[offset + 1] if len(args) > offset + 1 else 1.0)
    if (np.ndim(mean) != 0 or np.ndim(scale) != 0
            or not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0):
        raise ValueError("Distribution location and scale must be finite scalars with scale > 0")
    mean, scale = float(mean), float(scale)
    if family == "t":
        nu = args[0] if args else kwargs["df"]
        if np.ndim(nu) != 0 or np.isnan(nu) or nu <= 1:
            raise ValueError("Student t ES requires nu > 1 (including the Normal limit)")
        nu = float(nu)
    if family == "norm" or np.isposinf(nu):
        z = stats.norm.ppf(alpha)
        quantile = mean + scale * z
        tail_factor = np.exp(stats.norm.logpdf(z) - np.log(alpha))
        es = -mean + scale * tail_factor
    else:
        z = stats.t.ppf(alpha, nu)
        quantile = mean + scale * z
        # Combine products in log space so a small scale and tail density do
        # not underflow before division by the small tail probability.
        with np.errstate(divide="ignore"):
            log_second_moment = np.logaddexp(np.log(nu), 2 * np.log(abs(z)))
        log_tail_factor = (log_second_moment - np.log(nu - 1)
                           + stats.t.logpdf(z, nu) - np.log(alpha))
        es = -mean + np.exp(np.log(scale) + log_tail_factor)
    return {"var": float(-quantile), "es": float(es)}


def batch_standard_errors(pnl, alpha=0.05, batches=20):
    """Standard errors from independent equal-sized scenario batches."""
    sample = np.asarray(pnl, dtype=float)
    if (sample.ndim != 1 or not np.isfinite(sample).all()
            or not isinstance(batches, int) or batches < 2
            or len(sample) < batches * 20):
        raise ValueError("At least 20 finite scenarios per batch and two batches are required")
    estimates = [historical_risk(part, alpha) for part in np.array_split(sample, batches)]
    return {
        metric: float(np.std([estimate[metric] for estimate in estimates], ddof=1) / np.sqrt(batches))
        for metric in ("var", "es")
    }
