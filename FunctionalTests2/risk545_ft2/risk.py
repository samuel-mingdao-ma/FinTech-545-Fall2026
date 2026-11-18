"""Loss VaR/ES and Monte Carlo uncertainty using the instructor's conventions."""

import numpy as np
from scipy import stats


def _alpha(alpha):
    if not np.isfinite(alpha) or not 0 < alpha < 1:
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
    threshold = 0.5 * (ordered[low - 1] + ordered[high - 1])
    return {"var": float(-threshold), "es": float(-ordered[ordered <= threshold].mean())}


def parametric_risk(distribution, alpha=0.05):
    """Closed-form Normal or location-scale t VaR and lower-tail ES."""
    _alpha(alpha)
    family = distribution.dist.name
    quantile = float(distribution.ppf(alpha))
    if family == "norm":
        mean, scale = float(distribution.mean()), float(distribution.std())
        es = -mean + scale * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
    elif family == "t":
        nu = float(distribution.args[0] if distribution.args else distribution.kwds["df"])
        if nu <= 1:
            raise ValueError("Student t ES requires nu > 1")
        args = distribution.args
        mean = float(distribution.kwds.get("loc", args[1] if len(args) > 1 else 0.0))
        scale = float(distribution.kwds.get("scale", args[2] if len(args) > 2 else 1.0))
        z = stats.t.ppf(alpha, nu)
        es = -mean + scale * (nu + z * z) / (nu - 1) * stats.t.pdf(z, nu) / alpha
    else:
        raise ValueError("Closed-form risk supports Normal and Student t distributions")
    return {"var": -quantile, "es": float(es)}


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
