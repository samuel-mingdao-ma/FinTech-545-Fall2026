"""Covariance estimators with observations in rows and variables in columns."""
import numpy as np


def _data(x, allow_missing=False):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or min(x.shape) == 0:
        raise ValueError("Expected a nonempty observations-by-variables matrix")
    if np.isinf(x).any() or (not allow_missing and np.isnan(x).any()):
        raise ValueError("Input contains non-finite observations")
    return x


def correlation(cov):
    """Convert a covariance matrix to correlation (positive variances required)."""
    cov = _data(cov)
    if cov.shape[0] != cov.shape[1] or not np.allclose(cov, cov.T):
        raise ValueError("Expected a symmetric square covariance matrix")
    sd = np.sqrt(np.diag(cov)) if np.all(np.diag(cov) > 0) else None
    if sd is None:
        raise ValueError("Correlation requires positive variances")
    return cov / np.outer(sd, sd)


def missing_covariance(x, *, pairwise=False, corr=False):
    """Sample covariance/correlation using listwise or pairwise deletion.

    Covariance divides by n-1. Pairwise means and variances are computed
    separately on the overlapping rows for each pair, including correlation.
    """
    x = _data(x, allow_missing=True)
    if not pairwise:
        complete = x[~np.isnan(x).any(axis=1)]
        if len(complete) < 2:
            raise ValueError("At least two complete observations are required")
        cov = np.atleast_2d(np.cov(complete, rowvar=False, ddof=1))
        return correlation(cov) if corr else cov
    out = np.empty((x.shape[1], x.shape[1]))
    for i in range(x.shape[1]):
        for j in range(i + 1):
            keep = ~np.isnan(x[:, [i, j]]).any(axis=1)
            if keep.sum() < 2:
                raise ValueError(f"Fewer than two overlapping observations for ({i}, {j})")
            pair = x[keep][:, [i, j]]
            cov = np.cov(pair, rowvar=False, ddof=1)
            value = correlation(cov)[0, 1] if corr else cov[0, 1]
            out[i, j] = out[j, i] = value
    return out


def ew_covariance(x, lam=0.97):
    """Normalized exponential weights; last row is newest; weighted centering.

    Uses sum(w * deviations * deviations.T), without a Bessel correction,
    matching the course ewCovar implementation.
    """
    x = _data(x)
    if not 0 < lam <= 1:
        raise ValueError("lambda must be in (0, 1]")
    w = lam ** np.arange(len(x) - 1, -1, -1, dtype=float)
    w /= w.sum()
    centered = x - w @ x
    return (centered * w[:, None]).T @ centered


def mixed_ew_covariance(x, variance_lam=0.97, correlation_lam=0.94):
    """Combine EW variances at one decay with EW correlations at another."""
    sd = np.sqrt(np.diag(ew_covariance(x, variance_lam)))
    return np.outer(sd, sd) * correlation(ew_covariance(x, correlation_lam))
