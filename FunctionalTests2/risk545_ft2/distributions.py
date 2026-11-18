"""Normal moment fitting and constrained location-scale Student t likelihood."""

import numpy as np
from scipy import optimize, stats


def _sample(values):
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or sample.size < 2 or not np.isfinite(sample).all():
        raise ValueError("A finite one-dimensional sample with at least two values is required")
    if np.std(sample, ddof=1) <= 0:
        raise ValueError("A nonconstant sample is required")
    return sample


def fit_normal(values):
    """Course normal fit, using the mean and sample standard deviation (n-1)."""
    sample = _sample(values)
    return stats.norm(loc=float(sample.mean()), scale=float(sample.std(ddof=1)))


def fit_student_t(values):
    """Fit location, scale and nu >= 2.0001 by maximum likelihood.

    Standardizing the optimization follows the reusable routine from the first
    functional-test package. The t scale differs from its standard deviation.
    Multiple initial nu values reduce sensitivity to optimizer initialization.
    """
    sample = _sample(values)
    center, spread = sample.mean(), sample.std(ddof=1)
    standardized = (sample - center) / spread

    def objective(parameters):
        location, log_scale, nu = parameters
        return -float(stats.t.logpdf(
            standardized, nu, loc=location, scale=np.exp(log_scale)
        ).sum())

    candidates = []
    for nu in (3.0, 6.0, 12.0, 30.0):
        initial = [0.0, np.log(np.sqrt((nu - 2) / nu)), nu]
        result = optimize.minimize(
            objective, initial, method="L-BFGS-B",
            bounds=[(None, None), (-20, 20), (2.0001, None)],
            options={"ftol": 1e-13, "gtol": 1e-7, "maxiter": 3000},
        )
        if result.success and np.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("Student t maximum likelihood did not converge")
    location, log_scale, nu = min(candidates, key=lambda item: item.fun).x
    return stats.t(float(nu), loc=float(center + spread * location),
                   scale=float(spread * np.exp(log_scale)))
