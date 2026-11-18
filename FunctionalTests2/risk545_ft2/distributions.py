"""Normal moment fitting and constrained location-scale Student t likelihood."""

import numpy as np
from scipy import optimize, stats


def _sample(values):
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or sample.size < 2 or not np.isfinite(sample).all():
        raise ValueError("A finite one-dimensional sample with at least two values is required")
    if np.all(sample == sample[0]):
        raise ValueError("A nonconstant sample is required")
    return sample


def _standardize(sample):
    # Scale before moments: squaring tiny/large observations can otherwise
    # underflow/overflow even when their mean and standard deviation exist.
    magnitude = np.max(np.abs(sample))
    reduced = sample / magnitude
    center, spread = reduced.mean(), reduced.std(ddof=1)
    location, scale = float(center * magnitude), float(spread * magnitude)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The sample standard deviation must be positive and finite")
    return location, scale, (reduced - center) / spread


def fit_normal(values):
    """Course normal fit, using the mean and sample standard deviation (n-1)."""
    sample = _sample(values)
    center, spread, _ = _standardize(sample)
    return stats.norm(loc=center, scale=spread)


def fit_student_t(values):
    """Fit location, scale and nu >= 2.0001 by maximum likelihood.

    Standardizing the optimization follows the reusable routine from the first
    functional-test package. The t scale differs from its standard deviation.
    Multiple starts reduce sensitivity to initialization. Optimizing inverse
    degrees of freedom also includes the Normal limit, represented by nu=inf:
    some samples have no finite maximum-likelihood degrees of freedom.
    """
    sample = _sample(values)
    center, spread, standardized = _standardize(sample)

    def objective(parameters):
        location, log_scale, inverse_nu = parameters
        nu = np.inf if inverse_nu == 0 else 1.0 / inverse_nu
        return -float(stats.t.logpdf(
            standardized, nu, loc=location, scale=np.exp(log_scale)
        ).sum())

    candidates = []
    for nu in (3.0, 6.0, 12.0, 30.0):
        initial = [0.0, np.log(np.sqrt((nu - 2) / nu)), 1.0 / nu]
        result = optimize.minimize(
            objective, initial, method="L-BFGS-B",
            bounds=[(None, None), (-20, 20), (0.0, 1.0 / 2.0001)],
            options={"ftol": 1e-13, "gtol": 1e-7, "maxiter": 3000},
        )
        if result.success and np.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("Student t maximum likelihood did not converge")
    location, log_scale, inverse_nu = min(candidates, key=lambda item: item.fun).x
    nu = np.inf if inverse_nu == 0 else 1.0 / inverse_nu
    if np.isposinf(nu):
        # The boundary model has an exact Normal MLE; do not retain numerical
        # optimizer error in its otherwise analytically known location/scale.
        return stats.t(np.inf, loc=center, scale=spread * np.sqrt((sample.size - 1) / sample.size))
    return stats.t(float(nu), loc=float(center + spread * location),
                   scale=float(spread * np.exp(log_scale)))
