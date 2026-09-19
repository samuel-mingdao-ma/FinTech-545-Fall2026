"""PSD repair, factorization and Monte Carlo sampling."""
import numpy as np


def _matrix(a):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) == 0:
        raise ValueError("Expected a nonempty square matrix")
    if not np.isfinite(a).all() or not np.allclose(a, a.T, atol=1e-12, rtol=1e-10):
        raise ValueError("Expected a finite symmetric matrix")
    return (a + a.T) / 2


def _standardize(a):
    a = _matrix(a)
    if np.any(np.diag(a) <= 0):
        raise ValueError("Matrix repair requires positive diagonal entries")
    sd = np.sqrt(np.diag(a))
    return a / np.outer(sd, sd), sd


def _positive(a, epsilon=0):
    vals, vecs = np.linalg.eigh(a)
    return (vecs * np.maximum(vals, epsilon)) @ vecs.T


def near_psd(a, epsilon=0.0):
    """Clip correlation eigenvalues, renormalize diagonal, restore variances."""
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    c, sd = _standardize(a)
    c = _positive(c, epsilon)
    scale = np.sqrt(np.diag(c))
    c /= np.outer(scale, scale)
    return c * np.outer(sd, sd)


def higham_psd(a, *, tol=1e-9, max_iter=1000):
    """Higham alternating projections with Dykstra correction and unit weights.

    Work in correlation space and preserve the input variances. Stop only
    after both distance convergence and numerical positive semidefiniteness.
    """
    if tol <= 0 or max_iter < 1:
        raise ValueError("tol and max_iter must be positive")
    original, sd = _standardize(a)
    y, correction = original.copy(), np.zeros_like(original)
    previous = np.inf
    for _ in range(max_iter):
        r = y - correction
        x = _positive(r)
        correction = x - r
        y = x.copy()
        np.fill_diagonal(y, 1.0)
        distance = np.sum((y - original) ** 2)
        if abs(distance - previous) < tol and np.linalg.eigvalsh(y)[0] > -tol:
            return y * np.outer(sd, sd)
        previous = distance
    raise RuntimeError("Higham iteration did not converge")


def chol_psd(a, tol=1e-8):
    """Lower triangular root L with L @ L.T = a, including singular PSD."""
    a = _matrix(a)
    if tol <= 0:
        raise ValueError("tol must be positive")
    if np.linalg.eigvalsh(a)[0] < -tol:
        raise ValueError("Matrix is not positive semidefinite")
    root = np.zeros_like(a)
    for j in range(len(a)):
        pivot = a[j, j] - root[j, :j] @ root[j, :j]
        if pivot < -tol:
            raise ValueError("Negative Cholesky pivot")
        # Tiny positive/negative residuals at a zero pivot are rounding error.
        if abs(pivot) <= np.finfo(float).eps * max(1.0, abs(a[j, j])) * len(a) or pivot < 0:
            residual = a[j+1:, j] - root[j+1:, :j] @ root[j, :j]
            if np.any(np.abs(residual) > tol):
                raise ValueError("Inconsistent zero Cholesky pivot")
            continue
        root[j, j] = np.sqrt(pivot)
        root[j+1:, j] = (a[j+1:, j] - root[j+1:, :j] @ root[j, :j]) / root[j, j]
    return root


def _sampling_inputs(a, n, mean):
    a = _matrix(a)
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n must be an integer >= 2")
    mean = np.zeros(len(a)) if mean is None else np.asarray(mean, dtype=float)
    if mean.shape != (len(a),) or not np.isfinite(mean).all():
        raise ValueError("mean must match the covariance dimension")
    return a, mean


def simulate_normal(a, n=100000, *, mean=None, seed=1234, repair=near_psd):
    """Return n rows from N(mean, covariance); repair indefinite inputs."""
    a, mean = _sampling_inputs(a, n, mean)
    try:
        root = chol_psd(a)
    except ValueError:
        if repair is None:
            raise
        root = chol_psd(repair(a))
    return np.random.default_rng(seed).standard_normal((n, len(a))) @ root.T + mean


def _pca_root(a, explained):
    a = _matrix(a)
    if not 0 < explained <= 1:
        raise ValueError("explained must be in (0, 1]")
    vals, vecs = np.linalg.eigh(a)
    if vals[0] < -1e-8:
        raise ValueError("PCA input must be PSD")
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    keep = vals >= 1e-8  # Course numerical zero cutoff.
    vals, vecs = vals[keep], vecs[:, keep]
    if not len(vals):
        return np.zeros((len(a), 0))
    k = min(len(vals), np.searchsorted(np.cumsum(vals) / vals.sum(), explained) + 1)
    return vecs[:, :k] * np.sqrt(vals[:k])


def pca_target(a, explained=0.99):
    """Analytic covariance after truncation, distinct from sampling error."""
    root = _pca_root(a, explained)
    return root @ root.T


def simulate_pca(a, n=100000, *, explained=0.99, mean=None, seed=1234):
    a, mean = _sampling_inputs(a, n, mean)
    root = _pca_root(a, explained)
    return np.random.default_rng(seed).standard_normal((n, root.shape[1])) @ root.T + mean
