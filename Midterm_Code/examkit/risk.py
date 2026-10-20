"""Numerical routines; observations are rows and variables are columns."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats


def _vector(values, minimum=1):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < minimum or not np.isfinite(x).all():
        raise ValueError(f"Expected at least {minimum} finite observations in one dimension")
    return x


def _matrix(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite matrix")
    return x


def _symmetric(values):
    a = _matrix(values)
    if a.shape[0] != a.shape[1] or not np.allclose(a, a.T, rtol=1e-12, atol=1e-12):
        raise ValueError("Expected a symmetric square matrix")
    return (a+a.T)/2


def information_criteria(ll, k, n):
    if (not np.isfinite(ll) or not isinstance(k, (int, np.integer)) or k < 0
            or not isinstance(n, (int, np.integer)) or n <= k+1):
        raise ValueError("AICc requires finite likelihood, integer k >= 0, and integer n > k+1")
    return dict(aicc=float(-2*ll+2*k+2*k*(k+1)/(n-k-1)), bic=float(-2*ll+k*np.log(n)))


@dataclass
class Fit:
    name: str
    distribution: object
    parameters: dict
    ll: float
    aicc: float
    bic: float


def fit_distribution(values, family, normal_sample_std=False):
    """Fit the explicitly requested family; no family comparison or selection."""
    x = _vector(values, minimum=6)
    if x.std() <= 0:
        raise ValueError("Cannot fit a constant series")
    if family == "normal":
        mu, scale = x.mean(), x.std(ddof=int(normal_sample_std))
        d = stats.norm(loc=mu, scale=scale)
        parameters, count = dict(mu=mu, sigma=scale), 2
    elif family == "t":
        nu, mu, scale = stats.t.fit(x)
        if nu <= 0:
            raise RuntimeError("Nonpositive fitted t degrees of freedom")
        d = stats.t(nu, loc=mu, scale=scale)
        parameters, count = dict(mu=mu, scale=scale, nu=nu), 3
    elif family == "nig":
        a, b, mu, scale = stats.norminvgauss.fit(x)
        if a <= abs(b):
            raise RuntimeError("Invalid fitted NIG shape")
        d = stats.norminvgauss(a, b, loc=mu, scale=scale)
        parameters, count = dict(mu=mu, alpha=a/scale, beta=b/scale, delta=scale), 4
    else:
        raise ValueError("Family must be normal, t, or nig")
    if scale <= 0 or not np.isfinite(list(parameters.values())).all():
        raise RuntimeError("Nonfinite fitted parameters or nonpositive scale")
    ll = float(d.logpdf(x).sum())
    return Fit(family, d, parameters, ll, **information_criteria(ll, count, len(x)))


def historical_risk(values, alpha=.05, convention="course"):
    """VaR/ES of P&L, using positive losses and the course order statistics."""
    x = np.sort(_vector(values))
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if convention == "course":
        lo = max(1, int(np.floor(len(x)*alpha)))
        hi = max(1, int(np.ceil(len(x)*alpha)))
        q = (x[lo-1]+x[hi-1])/2
        es = -x[x <= q].mean()
    elif convention == "linear":
        q = np.quantile(x, alpha)
        mass = len(x)*alpha
        whole = int(np.floor(mass))
        total = x[:whole].sum()+(mass-whole)*x[min(whole, len(x)-1)]
        es = -total/mass
    else:
        raise ValueError("Convention must be course or linear")
    return dict(var=float(-q), es=float(es))


def parametric_risk(fit, alpha=.05):
    """Marginal return VaR/ES, relative to zero."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    d = fit.distribution
    q = d.ppf(alpha)
    if fit.name == "normal":
        es = -d.mean()+d.std()*stats.norm.pdf(stats.norm.ppf(alpha))/alpha
    elif fit.name == "t":
        nu, mu, scale = (fit.parameters[k] for k in ("nu", "mu", "scale"))
        if nu <= 1:
            es = np.inf
        else:
            z = stats.t.ppf(alpha, nu)
            es = -mu+scale*(nu+z*z)/(nu-1)*stats.t.pdf(z, nu)/alpha
    elif fit.name == "nig":
        es = -d.expect(lb=-np.inf, ub=q)/alpha
    else:
        raise ValueError("Unsupported fitted family")
    return dict(var=float(-q), es=float(es))


def returns(prices, demean=False):
    """Arithmetic returns; pass asset-price columns only."""
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("prices must be a DataFrame")
    p = prices.astype(float)
    if len(p) < 2 or p.shape[1] < 1 or not np.isfinite(p.to_numpy()).all() or (p <= 0).any().any():
        raise ValueError("Prices must be finite, positive, and complete")
    r = p.pct_change(fill_method=None).iloc[1:]
    return r-r.mean() if demean else r


def fit_ols(y, x):
    """OLS with intercept, coefficient standard errors, and residuals."""
    x, y = np.asarray(x, dtype=float), _vector(y)
    if x.ndim == 1:
        x = x[:, None]
    x = _matrix(x)
    if len(x) != len(y) or len(y) <= x.shape[1]+1:
        raise ValueError("Regression needs matching lengths and positive residual degrees of freedom")
    design = np.column_stack([np.ones(len(x)), x])
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < design.shape[1]:
        raise ValueError("Rank deficient regression")
    residuals = y-design @ beta
    variance = residuals @ residuals/(len(y)-len(beta))
    se = np.sqrt(np.diag(np.linalg.inv(design.T @ design))*variance)
    return dict(coefficients=beta, standard_errors=se, residuals=residuals, residual_std=np.sqrt(variance))


def correlation(values):
    a = _symmetric(values)
    if np.any(np.diag(a) <= 0):
        raise ValueError("Positive diagonal variances required")
    sd = np.sqrt(np.diag(a))
    return a/np.outer(sd, sd)


def _positive(a):
    eigenvalues, eigenvectors = np.linalg.eigh((a+a.T)/2)
    return (eigenvectors*np.maximum(eigenvalues, 0)) @ eigenvectors.T


def repair_psd(values, method="higham", tol=1e-10, max_iter=1000):
    """Correlation projection with covariance-diagonal restoration."""
    a = _symmetric(values)
    if (not np.isfinite(tol) or tol <= 0 or not isinstance(max_iter, (int, np.integer))
            or isinstance(max_iter, bool) or max_iter < 1):
        raise ValueError("Positive tolerance and iteration limit required")
    r = correlation(a)
    sd = np.sqrt(np.diag(a))
    if method == "near":
        z = correlation(_positive(r))
    elif method == "higham":
        z, correction = r.copy(), np.zeros_like(r)
        for _ in range(max_iter):
            old = z.copy()
            residual = z-correction
            projection = _positive(residual)
            correction = projection-residual
            z = projection.copy()
            np.fill_diagonal(z, 1)
            if np.linalg.norm(z-old) < tol and np.linalg.eigvalsh(z).min() > -tol:
                break
        else:
            raise RuntimeError("Higham iteration did not converge")
    else:
        raise ValueError("Method must be higham or near")
    z = correlation(_positive(z))
    return z*np.outer(sd, sd)


def simulate_normal(values, n=100000, seed=545):
    a = _symmetric(values)
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    eigenvalues, eigenvectors = np.linalg.eigh(a)
    if eigenvalues.min() < -1e-8*max(1, abs(eigenvalues).max()):
        raise ValueError("Covariance is not PSD")
    factor = eigenvectors*np.sqrt(np.maximum(eigenvalues, 0))
    return np.random.default_rng(seed).standard_normal((n, len(a))) @ factor.T


def kendall_correlation(values):
    x = _matrix(values)
    if len(x) < 2 or x.shape[1] < 2 or np.any(x.std(axis=0) <= 0):
        raise ValueError("Kendall correlation needs at least two nonconstant series")
    r = np.eye(x.shape[1])
    for i in range(x.shape[1]):
        for j in range(i):
            r[i, j] = r[j, i] = np.sin(np.pi*stats.kendalltau(x[:, i], x[:, j]).statistic/2)
    if np.linalg.eigvalsh(r).min() <= 1e-10:
        r = repair_psd(r)
        # Density evaluation requires a nonsingular matrix.
        ridge = max(0., 1e-10-np.linalg.eigvalsh(r).min())
        r = (r+ridge*np.eye(len(r)))/(1+ridge)
    return r


def copula_loglik(uniforms, values, nu=None, per_observation=False):
    u, r = _matrix(uniforms), _symmetric(values)
    if u.shape[1] != len(r) or np.any(u < 0) or np.any(u > 1):
        raise ValueError("Uniform data must be in [0,1] and match the correlation dimension")
    if not np.allclose(np.diag(r), 1., rtol=0, atol=1e-10) or np.linalg.eigvalsh(r).min() <= 0:
        raise ValueError("Copula density requires a positive definite unit-diagonal correlation matrix")
    u = np.clip(u, 1e-12, 1-1e-12)
    if nu is None:
        z = stats.norm.ppf(u)
        ll = stats.multivariate_normal.logpdf(z, cov=r)-stats.norm.logpdf(z).sum(axis=1)
    else:
        if not np.isfinite(nu) or nu <= 0:
            raise ValueError("Positive finite degrees of freedom required")
        z = stats.t.ppf(u, nu)
        ll = stats.multivariate_t.logpdf(z, shape=r, df=nu)-stats.t.logpdf(z, nu).sum(axis=1)
    return ll if per_observation else float(ll.sum())


def profile_nu(objective):
    """Fit a t degree-of-freedom parameter over the course reciprocal-nu grid."""
    grid = np.linspace(.01, .49, 200)
    values = np.array([objective(1/theta) for theta in grid])
    if not np.isfinite(values).any():
        raise RuntimeError("No finite t likelihood on the degree-of-freedom grid")
    j = int(np.argmax(np.where(np.isfinite(values), values, -np.inf)))
    result = optimize.minimize_scalar(
        lambda theta: -objective(1/theta),
        bounds=(grid[max(j-1, 0)], grid[min(j+1, len(grid)-1)]), method="bounded",
    )
    nu, ll = 1/grid[j], values[j]
    if result.success and np.isfinite(result.fun) and -result.fun > ll:
        nu, ll = 1/result.x, -result.fun
    return float(nu), float(ll)


def fit_copulas(values, fits):
    """Return both candidate copula fits without choosing a family."""
    x = _matrix(values)
    if len(fits) != x.shape[1] or len(x) <= 2:
        raise ValueError("Supply one fitted margin per variable and at least three observations")
    u = np.column_stack([f.distribution.cdf(x[:, j]) for j, f in enumerate(fits)])
    r = kendall_correlation(x)
    nu, ll = profile_nu(lambda n: copula_loglik(u, r, n))
    gaussian_ll = copula_loglik(u, r)
    return dict(
        r=r, u=u, nu=nu,
        gaussian=dict(ll=gaussian_ll, **information_criteria(gaussian_ll, 0, len(x))),
        t=dict(ll=ll, **information_criteria(ll, 1, len(x))),
    )
