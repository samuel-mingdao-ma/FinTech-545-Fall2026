"""Distribution fitting with the parameter conventions used by FINTECH 545."""
import numpy as np
from scipy import optimize, stats


def _sample(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or np.all(x == x[0]):
        raise ValueError("Expected at least two finite, nonconstant observations")
    return x


def _standardize(sample, *, ddof=1, axis=None):
    # Reduce before squaring so that representable standard deviations do not
    # vanish or overflow merely because the observations use extreme units.
    magnitude = np.max(np.abs(sample), axis=axis)
    reduced = sample / magnitude
    center, spread = reduced.mean(axis=axis), reduced.std(axis=axis, ddof=ddof)
    location, scale = center*magnitude, spread*magnitude
    if not np.isfinite(location).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Sample mean and positive standard deviation must be representable")
    return location, scale, (reduced-center)/spread


def fit_normal(x):
    """Moment fit using sample standard deviation, as in the course reference."""
    x = _sample(x)
    center, spread, _ = _standardize(x)
    return dict(mu=float(center), sigma=float(spread))


def fit_t(x):
    """MLE of location-scale t, constrained to nu >= 2.0001 like the reference.

    sigma is t scale, not standard deviation: sd = sigma*sqrt(nu/(nu-2)).
    The Gaussian boundary is represented by nu=np.inf.
    """
    x = _sample(x)
    center, spread, z = _standardize(x)

    def objective(p):
        nu = np.inf if p[2] == 0 else 1/p[2]
        return -stats.t.logpdf(z, df=nu, loc=p[0], scale=np.exp(p[1])).sum()

    candidates = []
    for nu in (3.0, 6.0, 12.0, 30.0):
        result = optimize.minimize(objective, [0, np.log(np.sqrt((nu-2)/nu)), 1/nu],
                                   method="L-BFGS-B", bounds=[(None, None), (-20, 20), (0, 1/2.0001)],
                                   options={"ftol": 1e-13, "gtol": 1e-7, "maxiter": 3000})
        if result.success and np.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("Student t maximum likelihood did not converge")
    p = min(candidates, key=lambda r: r.fun).x
    if p[2] == 0:
        return dict(mu=float(center), sigma=float(spread*np.sqrt((len(x)-1)/len(x))), nu=float(np.inf))
    return dict(mu=float(center + spread*p[0]), sigma=float(spread*np.exp(p[1])),
                nu=float(1/p[2]))


def fit_t_regression(y, x):
    """MLE y = Alpha + X beta + sigma*T(nu); zero error location, nu=inf allowed."""
    y = _sample(y)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError("X must contain finite observations matching y")
    if np.any(np.all(x == x[0], axis=0)):
        raise ValueError("X must not contain constant predictors")
    xm, xs, standardized_x = _standardize(x, ddof=0, axis=0)
    design = np.column_stack([np.ones(len(y)), standardized_x])
    if np.linalg.matrix_rank(design) < design.shape[1] or len(y) <= design.shape[1]:
        raise ValueError("Regression design must have full column rank and positive residual degrees of freedom")
    ym, ys, z = _standardize(y)
    initial_beta = np.linalg.lstsq(design, z, rcond=None)[0]
    residual_sd = np.std(z-design@initial_beta, ddof=design.shape[1])
    if residual_sd <= np.finfo(float).eps:
        raise ValueError("Degenerate zero-noise regression")

    def objective(p):
        nu = np.inf if p[-1] == 0 else 1/p[-1]
        return -stats.t.logpdf(z-design@p[:-2], df=nu, scale=np.exp(p[-2])).sum()

    candidates = []
    for nu in (3.0, 6.0, 12.0):
        start = np.r_[initial_beta, np.log(residual_sd*np.sqrt((nu-2)/nu)), 1/nu]
        result = optimize.minimize(objective, start, method="L-BFGS-B",
            bounds=[(None, None)]*len(initial_beta)+[(-20, 20), (0, 1/2.0001)],
            options={"ftol": 1e-13, "gtol": 1e-7, "maxiter": 5000})
        if result.success and np.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("Student t regression did not converge")
    p = min(candidates, key=lambda r: r.fun).x
    if p[-1] == 0:
        # At the Gaussian boundary, use the exact least-squares/ML solution.
        p = np.r_[initial_beta, np.log(np.std(z-design@initial_beta, ddof=0)), 0.0]
    beta = (ys/xs)*p[1:-2]
    alpha = ym + ys*p[0] - xm@beta
    result = dict(mu=0.0, sigma=float(ys*np.exp(p[-2])),
                  nu=float(np.inf if p[-1] == 0 else 1/p[-1]), Alpha=float(alpha))
    result.update({f"B{i+1}": float(b) for i, b in enumerate(beta)})
    return result


def aicc(log_likelihood, n, k):
    """Small-sample AIC; count every fitted parameter, including scale and nu."""
    integers = all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in (n, k))
    if not integers or k < 0 or n <= k+1:
        raise ValueError("AICc requires integer n > k+1 and a nonnegative integer k")
    try:
        ll = np.asarray(log_likelihood, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Log likelihood must be a finite scalar") from exc
    if ll.ndim != 0 or not np.isfinite(ll):
        raise ValueError("Log likelihood must be a finite scalar")
    n, k = int(n), int(k)
    return float(-2*float(ll) + 2*k + 2*k*(k+1)/(n-k-1))


def fit_nig_moments(x):
    """Return mu, alpha, beta, delta from the first four sample moments.

    Match Julia: sample variance (n-1), uncorrected skew and excess kurtosis.
    """
    x = _sample(x)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        m, v = x.mean(), x.var(ddof=1)
        s, k = stats.skew(x, bias=True), stats.kurtosis(x, fisher=True, bias=True)
    if not np.isfinite([m, v, s, k]).all() or v <= 0:
        raise ValueError("NIG sample moments must be finite with representable positive variance")
    if k <= 0 or k <= (5/3)*s*s:
        raise ValueError("Moments lie outside the admissible NIG region")
    t = s*s/k
    rho2 = t/(3-4*t)
    rho = np.sign(s)*np.sqrt(rho2)
    dgamma = 3*(1+4*rho2)/k
    alpha = np.sqrt(dgamma/(v*(1-rho2)**2))
    beta = rho*alpha
    gamma = alpha*np.sqrt(1-rho2)
    delta = dgamma/gamma
    return dict(mu=float(m-delta*beta/gamma), alpha=float(alpha), beta=float(beta), delta=float(delta))


def fit_nig_mle(x):
    """SciPy MLE converted from (a,b,loc,scale) to (mu,alpha,beta,delta)."""
    x = _sample(x)
    a, b, loc, scale = stats.norminvgauss.fit(x)
    if not np.isfinite([a, b, loc, scale]).all() or scale <= 0 or a <= abs(b):
        raise RuntimeError("Invalid NIG maximum-likelihood fit")
    return dict(mu=float(loc), alpha=float(a/scale), beta=float(b/scale), delta=float(scale))
