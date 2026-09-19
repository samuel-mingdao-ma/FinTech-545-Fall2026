"""Distribution fitting with the parameter conventions used by FINTECH 545."""
import numpy as np
from scipy import optimize, stats


def _sample(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or np.std(x) == 0:
        raise ValueError("Expected at least two finite, nonconstant observations")
    return x


def fit_normal(x):
    """Moment fit using sample standard deviation, as in the course reference."""
    x = _sample(x)
    return dict(mu=float(x.mean()), sigma=float(x.std(ddof=1)))


def fit_t(x):
    """MLE of location-scale t, constrained to nu >= 2.0001 like the reference.

    sigma is t scale, not standard deviation: sd = sigma*sqrt(nu/(nu-2)).
    """
    x = _sample(x)
    center, spread = x.mean(), x.std(ddof=1)
    z = (x - center) / spread

    def objective(p):
        return -stats.t.logpdf(z, df=p[2], loc=p[0], scale=np.exp(p[1])).sum()

    candidates = []
    for nu in (3.0, 6.0, 12.0, 30.0):
        result = optimize.minimize(objective, [0, np.log(np.sqrt((nu-2)/nu)), nu],
                                   method="L-BFGS-B", bounds=[(None, None), (-20, 20), (2.0001, None)],
                                   options={"ftol": 1e-13, "gtol": 1e-7, "maxiter": 3000})
        if result.success and np.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("Student t maximum likelihood did not converge")
    p = min(candidates, key=lambda r: r.fun).x
    return dict(mu=float(center + spread*p[0]), sigma=float(spread*np.exp(p[1])), nu=float(p[2]))


def fit_t_regression(y, x):
    """MLE y = Alpha + X beta + sigma*T(nu), with error location fixed at zero."""
    y = _sample(y)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError("X must contain finite observations matching y")
    xm, xs = x.mean(axis=0), x.std(axis=0)
    if np.any(xs == 0):
        raise ValueError("X must not contain constant predictors")
    design = np.column_stack([np.ones(len(y)), (x-xm)/xs])
    if np.linalg.matrix_rank(design) < design.shape[1] or len(y) <= design.shape[1]:
        raise ValueError("Regression design must have full column rank and positive residual degrees of freedom")
    ym, ys = y.mean(), y.std(ddof=1)
    z = (y-ym)/ys
    initial_beta = np.linalg.lstsq(design, z, rcond=None)[0]
    residual_sd = np.std(z-design@initial_beta, ddof=design.shape[1])
    if residual_sd <= np.finfo(float).eps:
        raise ValueError("Degenerate zero-noise regression")

    def objective(p):
        return -stats.t.logpdf(z-design@p[:-2], df=p[-1], scale=np.exp(p[-2])).sum()

    candidates = []
    for nu in (3.0, 6.0, 12.0):
        start = np.r_[initial_beta, np.log(residual_sd*np.sqrt((nu-2)/nu)), nu]
        result = optimize.minimize(objective, start, method="L-BFGS-B",
            bounds=[(None, None)]*len(initial_beta)+[(-20, 20), (2.0001, None)],
            options={"ftol": 1e-13, "gtol": 1e-7, "maxiter": 5000})
        if result.success and np.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("Student t regression did not converge")
    p = min(candidates, key=lambda r: r.fun).x
    beta = ys*p[1:-2]/xs
    alpha = ym + ys*p[0] - xm@beta
    result = dict(mu=0.0, sigma=float(ys*np.exp(p[-2])), nu=float(p[-1]), Alpha=float(alpha))
    result.update({f"B{i+1}": float(b) for i, b in enumerate(beta)})
    return result


def aicc(log_likelihood, n, k):
    """Small-sample AIC; count every fitted parameter, including scale and nu."""
    if not isinstance(k, (int, np.integer)) or k < 0 or n <= k+1:
        raise ValueError("AICc requires n > k+1 and a nonnegative integer k")
    return float(-2*log_likelihood + 2*k + 2*k*(k+1)/(n-k-1))


def fit_nig_moments(x):
    """Return mu, alpha, beta, delta from the first four sample moments.

    Match Julia: sample variance (n-1), uncorrected skew and excess kurtosis.
    """
    x = _sample(x)
    m, v = x.mean(), x.var(ddof=1)
    s, k = stats.skew(x, bias=True), stats.kurtosis(x, fisher=True, bias=True)
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
