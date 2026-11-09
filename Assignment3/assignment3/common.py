"""Statistical conventions and common output helpers."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy import stats


def moments(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError("Need at least four finite observations")
    return dict(n=len(x), mean=float(x.mean()), variance=float(x.var(ddof=1)),
                sd=float(x.std(ddof=1)), skewness=float(stats.skew(x, bias=True)),
                excess_kurtosis=float(stats.kurtosis(x, fisher=True, bias=True)))


def information_criteria(ll, k, n):
    if not np.isfinite(ll) or n <= k+1:
        raise ValueError("AICc needs finite log likelihood and n > k+1")
    return dict(aicc=float(-2*ll+2*k+2*k*(k+1)/(n-k-1)),
                bic=float(-2*ll+k*np.log(n)))


@dataclass
class Fit:
    name: str
    distribution: object
    parameters: dict
    ll: float
    aicc: float
    bic: float


def fit_distribution(values, family):
    """Normal and location-scale t maximum likelihood fits."""
    x = np.asarray(values, float)
    if x.ndim != 1 or len(x) < 6 or not np.isfinite(x).all() or x.std() <= 0:
        raise ValueError("Finite, nonconstant observations required")
    if family == "normal":
        mu, scale = x.mean(), x.std(ddof=0)
        d, parameters, k = stats.norm(mu, scale), dict(mu=float(mu), sigma=float(scale)), 2
    elif family == "t":
        nu, mu, scale = stats.t.fit(x)
        d = stats.t(nu, loc=mu, scale=scale)
        parameters, k = dict(mu=float(mu), scale=float(scale), nu=float(nu)), 3
    else:
        raise ValueError("Family must be normal or t")
    ll = float(d.logpdf(x).sum())
    return Fit(family, d, parameters, ll, **information_criteria(ll, k, len(x)))


def normal_risk(mu, sd, alpha=.05):
    if not 0 < alpha < 1 or sd < 0:
        raise ValueError("Invalid normal risk arguments")
    z = stats.norm.ppf(alpha)
    return dict(var=float(-mu-sd*z), es=float(-mu+sd*stats.norm.pdf(z)/alpha))


def historical_risk(values, alpha=.05, method="course"):
    """Positive-loss VaR and ES; main convention matches course RiskStats.jl.

    Course VaR averages the floor(n*alpha) and ceil(n*alpha) 1-based order
    statistics. ES averages all P&L observations at or below that threshold.
    Fractional provides a coherent empirical tail-mass sensitivity alternative.
    """
    x = np.sort(np.asarray(values, float))
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all() or not 0 < alpha < 1:
        raise ValueError("Finite P&L and 0 < alpha < 1 required")
    mass = len(x)*alpha
    if method == "course":
        lo, hi = max(1, int(np.floor(mass))), max(1, int(np.ceil(mass)))
        q = (x[lo-1]+x[hi-1])/2
        es = -x[x <= q].mean()
    elif method == "fractional":
        q = np.quantile(x, alpha)
        whole = int(np.floor(mass))
        es = -(x[:whole].sum()+(mass-whole)*x[min(whole, len(x)-1)])/mass
    else:
        raise ValueError("method must be course or fractional")
    return dict(var=float(-q), es=float(es))


def ols(y, x, names=None):
    """OLS with an intercept; Gaussian AICc counts the residual scale parameter."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim != 1 or x.ndim != 2 or len(y) != len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite, aligned regression arrays required")
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != design.shape[1] or len(y) <= design.shape[1]+1:
        raise ValueError("Full rank and positive degrees of freedom required")
    residuals = y-design @ coefficients
    sse = float(residuals @ residuals)
    se = np.sqrt(np.diag(np.linalg.inv(design.T @ design))*sse/(len(y)-len(coefficients)))
    t = coefficients/se
    ll = float(stats.norm.logpdf(residuals, scale=np.sqrt(sse/len(y))).sum())
    labels = ["alpha"]+(list(names) if names is not None else [f"x{i+1}" for i in range(x.shape[1])])
    return dict(coefficients=coefficients, standard_errors=se, t_statistics=t,
                p_values=2*stats.t.sf(np.abs(t), len(y)-len(coefficients)), names=labels,
                residuals=residuals, fitted=design @ coefficients,
                rsquared=float(1-sse/np.sum((y-y.mean())**2)), ll=ll,
                n=len(y), k=len(coefficients)+1,
                **information_criteria(ll, len(coefficients)+1, len(y)))


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (Path, datetime)):
        return str(value)
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(payload), indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_predictions(path, predictions):
    """Retain the first analytical prediction draft; never claim student authorship."""
    path = Path(path)
    if not path.exists():
        write_json(path, dict(recorded_at=datetime.now(timezone.utc).isoformat(),
                             origin="AI-assisted analytical prediction draft; chronology is not verified and this is not an unaided student prediction",
                             predictions=predictions))
