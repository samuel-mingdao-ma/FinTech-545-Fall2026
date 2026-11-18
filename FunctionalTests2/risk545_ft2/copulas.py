"""Rank-profile multivariate t fits and Gaussian/t copula risk models.

The estimation route follows ``library/multivariate_t.jl``: the sample mean
and standard deviations are fixed, Kendall's tau supplies correlation, and
only the degrees of freedom is profiled. A multivariate t's scale matrix is
not its covariance matrix: ``covariance = scale * nu / (nu - 2)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import linalg, special, stats


def _matrix(values: np.ndarray, *, name: str = "data") -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or min(result.shape) == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _square(values: np.ndarray, *, name: str = "matrix") -> np.ndarray:
    result = _matrix(values, name=name)
    if result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.allclose(result, result.T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    return result


def _correlation(values: np.ndarray) -> np.ndarray:
    result = _square(values, name="correlation")
    if not np.allclose(np.diag(result), 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("correlation must have a unit diagonal")
    return result


def fix_correlation(
    correlation: np.ndarray, *, tol: float = 1e-8, ridge: float = 1e-8
) -> np.ndarray:
    """Apply the course Higham projection, then make correlation definite.

    Higham's alternating projections include the Dykstra correction. The
    convergence rule and final diagonal ridge match the teacher's code.
    """
    result = _correlation(correlation).copy()
    if (np.ndim(ridge) != 0 or np.ndim(tol) != 0 or not np.isfinite(ridge)
            or not np.isfinite(tol) or ridge <= 0 or tol < 0):
        raise ValueError("ridge must be finite and positive and tol finite and nonnegative")
    if np.linalg.eigvalsh(result)[0] < -tol:
        original = result.copy()
        correction = np.zeros_like(result)
        previous_norm = np.inf
        for _ in range(100):
            residual = result - correction
            eigenvalues, eigenvectors = np.linalg.eigh(residual)
            positive = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
            correction = positive - residual
            result = positive.copy()
            np.fill_diagonal(result, 1.0)
            distance = float(np.sum((result - original) ** 2))
            if (
                abs(distance - previous_norm) < 1e-9
                and np.linalg.eigvalsh(result)[0] > -1e-9
            ):
                break
            previous_norm = distance
    result = (result + result.T) / 2.0
    if np.linalg.eigvalsh(result)[0] < ridge:
        result = (result + ridge * np.eye(len(result))) / (1.0 + ridge)
    # The likelihoods require Cholesky, so report a failed repair immediately.
    linalg.cholesky(result, lower=True)
    return result


def kendall_correlation(data: np.ndarray) -> np.ndarray:
    """Estimate elliptical correlation as sin(pi * Kendall tau / 2)."""
    data = _matrix(data)
    if len(data) < 2:
        raise ValueError("at least two observations are required")
    dimensions = data.shape[1]
    correlation = np.eye(dimensions)
    for i in range(dimensions):
        if np.ptp(data[:, i]) == 0:
            raise ValueError("Kendall correlation is undefined for a constant margin")
        for j in range(i + 1, dimensions):
            tau = stats.kendalltau(data[:, i], data[:, j]).statistic
            correlation[i, j] = correlation[j, i] = np.sin(np.pi * tau / 2.0)
    return fix_correlation(correlation)


def mvt_scale(sd: np.ndarray, correlation: np.ndarray, nu: float) -> np.ndarray:
    """Scale pinned to sample standard deviations at a given nu > 2."""
    correlation = _correlation(correlation)
    sd = np.asarray(sd, dtype=float)
    if sd.shape != (len(correlation),) or not np.isfinite(sd).all() or (sd <= 0).any():
        raise ValueError("sd must give one positive finite standard deviation per margin")
    if np.ndim(nu) != 0 or not np.isfinite(nu) or nu <= 2:
        raise ValueError("nu must be finite and greater than two")
    return np.outer(sd, sd) * correlation * ((nu - 2.0) / nu)


def mvt_loglikelihood(
    data: np.ndarray,
    mean: np.ndarray,
    sd: np.ndarray,
    correlation: np.ndarray,
    nu: float,
) -> float:
    """Joint t log likelihood with covariance fixed at diag(sd) R diag(sd)."""
    data = _matrix(data)
    mean = np.asarray(mean, dtype=float)
    if mean.shape != (data.shape[1],) or not np.isfinite(mean).all():
        raise ValueError("mean must give one finite value per margin")
    scale = mvt_scale(sd, correlation, nu)
    if scale.shape != (data.shape[1], data.shape[1]):
        raise ValueError("scale and observation dimensions must agree")
    return float(np.sum(_multivariate_t_logpdf(data - mean, scale, nu)))


def _multivariate_t_logpdf(centered: np.ndarray, scale: np.ndarray, nu: float) -> np.ndarray:
    """Evaluate t log densities through a Cholesky root and log-gamma terms."""
    dimensions = centered.shape[1]
    root = linalg.cholesky(scale, lower=True)
    whitened = linalg.solve_triangular(root, centered.T, lower=True)
    mahalanobis = np.sum(whitened**2, axis=0)
    normalizer = (
        special.gammaln((nu + dimensions) / 2.0)
        - special.gammaln(nu / 2.0)
        - dimensions / 2.0 * np.log(nu * np.pi)
        - np.log(np.diag(root)).sum()
    )
    return normalizer - (nu + dimensions) / 2.0 * np.log1p(mahalanobis / nu)


def profile_nu(
    loglikelihood: Callable[[float], float],
    *,
    lo: float = 0.01,
    hi: float = 0.49,
    n: int = 200,
) -> tuple[float, float]:
    """Search 1/nu on the course's 200 coarse and 200 local fine grid.

    Deliberately do not substitute a continuous optimizer: the two explicit
    grids define the numerical result in the functional-test specification.
    """
    if (np.ndim(lo) != 0 or np.ndim(hi) != 0 or not 0 < lo < hi < 0.5
            or not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 2):
        raise ValueError("require 0 < lo < hi < 0.5 and at least two grid points")
    coarse = np.linspace(lo, hi, n)
    likelihoods = np.array([loglikelihood(1.0 / theta) for theta in coarse])
    if not np.isfinite(likelihoods).all():
        raise ValueError("profile log likelihood must be finite on the search grid")
    best = int(np.argmax(likelihoods))
    fine = np.linspace(coarse[max(best - 1, 0)], coarse[min(best + 1, n - 1)], n)
    fine_likelihoods = np.array([loglikelihood(1.0 / theta) for theta in fine])
    if not np.isfinite(fine_likelihoods).all():
        raise ValueError("profile log likelihood must be finite on the fine grid")
    best = int(np.argmax(fine_likelihoods))
    return float(1.0 / fine[best]), float(fine_likelihoods[best])


def fit_multivariate_t(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return mean, scale, nu and log likelihood for the rank-profile fit."""
    data = _matrix(data)
    if len(data) < 2:
        raise ValueError("at least two observations are required")
    mean = data.mean(axis=0)
    sd = data.std(axis=0, ddof=1)
    correlation = kendall_correlation(data)
    nu, likelihood = profile_nu(
        lambda df: mvt_loglikelihood(data, mean, sd, correlation, df)
    )
    return mean, mvt_scale(sd, correlation, nu), nu, likelihood


def _uniforms(uniforms: np.ndarray) -> np.ndarray:
    uniforms = _matrix(uniforms, name="uniforms")
    if (uniforms <= 0).any() or (uniforms >= 1).any():
        raise ValueError("copula observations must lie strictly between zero and one")
    return uniforms


def gaussian_copula_ll(uniforms: np.ndarray, correlation: np.ndarray) -> float:
    """Gaussian copula joint log density minus standard-normal margins."""
    uniforms = _uniforms(uniforms)
    correlation = _correlation(correlation)
    if correlation.shape[0] != uniforms.shape[1]:
        raise ValueError("correlation and observation dimensions must agree")
    root = linalg.cholesky(correlation, lower=True)
    transformed = stats.norm.ppf(uniforms)
    whitened = linalg.solve_triangular(root, transformed.T, lower=True)
    logdet = 2.0 * np.log(np.diag(root)).sum()
    return float(
        -0.5 * len(uniforms) * logdet
        - 0.5 * (np.sum(whitened**2) - np.sum(transformed**2))
    )


def t_copula_ll(uniforms: np.ndarray, correlation: np.ndarray, nu: float) -> float:
    """t copula joint log density minus its univariate t margins."""
    uniforms = _uniforms(uniforms)
    correlation = _correlation(correlation)
    if correlation.shape[0] != uniforms.shape[1]:
        raise ValueError("correlation and observation dimensions must agree")
    if np.ndim(nu) != 0 or not np.isfinite(nu) or nu <= 0:
        raise ValueError("nu must be positive and finite")
    transformed = stats.t.ppf(uniforms, df=nu)
    joint = _multivariate_t_logpdf(transformed, correlation, nu)
    margins = stats.t.logpdf(transformed, df=nu)
    return float(np.sum(joint) - np.sum(margins))


def fit_gaussian_copula(uniforms: np.ndarray) -> tuple[np.ndarray, float]:
    uniforms = _uniforms(uniforms)
    correlation = kendall_correlation(uniforms)
    return correlation, gaussian_copula_ll(uniforms, correlation)


def fit_t_copula(uniforms: np.ndarray) -> tuple[np.ndarray, float, float]:
    uniforms = _uniforms(uniforms)
    correlation = kendall_correlation(uniforms)
    nu, likelihood = profile_nu(lambda df: t_copula_ll(uniforms, correlation, df))
    return correlation, nu, likelihood


def simulate_pca(covariance: np.ndarray, draws: int, seed: int = 1234) -> np.ndarray:
    """Simulate centered normal rows using descending principal components.

    As in the Julia implementation, components below 1e-8 are omitted. NumPy
    and Julia use different random streams, so equal seeds do not mean equal
    paths across languages.
    """
    covariance = _square(covariance, name="covariance")
    if (not isinstance(draws, (int, np.integer)) or isinstance(draws, (bool, np.bool_))
            or draws <= 0):
        raise ValueError("draws must be a positive integer")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[0] < -1e-8:
        raise ValueError("covariance must be positive semidefinite")
    order = np.argsort(eigenvalues)[::-1]
    order = order[eigenvalues[order] >= 1e-8]
    loading = eigenvectors[:, order] * np.sqrt(eigenvalues[order])
    normals = np.random.default_rng(seed).standard_normal((len(order), draws))
    return normals.T @ loading.T


def simulate_mvt(
    mean: np.ndarray, scale: np.ndarray, nu: float, draws: int, seed: int = 1234
) -> np.ndarray:
    """Simulate a t using one common chi-square scale for each entire row."""
    scale = _square(scale, name="scale")
    mean = np.asarray(mean, dtype=float)
    if mean.shape != (len(scale),) or not np.isfinite(mean).all():
        raise ValueError("mean must give one finite value per margin")
    if np.ndim(nu) != 0 or not np.isfinite(nu) or nu <= 0:
        raise ValueError("nu must be positive and finite")
    normals = simulate_pca(scale, draws, seed)
    # Separate seeded streams follow the teacher's normal/chi-square sequence.
    common_scale = np.sqrt(nu / np.random.default_rng(seed + 1).chisquare(nu, draws))
    return mean + normals * common_scale[:, None]


def simulate_gaussian_copula(
    correlation: np.ndarray, draws: int, seed: int = 1234
) -> np.ndarray:
    correlation = _correlation(correlation)
    return stats.norm.cdf(simulate_pca(correlation, draws, seed))


def simulate_t_copula(
    correlation: np.ndarray, nu: float, draws: int, seed: int = 1234
) -> np.ndarray:
    correlation = _correlation(correlation)
    simulated = simulate_mvt(np.zeros(len(correlation)), correlation, nu, draws, seed)
    return stats.t.cdf(simulated, df=nu)


def tail_dependence_t(rho: float | np.ndarray, nu: float) -> float | np.ndarray:
    """Equal upper/lower asymptotic tail dependence of a bivariate t copula."""
    rho = np.asarray(rho, dtype=float)
    if not np.isfinite(rho).all() or (rho < -1).any() or (rho > 1).any():
        raise ValueError("rho must be finite and between minus one and one")
    if np.ndim(nu) != 0 or not np.isfinite(nu) or nu <= 0:
        raise ValueError("nu must be positive and finite")
    with np.errstate(divide="ignore"):
        threshold = -np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    result = 2.0 * stats.t.cdf(threshold, df=nu + 1.0)
    return float(result) if result.ndim == 0 else result


def copula_aicc(likelihood: float, parameters: int, observations: int) -> float:
    _information_inputs(likelihood, parameters, observations)
    if parameters < 0 or observations <= parameters + 1:
        raise ValueError("AICc requires k >= 0 and observations > k + 1")
    return float(
        -2.0 * likelihood
        + 2 * parameters
        + (2 * parameters**2 + 2 * parameters) / (observations - parameters - 1)
    )


def copula_bic(likelihood: float, parameters: int, observations: int) -> float:
    _information_inputs(likelihood, parameters, observations)
    if parameters < 0 or observations <= 0:
        raise ValueError("BIC requires k >= 0 and positive observations")
    return float(parameters * np.log(observations) - 2.0 * likelihood)


def _information_inputs(likelihood, parameters, observations):
    if np.ndim(likelihood) != 0 or not np.isfinite(likelihood):
        raise ValueError("log likelihood must be a finite scalar")
    if any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
           for value in (parameters, observations)):
        raise ValueError("parameter and observation counts must be integers")


def _analytic_margin_risk(model, value: float, alpha: float = 0.05) -> dict[str, float]:
    from .risk import parametric_risk

    if np.ndim(value) != 0 or not np.isfinite(value) or value <= 0:
        raise ValueError("current value must be positive and finite")
    risk = parametric_risk(model, alpha)
    return {"VaR95": float(value * risk["var"]), "ES95": float(value * risk["es"]),
            "VaR95_Pct": risk["var"], "ES95_Pct": risk["es"]}


def build_cases(
    data_dir: str | Path, *, simulations: int = 100_000, seed: int = 13
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Compute 13.1--13.9 from input CSVs, without reading reference outputs.

    Diagnostics include twenty-batch Monte Carlo standard errors for every
    stock and the portfolio, and exact fitted-marginal t risk benchmarks.
    """
    from .distributions import fit_student_t
    from .risk import historical_risk

    if not isinstance(simulations, (int, np.integer)) or simulations < 10_000:
        raise ValueError("simulations must be an integer >= 10000 for twenty risk batches")
    data_dir = Path(data_dir)
    returns = pd.read_csv(data_dir / "test13_returns.csv")
    portfolio = pd.read_csv(data_dir / "test13_portfolio.csv", dtype={"Stock": str})
    data = _matrix(returns.to_numpy(dtype=float))
    if (portfolio["Stock"].isna().any() or portfolio["Stock"].duplicated().any()
            or portfolio["Stock"].eq("Total").any()):
        raise ValueError("portfolio stock names must be nonmissing, unique, and cannot be Total")
    stocks = portfolio["Stock"].tolist()
    if set(stocks) != set(returns.columns):
        raise ValueError("portfolio and return asset names must agree")
    values = portfolio["currentValue"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("portfolio current values must be positive and finite")

    correlation = kendall_correlation(data)
    mean, scale, nu, likelihood = fit_multivariate_t(data)
    columns = [f"x{i + 1}" for i in range(data.shape[1])]
    cases = {
        "13.1": pd.DataFrame(correlation, columns=columns),
        "13.2": pd.DataFrame({"mu": mean}),
        "13.3": pd.DataFrame(scale, columns=columns),
        "13.4": pd.DataFrame({"nu": [nu], "ll": [likelihood]}),
    }
    margins = [fit_student_t(data[:, j]) for j in range(data.shape[1])]
    uniforms = np.column_stack([model.cdf(data[:, j]) for j, model in enumerate(margins)])
    copula_correlation, gaussian_ll = fit_gaussian_copula(uniforms)
    t_correlation, copula_nu, t_ll = fit_t_copula(uniforms)
    cases["13.5"] = pd.DataFrame({"ll": [gaussian_ll]})
    cases["13.6"] = pd.DataFrame({"nu": [copula_nu], "ll": [t_ll]})
    cases["13.7"] = pd.DataFrame(
        {
            "Copula": ["Gaussian", "T"],
            "LL": [gaussian_ll, t_ll],
            "K": [0, 1],
            "AICC": [copula_aicc(gaussian_ll, 0, len(data)), copula_aicc(t_ll, 1, len(data))],
            "BIC": [copula_bic(gaussian_ll, 0, len(data)), copula_bic(t_ll, 1, len(data))],
        }
    )
    pairs = [
        {"I": i + 1, "J": j + 1, "Rho": t_correlation[i, j], "Lambda": tail_dependence_t(t_correlation[i, j], copula_nu)}
        for i in range(data.shape[1])
        for j in range(i + 1, data.shape[1])
    ]
    cases["13.8"] = pd.DataFrame(pairs, columns=["I", "J", "Rho", "Lambda"])

    simulated_u = simulate_t_copula(t_correlation, copula_nu, simulations, seed)
    simulated_u = np.clip(simulated_u, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
    simulated_returns = np.column_stack(
        [model.ppf(simulated_u[:, j]) for j, model in enumerate(margins)]
    )
    stock_indices = [returns.columns.get_loc(stock) for stock in stocks]
    pnl = simulated_returns[:, stock_indices] * values
    pnl = np.column_stack([pnl, pnl.sum(axis=1)])
    all_values = np.append(values, values.sum())
    risk_rows, mcse_rows, benchmark_rows = [], [], []
    batches = np.array_split(np.arange(simulations), 20)
    for j, (stock, current_value) in enumerate(zip(stocks + ["Total"], all_values)):
        risk = historical_risk(pnl[:, j], alpha=0.05)
        var, es = float(risk["var"]), float(risk["es"])
        risk_rows.append(
            {"Stock": stock, "VaR95": var, "ES95": es, "VaR95_Pct": var / current_value, "ES95_Pct": es / current_value}
        )
        batch_risk = [historical_risk(pnl[indices, j], alpha=0.05) for indices in batches]
        var_mcse = float(np.std([row["var"] for row in batch_risk], ddof=1) / np.sqrt(20))
        es_mcse = float(np.std([row["es"] for row in batch_risk], ddof=1) / np.sqrt(20))
        mcse_rows.append(
            {"Stock": stock, "VaR95_MCSE": var_mcse, "ES95_MCSE": es_mcse, "VaR95_Pct_MCSE": var_mcse / current_value, "ES95_Pct_MCSE": es_mcse / current_value}
        )
        if stock != "Total":
            benchmark = _analytic_margin_risk(margins[stock_indices[j]], current_value)
            benchmark_rows.append({"Stock": stock, **benchmark})
    cases["13.9"] = pd.DataFrame(risk_rows)
    diagnostics = {
        "13.4": {
            "estimator": "sample mean; ddof=1 sd; Kendall correlation; two-stage theta grid",
            "grid_points_per_stage": 200,
            "covariance_scale_identity_max_error": float(
                np.max(np.abs(scale * nu / (nu - 2.0) - np.outer(data.std(axis=0, ddof=1), data.std(axis=0, ddof=1)) * correlation))
            ),
        },
        "13.6": {
            "correlation_matches_gaussian": bool(np.array_equal(copula_correlation, t_correlation)),
            "marginal_df": [float(model.args[0]) if np.isfinite(model.args[0]) else None
                            for model in margins],
            "normal_limit_margins": [str(returns.columns[j]) for j, model in enumerate(margins)
                                     if np.isposinf(model.args[0])],
        },
        "13.9": {
            "simulations": int(simulations),
            "seed": int(seed),
            "batches": 20,
            "batch_sizes": [int(len(indices)) for indices in batches],
            "rng": "NumPy default_rng; independent normal seed and common chi-square seed+1 streams",
            "risk_mcse": mcse_rows,
            "analytic_marginal_risk": benchmark_rows,
            "mcse_method": "sample standard deviation of twenty batch estimates / sqrt(20)",
            "common_chi_square_per_path": True,
            "all_risks_positive": bool((cases["13.9"].iloc[:, 1:] > 0).all().all()),
        },
    }
    return cases, diagnostics
