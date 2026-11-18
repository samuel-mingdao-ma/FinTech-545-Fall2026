"""Constrained portfolios and realized factor attribution (functional groups 10–11).

The maximum-Sharpe function deliberately follows the course convention of using
the supplied symmetric matrix directly.  In particular, the course test5_3
matrix is indefinite.  Variances must be positive at the evaluated feasible
portfolios; the function never silently repairs the matrix.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize


def _vector(value, name, size=None):
    x = np.asarray(value, dtype=float)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    if size is not None and x.size != size:
        raise ValueError(f"{name} must have length {size}")
    return x


def _covariance(value, *, require_psd):
    covariance = np.asarray(value, dtype=float)
    if (covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]
            or not covariance.size or not np.isfinite(covariance).all()):
        raise ValueError("covariance must be a nonempty finite square matrix")
    scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12 * scale):
        raise ValueError("covariance must be symmetric")
    covariance = (covariance + covariance.T) / 2
    minimum_eigenvalue = float(np.linalg.eigvalsh(covariance)[0])
    if require_psd and minimum_eigenvalue < -1e-10 * scale:
        raise ValueError("covariance must be positive semidefinite")
    return covariance, minimum_eigenvalue


def risk_parity(covariance, risk_budget=None, *, tolerance=1e-12, max_iterations=20000):
    """Return long-only weights with component risk proportional to a budget.

    Cyclic coordinate descent minimizes the convex objective
    ``x' C x / 2 - sum(b * log(x))`` before normalizing the weights.  Singular
    positive-semidefinite covariance matrices are accepted when a positive-risk
    solution exists.  Budgets must be strictly positive.
    """
    covariance, minimum_eigenvalue = _covariance(covariance, require_psd=True)
    n = len(covariance)
    budget = np.ones(n) if risk_budget is None else _vector(risk_budget, "risk budget", n)
    if np.any(budget <= 0):
        raise ValueError("risk budget must be strictly positive")
    if (not np.isscalar(tolerance) or not np.isfinite(tolerance) or tolerance <= 0
            or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1):
        raise ValueError("tolerance and max_iterations must be positive")
    diagonal = np.diag(covariance)
    if np.any(diagonal <= 0):
        raise ValueError("positive risk budgets require positive asset variances")
    budget = budget / budget.sum()
    scaled = covariance / diagonal.max()
    x = np.sqrt(budget / np.diag(scaled))
    for iteration in range(1, max_iterations + 1):
        for i in range(n):
            other = float(scaled[i] @ x - scaled[i, i] * x[i])
            root = np.hypot(other, 2 * np.sqrt(scaled[i, i] * budget[i]))
            # The alternate quadratic root avoids cancellation for positive other.
            x[i] = (2 * budget[i] / (root + other) if other >= 0 else
                    (root - other) / (2 * scaled[i, i]))
        component_variance = x * (scaled @ x)
        total_variance = component_variance.sum()
        if total_variance <= 0 or not np.isfinite(total_variance):
            raise ValueError("no positive-risk risk-budget solution")
        error = float(np.max(np.abs(component_variance / total_variance - budget)))
        if error <= tolerance:
            weights = x / x.sum()
            volatility = float(np.sqrt(weights @ covariance @ weights))
            return weights, {
                "method": "convex cyclic coordinate descent",
                "iterations": iteration,
                "risk_budget_max_abs_error": error,
                "volatility": volatility,
                "minimum_covariance_eigenvalue": minimum_eigenvalue,
            }
    raise RuntimeError("risk parity did not converge; a positive-risk solution may not exist")


def _bounded_simplex_projection(value, lower, upper):
    """Euclidean projection onto box bounds and sum(weights)=1."""
    value = np.asarray(value, dtype=float)
    left = float(np.min(value - upper))
    right = float(np.max(value - lower))
    for _ in range(100):
        midpoint = (left + right) / 2
        if np.clip(value - midpoint, lower, upper).sum() > 1:
            left = midpoint
        else:
            right = midpoint
    return np.clip(value - (left + right) / 2, lower, upper)


def maximum_sharpe(covariance, means, risk_free, bounds=None, *, seed=545):
    """Maximize Sharpe subject to a unit budget and explicit box bounds.

    Default bounds are [0, 1].  Finite symmetric input is used without a PSD
    repair, matching the teacher's indefinite covariance fixture.  Nonpositive
    variance at a feasible point evaluated by the solver is rejected.  Analytic
    gradients and deterministic multiple starts check numerical convergence;
    when the matrix is indefinite this is not a general proof of globality.
    """
    covariance, minimum_eigenvalue = _covariance(covariance, require_psd=False)
    n = len(covariance)
    means = _vector(means, "means", n)
    if not np.isscalar(risk_free) or not np.isfinite(risk_free):
        raise ValueError("risk_free must be finite")
    if bounds is None:
        bounds = np.column_stack((np.zeros(n), np.ones(n)))
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (n, 2) or not np.isfinite(bounds).all():
        raise ValueError("bounds must be a finite n by 2 matrix")
    lower, upper = bounds[:, 0], bounds[:, 1]
    if np.any(lower > upper) or lower.sum() > 1 + 1e-12 or upper.sum() < 1 - 1e-12:
        raise ValueError("bounds are infeasible for weights summing to one")
    excess = means - float(risk_free)
    scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
    variance_floor = np.finfo(float).eps * scale * 0.01

    def objective(weights):
        variance = float(weights @ covariance @ weights)
        if variance <= variance_floor:
            raise ValueError("nonpositive portfolio variance at an evaluated feasible portfolio")
        return -float(weights @ excess) / np.sqrt(variance)

    def gradient(weights):
        variance = float(weights @ covariance @ weights)
        if variance <= variance_floor:
            raise ValueError("nonpositive portfolio variance at an evaluated feasible portfolio")
        sd = np.sqrt(variance)
        return -excess / sd + float(weights @ excess) * (covariance @ weights) / sd**3

    rng = np.random.default_rng(seed)
    starts = [np.ones(n) / n, *np.eye(n), *rng.dirichlet(np.ones(n), size=8)]
    candidates = []
    for start in starts:
        feasible_start = _bounded_simplex_projection(start, lower, upper)
        objective(feasible_start)
        result = optimize.minimize(
            objective, feasible_start, jac=gradient, method="SLSQP",
            bounds=list(zip(lower, upper)),
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1,
                          "jac": lambda w: np.ones(n)}],
            options={"ftol": 1e-13, "maxiter": 2000},
        )
        feasible = (abs(result.x.sum() - 1) < 1e-9
                    and np.all(result.x >= lower - 1e-10)
                    and np.all(result.x <= upper + 1e-10))
        if result.success and feasible and np.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("no maximum-Sharpe optimization start converged")
    best = min(candidates, key=lambda result: result.fun)
    weights = _bounded_simplex_projection(best.x, lower, upper)
    return weights, {
        "method": "SLSQP with analytic gradients and deterministic multiple starts",
        "sharpe_ratio": float(-objective(weights)),
        "successful_starts": len(candidates),
        "start_objective_spread": float(np.ptp([result.fun for result in candidates])),
        "budget_error": float(abs(weights.sum() - 1)),
        "minimum_covariance_eigenvalue": minimum_eigenvalue,
        "covariance_is_psd": minimum_eigenvalue >= -1e-10 * scale,
        "active_lower_bounds": np.flatnonzero(np.isclose(weights, lower, atol=1e-8)).tolist(),
        "active_upper_bounds": np.flatnonzero(np.isclose(weights, upper, atol=1e-8)).tolist(),
    }


@dataclass
class ExPostAttribution:
    attribution: pd.DataFrame
    weights: np.ndarray
    factor_weights: np.ndarray
    residual_individual: np.ndarray
    residual_return: np.ndarray
    portfolio_return: np.ndarray
    carino_k: np.ndarray


def _return_frame(value, name):
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    else:
        array = np.asarray(value, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional return matrix")
        frame = pd.DataFrame(array, columns=[f"x{i + 1}" for i in range(array.shape[1])])
    array = frame.to_numpy(dtype=float)
    if (array.ndim != 2 or array.shape[0] < 2 or array.shape[1] == 0
            or not np.isfinite(array).all()):
        raise ValueError(f"{name} needs at least two finite observations and one column")
    if np.any(array <= -1):
        raise ValueError(f"{name} simple returns must exceed -1")
    if not frame.columns.is_unique:
        raise ValueError(f"{name} column labels must be unique")
    return frame, array


def _log_return_ratio(value):
    value = np.asarray(value, dtype=float)
    result = np.ones_like(value)
    np.divide(np.log1p(value), value, out=result, where=value != 0)
    return result


def expost_factor(weights, stock_returns, factor_returns, beta):
    """Buy-and-hold return and sample-volatility attribution with Carino linking.

    Beta has assets in rows and factors in columns.  Return attribution sums to
    the portfolio's compounded total return.  Volatility attribution is each
    component's sample covariance with portfolio returns divided by portfolio
    sample volatility, equivalent to the teacher's regression with intercept.
    """
    _, stock_values = _return_frame(stock_returns, "stock returns")
    factors, factor_values = _return_frame(factor_returns, "factor returns")
    observations, assets = stock_values.shape
    if factor_values.shape[0] != observations:
        raise ValueError("stock and factor returns must have equal observation counts")
    if any(label in {"Value", "Alpha", "Portfolio"} for label in factors.columns):
        raise ValueError("factor names cannot use reserved output column labels")
    weights = _vector(weights, "weights", assets)
    if not np.isclose(weights.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError("starting weights must sum to one")
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (assets, factor_values.shape[1]) or not np.isfinite(beta).all():
        raise ValueError("beta must be a finite asset by factor matrix")

    drifted_weights = np.empty_like(stock_values)
    factor_weights = np.empty_like(factor_values)
    portfolio_returns = np.empty(observations)
    residual_returns = np.empty(observations)
    last_weights = weights.copy()
    for period in range(observations):
        drifted_weights[period] = last_weights
        factor_weights[period] = last_weights @ beta
        updated = last_weights * (1 + stock_values[period])
        gross_return = float(updated.sum())
        if gross_return <= 0 or not np.isfinite(gross_return):
            raise ValueError("portfolio wealth must remain strictly positive")
        last_weights = updated / gross_return
        portfolio_returns[period] = gross_return - 1
        residual_returns[period] = (portfolio_returns[period]
                                    - factor_weights[period] @ factor_values[period])
    if np.any(residual_returns <= -1):
        raise ValueError("Alpha returns must exceed -1 for geometric compounding")
    total_return = float(np.expm1(np.log1p(portfolio_returns).sum()))
    carino_k = _log_return_ratio(portfolio_returns) / _log_return_ratio(total_return)
    components = np.column_stack((factor_values * factor_weights, residual_returns))
    linked_return = np.sum(components * carino_k[:, None], axis=0)
    compounded_components = np.expm1(np.log1p(np.column_stack((factor_values, residual_returns))).sum(axis=0))
    centered_portfolio = portfolio_returns - portfolio_returns.mean()
    volatility = float(np.std(portfolio_returns, ddof=1))
    if volatility <= 10 * np.finfo(float).eps * max(1, np.max(np.abs(portfolio_returns))):
        volatility = 0.0
        component_volatility = np.zeros(components.shape[1])
    else:
        component_volatility = ((components - components.mean(axis=0)).T @ centered_portfolio
                                / (observations - 1) / volatility)
    attribution = pd.DataFrame({"Value": ["TotalReturn", "Return Attribution", "Vol Attribution"]})
    for i, label in enumerate([*factors.columns, "Alpha"]):
        attribution[label] = [compounded_components[i], linked_return[i], component_volatility[i]]
    attribution["Portfolio"] = [total_return, total_return, volatility]
    residual_individual = (stock_values - factor_values @ beta.T) * drifted_weights
    return ExPostAttribution(attribution, drifted_weights, factor_weights, residual_individual,
                             residual_returns, portfolio_returns, carino_k)


def build_cases(data_dir, *, simulations=100000, seed=545):
    """Compute teacher-schema tables 10.1–10.4 and 11.1–11.2 from input CSVs."""
    data_dir = Path(data_dir)
    covariance_rp = pd.read_csv(data_dir / "test5_2.csv").to_numpy(dtype=float)
    covariance_sr = pd.read_csv(data_dir / "test5_3.csv").to_numpy(dtype=float)
    means = pd.read_csv(data_dir / "test10_3_means.csv")["Mean"].to_numpy(dtype=float)
    cases, diagnostics = {}, {}
    configurations = [
        ("10.1", risk_parity, (covariance_rp,), {}),
        ("10.2", risk_parity, (covariance_rp, [1, 1, 1, 1, 0.5]), {}),
        ("10.3", maximum_sharpe, (covariance_sr, means, 0.04), {"seed": seed}),
        ("10.4", maximum_sharpe,
         (covariance_sr, means, 0.04, np.tile([0.1, 0.5], (len(means), 1))), {"seed": seed}),
    ]
    for key, function, args, kwargs in configurations:
        weights, detail = function(*args, **kwargs)
        cases[key] = pd.DataFrame({"W": weights})
        diagnostics[key] = detail
    weights = pd.read_csv(data_dir / "test11_1_weights.csv")["W"].to_numpy(dtype=float)
    returns = pd.read_csv(data_dir / "test11_1_returns.csv")
    asset_attribution = expost_factor(weights, returns, returns, np.eye(len(weights)))
    cases["11.1"] = asset_attribution.attribution.drop(columns="Alpha")
    weights = pd.read_csv(data_dir / "test11_2_weights.csv")["W"].to_numpy(dtype=float)
    stocks = pd.read_csv(data_dir / "test11_2_stock_returns.csv")
    factors = pd.read_csv(data_dir / "test11_2_factor_returns.csv")
    beta = pd.read_csv(data_dir / "test11_2_beta.csv").iloc[:, 1:].to_numpy(dtype=float)
    factor_attribution = expost_factor(weights, stocks, factors, beta)
    cases["11.2"] = factor_attribution.attribution
    for key, result in [("11.1", asset_attribution), ("11.2", factor_attribution)]:
        table = result.attribution
        components = table.drop(columns=["Value", "Portfolio"]).to_numpy(dtype=float)
        diagnostics[key] = {
            "return_attribution_sum_error": float(abs(components[1].sum() - table.Portfolio.iloc[1])),
            "volatility_attribution_sum_error": float(abs(components[2].sum() - table.Portfolio.iloc[2])),
            "weight_budget_max_abs_error": float(np.max(np.abs(result.weights.sum(axis=1) - 1))),
            "volatility_ddof": 1,
            "linking": "Carino with drifting buy-and-hold weights",
        }
    return cases, diagnostics
