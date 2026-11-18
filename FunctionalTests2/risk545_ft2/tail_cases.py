"""Functional cases 8 and 9 using fitted margins and simulated scenario P&Ls."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .copulas import simulate_gaussian_copula
from .distributions import fit_normal, fit_student_t
from .risk import batch_standard_errors, historical_risk, parametric_risk


def _risk_row(distribution, metric):
    risk = parametric_risk(distribution)[metric]
    # Both fitted families are symmetric and have finite means.
    return {f"{metric.upper() if metric == 'es' else 'VaR'} Absolute": risk,
            f"{metric.upper() if metric == 'es' else 'VaR'} Diff from Mean": risk + float(distribution.mean())}


def _simulation_case(distribution, metric, rng):
    """Keep the course's 10,000-draw univariate simulations unchanged."""
    sample = distribution.ppf(rng.uniform(size=10_000))
    name = "VaR" if metric == "var" else "ES"
    absolute = historical_risk(sample)[metric]
    centered = historical_risk(sample - sample.mean())[metric]
    columns = [f"{name} Absolute", f"{name} Diff from Mean"]
    estimates = []
    for part in np.array_split(sample, 20):
        estimates.append([historical_risk(part)[metric],
                          historical_risk(part - part.mean())[metric]])
    errors = np.std(estimates, axis=0, ddof=1) / np.sqrt(20)
    benchmark = _risk_row(distribution, metric)
    diagnostic = {
        "simulations": 10_000, "batches": 20,
        "mcse": dict(zip(columns, map(float, errors))),
        "analytic_risk": benchmark,
    }
    return pd.DataFrame([[absolute, centered]], columns=columns), diagnostic


def _group8(data, seed):
    normal = fit_normal(pd.read_csv(data / "test7_1.csv").iloc[:, 0])
    student = fit_student_t(pd.read_csv(data / "test7_2.csv").iloc[:, 0])
    cases = {
        "8.1": pd.DataFrame([_risk_row(normal, "var")]),
        "8.2": pd.DataFrame([_risk_row(student, "var")]),
        "8.4": pd.DataFrame([_risk_row(normal, "es")]),
        "8.5": pd.DataFrame([_risk_row(student, "es")]),
    }
    rng = np.random.default_rng(seed)
    cases["8.3"], var_diagnostic = _simulation_case(student, "var", rng)
    cases["8.6"], es_diagnostic = _simulation_case(student, "es", rng)
    diagnostics = {"8.3": var_diagnostic, "8.6": es_diagnostic,
                   "8_fits": {"normal": {"mu": float(normal.mean()), "sigma": float(normal.std())},
                              "t": {"mu": float(student.mean()), "scale": student.kwds["scale"],
                                    "nu": float(student.args[0])}}}
    return cases, diagnostics


def _group9(data, simulations, seed):
    returns = pd.read_csv(data / "test9_1_returns.csv")
    portfolio = pd.read_csv(data / "test9_1_portfolio.csv", encoding="utf-8-sig")
    if returns.columns.duplicated().any() or portfolio.Stock.duplicated().any():
        raise ValueError("Return and portfolio asset labels must be unique")
    if set(portfolio.Stock) != set(returns.columns):
        raise ValueError("The portfolio and return assets must match")
    portfolio = portfolio.set_index("Stock").loc[returns.columns].reset_index()
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("Return observations must be finite")
    models = []
    for column, family in zip(returns.columns, portfolio.Distribution):
        if str(family).lower() == "normal":
            models.append(fit_normal(returns[column]))
        elif str(family).lower() == "t":
            models.append(fit_student_t(returns[column]))
        else:
            raise ValueError("Portfolio Distribution must be Normal or T")
    # Spearman is computed on fitted PITs, as in test_setup.jl. Monotone
    # marginal transformations leave the ranks unchanged.
    pit = np.column_stack([model.cdf(returns[name]) for name, model in zip(returns.columns, models)])
    correlation = np.corrcoef(stats.rankdata(pit, axis=0), rowvar=False)
    uniform = simulate_gaussian_copula(correlation, simulations, seed)
    uniform = np.clip(uniform, np.finfo(float).tiny, 1 - np.finfo(float).eps)
    simulated = np.column_stack([model.ppf(uniform[:, j]) for j, model in enumerate(models)])
    values = (portfolio.Holding * portfolio["Starting Price"]).to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("The course portfolio requires positive finite current values")
    pnl = simulated * values
    pnl = np.column_stack([pnl, pnl.sum(axis=1)])
    stocks = list(returns.columns) + ["Total"]
    current_values = np.append(values, values.sum())
    risk_rows, error_rows, benchmark_rows = [], [], []
    for j, (stock, value) in enumerate(zip(stocks, current_values)):
        risk = historical_risk(pnl[:, j])
        mcse = batch_standard_errors(pnl[:, j])
        risk_rows.append(dict(Stock=stock, VaR95=risk["var"], ES95=risk["es"],
                              VaR95_Pct=risk["var"] / value, ES95_Pct=risk["es"] / value))
        error_rows.append(dict(Stock=stock, VaR95_MCSE=mcse["var"], ES95_MCSE=mcse["es"],
                               VaR95_Pct_MCSE=mcse["var"] / value, ES95_Pct_MCSE=mcse["es"] / value))
        if j < len(models):
            analytic = parametric_risk(models[j])
            benchmark_rows.append(dict(Stock=stock, VaR95=value * analytic["var"],
                                       ES95=value * analytic["es"],
                                       VaR95_Pct=analytic["var"], ES95_Pct=analytic["es"]))
    diagnostics = {
        "9.1": {"simulations": int(simulations), "seed": int(seed), "batches": 20,
                "spearman_correlation": correlation.tolist(),
                "risk_mcse": error_rows, "analytic_marginal_risk": benchmark_rows,
                "aggregation_max_error": float(np.max(np.abs(pnl[:, :-1].sum(axis=1) - pnl[:, -1])))},
    }
    return {"9.1": pd.DataFrame(risk_rows)}, diagnostics


def build_cases(data_dir, *, simulations=100_000, seed=545):
    """Compute every case in groups 8 and 9 from input data only."""
    if not isinstance(simulations, (int, np.integer)) or simulations < 10_000:
        raise ValueError("At least 10000 integer portfolio simulation draws are required")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("The random seed must be a nonnegative integer")
    data = Path(data_dir)
    cases, diagnostic = _group8(data, seed)
    group9, more = _group9(data, simulations, seed)
    cases.update(group9)
    diagnostic.update(more)
    return cases, diagnostic
