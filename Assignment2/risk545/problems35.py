"""Solutions for Assignment 2 Problems 3 and 5.

The public functions return only JSON-serializable Python objects.  Monetary
results are in dollars, and VaR/ES use the course sign convention: a loss is
reported as a positive number.  See each result's ``conventions`` entry for the
finite-sample details that matter when reproducing the numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .common import (
    arithmetic_returns,
    historical_var_es,
    normal_var,
    sample_moments,
)


DEFAULT_SEED = 545
DEFAULT_ALPHA = 0.05
N_SIMULATIONS = 100_000
ONE_MILLION = 1_000_000.0


def _validate_columns(data: pd.DataFrame, required: tuple[str, ...]) -> None:
    """Raise a useful error if an input CSV is incomplete."""

    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if data.loc[:, list(required)].isna().any().any():
        raise ValueError("Required input columns contain missing values")


def _tail_count(pnl: np.ndarray, alpha: float) -> int:
    """Count observations entering ES under the shared empirical convention."""

    threshold = _course_empirical_threshold(pnl, alpha)
    return int(np.count_nonzero(np.asarray(pnl) <= threshold))


def _course_empirical_threshold(pnl: np.ndarray, alpha: float) -> float:
    """Return the finite-sample quantile used in the course RiskStats.jl.

    If ``m = n*alpha``, the course code averages sorted observations at the
    one-based ranks ``floor(m)`` and ``ceil(m)``.  When ``m`` is an integer this
    is simply the observation at one-based rank ``m``.  This is intentionally
    not NumPy's default linear quantile.
    """

    values = np.sort(np.asarray(pnl, dtype=float))
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Historical VaR/ES requires a non-empty one-dimensional array")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    fractional_rank = len(values) * alpha
    lower_rank = int(np.floor(fractional_rank))
    upper_rank = int(np.ceil(fractional_rank))
    if lower_rank < 1:
        raise ValueError("alpha*n must be at least one under the course convention")
    return float(
        0.5 * (values[lower_rank - 1] + values[upper_rank - 1])
    )


def _position_risk(pnl: np.ndarray) -> dict[str, Any]:
    """Calculate the P&L summaries required for one Problem 3 position."""

    values = np.asarray(pnl, dtype=float)
    mean_pnl = float(np.mean(values))
    standard_deviation_pnl = float(np.std(values, ddof=1))
    risk: dict[str, Any] = {
        "mean_pnl": mean_pnl,
        "standard_deviation_pnl": standard_deviation_pnl,
        "normal_var_5pct": normal_var(
            mean_pnl, standard_deviation_pnl, alpha=DEFAULT_ALPHA
        ),
        "historical": {},
    }
    for alpha, label in ((0.05, "5pct"), (0.01, "1pct")):
        var, es = historical_var_es(values, alpha=alpha)
        risk["historical"][label] = {
            "alpha": alpha,
            "var": var,
            "es": es,
            "tail_observations_in_es": _tail_count(values, alpha),
        }
    return risk


def _plot_problem3(
    concentrated_pnl: np.ndarray,
    diversified_pnl: np.ndarray,
    figure_dir: str | Path,
) -> str:
    """Plot both investment choices on a common P&L scale."""

    output_dir = Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "problem3_pnl_distributions.png"

    combined = np.concatenate([concentrated_pnl, diversified_pnl]) / ONE_MILLION
    bins = np.linspace(float(np.min(combined)), float(np.max(combined)), 85)
    figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    plots = (
        (
            axes[0],
            concentrated_pnl / ONE_MILLION,
            "Concentrated: USD 2 million in A",
            "#355c7d",
        ),
        (
            axes[1],
            diversified_pnl / ONE_MILLION,
            "Diversified: USD 1 million in A and USD 1 million in B",
            "#c06c84",
        ),
    )
    for axis, values, title, color in plots:
        axis.hist(values, bins=bins, color=color, alpha=0.88, edgecolor="white")
        axis.axvline(0.0, color="#333333", linewidth=1.0, linestyle="--")
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.set_ylabel("Scenario count")
        axis.grid(axis="y", alpha=0.20)
    axes[1].set_xlabel("One-year P&L ($ millions)")
    figure.suptitle(
        "Problem 3: diversification moves default losses across the 5% threshold",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return str(output_path)


def solve_problem3(
    csv_path: str | Path,
    figure_dir: str | Path | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Solve the diversification, VaR, and ES exercise in Problem 3.

    ``seed`` is accepted for a uniform solver interface.  Problem 3 itself is a
    deterministic historical-scenario calculation, so the seed is recorded but
    is not consumed.
    """

    data = pd.read_csv(csv_path)
    _validate_columns(data, ("A", "B"))
    bond_a = data["A"].to_numpy(dtype=float)
    bond_b = data["B"].to_numpy(dtype=float)
    n_scenarios = int(len(data))
    if n_scenarios == 0:
        raise ValueError("Problem 3 requires at least one scenario")

    loss_a = bond_a < -0.20
    loss_b = bond_b < -0.20
    large_loss_counts = {
        "loss_threshold_return": -0.20,
        "A": int(np.count_nonzero(loss_a)),
        "B": int(np.count_nonzero(loss_b)),
        "at_least_one": int(np.count_nonzero(loss_a | loss_b)),
        "both": int(np.count_nonzero(loss_a & loss_b)),
    }
    large_loss_counts["A_rate"] = large_loss_counts["A"] / n_scenarios
    large_loss_counts["B_rate"] = large_loss_counts["B"] / n_scenarios
    large_loss_counts["at_least_one_rate"] = (
        large_loss_counts["at_least_one"] / n_scenarios
    )
    large_loss_counts["both_rate"] = (
        large_loss_counts["both"] / n_scenarios
    )

    pnl = {
        "one_million_A": ONE_MILLION * bond_a,
        "one_million_B": ONE_MILLION * bond_b,
        "two_million_A": 2.0 * ONE_MILLION * bond_a,
        "one_million_each": ONE_MILLION * (bond_a + bond_b),
    }
    positions = {name: _position_risk(values) for name, values in pnl.items()}

    combined_5 = positions["one_million_each"]["historical"]["5pct"]
    a_5 = positions["one_million_A"]["historical"]["5pct"]
    b_5 = positions["one_million_B"]["historical"]["5pct"]
    combined_1 = positions["one_million_each"]["historical"]["1pct"]
    a_1 = positions["one_million_A"]["historical"]["1pct"]
    b_1 = positions["one_million_B"]["historical"]["1pct"]

    def subadditivity(
        combined: dict[str, float],
        standalone_a: dict[str, float],
        standalone_b: dict[str, float],
        measure: str,
    ) -> dict[str, float | bool]:
        lhs = float(combined[measure])
        rhs = float(standalone_a[measure] + standalone_b[measure])
        return {
            "combined_A_plus_B": lhs,
            "standalone_A_plus_standalone_B": rhs,
            "holds": bool(lhs <= rhs + 1e-10),
            "lhs_minus_rhs": lhs - rhs,
        }

    subadditivity_5pct = {
        "var": subadditivity(combined_5, a_5, b_5, "var"),
        "es": subadditivity(combined_5, a_5, b_5, "es"),
    }
    subadditivity_1pct = {
        "var": subadditivity(combined_1, a_1, b_1, "var"),
        "es": subadditivity(combined_1, a_1, b_1, "es"),
    }

    concentrated = positions["two_million_A"]
    diversified = positions["one_million_each"]
    comparison = {
        "historical_5pct": {
            "concentrated_var": concentrated["historical"]["5pct"]["var"],
            "diversified_var": diversified["historical"]["5pct"]["var"],
            "concentrated_es": concentrated["historical"]["5pct"]["es"],
            "diversified_es": diversified["historical"]["5pct"]["es"],
            "lower_var_choice": (
                "concentrated_2m_A"
                if concentrated["historical"]["5pct"]["var"]
                < diversified["historical"]["5pct"]["var"]
                else "diversified_1m_each"
            ),
            "lower_es_choice": (
                "concentrated_2m_A"
                if concentrated["historical"]["5pct"]["es"]
                < diversified["historical"]["5pct"]["es"]
                else "diversified_1m_each"
            ),
        },
        "normal_5pct": {
            "concentrated_var": concentrated["normal_var_5pct"],
            "diversified_var": diversified["normal_var_5pct"],
            "lower_var_choice": (
                "concentrated_2m_A"
                if concentrated["normal_var_5pct"]
                < diversified["normal_var_5pct"]
                else "diversified_1m_each"
            ),
        },
        "historical_1pct": {
            "concentrated_var": concentrated["historical"]["1pct"]["var"],
            "diversified_var": diversified["historical"]["1pct"]["var"],
            "concentrated_es": concentrated["historical"]["1pct"]["es"],
            "diversified_es": diversified["historical"]["1pct"]["es"],
            "lower_var_choice": (
                "concentrated_2m_A"
                if concentrated["historical"]["1pct"]["var"]
                < diversified["historical"]["1pct"]["var"]
                else "diversified_1m_each"
            ),
            "lower_es_choice": (
                "concentrated_2m_A"
                if concentrated["historical"]["1pct"]["es"]
                < diversified["historical"]["1pct"]["es"]
                else "diversified_1m_each"
            ),
        },
    }

    figure_path = None
    if figure_dir is not None:
        figure_path = _plot_problem3(
            pnl["two_million_A"], pnl["one_million_each"], figure_dir
        )

    return {
        "conventions": {
            "pnl": "notional in dollars multiplied by one-year arithmetic return",
            "moments": (
                "mean, sample variance (n-1), bias-corrected skewness, and "
                "bias-corrected excess kurtosis"
            ),
            "var_sign": "VaR = -q_alpha(P&L), so losses are positive",
            "historical_var": (
                "course RiskStats.jl rule: if m=n*alpha, negate the average "
                "of sorted P&L at one-based ranks floor(m) and ceil(m)"
            ),
            "historical_es": (
                "negative mean of all P&L observations at or below the "
                "historical VaR quantile; ties are included"
            ),
            "normal_var": (
                "-(sample mean P&L + Phi^{-1}(alpha) times sample P&L SD)"
            ),
        },
        "seed": int(seed),
        "seed_used": False,
        "n_scenarios": n_scenarios,
        "moments": {
            "A": sample_moments(bond_a),
            "B": sample_moments(bond_b),
        },
        "large_loss_counts": large_loss_counts,
        "prediction": {
            "var_5pct": "diversified_larger",
            "es_5pct": "diversified_smaller",
            "basis": (
                "Each bond's >20% loss rate is below 5%, but the union rate is "
                "above 5%; the 5% quantile crosses a default-loss jump only "
                "after diversification. ES averages the worst tail, where the "
                "split position usually contains only one default."
            ),
        },
        "positions": positions,
        "choice_comparison": comparison,
        "subadditivity_5pct": subadditivity_5pct,
        "subadditivity_1pct": subadditivity_1pct,
        "figure": figure_path,
    }


def _ols_with_intercept(
    market: np.ndarray, stock: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return OLS [alpha, beta] and residuals for one stock."""

    design = np.column_stack([np.ones(len(market)), market])
    coefficients, _, _, _ = np.linalg.lstsq(design, stock, rcond=None)
    residuals = stock - design @ coefficients
    return coefficients, residuals


def _simulation_var(pnl: np.ndarray, alpha: float = DEFAULT_ALPHA) -> float:
    """Calculate empirical VaR from a simulated P&L vector."""

    var, _ = historical_var_es(np.asarray(pnl, dtype=float), alpha=alpha)
    return var


def _plot_problem5(
    pnl_by_model: dict[str, dict[str, np.ndarray]],
    var_by_model: dict[str, dict[str, float]],
    figure_dir: str | Path,
) -> str:
    """Plot how the residual-independence shortcut changes both portfolios."""

    output_dir = Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "problem5_residual_correlation.png"
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    labels = {
        "P1_long_A_long_B": "P1: long A + long B",
        "P2_long_A_short_B": "P2: long A - short B",
    }
    colors = {
        "full_residual_covariance": "#355c7d",
        "diagonal_residual_covariance": "#c06c84",
    }
    display = {
        "full_residual_covariance": "Full residual covariance",
        "diagonal_residual_covariance": "Independent residuals",
    }
    for axis, portfolio in zip(axes, labels, strict=True):
        all_values = np.concatenate(
            [
                pnl_by_model[portfolio]["full_residual_covariance"],
                pnl_by_model[portfolio]["diagonal_residual_covariance"],
            ]
        ) / 1_000.0
        bins = np.linspace(float(np.min(all_values)), float(np.max(all_values)), 80)
        for model in ("full_residual_covariance", "diagonal_residual_covariance"):
            values = pnl_by_model[portfolio][model] / 1_000.0
            axis.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.5,
                color=colors[model],
                label=display[model],
            )
            axis.axvline(
                -var_by_model[portfolio][model] / 1_000.0,
                color=colors[model],
                linewidth=1.1,
                linestyle="--",
            )
        axis.set_title(labels[portfolio], loc="left", fontweight="bold")
        axis.set_xlabel("One-day P&L ($ thousands)")
        axis.set_ylabel("Density")
        axis.grid(axis="y", alpha=0.20)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Problem 5: positive residual correlation raises P1 risk and lowers P2 risk",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return str(output_path)


def solve_problem5(
    csv_path: str | Path,
    figure_dir: str | Path | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Solve the model-based simulation exercise in Problem 5.

    The assignment says to assume zero expected returns.  Accordingly, the
    fitted intercepts are reported but risk draws are centered: the simulated
    market and residuals have mean zero, and stock draws use ``beta*r_M + eps``.
    This makes the simulation directly comparable with zero-mean delta-normal
    VaR.  Both residual models use the same Gaussian random numbers, isolating
    the effect of the off-diagonal residual covariance.
    """

    data = pd.read_csv(csv_path)
    _validate_columns(data, ("MKT", "A", "B"))
    prices = data[["MKT", "A", "B"]].to_numpy(dtype=float)
    if len(prices) < 3:
        raise ValueError("Problem 5 requires at least three price observations")
    if np.any(prices <= 0.0):
        raise ValueError("Arithmetic returns require strictly positive prices")
    returns = arithmetic_returns(prices)
    market = returns[:, 0]
    stocks = returns[:, 1:]
    n_returns = int(len(returns))

    ols_coefficients: list[np.ndarray] = []
    residual_columns: list[np.ndarray] = []
    for column in range(stocks.shape[1]):
        coefficients, residuals = _ols_with_intercept(market, stocks[:, column])
        ols_coefficients.append(coefficients)
        residual_columns.append(residuals)
    coefficients = np.vstack(ols_coefficients)
    alphas = coefficients[:, 0]
    betas = coefficients[:, 1]
    residuals = np.column_stack(residual_columns)

    residual_covariance = np.cov(residuals, rowvar=False, ddof=1)
    diagonal_residual_covariance = np.diag(np.diag(residual_covariance))
    residual_correlation = float(np.corrcoef(residuals, rowvar=False)[0, 1])
    market_variance = float(np.var(market, ddof=1))
    market_standard_deviation = float(np.sqrt(market_variance))
    sample_stock_covariance = np.cov(stocks, rowvar=False, ddof=1)

    # OLS makes sample cov(market, residual) zero.  Therefore this factor
    # reconstruction should equal the direct stock covariance up to rounding.
    factor_covariance_full = (
        np.outer(betas, betas) * market_variance + residual_covariance
    )
    factor_covariance_diagonal = (
        np.outer(betas, betas) * market_variance
        + diagonal_residual_covariance
    )
    covariance_reconstruction_error = float(
        np.max(np.abs(factor_covariance_full - sample_stock_covariance))
    )

    rng = np.random.default_rng(seed)
    standard_normals = rng.standard_normal((N_SIMULATIONS, 3))
    simulated_market = market_standard_deviation * standard_normals[:, 0]
    residual_cholesky = np.linalg.cholesky(residual_covariance)
    simulated_residuals_full = standard_normals[:, 1:] @ residual_cholesky.T
    simulated_residuals_diagonal = standard_normals[:, 1:] * np.sqrt(
        np.diag(residual_covariance)
    )

    simulated_stock_returns_full = (
        simulated_market[:, None] * betas[None, :] + simulated_residuals_full
    )
    simulated_stock_returns_diagonal = (
        simulated_market[:, None] * betas[None, :] + simulated_residuals_diagonal
    )

    portfolio_weights = {
        "P1_long_A_long_B": np.array([ONE_MILLION, ONE_MILLION]),
        "P2_long_A_short_B": np.array([ONE_MILLION, -ONE_MILLION]),
    }
    normal_multiplier = float(-stats.norm.ppf(DEFAULT_ALPHA))
    portfolio_results: dict[str, Any] = {}
    pnl_by_model: dict[str, dict[str, np.ndarray]] = {}
    var_by_model: dict[str, dict[str, float]] = {}
    for name, weights in portfolio_weights.items():
        pnl_full = simulated_stock_returns_full @ weights
        pnl_diagonal = simulated_stock_returns_diagonal @ weights
        simulation_var_full = _simulation_var(pnl_full)
        simulation_var_diagonal = _simulation_var(pnl_diagonal)
        delta_normal_var = normal_multiplier * float(
            np.sqrt(weights @ sample_stock_covariance @ weights)
        )
        analytic_factor_var_full = normal_multiplier * float(
            np.sqrt(weights @ factor_covariance_full @ weights)
        )
        analytic_factor_var_diagonal = normal_multiplier * float(
            np.sqrt(weights @ factor_covariance_diagonal @ weights)
        )
        difference = simulation_var_diagonal - simulation_var_full
        portfolio_results[name] = {
            "weights_dollars_A_B": [float(weights[0]), float(weights[1])],
            "simulation_var_5pct": {
                "full_residual_covariance": simulation_var_full,
                "diagonal_residual_covariance": simulation_var_diagonal,
                "diagonal_minus_full": difference,
                "absolute_difference": abs(difference),
                "diagonal_minus_full_pct_of_full": (
                    100.0 * difference / simulation_var_full
                ),
            },
            "analytic_normal_var_5pct": {
                "factor_model_full_residual_covariance": analytic_factor_var_full,
                "factor_model_diagonal_residual_covariance": (
                    analytic_factor_var_diagonal
                ),
                "direct_sample_stock_covariance_delta_normal": delta_normal_var,
                "simulation_full_minus_delta_normal": (
                    simulation_var_full - delta_normal_var
                ),
            },
        }
        pnl_by_model[name] = {
            "full_residual_covariance": pnl_full,
            "diagonal_residual_covariance": pnl_diagonal,
        }
        var_by_model[name] = {
            "full_residual_covariance": simulation_var_full,
            "diagonal_residual_covariance": simulation_var_diagonal,
        }

    p1_difference = portfolio_results["P1_long_A_long_B"][
        "simulation_var_5pct"
    ]["absolute_difference"]
    p2_difference = portfolio_results["P2_long_A_short_B"][
        "simulation_var_5pct"
    ]["absolute_difference"]
    more_exposed = (
        "P1_long_A_long_B" if p1_difference > p2_difference else "P2_long_A_short_B"
    )
    direction_check = {
        "P1_independence_understates_var": bool(
            portfolio_results["P1_long_A_long_B"]["simulation_var_5pct"][
                "diagonal_residual_covariance"
            ]
            < portfolio_results["P1_long_A_long_B"]["simulation_var_5pct"][
                "full_residual_covariance"
            ]
        ),
        "P2_independence_overstates_var": bool(
            portfolio_results["P2_long_A_short_B"]["simulation_var_5pct"][
                "diagonal_residual_covariance"
            ]
            > portfolio_results["P2_long_A_short_B"]["simulation_var_5pct"][
                "full_residual_covariance"
            ]
        ),
        "more_exposed_by_absolute_var_change": more_exposed,
    }

    figure_path = None
    if figure_dir is not None:
        figure_path = _plot_problem5(
            pnl_by_model, var_by_model, figure_dir=figure_dir
        )

    return {
        "conventions": {
            "returns": "daily arithmetic returns P_t/P_(t-1)-1",
            "ols": "ordinary least squares with an intercept",
            "residual_standard_deviation": "sample SD with denominator n-1",
            "covariance": "sample covariance with denominator n-1",
            "expected_returns": (
                "zero as required: fitted alpha is reported but omitted from "
                "risk draws; market and residual simulations are centered at zero"
            ),
            "simulation_var": (
                "course RiskStats.jl rule: negate the average of sorted P&L at "
                "one-based ranks floor(n*0.05) and ceil(n*0.05)"
            ),
            "delta_normal_var": (
                "-Phi^{-1}(0.05)*sqrt(w' Sigma w), with zero mean"
            ),
            "random_numbers": (
                "full and diagonal residual models use common Gaussian draws"
            ),
        },
        "seed": int(seed),
        "n_prices": int(len(prices)),
        "n_returns": n_returns,
        "n_simulations": N_SIMULATIONS,
        "sample_return_means": {
            "MKT": float(np.mean(market)),
            "A": float(np.mean(stocks[:, 0])),
            "B": float(np.mean(stocks[:, 1])),
        },
        "market_standard_deviation": market_standard_deviation,
        "ols": {
            "A": {
                "alpha": float(alphas[0]),
                "beta": float(betas[0]),
                "residual_standard_deviation": float(
                    np.std(residuals[:, 0], ddof=1)
                ),
            },
            "B": {
                "alpha": float(alphas[1]),
                "beta": float(betas[1]),
                "residual_standard_deviation": float(
                    np.std(residuals[:, 1], ddof=1)
                ),
            },
        },
        "residual_correlation": residual_correlation,
        "residual_covariance_full": residual_covariance.tolist(),
        "residual_covariance_diagonal": diagonal_residual_covariance.tolist(),
        "sample_stock_covariance": sample_stock_covariance.tolist(),
        "factor_reconstructed_stock_covariance": factor_covariance_full.tolist(),
        "maximum_absolute_covariance_reconstruction_error": (
            covariance_reconstruction_error
        ),
        "prediction": {
            "P1": "independence_understates_var",
            "P2": "independence_overstates_var",
            "basis": (
                "Positive residual covariance enters 2*w_A*w_B*cov(e_A,e_B): "
                "the term is positive for P1 and negative for P2."
            ),
        },
        "portfolios": portfolio_results,
        "direction_check": direction_check,
        "figure": figure_path,
    }
