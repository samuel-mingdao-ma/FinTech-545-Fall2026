"""Problem 4: marginal selection, copula fitting, and portfolio tail risk.

The public entry point is :func:`solve_problem4`.  It returns only ordinary
Python objects so that ``analysis.py`` can serialize the result directly to
JSON.  The implementation deliberately keeps marginal likelihoods separate
from copula likelihoods: the information-criterion comparison in part (d)
scores only the dependence model after the margins have been selected.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .common import corrected_aic, historical_var_es, sample_moments


PORTFOLIO_NOTIONAL_PER_ASSET = 1_000_000.0
N_SIMULATIONS = 100_000
LOWER_TAIL = 0.025
OUTER_REGION = 0.05
CDF_EPSILON = 1.0e-12


def _normal_margin_fit(values: np.ndarray) -> dict[str, float | int]:
    """Fit a Normal margin by maximum likelihood (MLE scale uses ``ddof=0``)."""

    location = float(np.mean(values))
    scale = float(np.sqrt(np.mean((values - location) ** 2)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("A marginal series must have positive finite variance.")
    log_likelihood = float(
        np.sum(stats.norm.logpdf(values, loc=location, scale=scale))
    )
    return {
        "location": location,
        "scale": scale,
        "log_likelihood": log_likelihood,
        "n_parameters": 2,
        "aicc": corrected_aic(log_likelihood, 2, values.size),
    }


def _student_t_margin_fit(values: np.ndarray) -> dict[str, float | int]:
    """Fit a location-scale Student-t margin by maximum likelihood."""

    degrees_of_freedom, location, scale = stats.t.fit(values)
    degrees_of_freedom = float(degrees_of_freedom)
    location = float(location)
    scale = float(scale)
    if (
        not np.isfinite(degrees_of_freedom)
        or degrees_of_freedom <= 0.0
        or not np.isfinite(scale)
        or scale <= 0.0
    ):
        raise RuntimeError("Student-t marginal MLE did not produce valid parameters.")
    log_likelihood = float(
        np.sum(
            stats.t.logpdf(
                values,
                degrees_of_freedom,
                loc=location,
                scale=scale,
            )
        )
    )
    implied_standard_deviation = (
        scale * np.sqrt(degrees_of_freedom / (degrees_of_freedom - 2.0))
        if degrees_of_freedom > 2.0
        else None
    )
    return {
        "degrees_of_freedom": degrees_of_freedom,
        "location": location,
        "scale": scale,
        "implied_standard_deviation": (
            None
            if implied_standard_deviation is None
            else float(implied_standard_deviation)
        ),
        "log_likelihood": log_likelihood,
        "n_parameters": 3,
        "aicc": corrected_aic(log_likelihood, 3, values.size),
    }


def _margin_cdf(
    values: np.ndarray, selected: str, fit: dict[str, Any]
) -> np.ndarray:
    """Transform observations through the selected fitted marginal CDF."""

    if selected == "normal":
        uniforms = stats.norm.cdf(
            values,
            loc=fit["location"],
            scale=fit["scale"],
        )
    elif selected == "student_t":
        uniforms = stats.t.cdf(
            values,
            fit["degrees_of_freedom"],
            loc=fit["location"],
            scale=fit["scale"],
        )
    else:  # pragma: no cover - guarded by the model-selection code
        raise ValueError(f"Unsupported margin: {selected}")
    return np.clip(np.asarray(uniforms, dtype=float), CDF_EPSILON, 1.0 - CDF_EPSILON)


def _margin_ppf(
    uniforms: np.ndarray, selected: str, fit: dict[str, Any]
) -> np.ndarray:
    """Map copula uniforms back through a selected fitted marginal."""

    u = np.clip(np.asarray(uniforms, dtype=float), CDF_EPSILON, 1.0 - CDF_EPSILON)
    if selected == "normal":
        return stats.norm.ppf(u, loc=fit["location"], scale=fit["scale"])
    if selected == "student_t":
        return stats.t.ppf(
            u,
            fit["degrees_of_freedom"],
            loc=fit["location"],
            scale=fit["scale"],
        )
    raise ValueError(f"Unsupported margin: {selected}")


def _make_strict_correlation(
    correlation: np.ndarray, eigenvalue_floor: float = 1.0e-10
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a symmetric, unit-diagonal, strictly positive-definite matrix.

    Pairwise Kendall estimates need not assemble into a positive-semidefinite
    matrix in every data set.  If repair is needed, negative/small eigenvalues
    are clipped and the result is rescaled back to a correlation matrix.
    """

    correlation_array = np.asarray(correlation, dtype=float)
    original = (correlation_array + correlation_array.T) / 2.0
    np.fill_diagonal(original, 1.0)
    eigenvalues_before = np.linalg.eigvalsh(original)
    repaired = original.copy()
    was_repaired = bool(np.min(eigenvalues_before) <= eigenvalue_floor)

    if was_repaired:
        eigenvalues, eigenvectors = np.linalg.eigh(repaired)
        eigenvalues = np.maximum(eigenvalues, eigenvalue_floor)
        repaired = (eigenvectors * eigenvalues) @ eigenvectors.T
        diagonal_scale = np.sqrt(np.diag(repaired))
        repaired = repaired / np.outer(diagonal_scale, diagonal_scale)
        repaired = (repaired + repaired.T) / 2.0
        np.fill_diagonal(repaired, 1.0)

        # Unit-diagonal rescaling can move the smallest eigenvalue by roundoff.
        minimum = float(np.min(np.linalg.eigvalsh(repaired)))
        if minimum <= eigenvalue_floor / 10.0:
            repaired += np.eye(repaired.shape[0]) * (eigenvalue_floor - minimum)
            diagonal_scale = np.sqrt(np.diag(repaired))
            repaired = repaired / np.outer(diagonal_scale, diagonal_scale)
            np.fill_diagonal(repaired, 1.0)

    eigenvalues_after = np.linalg.eigvalsh(repaired)
    return repaired, {
        "was_repaired": was_repaired,
        "eigenvalue_floor": float(eigenvalue_floor),
        "eigenvalues_before": [float(value) for value in eigenvalues_before],
        "eigenvalues_after": [float(value) for value in eigenvalues_after],
        "frobenius_distance": float(np.linalg.norm(repaired - original, ord="fro")),
    }


def _gaussian_copula_logpdf(
    uniforms: np.ndarray, correlation: np.ndarray
) -> np.ndarray:
    """Return observation-level Gaussian-copula log densities."""

    u = np.clip(np.asarray(uniforms, dtype=float), CDF_EPSILON, 1.0 - CDF_EPSILON)
    z = stats.norm.ppf(u)
    sign, log_determinant = np.linalg.slogdet(correlation)
    if sign <= 0:
        raise ValueError("Copula correlation matrix must be positive definite.")
    inverse = np.linalg.inv(correlation)
    quadratic = np.einsum("ni,ij,nj->n", z, inverse, z)
    independent_quadratic = np.einsum("ni,ni->n", z, z)
    # log c(u) = log phi_R(z) - sum_i log phi(z_i).
    return -0.5 * log_determinant - 0.5 * (
        quadratic - independent_quadratic
    )


def _student_t_copula_logpdf(
    uniforms: np.ndarray, correlation: np.ndarray, degrees_of_freedom: float
) -> np.ndarray:
    """Return observation-level Student-t-copula log densities.

    ``correlation`` is the multivariate-t shape matrix.  Subtracting the three
    univariate t log densities leaves the copula density, including all gamma
    and normalizing constants.
    """

    u = np.clip(np.asarray(uniforms, dtype=float), CDF_EPSILON, 1.0 - CDF_EPSILON)
    z = stats.t.ppf(u, degrees_of_freedom)
    dimension = z.shape[1]
    sign, log_determinant = np.linalg.slogdet(correlation)
    if sign <= 0:
        raise ValueError("Copula correlation matrix must be positive definite.")
    inverse = np.linalg.inv(correlation)
    quadratic = np.einsum("ni,ij,nj->n", z, inverse, z)
    multivariate_logpdf = (
        special.gammaln((degrees_of_freedom + dimension) / 2.0)
        - special.gammaln(degrees_of_freedom / 2.0)
        - 0.5 * log_determinant
        - dimension / 2.0 * np.log(degrees_of_freedom * np.pi)
        - (degrees_of_freedom + dimension)
        / 2.0
        * np.log1p(quadratic / degrees_of_freedom)
    )
    univariate_logpdf = np.sum(
        stats.t.logpdf(z, degrees_of_freedom), axis=1
    )
    return multivariate_logpdf - univariate_logpdf


def _profile_student_t_copula(
    uniforms: np.ndarray,
    correlation: np.ndarray,
    bounds: tuple[float, float] = (2.01, 200.0),
) -> tuple[float, np.ndarray]:
    """Profile the t-copula likelihood over its degrees of freedom."""

    objective = lambda degrees: -float(
        np.sum(_student_t_copula_logpdf(uniforms, correlation, degrees))
    )
    result = optimize.minimize_scalar(
        objective,
        bounds=bounds,
        method="bounded",
        options={"xatol": 1.0e-9, "maxiter": 500},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"t-copula profile optimization failed: {result.message}")
    degrees_of_freedom = float(result.x)
    if min(degrees_of_freedom - bounds[0], bounds[1] - degrees_of_freedom) < 1.0e-5:
        raise RuntimeError("t-copula optimum is on a profiling boundary.")
    contributions = _student_t_copula_logpdf(
        uniforms, correlation, degrees_of_freedom
    )
    return degrees_of_freedom, contributions


def _portfolio_risk(pnl: np.ndarray) -> dict[str, float]:
    """Return positive-loss 5% and 1% VaR and ES under the course convention."""

    var_5, es_5 = historical_var_es(pnl, alpha=0.05)
    var_1, es_1 = historical_var_es(pnl, alpha=0.01)
    if es_5 + 1.0e-9 < var_5 or es_1 + 1.0e-9 < var_1:
        raise RuntimeError("ES must not be smaller than VaR for these loss samples.")
    return {
        "var_5pct": float(var_5),
        "es_5pct": float(es_5),
        "var_1pct": float(var_1),
        "es_1pct": float(es_1),
        "mean_pnl": float(np.mean(pnl)),
        "standard_deviation_pnl": float(np.std(pnl, ddof=1)),
    }
def _pair_key(columns: list[str], first: int, second: int) -> str:
    return f"{columns[first]}-{columns[second]}"


def _plot_rank_pairs(
    rank_uniforms: np.ndarray,
    columns: list[str],
    pair_counts: dict[str, dict[str, float | int]],
    output_path: Path,
) -> None:
    """Plot every pair of pseudo-observations and highlight joint tails."""

    pairs = list(combinations(range(len(columns)), 2))
    figure, axes = plt.subplots(1, len(pairs), figsize=(13.2, 4.15), sharex=True, sharey=True)
    if len(pairs) == 1:
        axes = [axes]

    for axis, (first, second) in zip(axes, pairs):
        x = rank_uniforms[:, first]
        y = rank_uniforms[:, second]
        lower = (x <= LOWER_TAIL) & (y <= LOWER_TAIL)
        upper = (x >= 1.0 - LOWER_TAIL) & (y >= 1.0 - LOWER_TAIL)

        axis.scatter(x, y, s=10, alpha=0.28, color="#526777", linewidths=0)
        axis.scatter(x[lower], y[lower], s=24, color="#D95F4F", label="joint lower")
        axis.scatter(x[upper], y[upper], s=24, color="#168C87", label="joint upper")
        axis.axvline(LOWER_TAIL, color="#D95F4F", linewidth=0.8, alpha=0.75)
        axis.axhline(LOWER_TAIL, color="#D95F4F", linewidth=0.8, alpha=0.75)
        axis.axvline(1.0 - LOWER_TAIL, color="#168C87", linewidth=0.8, alpha=0.75)
        axis.axhline(1.0 - LOWER_TAIL, color="#168C87", linewidth=0.8, alpha=0.75)
        key = _pair_key(columns, first, second)
        counts = pair_counts[key]
        axis.set_title(
            f"{columns[first]} and {columns[second]}\n"
            f"lower={counts['observed_lower_count']}, upper={counts['observed_upper_count']}",
            fontsize=10,
            fontweight="bold",
        )
        axis.set_xlabel(f"{columns[first]} rank / (n + 1)")
        axis.set_ylabel(f"{columns[second]} rank / (n + 1)")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.16, linewidth=0.5)

    figure.suptitle(
        "Problem 4: marginal rank pairs (independence expects 0.625 days in each joint tail)",
        fontsize=12,
        fontweight="bold",
        color="#17324D",
        y=1.02,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def solve_problem4(
    csv_path: str | Path,
    figure_dir: str | Path | None = None,
    seed: int = 549,
) -> dict[str, Any]:
    """Solve every numerical and graphical part of Assignment 2 Problem 4.

    The three holdings are treated as dollar P&L approximations
    ``notional * return``.  Historical and simulated VaR/ES are positive loss
    amounts.  The two copula simulations use common correlated Normal draws to
    reduce Monte Carlo noise in the model comparison; the t copula adds one
    common chi-square scale draw per simulated day.
    """

    csv_path = Path(csv_path)
    data_frame = pd.read_csv(csv_path)
    if data_frame.shape[1] != 3:
        raise ValueError("Problem 4 requires exactly three return series.")
    if data_frame.shape[0] < 20:
        raise ValueError("Problem 4 requires enough observations to estimate tails.")
    if data_frame.isna().any().any():
        raise ValueError("Problem 4 returns must not contain missing values.")
    try:
        values = data_frame.astype(float).to_numpy()
    except (TypeError, ValueError) as error:
        raise ValueError("Problem 4 columns must be numeric.") from error
    if not np.isfinite(values).all():
        raise ValueError("Problem 4 returns must be finite.")

    n_observations, dimension = values.shape
    columns = [str(column) for column in data_frame.columns]
    moments: dict[str, dict[str, float]] = {}
    sensitivity: dict[str, dict[str, Any]] = {}
    for index, column in enumerate(columns):
        series = values[:, index]
        full_moments = sample_moments(series)
        # Distance from the median identifies one extreme observation without
        # allowing a shifted mean to influence which point is removed.
        extreme_index = int(np.argmax(np.abs(series - np.median(series))))
        without_extreme = np.delete(series, extreme_index)
        reduced_moments = sample_moments(without_extreme)
        moments[column] = full_moments
        sensitivity[column] = {
            "removed_observation_number_1_based": extreme_index + 1,
            "removed_value": float(series[extreme_index]),
            "moments_without_observation": reduced_moments,
            "change_without_minus_full": {
                key: float(reduced_moments[key] - full_moments[key])
                for key in full_moments
            },
        }

    # rank/(n+1) stays strictly inside (0,1), avoiding infinite inverse CDFs.
    rank_uniforms = np.column_stack(
        [stats.rankdata(values[:, index], method="average") / (n_observations + 1.0)
         for index in range(dimension)]
    )
    independence_expected = float(n_observations * LOWER_TAIL**2)
    pair_counts: dict[str, dict[str, float | int]] = {}
    for first, second in combinations(range(dimension), 2):
        lower = (rank_uniforms[:, first] <= LOWER_TAIL) & (
            rank_uniforms[:, second] <= LOWER_TAIL
        )
        upper = (rank_uniforms[:, first] >= 1.0 - LOWER_TAIL) & (
            rank_uniforms[:, second] >= 1.0 - LOWER_TAIL
        )
        pair_counts[_pair_key(columns, first, second)] = {
            "observed_lower_count": int(np.sum(lower)),
            "observed_upper_count": int(np.sum(upper)),
            "independence_expected_each_tail": independence_expected,
        }

    rank_figure: str | None = None
    if figure_dir is not None:
        figure_path = Path(figure_dir) / "problem4_rank_pairs.png"
        _plot_rank_pairs(rank_uniforms, columns, pair_counts, figure_path)
        rank_figure = str(figure_path)

    margin_fits: dict[str, dict[str, Any]] = {}
    selected_margin_uniforms = np.empty_like(values)
    selected_margin_details: list[tuple[str, dict[str, Any]]] = []
    for index, column in enumerate(columns):
        normal_fit = _normal_margin_fit(values[:, index])
        student_t_fit = _student_t_margin_fit(values[:, index])
        selected = (
            "normal"
            if normal_fit["aicc"] <= student_t_fit["aicc"]
            else "student_t"
        )
        selected_fit = normal_fit if selected == "normal" else student_t_fit
        selected_margin_uniforms[:, index] = _margin_cdf(
            values[:, index], selected, selected_fit
        )
        selected_margin_details.append((selected, selected_fit))
        margin_fits[column] = {
            "normal": normal_fit,
            "student_t": student_t_fit,
            "selected_by_aicc": selected,
            "aicc_advantage_of_selected": float(
                abs(normal_fit["aicc"] - student_t_fit["aicc"])
            ),
        }

    kendall_tau = np.eye(dimension)
    for first, second in combinations(range(dimension), 2):
        tau = float(
            stats.kendalltau(
                values[:, first], values[:, second], nan_policy="raise"
            ).statistic
        )
        kendall_tau[first, second] = tau
        kendall_tau[second, first] = tau
    correlation_raw = np.sin(np.pi * kendall_tau / 2.0)
    np.fill_diagonal(correlation_raw, 1.0)
    correlation, repair_details = _make_strict_correlation(correlation_raw)

    gaussian_daily_log_likelihood = _gaussian_copula_logpdf(
        selected_margin_uniforms, correlation
    )
    t_degrees_of_freedom, t_daily_log_likelihood = _profile_student_t_copula(
        selected_margin_uniforms, correlation
    )
    if not (
        np.isfinite(gaussian_daily_log_likelihood).all()
        and np.isfinite(t_daily_log_likelihood).all()
    ):
        raise RuntimeError("Copula likelihood contains a non-finite contribution.")

    gaussian_log_likelihood = float(np.sum(gaussian_daily_log_likelihood))
    t_log_likelihood = float(np.sum(t_daily_log_likelihood))
    n_pair_parameters = dimension * (dimension - 1) // 2
    gaussian_k = n_pair_parameters
    t_k = n_pair_parameters + 1
    gaussian_aicc = corrected_aic(
        gaussian_log_likelihood, gaussian_k, n_observations
    )
    t_aicc = corrected_aic(t_log_likelihood, t_k, n_observations)
    gaussian_bic = float(
        -2.0 * gaussian_log_likelihood + gaussian_k * np.log(n_observations)
    )
    t_bic = float(-2.0 * t_log_likelihood + t_k * np.log(n_observations))

    rng = np.random.default_rng(int(seed))
    cholesky = np.linalg.cholesky(correlation)
    independent_normals = rng.standard_normal((N_SIMULATIONS, dimension))
    correlated_normals = independent_normals @ cholesky.T
    gaussian_simulated_uniforms = stats.norm.cdf(correlated_normals)
    common_scale = np.sqrt(
        rng.chisquare(t_degrees_of_freedom, size=N_SIMULATIONS)
        / t_degrees_of_freedom
    )
    t_latent = correlated_normals / common_scale[:, None]
    t_simulated_uniforms = stats.t.cdf(t_latent, t_degrees_of_freedom)

    def simulated_returns(uniforms: np.ndarray) -> np.ndarray:
        result = np.empty_like(uniforms)
        for index, (selected, fit) in enumerate(selected_margin_details):
            result[:, index] = _margin_ppf(uniforms[:, index], selected, fit)
        return result

    gaussian_returns = simulated_returns(gaussian_simulated_uniforms)
    t_returns = simulated_returns(t_simulated_uniforms)
    historical_pnl = PORTFOLIO_NOTIONAL_PER_ASSET * np.sum(values, axis=1)
    gaussian_pnl = PORTFOLIO_NOTIONAL_PER_ASSET * np.sum(gaussian_returns, axis=1)
    t_pnl = PORTFOLIO_NOTIONAL_PER_ASSET * np.sum(t_returns, axis=1)
    portfolio_risk = {
        "historical": _portfolio_risk(historical_pnl),
        "gaussian_copula": _portfolio_risk(gaussian_pnl),
        "student_t_copula": _portfolio_risk(t_pnl),
    }

    for first, second in combinations(range(dimension), 2):
        key = _pair_key(columns, first, second)
        for model, uniforms in (
            ("gaussian_copula", gaussian_simulated_uniforms),
            ("student_t_copula", t_simulated_uniforms),
        ):
            lower_probability = float(
                np.mean(
                    (uniforms[:, first] <= LOWER_TAIL)
                    & (uniforms[:, second] <= LOWER_TAIL)
                )
            )
            upper_probability = float(
                np.mean(
                    (uniforms[:, first] >= 1.0 - LOWER_TAIL)
                    & (uniforms[:, second] >= 1.0 - LOWER_TAIL)
                )
            )
            pair_counts[key][f"{model}_lower_days_per_1000"] = float(
                1000.0 * lower_probability
            )
            pair_counts[key][f"{model}_upper_days_per_1000"] = float(
                1000.0 * upper_probability
            )

    likelihood_difference = (
        t_daily_log_likelihood - gaussian_daily_log_likelihood
    )
    central_day_mask = np.all(
        (rank_uniforms > OUTER_REGION)
        & (rank_uniforms < 1.0 - OUTER_REGION),
        axis=1,
    )
    total_difference = float(np.sum(likelihood_difference))
    central_difference = float(np.sum(likelihood_difference[central_day_mask]))
    top_positive_indices = np.argsort(likelihood_difference)[-10:][::-1]
    top_negative_indices = np.argsort(likelihood_difference)[:10]

    strongest_first, strongest_second = max(
        combinations(range(dimension), 2),
        key=lambda pair: correlation[pair[0], pair[1]],
    )
    strongest_rho = float(correlation[strongest_first, strongest_second])
    t_tail_dependence = float(
        2.0
        * stats.t.cdf(
            -np.sqrt(
                (t_degrees_of_freedom + 1.0)
                * (1.0 - strongest_rho)
                / (1.0 + strongest_rho)
            ),
            t_degrees_of_freedom + 1.0,
        )
    )

    return {
        "conventions": {
            "n_observations": int(n_observations),
            "holdings_usd_per_asset": PORTFOLIO_NOTIONAL_PER_ASSET,
            "var_es_reported_as": "positive loss",
            "var_quantile_method": "course RiskStats.jl: average the floor(n*alpha) and ceil(n*alpha) order statistics using one-based indices",
            "es_rule": "negative mean P&L at or below the empirical quantile, including ties",
            "rank_transform": "average rank / (n + 1)",
            "joint_tail_threshold_each_side": LOWER_TAIL,
            "central_day_definition": "all three empirical rank uniforms strictly between 0.05 and 0.95",
            "margin_parameter_counts": {"normal": 2, "student_t": 3},
            "copula_parameter_counts": {
                "gaussian": gaussian_k,
                "student_t": t_k,
                "explanation": "three pairwise correlations; t copula adds degrees of freedom",
            },
            "copula_correlation_estimator": "rho_ij = sin(pi * Kendall_tau_ij / 2)",
            "t_copula_df_profile_bounds": [2.01, 200.0],
            "n_simulations": N_SIMULATIONS,
            "simulation_seed": int(seed),
            "simulation_variance_reduction": "common correlated Normal draws for the two copulas",
        },
        "moments": moments,
        "single_observation_sensitivity": sensitivity,
        "rank_analysis": {
            "independence_expected_each_pair_each_tail": independence_expected,
            "pairs": pair_counts,
            "figure": rank_figure,
        },
        "margin_fits": margin_fits,
        "copula_fit": {
            "kendall_tau_matrix": kendall_tau.tolist(),
            "correlation_matrix_raw": correlation_raw.tolist(),
            "correlation_matrix_used": correlation.tolist(),
            "correlation_repair": repair_details,
            "gaussian": {
                "log_likelihood": gaussian_log_likelihood,
                "n_parameters": gaussian_k,
                "aicc": gaussian_aicc,
                "bic": gaussian_bic,
            },
            "student_t": {
                "degrees_of_freedom": t_degrees_of_freedom,
                "log_likelihood": t_log_likelihood,
                "n_parameters": t_k,
                "aicc": t_aicc,
                "bic": t_bic,
            },
            "winner_by_aicc": (
                "student_t" if t_aicc < gaussian_aicc else "gaussian"
            ),
            "winner_by_bic": (
                "student_t" if t_bic < gaussian_bic else "gaussian"
            ),
            "student_t_aicc_advantage": float(gaussian_aicc - t_aicc),
            "student_t_bic_advantage": float(gaussian_bic - t_bic),
        },
        "portfolio_risk_usd": portfolio_risk,
        "likelihood_contributions": {
            "daily_t_minus_gaussian": [
                float(value) for value in likelihood_difference
            ],
            "total_t_minus_gaussian": total_difference,
            "central_day_count": int(np.sum(central_day_mask)),
            "central_day_contribution": central_difference,
            "central_share_of_total": float(central_difference / total_difference),
            "noncentral_day_contribution": float(total_difference - central_difference),
            "largest_positive_days": [
                {
                    "observation_number_1_based": int(index) + 1,
                    "t_minus_gaussian": float(likelihood_difference[index]),
                    "is_central_day": bool(central_day_mask[index]),
                }
                for index in top_positive_indices
            ],
            "largest_negative_days": [
                {
                    "observation_number_1_based": int(index) + 1,
                    "t_minus_gaussian": float(likelihood_difference[index]),
                    "is_central_day": bool(central_day_mask[index]),
                }
                for index in top_negative_indices
            ],
        },
        "tail_dependence": {
            "most_correlated_pair": _pair_key(
                columns, strongest_first, strongest_second
            ),
            "rho": strongest_rho,
            "student_t_lower_and_upper": t_tail_dependence,
            "gaussian_lower_and_upper": 0.0,
        },
    }
