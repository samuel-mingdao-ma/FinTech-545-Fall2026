"""Numerical solutions for Assignment 2, Problems 1 and 2.

The public functions return only JSON-serializable Python objects.  Matrix
payloads keep their row/column labels next to ordinary nested lists so that a
caller can use the results in either a report or a machine-readable output
file without relying on pandas' JSON conventions.

The few conventions that materially affect the answers are stated in each
result dictionary.  In particular, the empirical quantile follows the course
``RiskStats.jl`` order-statistic convention rather than NumPy's default linear
interpolation.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats


ALPHA = 0.05
POSITION_VALUE = 1_000_000.0


def _matrix_payload(matrix: np.ndarray, labels: Iterable[str]) -> dict[str, Any]:
    """Return a labeled matrix made only from JSON-native objects."""

    names = [str(x) for x in labels]
    return {
        "rows": names,
        "columns": names,
        "values": np.asarray(matrix, dtype=float).tolist(),
    }


def _sample_moments(values: np.ndarray) -> dict[str, float]:
    """Mean, sample variance, corrected skewness, and excess kurtosis."""

    x = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "variance": float(np.var(x, ddof=1)),
        "standard_deviation": float(np.std(x, ddof=1)),
        "skewness": float(stats.skew(x, bias=False)),
        "excess_kurtosis": float(
            stats.kurtosis(x, fisher=True, bias=False)
        ),
    }


def _cholesky_report(matrix: np.ndarray) -> dict[str, Any]:
    """Attempt the ordinary positive-definite Cholesky factorization."""

    try:
        root = np.linalg.cholesky(np.asarray(matrix, dtype=float))
    except np.linalg.LinAlgError as exc:
        return {
            "success": False,
            "message": str(exc),
            "lower_triangular_factor": None,
        }
    return {
        "success": True,
        "message": "Ordinary Cholesky factorization succeeded.",
        "lower_triangular_factor": root.tolist(),
    }


def _project_psd(matrix: np.ndarray) -> np.ndarray:
    """Orthogonally project a symmetric matrix onto the PSD cone."""

    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T


def _rebonato_jackel(
    correlation: np.ndarray, eigenvalue_floor: float = 0.0
) -> np.ndarray:
    """Apply the Rebonato-Jackel spectral correlation repair.

    Negative eigenvalues are clipped, then the spectral root is rescaled row
    by row so the reconstructed matrix again has a unit diagonal.  This is the
    ``near_psd`` method presented in the Week 3 notes.
    """

    matrix = np.asarray(correlation, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    clipped = np.maximum(eigenvalues, eigenvalue_floor)

    # diag(S diag(lambda+) S') is (S**2) @ lambda+.  Its inverse square
    # root is precisely the row scaling in the Rebonato-Jackel construction.
    reconstructed_diagonal = (eigenvectors**2) @ clipped
    if np.any(reconstructed_diagonal <= 0.0):
        raise ValueError("Spectral repair produced a non-positive diagonal.")
    scale = 1.0 / np.sqrt(reconstructed_diagonal)
    root = (scale[:, None] * eigenvectors) * np.sqrt(clipped)[None, :]
    repaired = root @ root.T
    repaired = (repaired + repaired.T) / 2.0
    np.fill_diagonal(repaired, 1.0)
    return repaired


def _higham_nearest_correlation(
    correlation: np.ndarray,
    tolerance: float = 1.0e-14,
    max_iterations: int = 10_000,
) -> tuple[np.ndarray, int, float]:
    """Return the unweighted nearest correlation matrix via Higham/Dykstra.

    The iteration alternates the PSD and unit-diagonal projections and carries
    Dykstra's correction.  The unweighted Frobenius norm is used, matching the
    assignment and Week 3 notes.
    """

    y = np.asarray(correlation, dtype=float).copy()
    y = (y + y.T) / 2.0
    delta_s = np.zeros_like(y)
    final_change = math.inf

    for iteration in range(1, max_iterations + 1):
        previous = y.copy()
        r = y - delta_s
        x = _project_psd(r)
        delta_s = x - r
        y = x.copy()
        np.fill_diagonal(y, 1.0)
        y = (y + y.T) / 2.0
        final_change = float(np.linalg.norm(y - previous, ord="fro"))
        if final_change < tolerance:
            return y, iteration, final_change

    raise RuntimeError(
        "Higham nearest-correlation iteration did not converge within "
        f"{max_iterations} iterations (last change={final_change:.3e})."
    )


def _entry_changes(
    original: np.ndarray, repaired: np.ndarray, labels: list[str]
) -> list[dict[str, float | str]]:
    """Return unique off-diagonal entry changes, largest first."""

    changes: list[dict[str, float | str]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            signed = float(repaired[i, j] - original[i, j])
            changes.append(
                {
                    "row": labels[i],
                    "column": labels[j],
                    "original": float(original[i, j]),
                    "repaired": float(repaired[i, j]),
                    "signed_change": signed,
                    "absolute_change": abs(signed),
                }
            )
    changes.sort(key=lambda item: float(item["absolute_change"]), reverse=True)
    return changes


def _repair_payload(
    repaired: np.ndarray,
    original: np.ndarray,
    full_history_standard_deviations: np.ndarray,
    tracking_weights: np.ndarray,
    labels: list[str],
    *,
    iterations: int | None = None,
    final_iterate_change: float | None = None,
) -> dict[str, Any]:
    """Build the common diagnostics requested for one correlation repair."""

    eigenvalues = np.linalg.eigvalsh(repaired)
    covariance = (
        np.diag(full_history_standard_deviations)
        @ repaired
        @ np.diag(full_history_standard_deviations)
    )
    changes = _entry_changes(original, repaired, labels)
    payload: dict[str, Any] = {
        "correlation": _matrix_payload(repaired, labels),
        "eigenvalues_ascending": eigenvalues.tolist(),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "is_psd_at_1e_minus_12": bool(eigenvalues[0] >= -1.0e-12),
        "frobenius_distance_from_pairwise": float(
            np.linalg.norm(repaired - original, ord="fro")
        ),
        "maximum_absolute_off_diagonal_change": float(
            changes[0]["absolute_change"]
        ),
        "largest_off_diagonal_changes": changes,
        "tracking_portfolio_variance": float(
            tracking_weights @ covariance @ tracking_weights
        ),
    }
    if iterations is not None:
        payload["iterations"] = int(iterations)
    if final_iterate_change is not None:
        payload["final_iterate_frobenius_change"] = float(final_iterate_change)
    return payload


def solve_problem1(
    csv_path: str | os.PathLike[str],
    figure_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Solve Assignment 2 Problem 1 and return labeled diagnostics.

    ``figure_dir`` is accepted for a uniform public API with the other problem
    solvers.  Problem 1 requests tables rather than a plot, so no figure is
    created and the returned ``figure_path`` is ``None``.
    """

    del figure_dir  # Deliberately unused; Problem 1 has no required figure.
    path = Path(csv_path)
    frame = pd.read_csv(path)
    required = ["A", "B", "C", "D", "IDX"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Problem 1 data are missing columns: {missing}")
    data = frame[required].apply(pd.to_numeric, errors="coerce")

    observed = data.notna().astype(int)
    joint_counts = observed.T @ observed
    all_five_count = int(data.notna().all(axis=1).sum())

    complete_data = data.dropna(axis=0, how="any")
    complete_correlation = complete_data.corr().to_numpy(dtype=float)
    pairwise_correlation = data.corr(min_periods=2).to_numpy(dtype=float)
    complete_eigenvalues = np.linalg.eigvalsh(complete_correlation)
    pairwise_eigenvalues = np.linalg.eigvalsh(pairwise_correlation)

    full_history_sd = data.std(axis=0, ddof=1).to_numpy(dtype=float)
    sd_diagonal = np.diag(full_history_sd)
    pairwise_covariance = (
        sd_diagonal @ pairwise_correlation @ sd_diagonal
    )
    tracking_weights = np.array([-0.40, -0.30, -0.20, -0.10, 1.0])
    tracking_variance = float(
        tracking_weights @ pairwise_covariance @ tracking_weights
    )

    rebonato_jackel = _rebonato_jackel(pairwise_correlation)
    higham, iterations, final_change = _higham_nearest_correlation(
        pairwise_correlation
    )
    rj_payload = _repair_payload(
        rebonato_jackel,
        pairwise_correlation,
        full_history_sd,
        tracking_weights,
        required,
    )
    higham_payload = _repair_payload(
        higham,
        pairwise_correlation,
        full_history_sd,
        tracking_weights,
        required,
        iterations=iterations,
        final_iterate_change=final_change,
    )

    estimator_difference = complete_correlation - pairwise_correlation
    estimator_changes = _entry_changes(
        pairwise_correlation, complete_correlation, required
    )
    estimator_frobenius_gap = float(
        np.linalg.norm(estimator_difference, ord="fro")
    )
    higham_repair_distance = float(
        higham_payload["frobenius_distance_from_pairwise"]
    )

    # Lowest joint counts identify the least precisely estimated correlations.
    reliability_order: list[dict[str, int | str]] = []
    for i in range(len(required)):
        for j in range(i + 1, len(required)):
            reliability_order.append(
                {
                    "row": required[i],
                    "column": required[j],
                    "joint_observations": int(joint_counts.iloc[i, j]),
                }
            )
    reliability_order.sort(key=lambda item: int(item["joint_observations"]))

    return {
        "problem": 1,
        "source_file": str(path),
        "conventions": {
            "series_columns": required,
            "day_column_excluded": True,
            "correlation": (
                "Pearson correlation. Complete-case uses the same all-five "
                "rows for every entry; pairwise uses every jointly observed "
                "row separately for each entry."
            ),
            "standard_deviation": (
                "Sample standard deviation (ddof=1) from each series' full "
                "available history."
            ),
            "covariance_reconstruction": "Sigma = diag(sd) R diag(sd).",
            "tracking_weights_order_A_B_C_D_IDX": tracking_weights.tolist(),
            "repair_distance": "Unweighted Frobenius norm.",
            "rebonato_jackel": (
                "Clip correlation eigenvalues below zero, then rescale the "
                "spectral root to restore a unit diagonal."
            ),
            "higham": (
                "Unweighted Higham alternating PSD/unit-diagonal projections "
                "with Dykstra correction; convergence tolerance 1e-14."
            ),
            "numerical_psd_tolerance": 1.0e-12,
        },
        "sample": {
            "rows": int(len(data)),
            "complete_all_five_rows": all_five_count,
            "non_missing_by_series": {
                column: int(data[column].notna().sum()) for column in required
            },
        },
        "joint_observation_counts": {
            "rows": required,
            "columns": required,
            "values": joint_counts.to_numpy(dtype=int).tolist(),
        },
        "least_reliable_pairs_by_joint_count": reliability_order,
        "complete_case": {
            "n_rows": int(len(complete_data)),
            "correlation": _matrix_payload(complete_correlation, required),
            "eigenvalues_ascending": complete_eigenvalues.tolist(),
            "minimum_eigenvalue": float(complete_eigenvalues[0]),
            "is_psd_at_1e_minus_12": bool(
                complete_eigenvalues[0] >= -1.0e-12
            ),
            "cholesky": _cholesky_report(complete_correlation),
        },
        "pairwise": {
            "correlation": _matrix_payload(pairwise_correlation, required),
            "eigenvalues_ascending": pairwise_eigenvalues.tolist(),
            "minimum_eigenvalue": float(pairwise_eigenvalues[0]),
            "is_psd_at_1e_minus_12": bool(
                pairwise_eigenvalues[0] >= -1.0e-12
            ),
            "cholesky": _cholesky_report(pairwise_correlation),
            "full_history_standard_deviations": {
                required[i]: float(full_history_sd[i])
                for i in range(len(required))
            },
            "covariance": _matrix_payload(pairwise_covariance, required),
            "tracking_portfolio_variance": tracking_variance,
        },
        "repairs": {
            "rebonato_jackel": rj_payload,
            "higham": higham_payload,
            "frobenius_distance_between_repairs": float(
                np.linalg.norm(rebonato_jackel - higham, ord="fro")
            ),
        },
        "estimator_gap": {
            "complete_case_vs_pairwise_frobenius_distance": (
                estimator_frobenius_gap
            ),
            "maximum_absolute_off_diagonal_gap": float(
                estimator_changes[0]["absolute_change"]
            ),
            "largest_off_diagonal_gaps": estimator_changes,
            "gap_to_higham_repair_distance_ratio": float(
                estimator_frobenius_gap / higham_repair_distance
            ),
        },
        "reconcile": {
            "negative_variance_explanation": (
                "The pairwise matrix has a negative eigenvalue, so it is not "
                "PSD: there exists a weight vector with w' Sigma w < 0. The "
                "tracking position is close to that inconsistent direction "
                "and produces the reported negative variance."
            ),
            "higham_largest_move": higham_payload[
                "largest_off_diagonal_changes"
            ][0],
            "repair_uses": (
                "Matrix geometry and equal Frobenius cost per entry, not the "
                "number of observations behind each correlation."
            ),
            "dominant_decision": "estimator",
            "dominant_decision_explanation": (
                "The complete-case/pairwise gap is far larger than either "
                "repair distance, so estimator choice matters more here than "
                "the choice between the two repairs."
            ),
        },
        "figure_path": None,
    }


def _exponential_weights(n_obs: int, decay: float) -> np.ndarray:
    """Normalized finite-sample EW weights ordered oldest to newest."""

    if n_obs < 1:
        raise ValueError("n_obs must be positive.")
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must lie strictly between zero and one.")
    powers = np.arange(n_obs - 1, -1, -1, dtype=float)
    weights = decay**powers
    return weights / weights.sum()


def _course_empirical_quantile(values: np.ndarray, alpha: float) -> float:
    """Return the empirical quantile used by the course ``RiskStats.jl``.

    After sorting ``n`` observations, let ``ndn=floor(n*alpha)`` and
    ``nup=ceil(n*alpha)`` using Julia's one-based indexing.  The quantile is the
    50/50 average of those two order statistics.  When ``n*alpha`` is an
    integer, both refer to the same observation.
    """

    ordered = np.sort(np.asarray(values, dtype=float))
    n_obs = len(ordered)
    if n_obs == 0:
        raise ValueError("Cannot calculate a quantile from an empty sample.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one.")
    scaled = n_obs * alpha
    # Clamp only guards extreme alpha/n combinations that would otherwise
    # create Julia index zero; it does not affect this assignment (n*alpha=25).
    ndn = max(1, min(n_obs, math.floor(scaled)))
    nup = max(1, min(n_obs, math.ceil(scaled)))
    return float(0.5 * (ordered[ndn - 1] + ordered[nup - 1]))


def _profile_variance_break(
    demeaned_returns: np.ndarray, minimum_segment: int = 20
) -> dict[str, float | int]:
    """Profile a single Gaussian mean/variance break as a plot cross-check."""

    returns = np.asarray(demeaned_returns, dtype=float)
    best: tuple[float, int, float, float] | None = None
    for split in range(minimum_segment, len(returns) - minimum_segment + 1):
        left = returns[:split]
        right = returns[split:]
        left_mle_variance = float(np.mean((left - left.mean()) ** 2))
        right_mle_variance = float(np.mean((right - right.mean()) ** 2))
        if left_mle_variance <= 0.0 or right_mle_variance <= 0.0:
            continue
        log_likelihood = float(
            -0.5
            * len(left)
            * (math.log(2.0 * math.pi * left_mle_variance) + 1.0)
            - 0.5
            * len(right)
            * (math.log(2.0 * math.pi * right_mle_variance) + 1.0)
        )
        candidate = (
            log_likelihood,
            split,
            float(np.std(left, ddof=1)),
            float(np.std(right, ddof=1)),
        )
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise ValueError("No valid variance-break split was available.")
    return {
        "split_after_return_number": int(best[1]),
        "recent_days": int(len(returns) - best[1]),
        "profile_log_likelihood": float(best[0]),
        "earlier_sample_standard_deviation": float(best[2]),
        "recent_sample_standard_deviation": float(best[3]),
    }


def _plot_problem2_returns(
    return_days: np.ndarray,
    demeaned_returns: np.ndarray,
    recent_days: int,
    figure_dir: str | os.PathLike[str],
) -> str:
    """Create the Problem 2 return-series figure and return its absolute path."""

    # Matplotlib otherwise attempts to create a cache in the user's home.
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fintech545-mpl")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (output_dir / "problem2_demeaned_returns.png").resolve()

    fig, axis = plt.subplots(figsize=(9.0, 4.2))
    axis.plot(
        return_days,
        100.0 * demeaned_returns,
        color="#215a83",
        linewidth=0.85,
    )
    first_recent_index = len(demeaned_returns) - recent_days
    first_recent_day = float(return_days[first_recent_index])
    axis.axvspan(
        first_recent_day - 0.5,
        float(return_days[-1]) + 0.5,
        color="#d95f02",
        alpha=0.12,
        label=f"Recent high-volatility regime (last {recent_days} days)",
    )
    axis.axhline(0.0, color="#555555", linewidth=0.7)
    axis.set_title("Demeaned daily arithmetic returns")
    axis.set_xlabel("Day")
    axis.set_ylabel("Return (%)")
    axis.legend(loc="upper left", frameon=False)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def solve_problem2(
    csv_path: str | os.PathLike[str],
    figure_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Solve Assignment 2 Problem 2, optionally writing its required plot."""

    path = Path(csv_path)
    frame = pd.read_csv(path)
    if "Price" not in frame.columns:
        raise ValueError("Problem 2 data must contain a Price column.")
    prices = pd.to_numeric(frame["Price"], errors="raise").to_numpy(float)
    if len(prices) < 3 or np.any(prices <= 0.0):
        raise ValueError("Problem 2 requires at least three positive prices.")

    raw_returns = prices[1:] / prices[:-1] - 1.0
    removed_sample_mean = float(np.mean(raw_returns))
    demeaned_returns = raw_returns - removed_sample_mean
    n_obs = len(demeaned_returns)

    # The plot shows an abrupt increase at about day 461.  We use a round,
    # visually defensible 40-day recent regime.  The profile split below is a
    # diagnostic only and independently lands at 39 recent days.
    recent_regime_days = 40
    if n_obs <= recent_regime_days:
        raise ValueError("Not enough returns to define the 40-day recent regime.")
    earlier_returns = demeaned_returns[:-recent_regime_days]
    recent_returns = demeaned_returns[-recent_regime_days:]
    profile_break = _profile_variance_break(demeaned_returns)

    moments = _sample_moments(demeaned_returns)
    regime_moments = {
        "earlier": {
            "n_days": int(len(earlier_returns)),
            **_sample_moments(earlier_returns),
        },
        "recent": {
            "n_days": int(len(recent_returns)),
            **_sample_moments(recent_returns),
        },
    }

    # A two-zero-mean-Normal mixture has excess kurtosis even though each
    # component has none.  This diagnostic uses the two sample variances.
    earlier_fraction = len(earlier_returns) / n_obs
    earlier_variance = float(np.var(earlier_returns, ddof=1))
    recent_variance = float(np.var(recent_returns, ddof=1))
    mixture_variance = (
        earlier_fraction * earlier_variance
        + (1.0 - earlier_fraction) * recent_variance
    )
    mixture_implied_excess_kurtosis = float(
        3.0
        * (
            earlier_fraction * earlier_variance**2
            + (1.0 - earlier_fraction) * recent_variance**2
        )
        / mixture_variance**2
        - 3.0
    )

    z_loss = float(-stats.norm.ppf(ALPHA))
    equal_weight_sd = float(np.std(demeaned_returns, ddof=1))
    equal_weight_var = z_loss * equal_weight_sd * POSITION_VALUE

    ew_results: dict[str, Any] = {}
    ew_var_values: dict[float, float] = {}
    for decay in (0.94, 0.97):
        weights = _exponential_weights(n_obs, decay)
        variance = float(np.sum(weights * demeaned_returns**2))
        standard_deviation = math.sqrt(variance)
        effective_n = float(1.0 / np.sum(weights**2))
        infinite_horizon_effective_n = float((1.0 + decay) / (1.0 - decay))
        half_life = float(math.log(0.5) / math.log(decay))
        recent_weight = float(np.sum(weights[-recent_regime_days:]))
        volatility_standard_error = float(
            standard_deviation / math.sqrt(2.0 * effective_n)
        )
        var_standard_error = float(
            z_loss * volatility_standard_error * POSITION_VALUE
        )
        dollar_var = float(z_loss * standard_deviation * POSITION_VALUE)
        ew_var_values[decay] = dollar_var
        ew_results[f"lambda_{decay:.2f}"] = {
            "decay": decay,
            "normalized_weights_oldest_to_newest": weights.tolist(),
            "weight_sum": float(np.sum(weights)),
            "finite_sample_effective_n": effective_n,
            "infinite_horizon_effective_n": infinite_horizon_effective_n,
            "half_life_days": half_life,
            "weight_on_most_recent_40_days": recent_weight,
            "weight_on_chosen_recent_regime": recent_weight,
            "variance": variance,
            "standard_deviation": standard_deviation,
            "volatility_standard_error": volatility_standard_error,
            "var_standard_error_usd": var_standard_error,
            "var_usd": dollar_var,
        }

    fitted_df, fitted_location, fitted_scale = stats.t.fit(demeaned_returns)
    fitted_df = float(fitted_df)
    fitted_location = float(fitted_location)
    fitted_scale = float(fitted_scale)
    fitted_quantile = float(
        stats.t.ppf(ALPHA, fitted_df, loc=fitted_location, scale=fitted_scale)
    )
    student_t_var = float(-fitted_quantile * POSITION_VALUE)
    t_implied_sd = (
        float(fitted_scale * math.sqrt(fitted_df / (fitted_df - 2.0)))
        if fitted_df > 2.0
        else math.inf
    )

    empirical_quantile = _course_empirical_quantile(demeaned_returns, ALPHA)
    historical_var = float(-empirical_quantile * POSITION_VALUE)

    var_usd = {
        "normal_equal_weight": float(equal_weight_var),
        "normal_ew_lambda_0.97": float(ew_var_values[0.97]),
        "normal_ew_lambda_0.94": float(ew_var_values[0.94]),
        "student_t_mle": student_t_var,
        "historical_full_sample": historical_var,
    }
    actual_ranking = [
        name for name, _ in sorted(var_usd.items(), key=lambda item: item[1])
    ]
    predicted_ranking = [
        "normal_equal_weight",
        "historical_full_sample",
        "student_t_mle",
        "normal_ew_lambda_0.97",
        "normal_ew_lambda_0.94",
    ]

    ew_var_gap = abs(ew_var_values[0.94] - ew_var_values[0.97])
    se_94 = float(ew_results["lambda_0.94"]["var_standard_error_usd"])
    se_97 = float(ew_results["lambda_0.97"]["var_standard_error_usd"])
    # This combined number is a conservative scale comparison.  The estimates
    # use the same returns and are positively correlated, so it is not a formal
    # standard error for their difference.
    rss_noise_scale = float(math.sqrt(se_94**2 + se_97**2))

    if "Day" in frame.columns:
        return_days = pd.to_numeric(frame["Day"], errors="raise").to_numpy()[1:]
    else:
        return_days = np.arange(1, n_obs + 1)
    figure_path = None
    if figure_dir is not None:
        figure_path = _plot_problem2_returns(
            return_days, demeaned_returns, recent_regime_days, figure_dir
        )

    return {
        "problem": 2,
        "source_file": str(path),
        "conventions": {
            "returns": "Arithmetic return P_t/P_(t-1)-1.",
            "demeaning": (
                "Subtract the ordinary full-sample arithmetic-return mean "
                "before every subsequent calculation."
            ),
            "sample_variance_and_moments": (
                "Variance/standard deviation use ddof=1; skewness and excess "
                "kurtosis use finite-sample bias corrections."
            ),
            "normal_var": (
                "Positive loss = -Phi^{-1}(0.05) * sigma * $1,000,000; "
                "demeaned return mean is zero."
            ),
            "ew_variance": (
                "Finite-sample normalized weights proportional to lambda^lag, "
                "newest observation highest; sum(w*r^2), with no weighted "
                "recentering and no Bessel correction because returns were "
                "already demeaned as instructed."
            ),
            "student_t": (
                "Three-parameter maximum likelihood fit (df, location, scale) "
                "to demeaned returns; scale is not the standard deviation."
            ),
            "historical_quantile": (
                "Course RiskStats.jl convention: sort n values, set "
                "ndn=floor(n*alpha), nup=ceil(n*alpha) in one-based indexing, "
                "and average those two order statistics 50/50."
            ),
            "alpha": ALPHA,
            "position_value_usd": POSITION_VALUE,
            "reported_var_sign": "Positive dollar loss.",
        },
        "sample": {
            "price_observations": int(len(prices)),
            "return_observations": int(n_obs),
            "removed_arithmetic_return_mean": removed_sample_mean,
            "demeaned_return_mean": float(np.mean(demeaned_returns)),
        },
        "prediction": {
            "ranking_smallest_to_largest": predicted_ranking,
            "reason": (
                "The last roughly 40 days are much more volatile. EW(0.94) "
                "should react most, followed by EW(0.97). Before fitting, the "
                "full-sample heavy tails suggest Student-t above historical "
                "and both above equal-weight Normal at the 5% tail."
            ),
            "single_normal_regime_prediction": False,
        },
        "moments": moments,
        "regime": {
            "chosen_recent_days": recent_regime_days,
            "chosen_split_after_return_number": n_obs - recent_regime_days,
            "selection": (
                "Visual estimate of about 40 recent high-volatility days; a "
                "single-break Gaussian profile independently places the split "
                "one day later (39 recent observations)."
            ),
            "profile_break_cross_check": profile_break,
            "moments_by_regime": regime_moments,
            "standard_deviation_ratio_recent_to_earlier": float(
                regime_moments["recent"]["standard_deviation"]
                / regime_moments["earlier"]["standard_deviation"]
            ),
            "zero_mean_normal_mixture_implied_excess_kurtosis": (
                mixture_implied_excess_kurtosis
            ),
        },
        "exponentially_weighted": ew_results,
        "student_t_fit": {
            "degrees_of_freedom": fitted_df,
            "location": fitted_location,
            "scale": fitted_scale,
            "implied_standard_deviation_if_df_gt_2": t_implied_sd,
            "five_percent_return_quantile": fitted_quantile,
        },
        "historical": {
            "five_percent_return_quantile": empirical_quantile,
            "one_based_order_statistic_positions": {
                "floor_n_alpha": int(math.floor(n_obs * ALPHA)),
                "ceil_n_alpha": int(math.ceil(n_obs * ALPHA)),
            },
        },
        "var_usd": var_usd,
        "fit_ranking_smallest_to_largest": actual_ranking,
        "ew_noise_comparison": {
            "absolute_var_gap_usd": float(ew_var_gap),
            "lambda_0.94_var_standard_error_usd": se_94,
            "lambda_0.97_var_standard_error_usd": se_97,
            "root_sum_square_noise_scale_usd_not_formal_due_to_dependence": (
                rss_noise_scale
            ),
            "gap_over_lambda_0.94_standard_error": float(ew_var_gap / se_94),
            "gap_over_lambda_0.97_standard_error": float(ew_var_gap / se_97),
            "gap_over_root_sum_square_noise_scale": float(
                ew_var_gap / rss_noise_scale
            ),
            "conclusion": (
                "The VaR gap is not convincingly larger than estimation noise: "
                "it is below one lambda=0.94 standard error and below the "
                "conservative root-sum-square scale."
            ),
        },
        "reconcile": {
            "ranking_surprise": (
                "Student-t MLE is the smallest 5% VaR, contrary to the initial "
                "ranking. A fitted t is more peaked as well as more heavy-tailed; "
                "its fitted scale is below the sample standard deviation, and "
                "at 5% the central concentration dominates. The t tail would "
                "overtake farther out."
            ),
            "kurtosis_explanation": (
                "Each 40/460-day regime is close to Normal in kurtosis, while "
                "mixing their very different variances creates large full-sample "
                "excess kurtosis. The fitted t is mainly a one-distribution "
                "approximation to this volatility-regime mixture."
            ),
            "opinion_trading_limit": (
                "Opinion: use lambda=0.94 for a trading limit because quicker "
                "adaptation to the active high-volatility regime is worth the "
                "additional estimator noise."
            ),
            "opinion_capital": (
                "Opinion: use lambda=0.97 for a capital number because its larger "
                "effective sample size produces a more stable estimate, while "
                "still assigning most weight to the new regime."
            ),
        },
        "figure_path": figure_path,
    }


__all__ = ["solve_problem1", "solve_problem2"]
