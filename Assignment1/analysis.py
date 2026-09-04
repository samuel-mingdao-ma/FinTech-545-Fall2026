"""Reproduce every numerical result and figure for FinTech 545 Assignment 1.

Conventions are stated explicitly because several quantities (variance, kurtosis,
PACF, and AICc) have multiple common definitions.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)


COLORS = {
    "navy": "#16324F",
    "blue": "#2A6F97",
    "teal": "#2A9D8F",
    "orange": "#E76F51",
    "sand": "#E9C46A",
    "gray": "#64748B",
}


def configure_plots() -> None:
    """Apply one restrained style to all submitted figures."""

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def corrected_aic(log_likelihood: float, n_parameters: int, n_obs: int) -> float:
    """AICc using the Week 2 convention, counting every fitted parameter."""

    aic = -2.0 * log_likelihood + 2.0 * n_parameters
    return aic + 2.0 * n_parameters * (n_parameters + 1) / (
        n_obs - n_parameters - 1
    )


def problem_1() -> dict:
    data = pd.read_csv(ROOT / "problem1.csv")
    x = data["x"].to_numpy()
    n = len(x)

    # Use unbiased sample variance and the bias-corrected standardized moment
    # estimators. scipy's fisher=True reports excess rather than raw kurtosis.
    mean = float(np.mean(x))
    variance = float(np.var(x, ddof=1))
    skewness = float(stats.skew(x, bias=False))
    excess_kurtosis = float(stats.kurtosis(x, fisher=True, bias=False))
    standard_deviation = float(np.sqrt(variance))
    normal_q01 = float(stats.norm.ppf(0.01, loc=mean, scale=standard_deviation))
    observed_below = int(np.sum(x < normal_q01))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(x, bins=45, density=True, color=COLORS["blue"], alpha=0.72)
    grid = np.linspace(x.min() - 0.004, x.max() + 0.004, 500)
    ax.plot(
        grid,
        stats.norm.pdf(grid, loc=mean, scale=standard_deviation),
        color=COLORS["orange"],
        linewidth=2.2,
        label="Moment-matched Normal",
    )
    ax.axvline(normal_q01, color=COLORS["navy"], linestyle="--", linewidth=1.5)
    ax.set(title="Problem 1: sample shape and fitted Normal", xlabel="x", ylabel="Density")
    ax.legend(frameon=False)
    save_figure(fig, "problem1_distribution.png")

    return {
        "n": n,
        "mean": mean,
        "variance": variance,
        "standard_deviation": standard_deviation,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "normal_q01": normal_q01,
        "observed_below_q01": observed_below,
        "expected_below_q01": 0.01 * n,
    }


def problem_2() -> dict:
    data = pd.read_csv(ROOT / "problem2.csv")
    x = data["x"].to_numpy()
    y = data["y"].to_numpy()
    n = len(y)
    design = sm.add_constant(x)

    ols = sm.OLS(y, design).fit()
    alpha_ols, beta_ols = map(float, ols.params)
    se_alpha, se_beta = map(float, ols.bse)
    residuals_ols = y - (alpha_ols + beta_ols * x)
    sigma_ols = float(np.sqrt(np.sum(residuals_ols**2) / (n - 2)))

    # Under Normal errors, maximizing likelihood gives the OLS line. The MLE
    # scale uses denominator n, unlike the unbiased OLS residual standard error.
    alpha_normal = alpha_ols
    beta_normal = beta_ols
    sigma_normal = float(np.sqrt(np.mean(residuals_ols**2)))
    ll_normal = float(
        np.sum(stats.norm.logpdf(residuals_ols, loc=0.0, scale=sigma_normal))
    )
    aicc_normal = corrected_aic(ll_normal, n_parameters=3, n_obs=n)

    # Transform scale and df so optimization always stays in the valid region.
    # Constraining df > 2 gives the fitted error a finite variance.
    def negative_t_log_likelihood(parameters: np.ndarray) -> float:
        alpha, beta, log_scale, log_df_minus_two = parameters
        scale = np.exp(log_scale)
        degrees_freedom = 2.0 + np.exp(log_df_minus_two)
        residuals = y - alpha - beta * x
        return float(
            -np.sum(
                stats.t.logpdf(
                    residuals, df=degrees_freedom, loc=0.0, scale=scale
                )
            )
        )

    initial = np.array(
        [alpha_ols, beta_ols, np.log(np.std(residuals_ols)), np.log(4.0)]
    )
    t_fit = optimize.minimize(
        negative_t_log_likelihood,
        initial,
        method="Nelder-Mead",
        options={"maxiter": 20_000, "xatol": 1e-11, "fatol": 1e-9},
    )
    if not t_fit.success:
        raise RuntimeError(f"Student-t optimization failed: {t_fit.message}")

    alpha_t, beta_t, log_scale_t, log_df_minus_two = t_fit.x
    scale_t = float(np.exp(log_scale_t))
    df_t = float(2.0 + np.exp(log_df_minus_two))
    ll_t = float(-t_fit.fun)
    aicc_t = corrected_aic(ll_t, n_parameters=4, n_obs=n)

    quantiles = {}
    for probability in (0.95, 0.995):
        quantiles[str(probability)] = {
            "normal": float(stats.norm.ppf(probability, scale=sigma_normal)),
            "student_t": float(stats.t.ppf(probability, df=df_t, scale=scale_t)),
        }

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.scatter(x, y, s=18, color=COLORS["blue"], alpha=0.65, edgecolors="none")
    ax.set(title="Problem 2: y against x before fitting", xlabel="x", ylabel="y")
    save_figure(fig, "problem2_scatter.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(residuals_ols, bins=34, density=True, color=COLORS["blue"], alpha=0.55)
    grid = np.linspace(residuals_ols.min() - 0.5, residuals_ols.max() + 0.5, 500)
    ax.plot(
        grid,
        stats.norm.pdf(grid, scale=sigma_normal),
        color=COLORS["orange"],
        linewidth=2.0,
        label="Fitted Normal",
    )
    ax.plot(
        grid,
        stats.t.pdf(grid, df=df_t, scale=scale_t),
        color=COLORS["teal"],
        linewidth=2.0,
        label="Fitted Student-t",
    )
    ax.set(title="Problem 2: fitted error distributions", xlabel="Regression residual", ylabel="Density")
    ax.legend(frameon=False)
    save_figure(fig, "problem2_errors.png")

    return {
        "n": n,
        "ols": {
            "alpha": alpha_ols,
            "beta": beta_ols,
            "se_alpha": se_alpha,
            "se_beta": se_beta,
            "residual_standard_error": sigma_ols,
            "r_squared": float(ols.rsquared),
        },
        "normal_mle": {
            "alpha": alpha_normal,
            "beta": beta_normal,
            "sigma": sigma_normal,
            "log_likelihood": ll_normal,
            "aicc": aicc_normal,
        },
        "student_t_mle": {
            "alpha": float(alpha_t),
            "beta": float(beta_t),
            "scale": scale_t,
            "degrees_freedom": df_t,
            "implied_standard_deviation": float(
                scale_t * np.sqrt(df_t / (df_t - 2.0))
            ),
            "log_likelihood": ll_t,
            "aicc": aicc_t,
        },
        "aicc_difference_normal_minus_t": aicc_normal - aicc_t,
        "residual_skewness": float(stats.skew(residuals_ols, bias=False)),
        "residual_excess_kurtosis": float(
            stats.kurtosis(residuals_ols, fisher=True, bias=False)
        ),
        "quantiles": quantiles,
    }


def problem_3() -> dict:
    data = pd.read_csv(ROOT / "problem3.csv")
    columns = list(data.columns)
    pearson = data.corr(method="pearson")
    spearman = data.corr(method="spearman")
    gap = (pearson - spearman).abs()

    upper = np.triu(np.ones(gap.shape, dtype=bool), k=1)
    row, column = np.unravel_index(np.argmax(gap.to_numpy() * upper), gap.shape)

    fig, axes = plt.subplots(4, 4, figsize=(8.0, 8.0))
    for i, y_name in enumerate(columns):
        for j, x_name in enumerate(columns):
            ax = axes[i, j]
            if i == j:
                ax.hist(data[x_name], bins=24, color=COLORS["blue"], alpha=0.72)
            elif i > j:
                ax.scatter(
                    data[x_name],
                    data[y_name],
                    s=8,
                    color=COLORS["blue"],
                    alpha=0.45,
                    edgecolors="none",
                )
            else:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.58,
                    f"Pearson {pearson.loc[y_name, x_name]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=COLORS["navy"],
                )
                ax.text(
                    0.5,
                    0.38,
                    f"Spearman {spearman.loc[y_name, x_name]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=COLORS["teal"],
                )
            if i == 3:
                ax.set_xlabel(x_name)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(y_name)
            else:
                ax.set_yticklabels([])
    fig.suptitle("Problem 3: pairwise relationships", y=1.01, fontsize=13)
    save_figure(fig, "problem3_pairs.png")

    return {
        "n": len(data),
        "pearson": pearson.to_dict(),
        "spearman": spearman.to_dict(),
        "absolute_gap": gap.to_dict(),
        "largest_gap_pair": [columns[column], columns[row]],
        "largest_gap": float(gap.iloc[row, column]),
    }


def problem_4() -> dict:
    data = pd.read_csv(ROOT / "problem4.csv")
    x1 = data["x1"].to_numpy()
    x2 = data["x2"].to_numpy()
    n = len(data)
    means = data.mean()
    covariance = data.cov()

    sigma11 = float(covariance.loc["x1", "x1"])
    sigma12 = float(covariance.loc["x1", "x2"])
    sigma21 = float(covariance.loc["x2", "x1"])
    sigma22 = float(covariance.loc["x2", "x2"])
    conditional_slope = sigma21 / sigma11
    conditional_variance = sigma22 - sigma21 * sigma12 / sigma11
    variance_ratio = conditional_variance / sigma22
    standard_deviation_ratio = np.sqrt(variance_ratio)

    conditional_mean = means["x2"] + conditional_slope * (x1 - means["x1"])
    half_width = float(stats.norm.ppf(0.975) * np.sqrt(conditional_variance))
    inside = np.abs(x2 - conditional_mean) <= half_width
    standardized_distance = np.abs((x1 - means["x1"]) / np.sqrt(sigma11))

    bucket_masks = {
        "within_1_sd": standardized_distance < 1.0,
        "between_1_and_2_sd": (standardized_distance >= 1.0)
        & (standardized_distance < 2.0),
        "beyond_2_sd": standardized_distance >= 2.0,
    }
    bucket_coverage = {
        name: {
            "n": int(mask.sum()),
            "coverage": float(inside[mask].mean()),
            "residual_standard_deviation": float(
                np.std((x2 - conditional_mean)[mask], ddof=1)
            ),
        }
        for name, mask in bucket_masks.items()
    }

    order = np.argsort(x1)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]
    ax.scatter(x1, x2, s=12, color=COLORS["blue"], alpha=0.42, edgecolors="none")
    ax.plot(x1[order], conditional_mean[order], color=COLORS["navy"], linewidth=2.0)
    ax.fill_between(
        x1[order],
        conditional_mean[order] - half_width,
        conditional_mean[order] + half_width,
        color=COLORS["sand"],
        alpha=0.32,
        label="Constant-width 95% band",
    )
    ax.set(title="Conditional mean and 95% band", xlabel="x1", ylabel="x2")
    ax.legend(frameon=False, loc="upper left")

    labels = ["< 1 SD", "1-2 SD", ">= 2 SD"]
    coverage_values = [bucket_coverage[name]["coverage"] for name in bucket_masks]
    axes[1].bar(labels, coverage_values, color=[COLORS["teal"], COLORS["orange"], COLORS["gray"]])
    axes[1].axhline(0.95, color=COLORS["navy"], linestyle="--", linewidth=1.4)
    axes[1].set_ylim(0.80, 1.0)
    axes[1].set(title="Coverage by distance from mean", ylabel="Fraction inside band")
    axes[1].tick_params(axis="x", rotation=20)
    save_figure(fig, "problem4_conditional.png")

    return {
        "n": n,
        "means": means.to_dict(),
        "covariance": covariance.to_dict(),
        "conditional_slope": conditional_slope,
        "conditional_variance": conditional_variance,
        "conditional_standard_deviation": float(np.sqrt(conditional_variance)),
        "variance_ratio": float(variance_ratio),
        "variance_reduction_fraction": float(1.0 - variance_ratio),
        "standard_deviation_ratio": float(standard_deviation_ratio),
        "band_half_width": half_width,
        "overall_coverage": float(inside.mean()),
        "overall_inside_count": int(inside.sum()),
        "bucket_coverage": bucket_coverage,
    }


def problem_5() -> dict:
    data = pd.read_csv(ROOT / "problem5.csv")
    x = data["x"].to_numpy()
    n = len(x)
    nlags = 20
    significance_band = float(1.96 / np.sqrt(n))

    acf_values = acf(x, nlags=nlags, fft=False)
    # Yule-Walker without sample-size adjustment is a standard stable PACF
    # convention. The selected order is not sensitive to this choice.
    pacf_values = pacf(x, nlags=nlags, method="ywm")

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.2))
    axes[0].plot(np.arange(n), x, color=COLORS["blue"], linewidth=0.9)
    axes[0].set(title="Observed series", xlabel="Time index", ylabel="x")

    for ax, values, title in (
        (axes[1], acf_values, "ACF"),
        (axes[2], pacf_values, "PACF"),
    ):
        lags = np.arange(len(values))
        ax.axhspan(-significance_band, significance_band, color=COLORS["sand"], alpha=0.32)
        ax.axhline(0.0, color=COLORS["navy"], linewidth=0.8)
        ax.vlines(lags, 0.0, values, color=COLORS["blue"], linewidth=1.4)
        ax.scatter(lags, values, color=COLORS["blue"], s=18)
        ax.set_xlim(-0.5, nlags + 0.5)
        ax.set(title=title, xlabel="Lag", ylabel="Correlation")
    save_figure(fig, "problem5_acf_pacf.png")

    models = {}
    for label, order in {
        "AR(1)": (1, 0, 0),
        "AR(2)": (2, 0, 0),
        "AR(3)": (3, 0, 0),
        "MA(1)": (0, 0, 1),
        "MA(2)": (0, 0, 2),
        "MA(3)": (0, 0, 3),
    }.items():
        fit = ARIMA(
            x,
            order=order,
            trend="c",
            enforce_stationarity=True,
            enforce_invertibility=True,
        ).fit()
        k = len(fit.params)
        models[label] = {
            "order": order,
            "log_likelihood": float(fit.llf),
            "aic": float(fit.aic),
            "aicc": float(corrected_aic(fit.llf, k, n)),
            "parameters": {
                name: float(value)
                for name, value in zip(fit.param_names, fit.params)
            },
        }

    selected_model = min(models, key=lambda name: models[name]["aicc"])
    return {
        "n": n,
        "mean": float(np.mean(x)),
        "significance_band": significance_band,
        "acf": [float(value) for value in acf_values],
        "pacf": [float(value) for value in pacf_values],
        "models": models,
        "selected_model": selected_model,
    }


def main() -> None:
    configure_plots()
    results = {
        "problem1": problem_1(),
        "problem2": problem_2(),
        "problem3": problem_3(),
        "problem4": problem_4(),
        "problem5": problem_5(),
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
