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
from scipy import stats

from fintech545 import (
    compare_correlations,
    conditional_normal_summary,
    fit_arima_candidates,
    fit_regression_errors,
    sample_moments,
)


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


def problem_1() -> dict:
    data = pd.read_csv(ROOT / "problem1.csv")
    x = data["x"].to_numpy()
    result = sample_moments(x)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(x, bins=45, density=True, color=COLORS["blue"], alpha=0.72)
    grid = np.linspace(x.min() - 0.004, x.max() + 0.004, 500)
    ax.plot(
        grid,
        stats.norm.pdf(
            grid, loc=result["mean"], scale=result["standard_deviation"]
        ),
        color=COLORS["orange"],
        linewidth=2.2,
        label="Moment-matched Normal",
    )
    ax.axvline(
        result["normal_q01"],
        color=COLORS["navy"],
        linestyle="--",
        linewidth=1.5,
    )
    ax.set(title="Problem 1: sample shape and fitted Normal", xlabel="x", ylabel="Density")
    ax.legend(frameon=False)
    save_figure(fig, "problem1_distribution.png")

    return result


def problem_2() -> dict:
    data = pd.read_csv(ROOT / "problem2.csv")
    x = data["x"].to_numpy()
    y = data["y"].to_numpy()
    result = fit_regression_errors(x, y)
    residuals_ols = np.asarray(result.pop("residuals"))
    sigma_normal = result["normal_mle"]["sigma"]
    scale_t = result["student_t_mle"]["scale"]
    df_t = result["student_t_mle"]["degrees_freedom"]

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

    return result


def problem_3() -> dict:
    data = pd.read_csv(ROOT / "problem3.csv")
    columns = list(data.columns)
    result = compare_correlations(data)
    pearson = pd.DataFrame(result["pearson"])
    spearman = pd.DataFrame(result["spearman"])

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

    return result


def problem_4() -> dict:
    data = pd.read_csv(ROOT / "problem4.csv")
    x1 = data["x1"].to_numpy()
    x2 = data["x2"].to_numpy()
    result = conditional_normal_summary(x1, x2)
    conditional_mean = result.pop("conditional_mean")
    half_width = result["band_half_width"]

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
    coverage_values = [
        result["bucket_coverage"][name]["coverage"]
        for name in (
            "within_1_sd",
            "between_1_and_2_sd",
            "beyond_2_sd",
        )
    ]
    axes[1].bar(labels, coverage_values, color=[COLORS["teal"], COLORS["orange"], COLORS["gray"]])
    axes[1].axhline(0.95, color=COLORS["navy"], linestyle="--", linewidth=1.4)
    axes[1].set_ylim(0.80, 1.0)
    axes[1].set(title="Coverage by distance from mean", ylabel="Fraction inside band")
    axes[1].tick_params(axis="x", rotation=20)
    save_figure(fig, "problem4_conditional.png")

    return result


def problem_5() -> dict:
    data = pd.read_csv(ROOT / "problem5.csv")
    x = data["x"].to_numpy()
    nlags = 20
    result = fit_arima_candidates(x, nlags=nlags)
    n = result["n"]
    significance_band = result["significance_band"]
    acf_values = np.asarray(result["acf"])
    pacf_values = np.asarray(result["pacf"])

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

    return result


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
