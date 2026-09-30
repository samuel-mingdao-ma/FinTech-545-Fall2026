"""Small, reusable statistics library for Assignment 1.

The functions accept arrays/data frames and return plain dictionaries, so they
can be imported by another program without depending on the report generator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf


def corrected_aic(log_likelihood: float, n_parameters: int, n_obs: int) -> float:
    """Return AICc, counting every fitted parameter."""

    aic = -2.0 * log_likelihood + 2.0 * n_parameters
    return aic + 2.0 * n_parameters * (n_parameters + 1) / (
        n_obs - n_parameters - 1
    )


def sample_moments(values: np.ndarray) -> dict:
    """Compute the first four requested sample summaries and Normal 1% tail."""

    x = np.asarray(values, dtype=float)
    n = len(x)
    mean = float(np.mean(x))
    variance = float(np.var(x, ddof=1))
    standard_deviation = float(np.sqrt(variance))
    normal_q01 = float(stats.norm.ppf(0.01, loc=mean, scale=standard_deviation))
    return {
        "n": n,
        "mean": mean,
        "variance": variance,
        "standard_deviation": standard_deviation,
        "skewness": float(stats.skew(x, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(x, fisher=True, bias=False)),
        "normal_q01": normal_q01,
        "observed_below_q01": int(np.sum(x < normal_q01)),
        "expected_below_q01": 0.01 * n,
    }


def fit_regression_errors(x_values: np.ndarray, y_values: np.ndarray) -> dict:
    """Fit OLS, Normal-error MLE, and Student-t-error MLE regression models."""

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    n = len(y)
    design = sm.add_constant(x)

    ols = sm.OLS(y, design).fit()
    alpha_ols, beta_ols = map(float, ols.params)
    se_alpha, se_beta = map(float, ols.bse)
    residuals = y - (alpha_ols + beta_ols * x)
    sigma_ols = float(np.sqrt(np.sum(residuals**2) / (n - 2)))

    sigma_normal = float(np.sqrt(np.mean(residuals**2)))
    ll_normal = float(np.sum(stats.norm.logpdf(residuals, scale=sigma_normal)))
    aicc_normal = corrected_aic(ll_normal, n_parameters=3, n_obs=n)

    def negative_t_log_likelihood(parameters: np.ndarray) -> float:
        alpha, beta, log_scale, log_df_minus_two = parameters
        scale = np.exp(log_scale)
        degrees_freedom = 2.0 + np.exp(log_df_minus_two)
        errors = y - alpha - beta * x
        return float(
            -np.sum(stats.t.logpdf(errors, df=degrees_freedom, scale=scale))
        )

    initial = np.array(
        [alpha_ols, beta_ols, np.log(np.std(residuals)), np.log(4.0)]
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
    quantiles = {
        str(probability): {
            "normal": float(stats.norm.ppf(probability, scale=sigma_normal)),
            "student_t": float(stats.t.ppf(probability, df=df_t, scale=scale_t)),
        }
        for probability in (0.95, 0.995)
    }

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
            "alpha": alpha_ols,
            "beta": beta_ols,
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
        "residual_skewness": float(stats.skew(residuals, bias=False)),
        "residual_excess_kurtosis": float(
            stats.kurtosis(residuals, fisher=True, bias=False)
        ),
        "quantiles": quantiles,
        "residuals": residuals.tolist(),
    }


def compare_correlations(data: pd.DataFrame) -> dict:
    """Compare Pearson and Spearman correlations for every column pair."""

    columns = list(data.columns)
    pearson = data.corr(method="pearson")
    spearman = data.corr(method="spearman")
    gap = (pearson - spearman).abs()
    upper = np.triu(np.ones(gap.shape, dtype=bool), k=1)
    row, column = np.unravel_index(np.argmax(gap.to_numpy() * upper), gap.shape)
    return {
        "n": len(data),
        "pearson": pearson.to_dict(),
        "spearman": spearman.to_dict(),
        "absolute_gap": gap.to_dict(),
        "largest_gap_pair": [columns[column], columns[row]],
        "largest_gap": float(gap.iloc[row, column]),
    }


def conditional_normal_summary(x1_values: np.ndarray, x2_values: np.ndarray) -> dict:
    """Estimate X2|X1 under a bivariate Normal and check band coverage."""

    data = pd.DataFrame({"x1": x1_values, "x2": x2_values})
    x1 = data["x1"].to_numpy()
    x2 = data["x2"].to_numpy()
    means = data.mean()
    covariance = data.cov()
    sigma11 = float(covariance.loc["x1", "x1"])
    sigma12 = float(covariance.loc["x1", "x2"])
    sigma21 = float(covariance.loc["x2", "x1"])
    sigma22 = float(covariance.loc["x2", "x2"])
    slope = sigma21 / sigma11
    variance = sigma22 - sigma21 * sigma12 / sigma11
    variance_ratio = variance / sigma22
    conditional_mean = means["x2"] + slope * (x1 - means["x1"])
    half_width = float(stats.norm.ppf(0.975) * np.sqrt(variance))
    inside = np.abs(x2 - conditional_mean) <= half_width
    distance = np.abs((x1 - means["x1"]) / np.sqrt(sigma11))
    masks = {
        "within_1_sd": distance < 1.0,
        "between_1_and_2_sd": (distance >= 1.0) & (distance < 2.0),
        "beyond_2_sd": distance >= 2.0,
    }
    bucket_coverage = {
        name: {
            "n": int(mask.sum()),
            "coverage": float(inside[mask].mean()),
            "residual_standard_deviation": float(
                np.std((x2 - conditional_mean)[mask], ddof=1)
            ),
        }
        for name, mask in masks.items()
    }
    return {
        "n": len(data),
        "means": means.to_dict(),
        "covariance": covariance.to_dict(),
        "conditional_slope": slope,
        "conditional_variance": variance,
        "conditional_standard_deviation": float(np.sqrt(variance)),
        "variance_ratio": float(variance_ratio),
        "variance_reduction_fraction": float(1.0 - variance_ratio),
        "standard_deviation_ratio": float(np.sqrt(variance_ratio)),
        "band_half_width": half_width,
        "overall_coverage": float(inside.mean()),
        "overall_inside_count": int(inside.sum()),
        "bucket_coverage": bucket_coverage,
        "conditional_mean": conditional_mean,
    }


def fit_arima_candidates(values: np.ndarray, nlags: int = 20) -> dict:
    """Fit the Assignment 1 AR/MA candidate set and select it by AICc."""

    x = np.asarray(values, dtype=float)
    n = len(x)
    acf_values = acf(x, nlags=nlags, fft=False)
    pacf_values = pacf(x, nlags=nlags, method="ywm")
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
    return {
        "n": n,
        "mean": float(np.mean(x)),
        "significance_band": float(1.96 / np.sqrt(n)),
        "acf": [float(value) for value in acf_values],
        "pacf": [float(value) for value in pacf_values],
        "models": models,
        "selected_model": min(models, key=lambda name: models[name]["aicc"]),
    }
