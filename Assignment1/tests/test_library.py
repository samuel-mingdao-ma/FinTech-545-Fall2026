"""Smoke and invariant tests for the reusable Assignment 1 library."""

import numpy as np
import pandas as pd

from fintech545 import (
    compare_correlations,
    conditional_normal_summary,
    corrected_aic,
    sample_moments,
)


def test_corrected_aic_exceeds_aic() -> None:
    log_likelihood, k, n = -10.0, 3, 100
    assert corrected_aic(log_likelihood, k, n) > -2 * log_likelihood + 2 * k


def test_sample_moments_counts_normal_tail() -> None:
    result = sample_moments(np.arange(100.0))
    assert result["n"] == 100
    assert result["expected_below_q01"] == 1.0
    assert result["variance"] == np.var(np.arange(100.0), ddof=1)


def test_rank_correlation_detects_monotone_curve() -> None:
    x = np.linspace(-2, 2, 101)
    result = compare_correlations(pd.DataFrame({"x": x, "cube": x**3}))
    assert result["spearman"]["cube"]["x"] > result["pearson"]["cube"]["x"]


def test_conditional_variance_is_smaller() -> None:
    generator = np.random.default_rng(545)
    x1 = generator.normal(size=1_000)
    x2 = 2.0 * x1 + generator.normal(size=1_000)
    result = conditional_normal_summary(x1, x2)
    assert 0.0 < result["variance_ratio"] < 1.0
