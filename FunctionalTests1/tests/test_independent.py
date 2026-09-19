"""Additional checks on new inputs; these are not claimed to be hidden tests."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats
import risk545 as risk


def test_missing_data_against_independent_pandas():
    x = np.array([[1, 2, 7], [3, np.nan, 5], [5, 8, 3], [8, 9, np.nan], [10, 4, 1]], float)
    frame = pd.DataFrame(x)
    for pairwise in (False, True):
        source = frame if pairwise else frame.dropna()
        np.testing.assert_allclose(risk.missing_covariance(x, pairwise=pairwise), source.cov())
        np.testing.assert_allclose(risk.missing_covariance(x, pairwise=pairwise, corr=True), source.corr())


def test_ew_formula_and_translation():
    x = np.array([[1, 2], [3, 1], [8, 5]], float)
    w = np.array([0.25, 0.5, 1])/1.75
    mean = sum(w[i]*x[i] for i in range(3))
    expected = sum(w[i]*np.outer(x[i]-mean, x[i]-mean) for i in range(3))
    np.testing.assert_allclose(risk.ew_covariance(x, 0.5), expected)
    np.testing.assert_allclose(risk.ew_covariance(x+100, 0.5), expected)
    np.testing.assert_allclose(risk.ew_covariance(x, 1), np.cov(x, rowvar=False, ddof=0))


@pytest.mark.parametrize("repair", [risk.near_psd, risk.higham_psd])
def test_psd_repair_new_matrix(repair):
    c = np.array([[1, .9, .9], [.9, 1, -.8], [.9, -.8, 1]])
    a = c*np.outer([1, 2, 3], [1, 2, 3])
    result = repair(a)
    np.testing.assert_allclose(np.diag(result), np.diag(a))
    np.testing.assert_allclose(result, result.T, atol=1e-12)
    assert np.linalg.eigvalsh(result)[0] >= -1e-8
    np.testing.assert_allclose(repair(np.eye(3)), np.eye(3), atol=1e-12)


@pytest.mark.parametrize("a", [np.diag([0, 1, 2]), np.ones((3, 3)), np.array([[4, 2], [2, 3]]), np.zeros((2, 2))])
def test_cholesky_reconstructs(a):
    root = risk.chol_psd(a)
    np.testing.assert_allclose(root@root.T, a, atol=1e-12)
    np.testing.assert_allclose(root, np.tril(root))


@pytest.mark.parametrize("seed", [17, 81, 2026])
def test_simulation_new_matrix_seeds_and_means(seed):
    a = np.array([[4, 1.2], [1.2, 1]])
    mu = np.array([2, -3])
    n = 100000
    x = risk.simulate_normal(a, n, mean=mu, seed=seed)
    assert x.shape == (n, 2)
    assert np.max(np.abs(x.mean(axis=0)-mu)/np.sqrt(np.diag(a)/n)) < 6
    se = np.sqrt((np.outer(np.diag(a), np.diag(a))+a*a)/(n-1))
    assert np.max(np.abs(np.cov(x, rowvar=False)-a)/se) < 6
    np.testing.assert_array_equal(risk.simulate_normal(a, 20, seed=seed), risk.simulate_normal(a, 20, seed=seed))


def test_pca_truncation_and_full_covariance():
    a = np.diag([9, .9, .1])
    target = risk.pca_target(a, .95)
    np.testing.assert_allclose(target, np.diag([9, .9, 0]))
    np.testing.assert_allclose(risk.pca_target(a, 1), a)
    x = risk.simulate_pca(a, 100000, explained=.95, seed=481)
    np.testing.assert_allclose(np.cov(x, rowvar=False), target, atol=.06)


def test_normal_convention_and_aicc():
    fit = risk.fit_normal([1, 2, 3])
    assert fit == {"mu": 2.0, "sigma": 1.0}
    assert risk.aicc(-12, 20, 3) == 24+6+24/16


def test_t_fit_on_fresh_data_and_shift():
    x = 0.7*stats.t.rvs(5, size=2500, random_state=19)+1.5
    p = risk.fit_t(x)
    assert abs(p["mu"]-1.5) < .08 and abs(p["sigma"]-.7) < .08 and 3 < p["nu"] < 8
    shifted = risk.fit_t(2*x+5)
    np.testing.assert_allclose([shifted["mu"], shifted["sigma"], shifted["nu"]], [2*p["mu"]+5, 2*p["sigma"], p["nu"]], rtol=2e-4)


def test_t_regression_on_fresh_two_predictor_data():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(1800, 2))
    y = .4 + x@np.array([1.2, -2.1]) + .3*rng.standard_t(6, len(x))
    p = risk.fit_t_regression(y, x)
    np.testing.assert_allclose([p["Alpha"], p["B1"], p["B2"]], [.4, 1.2, -2.1], atol=.04)
    assert p["mu"] == 0 and p["sigma"] > 0 and p["nu"] > 2


def test_nig_parameters_reproduce_moments_and_likelihood():
    x = stats.norminvgauss.rvs(2, -.4, loc=.02, scale=.05, size=4000, random_state=79)
    moment_fit = risk.fit_nig_moments(x)
    mle_fit = risk.fit_nig_mle(x)
    def distribution(p):
        return stats.norminvgauss(p["alpha"]*p["delta"], p["beta"]*p["delta"], loc=p["mu"], scale=p["delta"])
    d = distribution(moment_fit)
    np.testing.assert_allclose(d.stats(moments="mvsk"), [x.mean(), x.var(ddof=1), stats.skew(x), stats.kurtosis(x)], rtol=1e-10)
    assert distribution(mle_fit).logpdf(x).sum() >= d.logpdf(x).sum()-1e-6


@pytest.mark.parametrize("call", [
    lambda: risk.ew_covariance([[1, 2]], 1.5),
    lambda: risk.missing_covariance([[np.nan, 1], [2, np.nan]]),
    lambda: risk.chol_psd([[1, 2], [2, 1]]),
    lambda: risk.chol_psd([[1, 1], [0, 1]]),
    lambda: risk.simulate_normal(np.eye(2), 10, mean=[1]),
    lambda: risk.simulate_pca(np.eye(2), 10, explained=0),
    lambda: risk.fit_normal([1, 1]),
    lambda: risk.fit_nig_moments([-1, 0, 1]),
    lambda: risk.fit_t_regression([1, 2, 4], np.ones((3, 1))),
    lambda: risk.aicc(-1, 4, 3),
])
def test_invalid_inputs_raise(call):
    with pytest.raises(ValueError):
        call()


def test_end_to_end_chain_without_reference_intermediates():
    # Input covariance -> repair -> root -> simulation. All stages calculated.
    x = np.array([[1, 2, np.nan], [3, 4, 5], [8, np.nan, 2], [7, 3, 1], [2, 6, 9]])
    cov = risk.missing_covariance(x, pairwise=True)
    fixed = risk.near_psd(cov)
    root = risk.chol_psd(fixed)
    np.testing.assert_allclose(root@root.T, fixed, atol=1e-10)
    draws = risk.simulate_normal(fixed, 100000, seed=518)
    assert np.linalg.norm(np.cov(draws, rowvar=False)-fixed, "fro")/np.linalg.norm(fixed, "fro") < .03
