"""Independent numerical checks using synthetic inputs only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, special, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examkit.risk import (
    Fit, copula_loglik, fit_copulas, fit_distribution, fit_ols,
    historical_risk, information_criteria, parametric_risk, profile_nu,
    repair_psd, returns, simulate_normal,
)


def test_arithmetic_returns_and_demeaning_preserve_labels():
    prices = pd.DataFrame({"X": [100., 110., 99., 108.9],
                           "Y": [80., 88., 88., 79.2]}, index=[5, 6, 7, 8])
    expected = pd.DataFrame({"X": [.1, -.1, .1], "Y": [.1, 0., -.1]}, index=[6, 7, 8])
    actual = returns(prices)
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    assert actual.index.equals(expected.index) and actual.columns.equals(expected.columns)
    centered = returns(prices, demean=True)
    np.testing.assert_allclose(centered, expected-expected.mean(), atol=1e-15)
    np.testing.assert_allclose(centered.mean(), 0., atol=1e-15)


def test_normal_fit_uses_mle_divisor_by_default_and_correct_criteria():
    x = np.array([-.035, -.022, -.01, -.003, .004, .012, .016, .03, .045, .06])
    fit = fit_distribution(x, "normal")
    mu = sum(x)/len(x)
    sigma = np.sqrt(sum((x-mu)**2)/len(x))
    ll = -len(x)/2*np.log(2*np.pi)-len(x)*np.log(sigma)-sum((x-mu)**2)/(2*sigma**2)
    np.testing.assert_allclose([fit.parameters["mu"], fit.parameters["sigma"], fit.ll], [mu, sigma, ll])
    assert fit.aicc == pytest.approx(-2*ll+4+12/(len(x)-3))
    assert fit.bic == pytest.approx(-2*ll+2*np.log(len(x)))
    sample = fit_distribution(x, "normal", normal_sample_std=True)
    assert sample.parameters["sigma"] == pytest.approx(np.std(x, ddof=1))
    assert fit.parameters["sigma"] != pytest.approx(sample.parameters["sigma"])


@pytest.mark.parametrize("family", ["t", "nig"])
def test_explicit_heavy_tail_fits_have_valid_parameters_and_likelihood(family):
    rng = np.random.default_rng(402)
    if family == "t":
        x = .002+.014*rng.standard_t(5.5, size=700)
    else:
        x = stats.norminvgauss.rvs(2.2, -.5, loc=.003, scale=.035, size=700, random_state=rng)
    fit = fit_distribution(x, family)
    assert fit.name == family and np.isfinite(list(fit.parameters.values())).all()
    assert np.isfinite([fit.ll, fit.aicc, fit.bic]).all()
    if family == "t":
        p = fit.parameters
        assert p["nu"] > 0 and p["scale"] > 0
        independent = stats.t(p["nu"], loc=p["mu"], scale=p["scale"])
    else:
        p = fit.parameters
        assert p["delta"] > 0 and p["alpha"] > abs(p["beta"])
        independent = stats.norminvgauss(p["alpha"]*p["delta"], p["beta"]*p["delta"],
                                        loc=p["mu"], scale=p["delta"])
    assert fit.ll == pytest.approx(independent.logpdf(x).sum(), abs=1e-9)
    probs = np.array([.01, .5, .99])
    np.testing.assert_allclose(independent.cdf(fit.distribution.ppf(probs)), probs, atol=1e-7)


def test_ols_coefficients_standard_errors_and_residuals_against_linregress():
    rng = np.random.default_rng(5)
    x = rng.normal(size=120)
    y = .8+1.7*x+rng.normal(scale=.4, size=len(x))
    fit = fit_ols(y, x)
    independent = stats.linregress(x, y)
    np.testing.assert_allclose(fit["coefficients"], [independent.intercept, independent.slope], rtol=1e-12)
    np.testing.assert_allclose(fit["standard_errors"], [independent.intercept_stderr, independent.stderr], rtol=1e-12)
    residuals = y-independent.intercept-independent.slope*x
    np.testing.assert_allclose(fit["residuals"], residuals, atol=1e-12)
    assert fit["residual_std"] == pytest.approx(np.sqrt(residuals@residuals/(len(y)-2)))


def test_higham_repair_preserves_diagonal_and_is_psd_symmetric_and_idempotent():
    raw_r = np.array([[1., .9, .9], [.9, 1., -.9], [.9, -.9, 1.]])
    scales = np.array([1., 2., 3.])
    raw = raw_r*np.outer(scales, scales)
    assert np.linalg.eigvalsh(raw).min() < 0
    repaired = repair_psd(raw)
    np.testing.assert_allclose(repaired, repaired.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(repaired), scales**2, atol=1e-12)
    assert np.linalg.eigvalsh(repaired).min() >= -1e-10
    np.testing.assert_allclose(repair_psd(repaired), repaired, atol=1e-9)


def test_normal_simulation_is_deterministic_and_supports_singular_psd_matrix():
    covariance = np.array([[4., 2.], [2., 1.]])
    simulated = simulate_normal(covariance, n=15000, seed=8)
    np.testing.assert_array_equal(simulate_normal(covariance, n=15000, seed=8), simulated)
    assert not np.array_equal(simulate_normal(covariance, n=15000, seed=9), simulated)
    np.testing.assert_allclose(simulated[:, 0], 2*simulated[:, 1], atol=1e-12)
    np.testing.assert_allclose(np.cov(simulated, rowvar=False), covariance, rtol=.08)


@pytest.mark.parametrize("alpha,var,es", [(.25, 8., 9.), (.2, 9., 10.), (.01, 10., 10.)])
def test_course_historical_risk_order_statistic_convention(alpha, var, es):
    sample = [3., -8., 2., -10., 1., 0., -1., -5.]
    risk = historical_risk(sample, alpha)
    assert risk == pytest.approx(dict(var=var, es=es))


def test_course_ties_and_linear_fractional_tail_mass():
    tied = historical_risk([-10., -10., -5., 0., 1.], .2)
    assert tied == pytest.approx(dict(var=10., es=10.))
    sample = np.array([-10., -8., -5., -1., 0., 1., 2., 3.])
    linear = historical_risk(sample, .2, convention="linear")
    assert linear["var"] == pytest.approx(-np.quantile(sample, .2))
    assert linear["es"] == pytest.approx(-(-10.+.6*-8.)/1.6)


@pytest.mark.parametrize("family", ["normal", "t"])
@pytest.mark.parametrize("alpha", [.05, .005])
def test_parametric_var_and_es_against_independent_tail_integral(family, alpha):
    if family == "normal":
        distribution = stats.norm(loc=.003, scale=.02)
        parameters = dict(mu=.003, sigma=.02)
    else:
        distribution = stats.t(4.7, loc=.003, scale=.02)
        parameters = dict(mu=.003, scale=.02, nu=4.7)
    fit = Fit(family, distribution, parameters, 0., 0., 0.)
    q = distribution.ppf(alpha)
    tail = integrate.quad(lambda value: value*distribution.pdf(value), -np.inf, q,
                          epsabs=1e-11, epsrel=1e-10)[0]
    risk = parametric_risk(fit, alpha)
    assert risk["var"] == pytest.approx(-q)
    assert risk["es"] == pytest.approx(-tail/alpha, rel=1e-8, abs=1e-10)


def test_t_es_is_infinite_when_mean_does_not_exist():
    fit = Fit("t", stats.t(.8), dict(mu=0., scale=1., nu=.8), 0., 0., 0.)
    assert np.isinf(parametric_risk(fit)["es"])


def test_gaussian_independence_copula_and_per_observation_sum():
    uniforms = np.random.default_rng(55).uniform(.001, .999, (60, 3))
    independent = copula_loglik(uniforms, np.eye(3), per_observation=True)
    np.testing.assert_allclose(independent, 0., atol=1e-13)
    r = np.array([[1., .3, -.15], [.3, 1., .2], [-.15, .2, 1.]])
    z = stats.norm.ppf(uniforms)
    expected = -.5*np.linalg.slogdet(r)[1]-.5*np.sum(z*(z@np.linalg.inv(r)-z), axis=1)
    actual = copula_loglik(uniforms, r, per_observation=True)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert copula_loglik(uniforms, r) == pytest.approx(expected.sum(), abs=1e-11)


def test_t_copula_likelihood_against_explicit_log_density_formula():
    uniforms = np.random.default_rng(56).uniform(.01, .99, (40, 2))
    r = np.array([[1., .45], [.45, 1.]])
    nu, dimension = 7., 2
    z = stats.t.ppf(uniforms, nu)
    quadratic = np.einsum("ij,jk,ik->i", z, np.linalg.inv(r), z)
    joint = (special.gammaln((nu+dimension)/2)-special.gammaln(nu/2)
             -dimension/2*np.log(nu*np.pi)-.5*np.linalg.slogdet(r)[1]
             -(nu+dimension)/2*np.log1p(quadratic/nu))
    expected = joint-stats.t.logpdf(z, nu).sum(axis=1)
    np.testing.assert_allclose(copula_loglik(uniforms, r, nu, per_observation=True), expected, atol=1e-12)
    assert copula_loglik(uniforms, r, nu) == pytest.approx(expected.sum())


def test_reciprocal_nu_profile_recovers_synthetic_interior_optimum_and_boundaries():
    nu, ll = profile_nu(lambda value: 4.-(value-7.3)**2)
    assert nu == pytest.approx(7.3, abs=5e-4)
    assert ll == pytest.approx(4., abs=1e-6)
    upper, _ = profile_nu(lambda value: value)
    lower, _ = profile_nu(lambda value: -value)
    assert upper == pytest.approx(100.) and lower == pytest.approx(1/.49)


def test_nu_profile_of_synthetic_t_copula_is_at_least_as_good_as_reference_grid():
    rng = np.random.default_rng(57)
    r = np.array([[1., .5], [.5, 1.]])
    sample = rng.multivariate_normal(np.zeros(2), r, size=160)
    sample /= np.sqrt(rng.chisquare(6., size=160)/6.)[:, None]
    uniforms = stats.t.cdf(sample, 6.)
    nu, ll = profile_nu(lambda value: copula_loglik(uniforms, r, value))
    assert 1/.49 <= nu <= 100.
    assert ll == pytest.approx(copula_loglik(uniforms, r, nu))
    grid_ll = [copula_loglik(uniforms, r, value) for value in [3., 4., 5., 6., 8., 10., 20., 50., 100.]]
    assert ll >= max(grid_ll)-1e-7


def test_copula_fit_returns_both_candidates_and_supplied_marginal_cdfs():
    rng = np.random.default_rng(59)
    x = rng.multivariate_normal([.001, -.002], [[.01, .003], [.003, .02]], size=100)
    margins = [fit_distribution(x[:, i], "normal") for i in range(2)]
    fitted = fit_copulas(x, margins)
    assert {"gaussian", "t", "nu", "r", "u"}.issubset(fitted)
    np.testing.assert_allclose(fitted["u"], np.column_stack([margins[i].distribution.cdf(x[:, i]) for i in range(2)]))
    np.testing.assert_allclose(np.diag(fitted["r"]), 1.)
    assert np.linalg.eigvalsh(fitted["r"]).min() > 0
    for family, df in [("gaussian", None), ("t", fitted["nu"])]:
        assert fitted[family]["ll"] == pytest.approx(copula_loglik(fitted["u"], fitted["r"], df))


@pytest.mark.parametrize("ll,k,n", [(np.nan, 1, 10), (np.inf, 1, 10), (0., -1, 10),
                                     (0., 1.5, 10), (0., 2, 3), (0., 1, 10.5)])
def test_invalid_information_criteria_arguments(ll, k, n):
    with pytest.raises(ValueError):
        information_criteria(ll, k, n)


@pytest.mark.parametrize("sample,family", [([1.] * 6, "normal"), ([1., 2.], "t"),
                                          ([1., 2., 3., 4., 5., np.nan], "nig"),
                                          (np.arange(6), "unsupported")])
def test_invalid_distribution_fit_arguments(sample, family):
    with pytest.raises(ValueError):
        fit_distribution(sample, family)


@pytest.mark.parametrize("alpha", [0., 1., -.01, np.nan])
def test_invalid_risk_probability(alpha):
    fit = Fit("normal", stats.norm(), dict(mu=0., sigma=1.), 0., 0., 0.)
    with pytest.raises(ValueError):
        historical_risk([1., 2.], alpha)
    with pytest.raises(ValueError):
        parametric_risk(fit, alpha)


def test_invalid_regression_covariance_and_copula_inputs():
    with pytest.raises(ValueError, match="Rank"):
        fit_ols(np.arange(6.), np.ones(6))
    with pytest.raises(ValueError):
        fit_ols(np.arange(6.), np.arange(5.))
    with pytest.raises(ValueError):
        simulate_normal([[1., 2.], [2., 1.]], n=10)
    with pytest.raises(ValueError):
        simulate_normal(np.eye(2), n=True)
    with pytest.raises(ValueError):
        repair_psd([[1., .2], [.3, 1.]])
    with pytest.raises(ValueError):
        copula_loglik([[.2, 1.1]], np.eye(2))
    with pytest.raises(ValueError):
        copula_loglik([[.2, .3]], np.eye(2), nu=0.)
    with pytest.raises(RuntimeError):
        profile_nu(lambda value: np.nan)


@pytest.mark.parametrize("matrix", [np.eye(2)*2., [[1., 1.], [1., 1.]], [[1., 2.], [2., 1.]]])
def test_copula_density_requires_positive_definite_correlation(matrix):
    with pytest.raises(ValueError):
        copula_loglik([[.2, .3], [.4, .7]], matrix)


@pytest.mark.parametrize("kwargs", [{"tol": np.nan}, {"tol": 0.}, {"max_iter": 0}, {"max_iter": 1.5}])
def test_invalid_psd_repair_tolerance_or_iterations(kwargs):
    with pytest.raises(ValueError):
        repair_psd(np.eye(2), **kwargs)


@pytest.mark.parametrize("frame", [pd.DataFrame({"X": [1., 0.]}),
                                   pd.DataFrame({"X": [1., np.inf]}),
                                   pd.DataFrame({"X": [1., np.nan]}),
                                   pd.DataFrame({"X": [1.]})])
def test_invalid_price_inputs(frame):
    with pytest.raises(ValueError):
        returns(frame)
