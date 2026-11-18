"""Independent distributional, tail-convention and aggregation checks."""
import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from risk545_ft2.distributions import fit_normal, fit_student_t
from risk545_ft2.risk import batch_standard_errors, historical_risk, parametric_risk


def test_course_order_statistics_and_ties():
    risk = historical_risk(np.array([-4.0, -3, -2, -1, 0]), 0.3)
    assert risk == {"var": 3.5, "es": 4.0}
    assert historical_risk(np.array([-2, -2, -1, 3]), 0.5) == {"var": 2, "es": 2}
    assert historical_risk(np.array([1, 2, 3]), 0.01) == {"var": -1, "es": -1}


@pytest.mark.parametrize("distribution", [stats.norm(.003, .02), stats.t(5, .002, .03)])
@pytest.mark.parametrize("alpha", [.01, .05, .2])
def test_closed_form_expected_shortfall_against_quadrature(distribution, alpha):
    risk = parametric_risk(distribution, alpha)
    integral = integrate.quad(lambda x: x * distribution.pdf(x), -np.inf,
                              distribution.ppf(alpha), epsabs=1e-11)[0]
    assert risk["es"] == pytest.approx(-integral / alpha, rel=1e-8, abs=1e-10)
    assert risk["var"] == pytest.approx(-distribution.ppf(alpha))


def test_risk_translation_scaling_and_subadditivity():
    rng = np.random.default_rng(101)
    x, y = rng.normal(size=(2, 20000))
    rx, shifted, scaled = historical_risk(x), historical_risk(x + 3), historical_risk(4 * x)
    for metric in ("var", "es"):
        assert shifted[metric] == pytest.approx(rx[metric] - 3)
        assert scaled[metric] == pytest.approx(4 * rx[metric])
    assert historical_risk(x + y)["es"] <= rx["es"] + historical_risk(y)["es"]
    assert batch_standard_errors(x)["var"] > 0


def test_distribution_fits_are_location_scale_equivariant():
    sample = stats.t.rvs(6, size=500, random_state=7)
    base = fit_student_t(sample)
    shifted = fit_student_t(2 * sample + 5)
    np.testing.assert_allclose(shifted.ppf([.01, .5, .95]),
                               2 * base.ppf([.01, .5, .95]) + 5, rtol=1e-5, atol=1e-5)
    normal = fit_normal(sample)
    assert normal.std() == pytest.approx(sample.std(ddof=1))
    assert base.args[0] > 2


@pytest.mark.parametrize("bad", [[], [1, 1, 1], [1, np.nan, 2]])
def test_invalid_fit_samples(bad):
    with pytest.raises(ValueError):
        fit_student_t(bad)


def test_course_t_fit_has_nearby_likelihood_maximum():
    sample = stats.t.rvs(7, loc=.01, scale=.03, size=700, random_state=341)
    fitted = fit_student_t(sample)
    original_ll = fitted.logpdf(sample).sum()
    mu, scale, nu = fitted.kwds["loc"], fitted.kwds["scale"], fitted.args[0]
    for parameter in [(mu + .001, scale, nu), (mu, scale * 1.05, nu), (mu, scale, nu * 1.2)]:
        altered = stats.t(parameter[2], loc=parameter[0], scale=parameter[1])
        assert altered.logpdf(sample).sum() < original_ll


@pytest.mark.parametrize("distribution", [stats.norm(scale=1e-300), stats.t(np.inf, scale=1e-300)])
def test_extreme_normal_tail_es_preserves_representable_risk(distribution):
    risk = parametric_risk(distribution, alpha=1e-310)
    threshold = -stats.norm.ppf(1e-310)
    # Mills-ratio bounds independently bracket E[-Z | Z <= -threshold].
    assert threshold * 1e-300 < risk["es"] < (threshold + 1 / threshold) * 1e-300
    assert np.isfinite(risk["es"])
    assert risk["es"] > risk["var"] > 0


def test_student_t_fit_reaches_normal_boundary_instead_of_false_finite_optimum():
    sample = np.random.default_rng(0).normal(size=100)
    fitted = fit_student_t(sample)
    normal_mle = stats.norm(sample.mean(), sample.std(ddof=0))
    assert np.isposinf(fitted.args[0])
    assert fitted.logpdf(sample).sum() == pytest.approx(normal_mle.logpdf(sample).sum(), abs=1e-10)
    np.testing.assert_allclose(fitted.cdf(sample), normal_mle.cdf(sample), atol=2e-8)
    np.testing.assert_allclose(fitted.ppf([.01, .5, .99]), normal_mle.ppf([.01, .5, .99]), atol=1e-7)
    assert parametric_risk(fitted) == pytest.approx(parametric_risk(normal_mle), abs=1e-7)


@pytest.mark.parametrize("magnitude", [1e-200, 1e200])
def test_distribution_fits_survive_rescaling_without_moment_underflow_or_overflow(magnitude):
    sample = np.array([-3., -1., 0., 1., 3.]) * magnitude
    normal = fit_normal(sample)
    assert normal.kwds["loc"] == pytest.approx(0, abs=magnitude * 1e-15)
    assert normal.kwds["scale"] / magnitude == pytest.approx(np.sqrt(5))
    student = fit_student_t(sample)
    assert np.isfinite(student.kwds["scale"]) and student.kwds["scale"] > 0
    assert np.isfinite(student.cdf(sample)).all()


@pytest.mark.parametrize("distribution", [
    stats.norm(loc=np.nan), stats.norm(scale=0), stats.norm(scale=-1),
    stats.norm(scale=np.inf), stats.t(np.nan), stats.t(0.5),
    stats.t(5, loc=np.inf), stats.t(5, scale=0),
])
def test_parametric_risk_rejects_invalid_frozen_parameters(distribution):
    with pytest.raises(ValueError):
        parametric_risk(distribution)


def test_historical_risk_finite_extreme_and_subnormal_losses():
    with np.errstate(over="raise", invalid="raise"):
        assert historical_risk([-1e308, -1e308]) == {"var": 1e308, "es": 1e308}
    smallest = np.nextafter(0., 1.)
    assert historical_risk([-smallest, -smallest]) == {"var": smallest, "es": smallest}
