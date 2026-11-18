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
