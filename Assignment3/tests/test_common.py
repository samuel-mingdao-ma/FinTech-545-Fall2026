"""Independent checks on shared statistical conventions."""
import numpy as np
import pytest
from scipy import stats

from assignment3.common import (fit_distribution, historical_risk, information_criteria,
                               moments, normal_risk, ols, write_predictions)


def test_moments_and_normal_mle():
    x = np.random.default_rng(6).normal(.003, .012, 200)
    m = moments(x)
    assert m["variance"] == pytest.approx(x.var(ddof=1))
    assert m["skewness"] == pytest.approx(stats.skew(x))
    fit = fit_distribution(x, "normal")
    assert fit.parameters["sigma"] == pytest.approx(x.std(ddof=0))
    assert fit.ll == pytest.approx(stats.norm.logpdf(x, loc=x.mean(), scale=x.std(ddof=0)).sum())


def test_information_criteria_counts():
    result = information_criteria(-22., 4, 100)
    assert result["aicc"] == pytest.approx(44+8+40/95)
    assert result["bic"] == pytest.approx(44+4*np.log(100))


def test_ols_independent_coefficients_and_errors():
    x = np.random.default_rng(5).normal(size=300)
    y = .1+1.7*x+np.random.default_rng(9).normal(scale=.3, size=300)
    result = ols(y, x, ["market"])
    reference = stats.linregress(x, y)
    np.testing.assert_allclose(result["coefficients"], [reference.intercept, reference.slope])
    np.testing.assert_allclose(result["standard_errors"], [reference.intercept_stderr, reference.stderr])
    assert result["k"] == 3  # intercept, slope, residual scale


def test_course_and_fractional_tail_conventions():
    x = np.arange(-100., 100.)
    result = historical_risk(x, .05)
    assert result["var"] == 91.
    assert result["es"] == 95.5
    alternate = historical_risk(x, .0125, method="fractional")
    assert alternate["es"] == pytest.approx((100+99+.5*98)/2.5)


def test_normal_tail_integrals():
    d = stats.norm(.01, .04)
    for alpha in [.01, .05]:
        result = normal_risk(.01, .04, alpha)
        assert result["var"] == pytest.approx(-d.ppf(alpha))
        assert result["es"] == pytest.approx(-d.expect(lb=-np.inf, ub=d.ppf(alpha))/alpha)


def test_original_prediction_draft_is_not_overwritten(tmp_path):
    path = tmp_path/"predictions.json"
    write_predictions(path, ["Original pre-fit draft"])
    original = path.read_bytes()
    write_predictions(path, ["Do not replace the original after seeing a fit"])
    assert path.read_bytes() == original
