"""Regression checks for reproduced numerical and harness failures."""
import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import risk545 as risk
from functional_cases import calculate, compare, output_name, simulation_target

# Both submissions have a run_functional_tests.py entry point. Load this one
# explicitly so collecting both test folders cannot select the other runner.
_spec = importlib.util.spec_from_file_location(
    "functionaltests1_runner", Path(__file__).resolve().parents[1] / "run_functional_tests.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


@pytest.mark.parametrize("case", ["1.5", "2.4", "3.5", "5.6", "7.7"])
def test_unknown_case_ids_are_rejected(case):
    with pytest.raises(ValueError, match="scope"):
        calculate(case)
    with pytest.raises(ValueError, match="scope"):
        output_name(case)


def test_simulation_target_rejects_non_simulation_case():
    with pytest.raises(ValueError, match="simulation"):
        simulation_target("7.1")


def test_simulation_comparison_rejects_nonfinite_reference():
    target = pd.DataFrame(simulation_target("5.1"))
    expected = target.copy()
    expected.iloc[0, 0] = np.nan
    with pytest.raises(AssertionError, match="Reference includes"):
        compare("5.1", target, expected)


@pytest.mark.parametrize("covariance", [
    np.diag([1e-20, 1e-18]),
    np.array([[1e-20, 5e-20], [5e-20, 1e-18]]),
])
def test_cholesky_preserves_small_positive_variances(covariance):
    root = risk.chol_psd(covariance)
    np.testing.assert_allclose(root @ root.T, covariance, rtol=1e-14, atol=0)
    assert (np.diag(root) > 0).all()


@pytest.mark.parametrize("call", [
    lambda value: risk.near_psd(np.eye(2), epsilon=value),
    lambda value: risk.chol_psd(np.eye(2), tol=value),
    lambda value: risk.higham_psd(np.eye(2), tol=value),
])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_repair_tolerances_are_rejected(call, value):
    with pytest.raises(ValueError):
        call(value)


def test_failed_rerun_removes_prior_case_csv(tmp_path, monkeypatch):
    data, output = tmp_path / "data", tmp_path / "output"
    data.mkdir()
    pd.DataFrame({"x": [1., 2., 3.]}).to_csv(data / "test7_1.csv", index=False)
    pd.DataFrame({"mu": [2.], "sigma": [1.]}).to_csv(data / "testout7_1.csv", index=False)
    monkeypatch.setattr(runner, "CASE_IDS", ["7.1"])
    monkeypatch.setattr("sys.argv", ["run_functional_tests.py", "--data-dir", str(data), "--output-dir", str(output)])
    assert runner.main() == 0
    assert (output / "testout7_1.csv").exists()

    pd.DataFrame({"x": [1., 1., 1.]}).to_csv(data / "test7_1.csv", index=False)
    assert runner.main() == 1
    assert not (output / "testout7_1.csv").exists()
    report = json.loads((output / "test_report.json").read_text())
    assert report["passed"] == 0
    assert report["cases"][0]["status"] == "FAIL"
    assert report["started_at_utc"] <= report["completed_at_utc"]


def test_t_fit_reaches_gaussian_likelihood_boundary():
    sample = np.random.default_rng(0).normal(size=100)
    fit = risk.fit_t(sample)
    normal_mu, normal_sigma = sample.mean(), sample.std(ddof=0)
    normal_ll = stats.norm.logpdf(sample, loc=normal_mu, scale=normal_sigma).sum()
    fitted_ll = stats.t.logpdf(sample, fit["nu"], loc=fit["mu"], scale=fit["sigma"]).sum()
    assert fitted_ll >= normal_ll - 1e-7
    np.testing.assert_allclose([fit["mu"], fit["sigma"]], [normal_mu, normal_sigma], atol=1e-6)


def test_t_regression_reaches_gaussian_likelihood_boundary():
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    y = .3 + 1.2*x + rng.normal(size=100)
    fit = risk.fit_t_regression(y, x)
    design = np.column_stack([np.ones(len(x)), x])
    ols = np.linalg.lstsq(design, y, rcond=None)[0]
    normal_sigma = (y - design @ ols).std(ddof=0)
    normal_ll = stats.norm.logpdf(y-design@ols, scale=normal_sigma).sum()
    fitted_ll = stats.t.logpdf(y-fit["Alpha"]-fit["B1"]*x, fit["nu"], scale=fit["sigma"]).sum()
    assert fitted_ll >= normal_ll - 1e-7
    np.testing.assert_allclose([fit["Alpha"], fit["B1"], fit["sigma"]], [*ols, normal_sigma], atol=1e-6)


@pytest.mark.parametrize("factor", [1e-200, 1e200])
def test_distribution_fits_preserve_extreme_but_representable_units(factor):
    sample = np.array([-2., -1., 0., 1., 2.])
    normal = risk.fit_normal(sample*factor)
    np.testing.assert_allclose([normal["mu"]/factor, normal["sigma"]/factor], [0, np.sqrt(2.5)])

    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    t = risk.fit_t(x*factor)
    np.testing.assert_allclose([t["mu"]/factor, t["sigma"]/factor], [x.mean(), x.std(ddof=0)], atol=1e-7)
    assert np.isposinf(t["nu"])

    y = .3 + 1.2*x + rng.normal(size=100)
    baseline = risk.fit_t_regression(y, x)
    fitted = risk.fit_t_regression(y*factor, x*factor)
    np.testing.assert_allclose([fitted["Alpha"]/factor, fitted["B1"], fitted["sigma"]/factor],
                               [baseline["Alpha"], baseline["B1"], baseline["sigma"]], atol=1e-7)
    assert np.isposinf(fitted["nu"])


@pytest.mark.parametrize("factor", [1e-200, 1e200])
def test_nig_moments_reject_unrepresentable_sample_variance(factor):
    with pytest.raises(ValueError, match="moments"):
        risk.fit_nig_moments(np.array([-2., -1., 0., 1., 2.])*factor)


@pytest.mark.parametrize("function", [risk.chol_psd, risk.correlation])
def test_small_covariance_asymmetry_is_not_averaged_away(function):
    with pytest.raises(ValueError, match="symmetric"):
        function([[1e-20, 2e-21], [0, 1e-20]])


def test_aicc_rejects_nonfinite_likelihood_and_noninteger_counts():
    for ll, n, k in [(np.nan, 20, 3), (np.inf, 20, 3), (-np.inf, 20, 3),
                     ([-12.], 20, 3), (-12., 20.5, 3), (-12., True, 0),
                     (-12., 20, True), (-12., 20, np.bool_(True))]:
        with pytest.raises(ValueError):
            risk.aicc(ll, n, k)
