"""Independent tests for portfolio constraints and factor data/regression units."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assignment3 import q4, q5
from assignment3.common import ols


def test_unrestricted_tangency_matches_linear_system():
    mean = np.array([.012, .008, .014, .018])
    covariance = np.diag([.002, .0004, .004, .0008])
    rf = .002
    w, diagnostic = q4.maximum_sharpe(mean, covariance, rf)
    raw = np.linalg.inv(covariance) @ (mean-rf)
    np.testing.assert_allclose(w, raw/raw.sum(), rtol=1e-13)
    assert w.sum() == pytest.approx(1)
    assert diagnostic["method"] == "analytic"
    bounded, _ = q4.maximum_sharpe(mean, covariance, rf, np.ones(4))
    np.testing.assert_allclose(bounded, w, atol=1e-6)


def test_active_putw_cap_and_multistart_constraints():
    mean = np.array([.009, .006, .01, .025])
    covariance = np.diag([.0025, .0003, .008, .0002])
    upper = np.array([1., 1., 1., .2])
    w, diagnostics = q4.maximum_sharpe(mean, covariance, .002, upper)
    assert w.sum() == pytest.approx(1, abs=1e-10)
    assert np.all(w >= -1e-10) and np.all(w <= upper+1e-10)
    assert w[3] == pytest.approx(.2, abs=1e-7)
    assert diagnostics["successful_starts"] >= 5
    best = q4.sharpe(w, mean, covariance, .002)
    # Independent coarse simplex grid with exactly the same cap.
    for putw in np.linspace(0, .2, 5):
        for eq in np.linspace(0, 1-putw, 11):
            for bond in np.linspace(0, 1-putw-eq, 11):
                test = np.array([eq, bond, 1-putw-eq-bond, putw])
                assert q4.sharpe(test, mean, covariance, .002) <= best+1e-8


def test_frontier_budget_nonnegative_risk_and_efficient_branch():
    mean = np.array([.009, .005, .013, .007])
    covariance = np.array([[.0025, .00004, .001, .0002], [.00004, .0003, .00005, 0],
                           [.001, .00005, .006, .0003], [.0002, 0, .0003, .0005]])
    frontier = q4.efficient_frontier(mean, covariance, points=17)
    np.testing.assert_allclose(frontier[q4.ASSETS].sum(axis=1), 1, atol=1e-9)
    assert (frontier[q4.ASSETS].to_numpy() >= -1e-9).all()
    assert np.all(np.diff(frontier["mean"]) >= -1e-10)
    assert np.all(np.diff(frontier.sd) >= -1e-7)
    np.testing.assert_allclose(frontier.iloc[-1][q4.ASSETS].to_numpy(float), [0, 0, 1, 0])


def test_common_ols_standard_errors_against_linregress():
    rng = np.random.default_rng(121)
    x = rng.normal(size=300)
    y = .002+.8*x+rng.normal(scale=.2, size=300)
    actual = ols(y, x, ["market"])
    reference = stats.linregress(x, y)
    np.testing.assert_allclose(actual["coefficients"], [reference.intercept, reference.slope], rtol=1e-12)
    np.testing.assert_allclose(actual["standard_errors"], [reference.intercept_stderr, reference.stderr], rtol=1e-12)
    assert actual["k"] == 3


def test_factor_percentage_conversion_and_whitespace(tmp_path):
    path = tmp_path/"factors.csv"
    pd.DataFrame({"Date": [20240102, 20240103], "Mom   ": [1.2, -.5], "RF": [.02, .02]}).to_csv(path, index=False)
    actual = q5.read_factor_file(path)
    assert list(actual.columns) == ["Date", "Mom", "RF"]
    np.testing.assert_allclose(actual.Mom, [.012, -.005])
    np.testing.assert_allclose(actual.RF, [.0002, .0002])
    assert actual.Date.iloc[0] == pd.Timestamp("2024-01-02")


def test_recent_window_uses_file_end_and_open_left_boundary():
    dates = pd.to_datetime(["2000-01-01", "2010-09-30", "2014-09-30", "2014-10-01", "2024-09-30"])
    frame = pd.DataFrame({"Date": dates, "Mom": np.arange(len(dates))})
    actual, start, end = q5.recent_factor_window(frame)
    assert start == pd.Timestamp("2014-09-30") and end == pd.Timestamp("2024-09-30")
    assert actual.Date.tolist() == list(pd.to_datetime(["2014-10-01", "2024-09-30"]))


def make_synthetic_factor_files(data):
    rng = np.random.default_rng(525)
    dates = pd.bdate_range("1998-01-01", "2025-09-30")
    values = rng.normal([.0003, .0001, .00008, .0002], [.01, .006, .007, .008], (len(dates), 4))
    values[100, 3] = -.25  # Stress event is outside both the fund and trailing-ten-year windows.
    ff = pd.DataFrame(values[:, :3]*100, columns=q5.FACTORS[:3])
    ff.insert(0, "Date", dates.strftime("%Y%m%d"))
    ff["RF"] = .01
    ff.to_csv(data/"F-F_Research_Data_Factors_daily.CSV", index=False)
    pd.DataFrame({"Date": dates.strftime("%Y%m%d"), "Mom   ": values[:, 3]*100}).to_csv(
        data/"F-F_Momentum_Factor_daily.CSV", index=False)
    mask = (dates >= "2006-01-01") & (dates < "2007-01-01")
    x = values[mask]
    alpha = np.array([.0002, .0003])
    beta = np.array([[1.1, .4, -.3, .8], [.8, .05, .1, -.1]])
    fund_returns = .0001+alpha+x @ beta.T+rng.normal(0, .00002, (len(x), 2))
    funds = pd.DataFrame(fund_returns, columns=["FUND1", "FUND2"])
    funds.insert(0, "Date", dates[mask].strftime("%Y-%m-%d"))
    funds.to_csv(data/"problem5.csv", index=False)
    return alpha, beta


def test_complete_factor_pipeline_alignment_coefficients_stress_and_attribution(tmp_path):
    data, out, figures = tmp_path/"data", tmp_path/"out", tmp_path/"figures"
    data.mkdir()
    alpha, beta = make_synthetic_factor_files(data)
    summary = q5.run(data, out, figures)
    coefficients = pd.read_csv(out/"q5_regression_coefficients.csv")
    for j, fund in enumerate(["FUND1", "FUND2"]):
        fitted = coefficients.loc[(coefficients.fund == fund) & (coefficients.model == "Carhart4")]
        np.testing.assert_allclose(fitted.coefficient.to_numpy(), np.r_[alpha[j], beta[j]], atol=.0006)
    comparison = pd.read_csv(out/"q5_model_comparison.csv")
    assert len(comparison) == 6 and set(comparison.k) == {3, 5, 6}
    stress = pd.read_csv(out/"q5_momentum_stress.csv")
    assert set(stress.worst_momentum_return) == {-.25}
    assert all(date < "2006-01-01" for date in stress.worst_momentum_date)
    assert summary["forecast_first_date"] > "2015-09-30"
    assert summary["fund_sample_end"] < summary["forecast_first_date"]
    np.testing.assert_allclose(stress.loss_dollars, -10_000_000*stress.momentum_beta*(-.25))
    attribution = pd.read_csv(out/"q5_alpha_attribution_checks.csv")
    np.testing.assert_allclose(attribution.residual, 0, atol=1e-12)
    expected = pd.read_csv(out/"q5_expected_returns.csv")
    np.testing.assert_allclose(expected.expected_annual_with_alpha-expected.expected_annual_without_alpha,
                               expected.annual_fitted_alpha)


def test_portfolio_pipeline_records_all_risk_methods_and_constraints(tmp_path):
    data, out, figures = tmp_path/"data", tmp_path/"out", tmp_path/"figures"
    data.mkdir()
    rng = np.random.default_rng(441)
    eq = rng.normal(.01, .05, 240)
    frame = pd.DataFrame({"Month": pd.date_range("2000-01-31", periods=240, freq="ME"),
                          "EQ": eq, "BOND": rng.normal(.004, .012, 240),
                          "STK": .003+1.1*eq+rng.normal(0, .04, 240),
                          "PUTW": .008+np.minimum(eq+.05, 0)*.7+rng.normal(0, .001, 240)})
    frame.to_csv(data/"problem4.csv", index=False)
    summary = q4.run(data, out, figures, n_boot=3)
    weights = pd.read_csv(out/"q4_portfolios.csv").set_index("portfolio")
    np.testing.assert_allclose(weights[q4.ASSETS].sum(axis=1), 1, atol=1e-8)
    assert weights.loc["PUTW20", "PUTW"] <= .2+1e-9
    assert (weights.loc[["LongOnly", "PUTW20"], q4.ASSETS] >= -1e-9).all().all()
    risks = pd.read_csv(out/"q4_portfolio_risk.csv")
    assert len(risks) == 6 and set(risks.alpha) == {.05, .01}
    assert {"historical_VaR", "historical_ES", "normal_VaR", "normal_ES"}.issubset(risks)
    assert summary["risk_free_monthly"] == pytest.approx(1.03**(1/12)-1)
    assert (out/"q4_tail_convention_sensitivity.csv").exists()


def test_missing_fund_factor_alignment_is_rejected(tmp_path):
    data = tmp_path/"data"; data.mkdir()
    make_synthetic_factor_files(data)
    funds = pd.read_csv(data/"problem5.csv")
    funds.loc[0, "Date"] = "1900-01-01"
    funds.to_csv(data/"problem5.csv", index=False)
    with pytest.raises(ValueError, match="Every fund date"):
        q5.aligned_data(data)
