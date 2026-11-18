"""Independent numerical checks for optimization and realized attribution."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risk545_ft2.portfolios import build_cases, expost_factor, maximum_sharpe, risk_parity


def test_diagonal_risk_budgets_have_closed_form_weights():
    variances = np.array([0.01, 0.04, 0.0225, 0.09])
    budget = np.array([1.0, 2.0, 0.5, 3.0])
    weights, detail = risk_parity(np.diag(variances), budget)
    expected = np.sqrt(budget / variances)
    expected /= expected.sum()
    np.testing.assert_allclose(weights, expected, atol=1e-13, rtol=1e-12)
    component_risk = weights * (variances * weights)
    np.testing.assert_allclose(component_risk / component_risk.sum(), budget / budget.sum(),
                               atol=1e-12, rtol=0)
    assert detail["risk_budget_max_abs_error"] < 1e-12


def test_risk_budget_ratio_permutation_and_scale():
    loadings = np.array([[0.2, 0.0], [0.1, 0.2], [0.3, 0.1], [0.05, 0.1]])
    covariance = loadings @ loadings.T + np.diag([0.01, 0.005, 0.003, 0.002])
    budget = np.array([1.0, 1.0, 1.0, 0.5])
    weights, _ = risk_parity(covariance, budget)
    component_sd = weights * (covariance @ weights) / np.sqrt(weights @ covariance @ weights)
    np.testing.assert_allclose(component_sd / component_sd[0], budget / budget[0], atol=1e-10)
    permutation = np.array([2, 0, 3, 1])
    reordered, _ = risk_parity(10000 * covariance[np.ix_(permutation, permutation)], budget[permutation])
    np.testing.assert_allclose(reordered, weights[permutation], atol=1e-12)


def test_singular_psd_risk_budget_is_supported():
    # A one-factor covariance has zero eigenvalues but a valid risk budget solution.
    exposure = np.array([1.0, 2.0, 4.0])
    weights, _ = risk_parity(np.outer(exposure, exposure), [1, 2, 1])
    expected = np.array([1, 2, 1]) / exposure
    np.testing.assert_allclose(weights, expected / expected.sum(), atol=1e-11)


def test_maximum_sharpe_matches_interior_tangency_and_permutation():
    covariance = np.diag([0.01, 0.04, 0.09])
    means, risk_free = np.array([0.07, 0.10, 0.13]), 0.02
    expected = np.linalg.solve(covariance, means - risk_free)
    expected /= expected.sum()
    weights, detail = maximum_sharpe(covariance, means, risk_free)
    np.testing.assert_allclose(weights, expected, atol=2e-8, rtol=0)
    assert detail["start_objective_spread"] < 1e-11
    permutation = np.array([2, 0, 1])
    reordered, _ = maximum_sharpe(covariance[np.ix_(permutation, permutation)],
                                 means[permutation], risk_free)
    np.testing.assert_allclose(reordered, expected[permutation], atol=2e-8, rtol=0)


def test_box_constraints_bind_and_match_independent_dense_search():
    covariance = np.diag([0.01, 0.04, 0.09])
    means, risk_free = np.array([0.15, 0.08, 0.04]), 0.02
    bounds = np.tile([0.1, 0.6], (3, 1))
    weights, detail = maximum_sharpe(covariance, means, risk_free, bounds)
    np.testing.assert_allclose(weights, [0.6, 0.3, 0.1], atol=1e-9, rtol=0)
    assert detail["active_lower_bounds"] == [2]
    assert detail["active_upper_bounds"] == [0]
    # Search the whole bounded simplex independently, including its edges.
    first, second = np.meshgrid(np.linspace(0.1, 0.6, 501), np.linspace(0.1, 0.6, 501))
    grid = np.column_stack((first.ravel(), second.ravel(), 1 - first.ravel() - second.ravel()))
    grid = grid[np.all((grid >= 0.1 - 1e-12) & (grid <= 0.6 + 1e-12), axis=1)]
    grid_sharpe = grid @ (means - risk_free) / np.sqrt(np.einsum("ij,jk,ik->i", grid, covariance, grid))
    assert detail["sharpe_ratio"] >= grid_sharpe.max() - 1e-10


def test_indefinite_course_convention_uses_the_original_matrix():
    # Positive entries guarantee positive variance for every nonnegative portfolio,
    # even though this symmetric matrix has a negative eigenvalue.
    covariance = np.array([[0.04, 0.06], [0.06, 0.04]])
    weights, detail = maximum_sharpe(covariance, [0.10, 0.05], 0.02)
    np.testing.assert_allclose(weights, [1, 0], atol=1e-9)
    assert not detail["covariance_is_psd"]
    assert detail["minimum_covariance_eigenvalue"] == pytest.approx(-0.02)
    assert detail["sharpe_ratio"] == pytest.approx(0.4, abs=1e-10)
    with pytest.raises(ValueError, match="positive semidefinite"):
        risk_parity(covariance)


@pytest.mark.parametrize("function,args,match", [
    (risk_parity, ([[1.0, 0.2], [0.1, 1.0]],), "symmetric"),
    (risk_parity, (np.eye(2), [1, 0]), "strictly positive"),
    (risk_parity, (np.eye(2), [1]), "length"),
    (maximum_sharpe, (np.eye(2), [0.1], 0.02), "length"),
    (maximum_sharpe, (np.eye(2), [0.1, 0.2], 0.02, [[0.6, 1], [0.6, 1]]), "infeasible"),
    (maximum_sharpe, (np.zeros((2, 2)), [0.1, 0.2], 0.02), "nonpositive"),
])
def test_optimizer_input_guards(function, args, match):
    with pytest.raises(ValueError, match=match):
        function(*args)


def test_buy_and_hold_geometric_total_and_return_attribution_additivity():
    returns = pd.DataFrame({"A": [0.10, -0.03, 0.04, -0.02],
                            "B": [-0.04, 0.08, -0.02, 0.03]})
    initial_weights = np.array([0.35, 0.65])
    result = expost_factor(initial_weights, returns, returns, np.eye(2))
    # Hold initial units: terminal wealth is a weighted sum of each asset's gross return.
    wealth = np.cumprod(1 + returns.to_numpy(), axis=0) * initial_weights
    portfolio_wealth = wealth.sum(axis=1)
    expected_returns = portfolio_wealth / np.r_[1, portfolio_wealth[:-1]] - 1
    np.testing.assert_allclose(result.portfolio_return, expected_returns, atol=3e-16, rtol=0)
    expected_start_wealth = np.vstack((initial_weights, wealth[:-1]))
    expected_weights = expected_start_wealth / expected_start_wealth.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(result.weights, expected_weights, atol=3e-16, rtol=0)
    table = result.attribution.set_index("Value")
    total_return = float(portfolio_wealth[-1] - 1)
    assert table.loc["TotalReturn", "Portfolio"] == pytest.approx(total_return, abs=8e-16)
    assert table.loc["Return Attribution", ["A", "B", "Alpha"]].sum() == pytest.approx(total_return, abs=8e-16)
    np.testing.assert_allclose(table["Alpha"], 0, atol=5e-16)
    # Distinct asset geometric returns do not simply add to portfolio performance.
    np.testing.assert_allclose(table.loc["TotalReturn", ["A", "B"]].to_numpy(dtype=float),
                               np.prod(1 + returns, axis=0) - 1, atol=5e-16)


def test_factor_attribution_volatility_matches_covariance_and_gradient():
    factors = pd.DataFrame({"Market": [0.02, -0.01, 0.015, -0.025, 0.005],
                            "Style": [-0.01, 0.02, 0.005, -0.015, 0.01]})
    beta = np.array([[1.2, -0.3], [0.5, 0.8]])
    residual = np.array([[0.005, -0.003], [-0.004, 0.002], [0.001, 0.006],
                         [-0.002, -0.001], [0.003, 0.004]])
    stocks = pd.DataFrame(factors.to_numpy() @ beta.T + residual, columns=["S1", "S2"])
    result = expost_factor([0.4, 0.6], stocks, factors, beta)
    components = np.column_stack((factors * result.factor_weights, result.residual_return))
    joint_covariance = np.cov(np.column_stack((components, result.portfolio_return)), rowvar=False, ddof=1)
    expected = joint_covariance[:-1, -1] / np.sqrt(joint_covariance[-1, -1])
    table = result.attribution.set_index("Value")
    actual = table.loc["Vol Attribution", ["Market", "Style", "Alpha"]].to_numpy(dtype=float)
    np.testing.assert_allclose(actual, expected, atol=1e-16, rtol=1e-12)
    assert actual.sum() == pytest.approx(result.portfolio_return.std(ddof=1), abs=1e-16)
    linked = table.loc["Return Attribution", ["Market", "Style", "Alpha"]].sum()
    assert linked == pytest.approx(np.prod(1 + result.portfolio_return) - 1, abs=8e-16)
    # Euler volatility contributions are derivatives under a component scaling.
    for i in range(components.shape[1]):
        step = 1e-5
        plus = (result.portfolio_return + step * components[:, i]).std(ddof=1)
        minus = (result.portfolio_return - step * components[:, i]).std(ddof=1)
        assert (plus - minus) / (2 * step) == pytest.approx(actual[i], abs=2e-12)
    reordered = expost_factor([0.6, 0.4], stocks.iloc[:, ::-1], factors.iloc[:, ::-1], beta[::-1, ::-1])
    pd.testing.assert_frame_equal(reordered.attribution[table.reset_index().columns],
                                  result.attribution, atol=1e-14, rtol=1e-12)


def test_zero_returns_have_finite_carino_limit_and_zero_volatility():
    returns = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]})
    result = expost_factor([0.3, 0.7], returns, returns, np.eye(2))
    np.testing.assert_array_equal(result.carino_k, 1)
    np.testing.assert_array_equal(result.attribution.drop(columns="Value"), 0)


def test_carino_limit_when_nonzero_period_returns_compound_to_zero():
    returns = pd.DataFrame({"Asset": [0.1, -1 / 11, 0.0]})
    result = expost_factor([1], returns, returns, [[1]])
    assert np.isfinite(result.carino_k).all()
    assert result.attribution.Portfolio.iloc[0] == pytest.approx(0, abs=3e-16)
    assert result.attribution.Asset.iloc[1] == pytest.approx(0, abs=3e-16)
    assert result.attribution.Portfolio.iloc[2] > 0


def test_attribution_input_guards():
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.02, -0.01]})
    with pytest.raises(ValueError, match="sum to one"):
        expost_factor([0.3, 0.3], returns, returns, np.eye(2))
    with pytest.raises(ValueError, match="beta"):
        expost_factor([0.3, 0.7], returns, returns, np.eye(3))
    with pytest.raises(ValueError, match="exceed -1"):
        expost_factor([0.3, 0.7], returns * -100, returns, np.eye(2))


def test_build_cases_has_teacher_schemas_and_numerical_identities():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    cases, diagnostics = build_cases(data_dir)
    assert set(cases) == {"10.1", "10.2", "10.3", "10.4", "11.1", "11.2"}
    for key in ["10.1", "10.2", "10.3", "10.4"]:
        assert list(cases[key]) == ["W"]
        assert cases[key].W.sum() == pytest.approx(1, abs=1e-12)
        assert (cases[key].W >= 0).all()
    assert list(cases["11.1"]) == ["Value", "x1", "x2", "x3", "Portfolio"]
    assert list(cases["11.2"]) == ["Value", "F1", "F2", "F3", "Alpha", "Portfolio"]
    for key in ["11.1", "11.2"]:
        assert cases[key].Value.tolist() == ["TotalReturn", "Return Attribution", "Vol Attribution"]
        assert diagnostics[key]["return_attribution_sum_error"] < 1e-14
        assert diagnostics[key]["volatility_attribution_sum_error"] < 1e-14
    assert diagnostics["10.3"]["minimum_covariance_eigenvalue"] < 0
    # Solve the surviving support analytically, independent of the constrained solver.
    weights = cases["10.3"].W.to_numpy()
    support = np.flatnonzero(weights > 1e-7)
    covariance = pd.read_csv(data_dir / "test5_3.csv").to_numpy()
    excess = pd.read_csv(data_dir / "test10_3_means.csv").Mean.to_numpy() - 0.04
    analytic = np.linalg.solve(covariance[np.ix_(support, support)], excess[support])
    analytic /= analytic.sum()
    np.testing.assert_allclose(weights[support], analytic, atol=3e-9, rtol=0)
    variance = weights @ covariance @ weights
    gradient = excess / np.sqrt(variance) - (weights @ excess) * (covariance @ weights) / variance**1.5
    np.testing.assert_allclose(gradient[support], gradient[support].mean(), atol=2e-8, rtol=0)
    assert np.all(gradient[weights < 1e-7] <= gradient[support].mean() + 1e-8)
