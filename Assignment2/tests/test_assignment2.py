"""End-to-end checks for the numerical conclusions in Assignment 2."""

from pathlib import Path

import pytest

from risk545.problem4 import solve_problem4
from risk545.problems12 import solve_problem1, solve_problem2
from risk545.problems35 import solve_problem3, solve_problem5


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    figures = tmp_path_factory.mktemp("figures")
    return {
        "p1": solve_problem1(ROOT / "problem1.csv", figures),
        "p2": solve_problem2(ROOT / "problem2.csv", figures),
        "p3": solve_problem3(ROOT / "problem3.csv", figures, seed=548),
        "p4": solve_problem4(ROOT / "problem4.csv", figures, seed=549),
        "p5": solve_problem5(ROOT / "problem5.csv", figures, seed=550),
    }


def test_problem1_pairwise_is_invalid_and_repairs_are_valid(results):
    p1 = results["p1"]
    assert p1["pairwise"]["minimum_eigenvalue"] < 0
    assert p1["pairwise"]["tracking_portfolio_variance"] < 0
    for method in ("rebonato_jackel", "higham"):
        assert p1["repairs"][method]["minimum_eigenvalue"] > -1e-10
        assert p1["repairs"][method]["tracking_portfolio_variance"] > 0


def test_problem2_recent_regime_drives_ew_var(results):
    p2 = results["p2"]
    assert p2["regime"]["standard_deviation_ratio_recent_to_earlier"] > 2
    assert p2["var_usd"]["normal_ew_lambda_0.94"] > p2["var_usd"]["normal_ew_lambda_0.97"]
    assert p2["exponentially_weighted"]["lambda_0.94"]["weight_on_most_recent_40_days"] > 0.9


def test_problem3_var_failure_and_es_coherence(results):
    sub = results["p3"]["subadditivity_5pct"]
    assert sub["var"]["lhs_minus_rhs"] > 0
    assert sub["es"]["lhs_minus_rhs"] < 0


def test_problem4_t_copula_wins_and_has_tail_dependence(results):
    p4 = results["p4"]
    assert p4["copula_fit"]["student_t_aicc_advantage"] > 100
    assert p4["tail_dependence"]["student_t_lower_and_upper"] > 0
    for model in p4["portfolio_risk_usd"].values():
        assert model["es_5pct"] >= model["var_5pct"]
        assert model["es_1pct"] >= model["var_1pct"]


def test_problem5_residual_correlation_has_opposite_portfolio_effects(results):
    p5 = results["p5"]
    assert p5["residual_correlation"] > 0
    p1 = p5["portfolios"]["P1_long_A_long_B"]["simulation_var_5pct"]
    p2 = p5["portfolios"]["P2_long_A_short_B"]["simulation_var_5pct"]
    assert p1["diagonal_minus_full"] < 0
    assert p2["diagonal_minus_full"] > 0
    assert p5["maximum_absolute_covariance_reconstruction_error"] < 1e-12
