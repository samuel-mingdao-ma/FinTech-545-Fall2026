"""Independent density, mixture, profile-grid, and group-13 checks."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from risk545_ft2 import copulas


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REFERENCE = ROOT / "reference"


def test_multivariate_t_density_matches_scipy_scale_parameterization():
    rng = np.random.default_rng(321)
    data = rng.normal(size=(37, 3))
    mean = np.array([0.3, -0.1, 0.8])
    sd = np.array([0.7, 1.3, 1.8])
    correlation = np.array([[1, 0.2, -0.1], [0.2, 1, 0.4], [-0.1, 0.4, 1]])
    nu = 6.4
    scale = np.diag(sd) @ correlation @ np.diag(sd) * (nu - 2) / nu
    expected = stats.multivariate_t.logpdf(data, loc=mean, shape=scale, df=nu).sum()
    actual = copulas.mvt_loglikelihood(data, mean, sd, correlation, nu)
    assert_allclose(actual, expected, rtol=1e-13, atol=1e-11)
    assert_allclose(copulas.mvt_scale(sd, correlation, nu), scale, rtol=1e-14)


def test_kendall_rank_correlation_and_positive_definite_repair():
    data = np.array([[1, 4, 2], [2, 3, 4], [3, 6, 3], [4, 1, 5], [5, 2, 1]])
    actual = copulas.kendall_correlation(data)
    expected_tau = stats.kendalltau(data[:, 0], data[:, 1]).statistic
    assert actual[0, 1] == pytest.approx(np.sin(np.pi * expected_tau / 2))
    assert_allclose(actual, copulas.kendall_correlation(np.exp(data)), atol=1e-14)
    indefinite = np.array([[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]])
    repaired = copulas.fix_correlation(indefinite)
    assert np.linalg.eigvalsh(repaired).min() > 0
    assert_allclose(repaired, repaired.T, atol=1e-14)
    assert_allclose(np.diag(repaired), 1.0, atol=1e-14)
    np.linalg.cholesky(repaired)


@pytest.mark.parametrize("optimum", [0.001, 0.173456, 0.499])
def test_profile_nu_uses_both_exact_grids_including_endpoints(optimum):
    evaluated = []

    def objective(nu):
        evaluated.append(nu)
        return -(1 / nu - optimum) ** 2

    actual_nu, actual_ll = copulas.profile_nu(objective)
    coarse = np.linspace(0.01, 0.49, 200)
    index = np.argmin(abs(coarse - optimum))
    fine = np.linspace(coarse[max(0, index - 1)], coarse[min(199, index + 1)], 200)
    best = fine[np.argmin(abs(fine - optimum))]
    assert len(evaluated) == 400
    assert_allclose(evaluated[:200], 1 / coarse, atol=0, rtol=0)
    assert_allclose(evaluated[200:], 1 / fine, atol=0, rtol=0)
    assert actual_nu == 1 / best
    assert actual_ll == pytest.approx(-(best - optimum) ** 2, abs=1e-16)


def test_gaussian_independence_loglikelihood_is_zero():
    uniforms = np.random.default_rng(1).uniform(0.0001, 0.9999, size=(53, 4))
    assert copulas.gaussian_copula_ll(uniforms, np.eye(4)) == pytest.approx(0, abs=1e-12)


def test_copula_densities_match_scipy_joint_minus_margins():
    uniforms = np.random.default_rng(8).uniform(0.001, 0.999, size=(29, 3))
    correlation = np.array([[1, 0.3, -0.2], [0.3, 1, 0.1], [-0.2, 0.1, 1]])
    z = stats.norm.ppf(uniforms)
    expected_gaussian = stats.multivariate_normal.logpdf(z, cov=correlation).sum() - stats.norm.logpdf(z).sum()
    assert_allclose(copulas.gaussian_copula_ll(uniforms, correlation), expected_gaussian, atol=1e-11)
    nu = 5.7
    t = stats.t.ppf(uniforms, nu)
    expected_t = stats.multivariate_t.logpdf(t, shape=correlation, df=nu).sum() - stats.t.logpdf(t, nu).sum()
    assert_allclose(copulas.t_copula_ll(uniforms, correlation, nu), expected_t, atol=1e-11)
    # Identity correlation still has t tail dependence through its common shock.
    assert abs(copulas.t_copula_ll(uniforms, np.eye(3), nu)) > 0.1


def test_mixture_covariance_and_common_chi_square_shock():
    nu = 8.0
    scale = np.array([[0.7, 0.2, -0.1], [0.2, 1.4, 0.3], [-0.1, 0.3, 1.0]])
    mean = np.array([0.1, -0.2, 0.3])
    sample = copulas.simulate_mvt(mean, scale, nu, 180_000, seed=41)
    assert_allclose(sample.mean(axis=0), mean, atol=0.012)
    assert_allclose(np.cov(sample, rowvar=False), scale * nu / (nu - 2), rtol=0.04, atol=0.015)
    independent_normals = copulas.simulate_mvt(np.zeros(2), np.eye(2), nu, 180_000, seed=72)
    # For t_nu with identity scale, Corr(T1^2,T2^2) = 1/(nu-1).
    # Separate chi-square draws per margin would incorrectly give zero.
    squared_correlation = np.corrcoef(independent_normals**2, rowvar=False)[0, 1]
    assert squared_correlation == pytest.approx(1 / (nu - 1), abs=0.025)


def test_simulated_copula_uniform_margins_and_reproducibility():
    correlation = np.array([[1, 0.4], [0.4, 1]])
    for sample in (
        copulas.simulate_gaussian_copula(correlation, 60_000, seed=12),
        copulas.simulate_t_copula(correlation, 6.0, 60_000, seed=12),
    ):
        assert ((sample > 0) & (sample < 1)).all()
        assert_allclose(sample.mean(axis=0), 0.5, atol=0.005)
        assert_allclose(np.quantile(sample, [0.05, 0.5, 0.95], axis=0), np.array([[0.05, 0.05], [0.5, 0.5], [0.95, 0.95]]), atol=0.008)
    first = copulas.simulate_t_copula(correlation, 6.0, 20, seed=12)
    assert_allclose(first, copulas.simulate_t_copula(correlation, 6.0, 20, seed=12), atol=0, rtol=0)


@pytest.mark.parametrize(
    "operation",
    [
        lambda matrix: copulas.simulate_gaussian_copula(matrix, 20),
        lambda matrix: copulas.simulate_t_copula(matrix, 6.0, 20),
        lambda matrix: copulas.gaussian_copula_ll(np.array([[0.3, 0.7]]), matrix),
        lambda matrix: copulas.t_copula_ll(np.array([[0.3, 0.7]]), matrix, 6.0),
        lambda matrix: copulas.mvt_scale(np.ones(2), matrix, 6.0),
    ],
    ids=["Gaussian simulation", "t simulation", "Gaussian likelihood", "t likelihood", "mvt scale"],
)
@pytest.mark.parametrize(
    "invalid, message",
    [
        (np.diag([4.0, 1.0]), "unit diagonal"),
        (np.array([[1.0, 0.4], [0.2, 1.0]]), "symmetric"),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), "finite"),
    ],
    ids=["nonunit", "asymmetric", "nonfinite"],
)
def test_correlation_apis_reject_invalid_correlations(operation, invalid, message):
    with pytest.raises(ValueError, match=message):
        operation(invalid)


def test_covariance_and_scale_simulators_keep_nonunit_diagonals():
    covariance = np.diag([4.0, 1.0])
    normals = copulas.simulate_pca(covariance, 60_000, seed=5)
    assert_allclose(np.var(normals, axis=0, ddof=1), [4.0, 1.0], rtol=0.025)
    student = copulas.simulate_mvt(np.zeros(2), covariance, 6.0, 60_000, seed=5)
    assert_allclose(np.var(student, axis=0, ddof=1), [6.0, 1.5], rtol=0.04)


def test_tail_dependence_formula_and_limits():
    rho, nu = 0.45, 5.3
    expected = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)
    assert copulas.tail_dependence_t(rho, nu) == pytest.approx(expected)
    assert copulas.tail_dependence_t(-1, nu) == 0
    assert copulas.tail_dependence_t(1, nu) == 1
    assert copulas.tail_dependence_t(0, 1) == pytest.approx(1 - 1 / np.sqrt(2))
    assert copulas.tail_dependence_t(rho, 100_000) < 1e-12
    assert np.all(np.diff(copulas.tail_dependence_t(np.array([-0.5, 0, 0.5, 0.9]), nu)) > 0)


@pytest.mark.parametrize("uniforms", [np.array([[0.0, 0.5]]), np.array([[0.5, 1.0]]), np.array([[np.nan, 0.3]])])
def test_copula_boundary_and_nonfinite_input_is_rejected(uniforms):
    with pytest.raises(ValueError):
        copulas.gaussian_copula_ll(uniforms, np.eye(2))
    with pytest.raises(ValueError):
        copulas.t_copula_ll(uniforms, np.eye(2), 6)


@pytest.fixture(scope="module")
def group13_result():
    # Fail if production accidentally tries to use any teacher output as input.
    original_read = pd.read_csv
    read_names = []

    def guarded_read(path, *args, **kwargs):
        name = Path(path).name
        assert name in {"test13_returns.csv", "test13_portfolio.csv"}
        read_names.append(name)
        return original_read(path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pd, "read_csv", guarded_read)
        result = copulas.build_cases(DATA)
    assert sorted(read_names) == ["test13_portfolio.csv", "test13_returns.csv"]
    return result


def test_group13_deterministic_reference_outputs(group13_result):
    cases, diagnostics = group13_result
    assert set(cases) == {f"13.{i}" for i in range(1, 10)}
    for number in range(1, 9):
        actual = cases[f"13.{number}"]
        expected = pd.read_csv(REFERENCE / f"testout13_{number}.csv")
        assert list(actual.columns) == list(expected.columns)
        assert actual.shape == expected.shape
        for column in actual:
            if pd.api.types.is_numeric_dtype(actual[column]):
                # Generalized-t numerical MLE varies slightly across solvers;
                # mean/rank/scale/profile checks receive tighter bounds below.
                assert_allclose(actual[column], expected[column], rtol=2e-5, atol=2e-4)
            else:
                assert actual[column].tolist() == expected[column].tolist()
    for number in [1, 2, 3, 4]:
        assert_allclose(cases[f"13.{number}"], pd.read_csv(REFERENCE / f"testout13_{number}.csv"), rtol=1e-11, atol=1e-10)
    assert diagnostics["13.6"]["correlation_matches_gaussian"]
    assert diagnostics["13.4"]["covariance_scale_identity_max_error"] < 1e-16


def test_group13_fit_is_sample_mean_sd_and_rank_profile(group13_result):
    cases, _ = group13_result
    data = pd.read_csv(DATA / "test13_returns.csv").to_numpy()
    nu = cases["13.4"]["nu"].iloc[0]
    sd = data.std(axis=0, ddof=1)
    assert_allclose(cases["13.2"]["mu"], data.mean(axis=0), atol=1e-16)
    fitted_covariance = cases["13.3"].to_numpy() * nu / (nu - 2)
    assert_allclose(np.diag(fitted_covariance), sd**2, rtol=1e-13)
    correlation = cases["13.1"].to_numpy()
    assert_allclose(fitted_covariance, np.outer(sd, sd) * correlation, rtol=1e-13)
    data_ll = lambda df: stats.multivariate_t.logpdf(data, loc=data.mean(axis=0), shape=np.outer(sd, sd) * correlation * (df - 2) / df, df=df).sum()
    # The profiled result should beat either neighboring coarse-grid point.
    grid = np.linspace(0.01, 0.49, 200)
    best_coarse = max(data_ll(1 / theta) for theta in grid)
    assert cases["13.4"]["ll"].iloc[0] >= best_coarse - 1e-10


def test_group13_model_selection_uses_one_extra_t_parameter(group13_result):
    cases, _ = group13_result
    comparison = cases["13.7"].set_index("Copula")
    assert comparison.loc["Gaussian", "K"] == 0
    assert comparison.loc["T", "K"] == 1
    assert comparison.loc["T", "AICC"] < comparison.loc["Gaussian", "AICC"]
    assert comparison.loc["T", "BIC"] < comparison.loc["Gaussian", "BIC"]
    ll = cases["13.6"]["ll"].iloc[0]
    assert comparison.loc["T", "BIC"] == pytest.approx(np.log(250) - 2 * ll)
    assert comparison.loc["T", "AICC"] == pytest.approx(-2 * ll + 2 + 4 / 248)
    pairs = cases["13.8"]
    assert len(pairs) == 10
    assert ((pairs["I"] < pairs["J"]) & (pairs["Lambda"] > 0)).all()


def test_group13_stochastic_risk_has_mcse_and_analytic_marginal_validation(group13_result):
    cases, diagnostics = group13_result
    risk = cases["13.9"].set_index("Stock")
    diag = diagnostics["13.9"]
    mcse = pd.DataFrame(diag["risk_mcse"]).set_index("Stock")
    analytic = pd.DataFrame(diag["analytic_marginal_risk"]).set_index("Stock")
    values = pd.read_csv(DATA / "test13_portfolio.csv").set_index("Stock")["currentValue"]
    values.loc["Total"] = values.sum()
    assert diag["batches"] == 20
    assert sum(diag["batch_sizes"]) == 100_000
    assert diag["common_chi_square_per_path"] and diag["all_risks_positive"]
    assert (risk > 0).all().all()
    assert (mcse > 0).all().all()
    assert (risk["ES95"] > risk["VaR95"]).all()
    for measure in ["VaR95", "ES95"]:
        assert_allclose(risk[f"{measure}_Pct"], risk[measure] / values, rtol=1e-14)
        assert_allclose(mcse[f"{measure}_Pct_MCSE"], mcse[f"{measure}_MCSE"] / values, rtol=1e-14)
        # Analytic univariate fitted-t risk is independent of copula simulation.
        assert (abs(risk.loc[analytic.index, measure] - analytic[measure]) < 6 * mcse.loc[analytic.index, f"{measure}_MCSE"]).all()
    # Julia and NumPy use different streams. Compare independent simulations
    # using their combined estimated Monte Carlo uncertainty, not decimals.
    teacher = pd.read_csv(REFERENCE / "testout13_9.csv").set_index("Stock")
    for measure in ["VaR95", "ES95"]:
        assert (abs(risk[measure] - teacher[measure]) < 6 * np.sqrt(2) * mcse[f"{measure}_MCSE"]).all()


def test_analytic_margin_es_matches_numerical_tail_integration():
    model = stats.t(7.3, loc=0.0012, scale=0.014)
    value, alpha = 2_000, 0.05
    actual = copulas._analytic_margin_risk(model, value, alpha)
    expected_es = -value * model.expect(lambda x: x, lb=-np.inf, ub=model.ppf(alpha)) / alpha
    assert actual["VaR95"] == pytest.approx(-value * model.ppf(alpha))
    assert actual["ES95"] == pytest.approx(expected_es, rel=1e-8)


def test_normal_limit_margin_risk_and_mixed_copula_simulation_preserve_stock_ids(tmp_path):
    # Normal-limit and finite-t marginal fits must both remain usable by the
    # inverse-CDF simulation. CSV identifiers must retain their leading zeros.
    returns = pd.DataFrame({
        "001": np.random.default_rng(0).normal(size=100) * .01,
        "010": stats.t.rvs(5, size=100, random_state=7) * .01,
    })
    returns.to_csv(tmp_path / "test13_returns.csv", index=False)
    pd.DataFrame({"Stock": ["010", "001"], "currentValue": [100., 200.]}).to_csv(
        tmp_path / "test13_portfolio.csv", index=False)
    cases, diagnostics = copulas.build_cases(tmp_path, simulations=10_000)
    assert diagnostics["13.6"]["marginal_df"][0] is None
    assert diagnostics["13.6"]["marginal_df"][1] > 2
    assert diagnostics["13.6"]["normal_limit_margins"] == ["001"]
    assert cases["13.9"]["Stock"].tolist() == ["010", "001", "Total"]
    assert np.isfinite(cases["13.9"].iloc[:, 1:]).all().all()
    json.dumps(diagnostics, allow_nan=False)
    normal = stats.norm(.002, .01)
    margin = copulas._analytic_margin_risk(stats.t(np.inf, loc=.002, scale=.01), 200.)
    expected_es = 200 * (-.002 + .01 * stats.norm.pdf(stats.norm.ppf(.05)) / .05)
    assert margin["ES95"] == pytest.approx(expected_es)
    assert margin["VaR95"] == pytest.approx(-200 * normal.ppf(.05))


@pytest.mark.parametrize("function", [copulas.copula_aicc, copulas.copula_bic])
@pytest.mark.parametrize("arguments", [(np.nan, 1, 20), (np.inf, 1, 20),
                                        (-3., .5, 20), (-3., 1, 20.5),
                                        (-3., True, 20)])
def test_information_criteria_reject_nonfinite_likelihood_and_noninteger_counts(function, arguments):
    with pytest.raises(ValueError):
        function(*arguments)


@pytest.mark.parametrize("keyword", ["ridge", "tol"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_correlation_repair_rejects_nonfinite_controls(keyword, value):
    with pytest.raises(ValueError, match="finite"):
        copulas.fix_correlation(np.eye(2), **{keyword: value})


@pytest.mark.parametrize("nu", [np.array([6.]), np.nan, np.inf])
def test_copula_degrees_of_freedom_must_be_finite_scalars(nu):
    operations = [
        lambda: copulas.mvt_scale(np.ones(2), np.eye(2), nu),
        lambda: copulas.t_copula_ll(np.array([[.3, .7]]), np.eye(2), nu),
        lambda: copulas.simulate_mvt(np.zeros(2), np.eye(2), nu, 20),
        lambda: copulas.tail_dependence_t(.3, nu),
    ]
    for operation in operations:
        with pytest.raises(ValueError):
            operation()


def test_pca_simulation_rejects_boolean_draw_count():
    with pytest.raises(ValueError, match="positive integer"):
        copulas.simulate_pca(np.eye(2), True)
