"""Regression checks for result alignment, tolerance handling and output safety."""
import pandas as pd
import pytest

from risk545_ft2.validation import normalize_groups, validate_cases


@pytest.mark.parametrize("groups", [[], [7], [14], [8.0], [True], "8", 8])
def test_group_selection_requires_supported_integers(groups):
    with pytest.raises(ValueError, match="Groups"):
        normalize_groups(groups)


def test_group_selection_is_complete_by_default_and_orders_subsets():
    assert normalize_groups() == [8, 9, 10, 11, 12, 13]
    assert normalize_groups([11, 8, 11]) == [8, 11]


@pytest.mark.parametrize("cases", [{}, {"8.0": pd.DataFrame({"Value": [1.]})}])
def test_empty_or_unknown_case_selection_cannot_pass(cases, tmp_path):
    with pytest.raises(ValueError, match="supported functional case"):
        validate_cases(cases, {}, tmp_path)


def test_option_values_are_comparisons_and_ids_are_keys(tmp_path):
    reference = pd.DataFrame({"ID": [1, 2], "Value": [4.0, 5.0]})
    reference.to_csv(tmp_path / "testout12_3.csv", index=False)
    calculated = pd.DataFrame({"ID": [2, 1], "Value": [5.000000001, 4.000000001]})
    summary, details = validate_cases({"12.3": calculated}, {}, tmp_path)
    assert summary.Status.tolist() == ["PASS"]
    assert len(details) == 2
    assert all(record["column"] == "Value" for record in details)


def test_duplicate_identifiers_are_rejected(tmp_path):
    reference = pd.DataFrame({"ID": [1, 2], "Value": [4., 5.]})
    reference.to_csv(tmp_path / "testout12_3.csv", index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_cases({"12.3": pd.DataFrame({"ID": [1, 1], "Value": [4., 5.]})}, {}, tmp_path)


def test_missing_identifiers_are_rejected(tmp_path):
    frame = pd.DataFrame({"ID": [1, None], "Value": [4., 5.]})
    frame.to_csv(tmp_path / "testout12_3.csv", index=False)
    with pytest.raises(ValueError, match="identifiers must not be missing"):
        validate_cases({"12.3": frame}, {}, tmp_path)


@pytest.mark.parametrize("frame", [pd.DataFrame({"Value": []}), pd.DataFrame({"Label": ["empty"]})])
def test_reference_requires_numeric_observations(frame, tmp_path):
    frame.to_csv(tmp_path / "testout12_3.csv", index=False)
    with pytest.raises(ValueError, match="numeric reference observations"):
        validate_cases({"12.3": frame}, {}, tmp_path)


def test_mc_tolerance_does_not_hide_biased_simulation(tmp_path):
    columns = ["VaR Absolute", "VaR Diff from Mean"]
    pd.DataFrame([[.04, .09]], columns=columns).to_csv(tmp_path / "testout8_3.csv", index=False)
    simulated = pd.DataFrame([[.2, .3]], columns=columns)
    diagnostic = {"8.3": {"simulations": 10000, "mcse": dict.fromkeys(columns, .001),
                           "analytic_risk": dict(zip(columns, [.04, .09]))}}
    summary, details = validate_cases({"8.3": simulated}, diagnostic, tmp_path)
    assert summary.Status.tolist() == ["FAIL"]
    assert not any(record["passed"] for record in details)


def test_missing_analytic_benchmarks_are_rejected(tmp_path):
    frame = pd.DataFrame({"Stock": ["A", "Total"], "VaR95": [1., 1.], "ES95": [2., 2.]})
    frame.to_csv(tmp_path / "testout13_9.csv", index=False)
    diagnostic = {"13.9": {"simulations": 100000, "analytic_marginal_risk": [],
                            "risk_mcse": [{"Stock": "A"}, {"Stock": "Total"}]}}
    with pytest.raises(ValueError, match="complete"):
        validate_cases({"13.9": frame}, diagnostic, tmp_path)


def test_reference_uncertainty_accounts_for_different_path_counts(tmp_path):
    frame = pd.DataFrame({"Stock": ["A", "Total"], "VaR95": [1., 1.], "ES95": [2., 2.]})
    frame.to_csv(tmp_path / "testout13_9.csv", index=False)
    diagnostic = {"13.9": {"simulations": 400000,
                            "analytic_marginal_risk": [{"Stock": "A", "VaR95": 1., "ES95": 2.}],
                            "risk_mcse": [{"Stock": name, "VaR95_MCSE": .01, "ES95_MCSE": .02}
                                          for name in ["A", "Total"]]}}
    _, details = validate_cases({"13.9": frame}, diagnostic, tmp_path)
    var = next(record for record in details if record["column"] == "VaR95")
    assert var["allowed_difference"] == pytest.approx(6 * .01 * 5 ** .5)


def test_publisher_preserves_existing_unowned_files(tmp_path):
    from functionaltests2_runner import _publish
    output = tmp_path / "existing"
    output.mkdir()
    (output / "user.txt").write_text("preserve this")
    staging = tmp_path / "new"
    staging.mkdir()
    with pytest.raises(ValueError):
        _publish(staging, output)
    assert (output / "user.txt").read_text() == "preserve this"
