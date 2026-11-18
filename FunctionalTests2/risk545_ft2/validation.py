"""Compare calculated cases with independent teacher outputs and model benchmarks."""

from pathlib import Path

import numpy as np
import pandas as pd


CASE_COUNTS = {8: 6, 9: 1, 10: 4, 11: 2, 12: 3, 13: 9}
CASE_IDS = [f"{group}.{case}" for group, count in CASE_COUNTS.items() for case in range(1, count + 1)]
STOCHASTIC = {"8.3", "8.6", "9.1", "13.9"}


def normalize_groups(groups=None):
    """Return a nonempty, ordered selection of the supported integer groups."""
    try:
        values = list(CASE_COUNTS if groups is None else groups)
    except TypeError as error:
        raise ValueError("Groups must be chosen from 8 through 13") from error
    if not values or any(not isinstance(group, int) or isinstance(group, bool)
                         or group not in CASE_COUNTS for group in values):
        raise ValueError("Groups must be chosen from 8 through 13")
    return sorted(set(values))


def output_name(case):
    return f"testout{case.replace('.', '_')}.csv"


def numerical_tolerance(case):
    """Declared tolerances cover rounding, finite differences and optimizer precision."""
    if case.startswith("10."):
        return 3e-7, 1e-6
    if case == "12.2":
        return 2e-6, 2e-6
    if case in {"13.5", "13.6", "13.7", "13.8"}:
        return 1e-5, 2e-7
    return 2e-8, 2e-6


def _aligned(actual, expected):
    if list(actual.columns) != list(expected.columns) or actual.shape != expected.shape:
        raise ValueError("Calculated and reference schemas must match exactly")
    keys = []
    for candidate in ("Stock", "ID", "Copula"):
        if candidate in expected:
            keys = [candidate]
            break
    if not keys and "Value" in expected and not pd.api.types.is_numeric_dtype(expected["Value"]):
        keys = ["Value"]
    if not keys and {"I", "J"} <= set(expected):
        keys = ["I", "J"]
    if keys:
        if actual[keys].isna().any().any() or expected[keys].isna().any().any():
            raise ValueError("Output row identifiers must not be missing")
        if actual.duplicated(keys).any() or expected.duplicated(keys).any():
            raise ValueError("Duplicate output row identifiers")
        actual = actual.set_index(keys)
        expected = expected.set_index(keys)
        if set(actual.index) != set(expected.index):
            raise ValueError("Calculated and reference row identifiers differ")
        actual = actual.reindex(expected.index)
    return actual, expected


def _se_for(case, diagnostics, row, column):
    if case.startswith("8."):
        return float(diagnostics[case]["mcse"][column])
    records = {record["Stock"]: record for record in diagnostics[case]["risk_mcse"]}
    return float(records[row][f"{column}_MCSE"])


def _check_diagnostics(case, diagnostics, actual):
    """Require every simulated row and analytic marginal benchmark to be present."""
    diagnostic = diagnostics[case]
    simulations = diagnostic["simulations"]
    if not isinstance(simulations, int) or isinstance(simulations, bool) or simulations < 10000:
        raise ValueError(f"Case {case} requires its actual simulation count")
    metrics = set(actual.columns)
    if case.startswith("8."):
        if set(diagnostic["mcse"]) != metrics or set(diagnostic["analytic_risk"]) != metrics:
            raise ValueError(f"Case {case} requires every simulated metric and analytic benchmark")
    else:
        benchmarks = diagnostic["analytic_marginal_risk"]
        names = [record["Stock"] for record in benchmarks]
        errors = diagnostic["risk_mcse"]
        error_names = [record["Stock"] for record in errors]
        if (len(names) != len(set(names)) or set(names) != set(actual.index) - {"Total"}
                or len(error_names) != len(set(error_names)) or set(error_names) != set(actual.index)):
            raise ValueError(f"Case {case} requires complete unique marginal benchmarks and row errors")
        if any(not metrics <= set(record) for record in benchmarks):
            raise ValueError(f"Case {case} has an incomplete analytic marginal benchmark")


def validate_cases(cases, diagnostics, reference_dir):
    """Validate all numeric cells and the analytic marginal Monte Carlo checks.

    Python and Julia use different pseudorandom streams. Stochastic reference
    comparisons include the noise in both estimates. Teacher reference sample
    counts are 10,000 for group 8 and 100,000 for portfolio risks. Estimate the
    reference SE as ownSE*sqrt(ownN/referenceN), then combine in quadrature.
    At equal sample sizes the limit is 6*sqrt(2)*ownSE. Comparisons with exact
    fitted-model marginal risk use 6*ownSE.
    """
    if not cases or not set(cases) <= set(CASE_IDS):
        raise ValueError("At least one supported functional case is required")
    summary, details = [], []
    for case in sorted(cases, key=lambda value: tuple(map(int, value.split(".")))):
        expected = pd.read_csv(Path(reference_dir) / output_name(case))
        actual, expected = _aligned(cases[case], expected)
        if expected.empty or not any(pd.api.types.is_numeric_dtype(expected[column]) for column in expected):
            raise ValueError(f"Case {case} requires numeric reference observations")
        if case in STOCHASTIC:
            _check_diagnostics(case, diagnostics, actual)
        absolute_tolerance, relative_tolerance = numerical_tolerance(case)
        records = []
        for column in expected:
            if pd.api.types.is_numeric_dtype(expected[column]):
                if not pd.api.types.is_numeric_dtype(actual[column]):
                    raise ValueError(f"Case {case} column {column} must be numeric")
                for row, reference in expected[column].items():
                    observed = float(actual.loc[row, column])
                    reference = float(reference)
                    difference = abs(observed - reference)
                    limit = absolute_tolerance + relative_tolerance * abs(reference)
                    z = None
                    if case in STOCHASTIC:
                        error = _se_for(case, diagnostics, row, column)
                        if not np.isfinite(error) or error <= 0:
                            raise ValueError(f"Case {case} requires a positive finite MC standard error")
                        reference_count = 10000 if case.startswith("8.") else 100000
                        combined_error = error * np.sqrt(1 + diagnostics[case]["simulations"] / reference_count)
                        z = difference / combined_error
                        limit = 6 * combined_error
                    passed = bool(np.isfinite(observed) and np.isfinite(reference) and difference <= limit)
                    records.append(dict(case=case, row=str(row), column=column, calculated=observed,
                                        reference=reference, absolute_difference=difference,
                                        allowed_difference=limit, combined_mcse_z=z, passed=passed,
                                        comparison="reference"))
            elif not actual[column].equals(expected[column]):
                raise ValueError(f"Case {case} categorical column {column} differs")

        if case in STOCHASTIC:
            if case.startswith("8."):
                benchmarks = [(0, diagnostics[case]["analytic_risk"])]
            else:
                benchmarks = [(record["Stock"], record) for record in diagnostics[case]["analytic_marginal_risk"]]
            for row, benchmark in benchmarks:
                for column in expected:
                    if column not in benchmark:
                        continue
                    observed = float(actual.loc[row, column])
                    reference = float(benchmark[column])
                    error = _se_for(case, diagnostics, row, column)
                    difference = abs(observed - reference)
                    records.append(dict(case=case, row=str(row), column=column, calculated=observed,
                                        reference=reference, absolute_difference=difference,
                                        allowed_difference=6 * error, combined_mcse_z=difference / error,
                                        passed=bool(np.isfinite(observed) and np.isfinite(reference) and difference <= 6 * error),
                                        comparison="analytic marginal"))
        reference_records = [record for record in records if record["comparison"] == "reference"]
        zs = [record["combined_mcse_z"] for record in reference_records if record["combined_mcse_z"] is not None]
        passed = all(record["passed"] for record in records)
        summary.append(dict(Test=case, Status="PASS" if passed else "FAIL",
                            Method="Monte Carlo with MCSE" if case in STOCHASTIC else "Deterministic",
                            Checks=len(records), MaxAbsoluteReferenceDifference=max(
                                record["absolute_difference"] for record in reference_records),
                            WorstReferenceMCSEZ=max(zs) if zs else None))
        details.extend(records)
    summary = pd.DataFrame(summary)
    summary["WorstReferenceMCSEZ"] = summary["WorstReferenceMCSEZ"].astype(float)
    return summary, details
