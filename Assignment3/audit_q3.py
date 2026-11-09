"""Reproduce the independent Q3 Monte Carlo and finite-difference audits.

Run the main assignment calculation first. This script reads its Q3 outputs,
reruns Q3 with a different random seed in a temporary directory, and writes only
the two named independent-audit files into results/. Existing numerical outputs
and prediction notes are never reused as the rerun's output directory.
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

# Keep optional Matplotlib font/cache files outside the submitted calculation.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "assignment3-mpl-cache"))

from assignment3 import q3
from assignment3.common import write_json
from assignment3.pricing import greeks, price

ROOT = Path(__file__).resolve().parent
AUDIT_JSON = "q3_independent_audit.json"
COMPARISON_CSV = "q3_independent_comparison.csv"
KEYS = ["family", "Portfolio", "alpha"]


def _production_hashes(results):
    """Hash all pre-existing Q3 results other than this script's own outputs."""
    return {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in results.glob("q3*")
        if path.is_file() and path.name not in {AUDIT_JSON, COMPARISON_CSV}
    }


def audit_greeks():
    """Synthetic derivative checks: call/put, three carries, four Greeks."""
    checks = []
    s, k, t, r, sigma = 103., 100., .7, .0425, .27
    h_s, h_t, h_sigma = .003, 1e-5, 1e-5
    for b in (.0425, .0365, 0.):
        for is_call in (True, False):
            analytical = greeks(s, k, t, r, b, sigma, is_call)
            up = price(s+h_s, k, t, r, b, sigma, is_call)
            down = price(s-h_s, k, t, r, b, sigma, is_call)
            numerical = dict(
                delta=(up-down)/(2*h_s),
                gamma=(up-2*analytical["price"]+down)/(h_s*h_s),
                theta=-(price(s, k, t+h_t, r, b, sigma, is_call)
                        -price(s, k, t-h_t, r, b, sigma, is_call))/(2*h_t),
                vega=(price(s, k, t, r, b, sigma+h_sigma, is_call)
                      -price(s, k, t, r, b, sigma-h_sigma, is_call))/(2*h_sigma),
            )
            for name, value in numerical.items():
                error = abs(value-analytical[name])/max(1e-12, abs(analytical[name]))
                checks.append(dict(cost_of_carry=b, is_call=is_call, greek=name,
                                   analytic=analytical[name], finite_difference=value,
                                   relative_error=float(error), passed=bool(error < 1e-5)))
    assert len(checks) == 24
    return checks


def run(simulations=1000000, seed=1545):
    if not isinstance(simulations, int) or simulations < 5000:
        raise ValueError("Use at least 5000 integer simulation paths")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Use a nonnegative integer random seed")
    results = ROOT / "results"
    primary = json.loads((results / "q3_summary.json").read_text(encoding="utf-8"))
    if seed == primary["parameters"]["seed"]:
        raise ValueError("The audit seed must differ from the production seed")
    original_hashes = _production_hashes(results)
    baseline = pd.read_csv(results / "q3_full_revaluation_risk.csv").set_index(KEYS)
    baseline_se = pd.read_csv(results / "q3_monte_carlo_errors.csv").set_index(KEYS)
    baseline_fits = pd.read_csv(results / "q3_distribution_fits.csv")
    if baseline.index.has_duplicates or len(baseline) != 16:
        raise ValueError("Expected all 16 production book/distribution/tail-level rows")

    # A different output directory also prevents rerunning the workflow from
    # changing the production prediction notes or any production chart/table.
    with tempfile.TemporaryDirectory(prefix="assignment3-q3-independent-") as temporary:
        work = Path(temporary)
        independent = q3.run(ROOT / "data", work / "results", work / "figures",
                             n_sim=simulations, seed=seed)
        rerun = pd.read_csv(work / "results/q3_full_revaluation_risk.csv").set_index(KEYS)
        rerun_se = pd.read_csv(work / "results/q3_monte_carlo_errors.csv").set_index(KEYS)
        rerun_fits = pd.read_csv(work / "results/q3_distribution_fits.csv")
        pd.testing.assert_frame_equal(baseline_fits, rerun_fits, check_exact=False,
                                      rtol=1e-10, atol=1e-12)
        if rerun.index.has_duplicates or set(baseline.index) != set(rerun.index):
            raise AssertionError("Independent rerun is missing required comparison rows")
        rows = []
        for key in baseline.index:
            for metric in ("VaR", "ES"):
                left, right = float(baseline.loc[key, metric]), float(rerun.loc[key, metric])
                se = float(np.hypot(baseline_se.loc[key, metric+"_MCSE"],
                                    rerun_se.loc[key, metric+"_MCSE"]))
                if not np.isfinite([left, right, se]).all() or se <= 0:
                    raise AssertionError("Finite risk values and positive Monte Carlo SE required")
                z = (left-right)/se
                rows.append(dict(family=key[0], Portfolio=key[1], alpha=key[2], metric=metric,
                                 production_value=left, independent_value=right,
                                 joint_MCSE=se, z=z, passed=bool(abs(z) < 5)))

    if _production_hashes(results) != original_hashes:
        raise AssertionError("An existing production Q3 output changed during the audit")
    comparison = pd.DataFrame(rows)
    if len(comparison) != 32:
        raise AssertionError("Expected 32 risk comparisons")
    greek_checks = audit_greeks()
    max_z = float(comparison.z.abs().max())
    max_greek_error = max(row["relative_error"] for row in greek_checks)
    proof = dict(
        final_paths=int(primary["parameters"]["paths"]),
        final_seed=int(primary["parameters"]["seed"]),
        independent_paths=int(independent["parameters"]["paths"]), independent_seed=seed,
        compared_risk_points=len(comparison), max_absolute_joint_MCSE_z=max_z,
        tolerance_z=5., independent_Greek_checks=len(greek_checks),
        max_Greek_relative_error=max_greek_error, greek_tolerance_relative=1e-5,
        production_files_unchanged=True, greek_checks=greek_checks,
        scope="Independent Monte Carlo random-seed rerun conditional on the same fitted models, plus synthetic central-finite-difference derivative checks",
        passed=bool(comparison.passed.all() and all(row["passed"] for row in greek_checks)),
    )
    comparison.to_csv(results / COMPARISON_CSV, index=False)
    write_json(results / AUDIT_JSON, proof)
    print(f"Risk comparisons: {len(comparison)}/32; maximum |joint-MCSE z|: {max_z:.8f}")
    print(f"Greek checks: {len(greek_checks)}/24; maximum relative error: {max_greek_error:.8g}")
    print("Production Q3 results and prediction notes unchanged")
    if not proof["passed"]:
        raise AssertionError("Independent Q3 audit tolerance exceeded; see audit JSON and CSV")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=1545)
    arguments = parser.parse_args()
    run(arguments.simulations, arguments.seed)
