"""Produce the final text answer from calculated CSV and machine test reports."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import pandas as pd

root = Path(__file__).resolve().parent
out = root / "output"
report = json.loads((out / "test_report.json").read_text())
suites = ET.parse(out / "pytest-results.xml").getroot()
tests = sum(int(s.get("tests", 0)) for s in suites.findall("testsuite"))
failures = sum(int(s.get("failures", 0))+int(s.get("errors", 0)) for s in suites.findall("testsuite"))
lines = ["# FINTECH 545 — Functional Test Answers", "", "Samuel Ma", "",
         "## Scope and verification", "",
         "Implemented: 1.1–1.4, 2.1–2.3, 3.1–3.4, 4.1, 5.1–5.5 and 7.1–7.6.", "",
         "This first-checkpoint scope is inferred from the September 14 lecture; the final Canvas list has not been verified.", "",
         f"Instructor reference comparisons: **{report['passed']}/{report['total']} passed**.",
         f"Pytest suite: **{tests-failures}/{tests} passed** (23 instructor cases plus 27 independent checks).",
         "This is not a claim that all 50 cases in the full-semester Tests.xlsx are implemented.", "",
         "All answers below were calculated by risk545. Full-precision matrices and parameters are in output/*.csv.", "",
         "## Results by test", "", "| Test | Result | Maximum absolute difference vs reference |", "|---|---|---:|"]
for row in report["cases"]:
    lines.append(f"| {row['test']} | {row['status']} | {row.get('max_abs_error', float('nan')):.6g} |")
lines += ["", "## Monte Carlo covariance comparisons", "",
          "100,000 observations per case; seed 1234. Python and Julia simulations have different random streams.",
          "The Frobenius differences below are divided by the norm of the analytic target covariance.", "",
          "| Test | Relative Frobenius difference | Largest error / MC standard error vs target |", "|---|---:|---:|"]
for r in report["cases"]:
    if r["test"].startswith("5."):
        lines.append(f"| {r['test']} | {100*r['relative_frobenius']:.4f}% | {r['max_mc_z']:.4f} |")
lines += ["", "Cases 5.3 and 5.4 use the repaired covariance; case 5.5 uses the covariance implied by the retained 99% PCA factors.",
          "Criteria: relative Frobenius error <= 3%; maximum target and independent-reference standardized error <= 7.",
          "These are documented implementation tolerances, not confirmed instructor grading thresholds.", "",
          "## Distribution and regression answers", ""]
for i in range(1, 7):
    frame = pd.read_csv(out / f"testout7_{i}.csv")
    lines += [f"### Test 7.{i}", "", "| Parameter | Calculated value |", "|---|---:|"]
    for name, value in frame.iloc[0].items():
        lines.append(f"| {name} | {value:.12g} |")
    lines.append("")
lines += ["The t scale is not its standard deviation. Error location in the regression is fixed to zero, while Alpha is estimated.",
          "NIG parameters use (mu, alpha, beta, delta), with SciPy a=alpha*delta and b=beta*delta.", "",
          "## Methods", "",
          "1. Missing-data covariance/correlation: listwise or pairwise deletion, sample covariance divisor n-1.",
          "2. EW estimators: normalized exponential weights and weighted mean; variance lambda 0.97 and correlation lambda 0.94 for the mixed estimator.",
          "3. PSD repair: eigenvalue clipping and diagonal normalization, or Higham alternating projections with Dykstra correction.",
          "4. PSD Cholesky: lower triangular factor with explicit handling of zero pivots.",
          "5. Normal and PCA simulation: transform independent standard-normal draws by the factor; retain at least 99% variance for PCA.",
          "6. Distribution fits: normal moment fit; constrained t MLE; t-error regression; three-parameter t AICc; NIG moment inversion and MLE.", "",
          "Deterministic comparisons use rtol=1e-7, atol=1e-8. Numerical fits 7.2, 7.3 and 7.6 use rtol=5e-4, atol=1e-6.", "",
          "Independent checks cover pandas agreement, hand-computed EW weights, translation invariance, repair eigenvalues and diagonals, singular Cholesky reconstruction, new-data and multi-seed simulation, PCA truncation, t parameter recovery, t regression, NIG moments/likelihood, invalid inputs, and a complete calculation chain.", "",
          "## Reproduce", "", "```bash", "python -m pip install -r requirements.txt", "python run_functional_tests.py", "python -m pytest -q --junitxml=output/pytest-results.xml", "python generate_answer.py", "```", "",
          "Environment: " + ", ".join(f"{k} {v}" for k,v in report["versions"].items()) + ".", "",
          "See README.md for function conventions, data provenance, naming corrections, installation, and grading assumptions.", ""]
(root / "ANSWER.md").write_text("\n".join(lines), encoding="utf-8")
print("Generated ANSWER.md from actual reports")
