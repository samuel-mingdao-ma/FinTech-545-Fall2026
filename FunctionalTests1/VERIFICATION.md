# Verification record

The following checks were executed on 18 November 2026. The primary environment
used Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1 and pandas 3.0.6.

| Check | Observed result |
|---|---|
| Instructor cases, seed 1234 | 25/25 passed |
| Pytest: instructor cases + independent checks | 85/85 passed = 25 + 60 |
| Instructor cases, seed 2026 | 25/25 passed |
| Build and install both libraries with `pip install --no-build-isolation --no-deps --target ...` | Both wheels built and installed successfully |
| Import installed packages from `/private/tmp`, outside the repository | Both imports resolved to the installation target |
| Installed-package numerical smoke test | Tiny-covariance Cholesky reconstruction, large-unit Normal fit, Normal-limit t fits and portfolio weights passed |
| Copy both complete folders outside the repository and run their CLIs | 25/25 first-checkpoint cases and 25/25 second-checkpoint cases passed |
| Run both test folders together, in either collection order | 281/281 passed = 85 + 196 |
| First-checkpoint cases using Python 3.12.2, NumPy 1.26.4, SciPy 1.15.2 and pandas 2.3.3 | 25/25 passed, including both NIG fits |

See `output/test_report.json`, `output/seed2026/test_report.json` and
`output/pytest-results.xml` for machine-readable numerical test results. The
additional SciPy 1.15.2 run is saved in `verification/compatibility_scipy115.json`.
`requirements-tested.txt` records the direct dependency versions.

This folder implements 25 instructor cases from groups 1–7; the separate
FunctionalTests2 folder implements the other 25 cases from groups 8–13. Unit and
regression checks are additional checks, not additional instructor cases.
Monte Carlo results are compared using the documented sampling-error tolerances;
they are not expected to match the instructor's independent random draws exactly.
No official grader was run. These results establish the checked cases,
environments and tolerances, rather than a guarantee about unseen tests.
