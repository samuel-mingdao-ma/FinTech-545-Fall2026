# Verification record

The following checks were actually executed in an isolated Python 3.12.3 environment:

| Check | Observed result |
|---|---|
| Instructor cases, seed 1234 | 23/23 passed |
| Pytest: instructor cases + independent checks | 50/50 passed = 23 + 27 |
| Instructor cases, seed 2026 | 23/23 passed |
| Build and install with `python -m pip install . --no-deps` after installing dependencies | Wheel built and package installed successfully |
| Import installed package using `python -I` from `/tmp`, outside the submission folder | Imported from the isolated environment's site-packages |
| Installed-package numerical smoke test | Cholesky reconstruction and normal sample fit passed |

See `output/test_report.json`, `output/test_report_seed2026.json` and
`output/pytest-results.xml` for machine-readable numerical test results.
`requirements-tested.txt` records the direct dependency versions.

The 50 pytest checks are not the 50 full-semester instructor cases. Only the 23
explicitly listed cases are implemented. No Canvas access or official grader run
was performed; this validates the documented first-checkpoint scope and tolerances.
