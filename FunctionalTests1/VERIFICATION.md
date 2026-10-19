# Verification record

The following checks were actually executed in an isolated Python 3.12.3 environment:

| Check | Observed result |
|---|---|
| Instructor cases, seed 1234 | 25/25 passed |
| Pytest: instructor cases + independent checks | 59/59 passed = 25 + 34 |
| Instructor cases, seed 2026 | 25/25 passed |
| Build and install with `python -m pip install . --no-deps` after installing dependencies | Wheel built and package installed successfully |
| Import installed package using `python -I` from `/tmp`, outside the submission folder | Imported from the isolated environment's site-packages |
| Installed-package numerical smoke test | Cholesky reconstruction and normal sample fit passed |

See `output/test_report.json`, `output/seed2026/test_report.json` and
`output/pytest-results.xml` for machine-readable numerical test results.
`requirements-tested.txt` records the direct dependency versions.

The 59 pytest checks are not full-semester instructor cases. Only the 25
explicitly listed cases are implemented. No Canvas access or official grader run
was performed; this validates the documented first-checkpoint scope and tolerances.
