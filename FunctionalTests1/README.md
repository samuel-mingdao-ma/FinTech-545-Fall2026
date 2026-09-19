# FINTECH 545 — Functional Tests, First Checkpoint

Author: Samuel Ma

This submission implements a reusable Python risk library and independently executable
functional tests against the instructor's supplied input and expected-output CSV files.

## Scope

Implemented instructor cases (23 total): **1.1–1.4, 2.1–2.3, 3.1–3.4, 4.1,
5.1–5.5, and 7.1–7.6**. This is the first-checkpoint scope inferred from the
September 14 lecture. The final Canvas list has not been supplied or verified.
Cases 6 and 8–13 are not implemented in this submission.

## Run

From this folder, using Python 3.10 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_functional_tests.py
python -m pytest -q
```

On Windows, activate with `.venv\Scripts\activate` instead.
To install the reusable library as a package:

```bash
python -m pip install .
```

The grader does not need Julia, Excel, the instructor's library, a network connection
after installing dependencies, or any files outside this submission. The runner returns
exit code 1 if any case fails and 0 if all pass. To use an alternative input/reference
directory or a different simulation seed:

```bash
python run_functional_tests.py --data-dir /path/to/data --output-dir /path/to/results --seed 2026
```

## Files

- `risk545/`: reusable functions, with docstrings and input validation. These functions
  never load or return values from expected-output files.
- `data/`: unmodified instructor input and reference CSVs for the 23 cases.
- `functional_cases.py`: maps each numbered case to inputs, function calls and comparison rules.
- `run_functional_tests.py`: exports calculated CSV answers and a per-case pass/fail report.
- `tests/test_functional.py`: 23 separately named instructor tests.
- `tests/test_independent.py`: 27 additional checks on independent data, mathematical
  properties, repeatability and invalid inputs. These are our own checks, not hidden instructor tests.
- `output/`: calculated answer CSVs, summary CSV, JSON report and pytest JUnit results.
- `ANSWER.md`: methods, generated numerical results, tolerances and verification summary.
- `generate_answer.py`: regenerate `ANSWER.md` from the actual test reports.
- `requirements-tested.txt`: exact direct dependency versions used for verification.
- `说明.md`: Chinese guide to using and submitting this folder.

## Public library example

```python
import numpy as np
from risk545 import missing_covariance, near_psd, simulate_normal, fit_t

x = np.array([[1., 2.], [2., np.nan], [4., 5.], [7., 3.]])
cov = missing_covariance(x, pairwise=True)
repaired = near_psd(cov)
draws = simulate_normal(repaired, n=100000, seed=1234)
parameters = fit_t(draws[:, 0])
```

## Conventions and tolerances

Observations are rows and variables are columns. CSVs retain the instructor's column
order and have no added row index. Matrix output row order matches variable column order.

- Missing-data covariance: sample covariance, divisor `n-1`. Pairwise correlations use
  pair-specific overlapping rows and pair-specific means and variances.
- EW covariance: last row is newest, normalized exponential weights, weighted mean,
  and no Bessel correction. Case 2.3 uses variance lambda 0.97 and correlation lambda 0.94.
  The reference generator's comment reverses these, but its executable code agrees with the workbook.
- Matrix repair: operate in correlation space and restore original variances. Higham uses
  alternating projections with Dykstra correction and identity weights.
- Cholesky: return the lower triangular factor, including singular PSD inputs.
- Simulations: 100,000 observations, zero mean, default seed 1234. Independent Julia/Python
  random-number generators do not give identical samples with the same seed.
- Case 5.4: the workbook says PSD but uses `test5_3.csv`; the generator explicitly repairs
  that non-PSD matrix using Higham. We follow the actual input and generator.
- PCA: retain enough descending eigenvalues for 99% explained variance; numerical positive
  eigenvalues use the course cutoff `1e-8`. Compare sampling error to the truncated covariance,
  not to the original full-rank covariance.
- Normal fit: sample standard deviation (`ddof=1`), not the normal MLE divisor `n`.
- Student t: estimate location, scale and nu; constrain `nu >= 2.0001`, as the reference does.
  Scale is not standard deviation. For nu > 2, variance is `scale**2 * nu/(nu-2)`.
- t regression: estimate intercept, slopes, scale and nu; error location is fixed at zero
  to avoid confounding it with the regression intercept.
- AICc: `-2*log_likelihood + 2*k + 2*k*(k+1)/(n-k-1)`; the univariate t fit has k=3.
- NIG moments: sample variance and uncorrected skewness/excess kurtosis match Julia conventions.
- NIG MLE: convert SciPy `(a,b,loc,scale)` to `mu=loc`, `delta=scale`,
  `alpha=a/scale`, `beta=b/scale`.

Deterministic cases use `rtol=1e-7, atol=1e-8`. Numerical fitting cases 7.2, 7.3 and
7.6 use `rtol=5e-4, atol=1e-6` to allow optimizer/version variation. These tolerances
were set before the first run, not increased to make a failing result pass.

Monte Carlo comparisons use the Gaussian sample-covariance standard error
`sqrt((Sigma_ii*Sigma_jj + Sigma_ij**2)/(N-1))`. Each entry must be within 7 standard
errors of the analytic covariance. Comparison to the independent reference simulation
uses sqrt(2) times that standard error. Relative Frobenius error versus the reference
must also be <= 3%. These are our documented verification tolerances; the instructor
asked for closeness and a Frobenius comparison but did not provide a numerical cutoff
in the supplied transcript.

## Source and naming notes

The methods and CSV contracts follow the instructor's `testfiles/test_setup.jl`,
`library/missing_cov.jl`, `library/ewCov.jl`, `library/simulate.jl`,
`library/fitted_model.jl` and `testfiles/new_test_rows.md` in the course materials.
Python implementation and test harness were prepared with AI assistance and verified
against the provided references. The lecture explicitly permits consulting the instructor's library.

The downloaded `Tests.xlsx` is the updated 50-case catalog; the local course copy has
39 cases. Neither is itself an executable test suite. The actual group 7 CSVs are named
`testout7_1.csv`, etc., although some workbook rows say `testout_7.1.csv`.
The adapters follow actual supplied filenames. Reference intermediates from cases 1 and
3 are used as inputs where the workbook explicitly specifies them; the independent suite
also validates a fully calculated covariance-to-repair-to-factorization-to-simulation chain.

Do not run the instructor's data generator over `data/`: the assignment consumes the
provided inputs. Calculated outputs are stored separately in `output/`.
