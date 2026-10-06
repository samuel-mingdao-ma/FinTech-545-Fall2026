# Assignment 2 - Covariance, VaR, and Copulas

This folder is the complete, reproducible submission for FinTech 545 Assignment 2.

## Contents

- `assignment2answer.pdf`: written Predict, Fit, and Reconcile answers.
- `risk545/`: reusable Python library for covariance repair, VaR/ES, copulas,
  regression, and simulation.
- `analysis.py`: runs all five analyses and writes `output/results.json`.
- `build_report.py`: builds the PDF from the calculated JSON and figures.
- `run_assignment.py`: one-command runner for every output and the PDF.
- `tests/`: independent numerical and reproducibility checks.
- `problem1.csv` through `problem5.csv`: instructor-supplied data.

## Reproduce everything

Use Python 3.10 or newer. From this `Assignment2` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_assignment.py --seed 545
python -m pytest -q
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The runner recreates every figure, `output/results.json`, and
`assignment2answer.pdf`. A fixed seed makes all simulations reproducible.

## Install and import the library

```bash
python -m pip install -e .
```

The calculations are functions of arrays or CSV paths and can be imported
without running the report builder.

## Conventions

- Arithmetic return is `price[t] / price[t-1] - 1`.
- Sample variance and standard deviation use `ddof=1` unless they are likelihood
  scale parameters or explicitly weighted population moments.
- Skewness is bias-corrected; kurtosis is bias-corrected excess kurtosis.
- VaR and ES are positive dollar losses. Historical VaR follows the course
  `RiskStats.jl` convention: average the sorted observations at floor(n*alpha)
  and ceil(n*alpha), using one-based ranks, then negate. Historical ES is the
  negative mean of P&L at or below that threshold, including ties.
- Exponentially weighted variances use normalized finite-sample weights, newest
  observation receiving the largest weight.
- AICc counts every estimated distribution or copula parameter. BIC is
  `-2*logLik + k*log(n)`.
- Copula pseudo-observations use `rank/(n+1)`. Elliptical copula correlations
  use `rho = sin(pi*tau/2)` and are repaired if numerical noise prevents positive
  definiteness.
- Monte Carlo calculations use 100,000 draws and seed 545 (with stable offsets
  by problem).
- Problem 5 follows the instruction to assume zero expected returns for risk;
  regression intercepts are still estimated and reported as diagnostics.
