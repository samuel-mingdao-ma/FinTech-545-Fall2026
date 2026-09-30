# Assignment 1 - Univariate and Multivariate Statistics

This folder contains the complete submission for FinTech 545 Assignment 1.

## Contents

- `assignment1answer.pdf` - written answers, tables, and figures
- `fintech545/` - reusable, importable Python library containing the statistical methods
- `analysis.py` - uses the library to reproduce every numerical result and figure
- `build_report.py` - builds the submitted PDF from the computed results
- `run_assignment.py` - one command that runs the analysis and rebuilds the PDF
- `pyproject.toml` - package metadata so the library can be installed with `pip`
- `tests/test_library.py` - independent checks of the public library functions
- `problem1.csv` through `problem5.csv` - supplied data
- `Assignment 1.pdf` - original assignment instructions
- `requirements.txt` - Python dependencies
- `output/results.json` - full-precision calculated results
- `output/*.png` - figures used in the report

## Reproduce the analysis

The code requires Python 3.10 or newer. From this `Assignment1` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 run_assignment.py
```

`run_assignment.py` prints all results, recreates `output/results.json` and every
figure, and rebuilds `assignment1answer.pdf`. The optimizer is deterministic and
does not use random simulation.

## Use the Python library

The analysis methods are not tied to the submitted CSV files. They can be
imported from the `fintech545` package, for example:

```python
from fintech545 import sample_moments, fit_regression_errors

summary = sample_moments(values)
models = fit_regression_errors(x, y)
```

The package can also be installed in editable mode and tested:

```bash
python3 -m pip install -e .
python3 -m pytest -q
```

## Rebuild the PDF

To rebuild only the PDF after running the analysis:

```bash
python3 build_report.py
```

This writes `assignment1answer.pdf` in the current directory. Run the analysis first
because the report reads `output/results.json` and the generated figures.

## Conventions

- Sample variance uses `ddof=1` unless a likelihood scale requires the MLE
  convention.
- Skewness and kurtosis use bias-corrected sample estimators.
- Kurtosis is reported as excess kurtosis, so a Normal has value zero.
- The Student-t regression uses a location-scale t distribution and constrains
  degrees of freedom above two so the fitted error variance exists.
- PACF uses the Yule-Walker `ywm` convention.
- AICc counts all fitted coefficients and distribution parameters.
