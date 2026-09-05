# Assignment 1 - Univariate and Multivariate Statistics

This folder contains the complete submission for FinTech 545 Assignment 1.

## Contents

- `assignment1answer.pdf` - written answers, tables, and figures
- `analysis.py` - reproduces every numerical result and figure
- `build_report.py` - builds the submitted PDF from the computed results
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
python3 analysis.py
```

`analysis.py` prints all results and recreates `output/results.json` plus every
figure. The optimizer is deterministic and does not use random simulation.

## Rebuild the PDF

After running the analysis:

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
