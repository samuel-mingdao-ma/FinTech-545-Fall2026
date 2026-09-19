# FINTECH 545 — Functional Test Answers

Samuel Ma

## Scope and verification

Implemented: 1.1–1.4, 2.1–2.3, 3.1–3.4, 4.1, 5.1–5.5 and 7.1–7.6.

This first-checkpoint scope is inferred from the September 14 lecture; the final Canvas list has not been verified.

Instructor reference comparisons: **23/23 passed**.
Pytest suite: **50/50 passed** (23 instructor cases plus 27 independent checks).
This is not a claim that all 50 cases in the full-semester Tests.xlsx are implemented.

All answers below were calculated by risk545. Full-precision matrices and parameters are in output/*.csv.

## Results by test

| Test | Result | Maximum absolute difference vs reference |
|---|---|---:|
| 1.1 | PASS | 2.22045e-16 |
| 1.2 | PASS | 2.22045e-16 |
| 1.3 | PASS | 4.44089e-16 |
| 1.4 | PASS | 2.22045e-16 |
| 2.1 | PASS | 6.66134e-16 |
| 2.2 | PASS | 3.33067e-16 |
| 2.3 | PASS | 8.88178e-16 |
| 3.1 | PASS | 1.27676e-15 |
| 3.2 | PASS | 1.4988e-15 |
| 3.3 | PASS | 1.09635e-15 |
| 3.4 | PASS | 1.60982e-15 |
| 4.1 | PASS | 1.66533e-16 |
| 5.1 | PASS | 0.000878503 |
| 5.2 | PASS | 0.00165907 |
| 5.3 | PASS | 0.000878503 |
| 5.4 | PASS | 0.000878503 |
| 5.5 | PASS | 0.000939396 |
| 7.1 | PASS | 2.77556e-17 |
| 7.2 | PASS | 3.9684e-07 |
| 7.3 | PASS | 3.35438e-08 |
| 7.4 | PASS | 5.68434e-14 |
| 7.5 | PASS | 7.81597e-14 |
| 7.6 | PASS | 1.46372e-12 |

## Monte Carlo covariance comparisons

100,000 observations per case; seed 1234. Python and Julia simulations have different random streams.
The Frobenius differences below are divided by the norm of the analytic target covariance.

| Test | Relative Frobenius difference | Largest error / MC standard error vs target |
|---|---:|---:|
| 5.1 | 0.6222% | 1.6923 |
| 5.2 | 1.0373% | 2.0477 |
| 5.3 | 0.5231% | 1.5257 |
| 5.4 | 0.5088% | 1.5257 |
| 5.5 | 0.7340% | 3.4842 |

Cases 5.3 and 5.4 use the repaired covariance; case 5.5 uses the covariance implied by the retained 99% PCA factors.
Criteria: relative Frobenius error <= 3%; maximum target and independent-reference standardized error <= 7.
These are documented implementation tolerances, not confirmed instructor grading thresholds.

## Distribution and regression answers

### Test 7.1

| Parameter | Calculated value |
|---|---:|
| mu | 0.0467719575164 |
| sigma | 0.0517897257241 |

### Test 7.2

| Parameter | Calculated value |
|---|---:|
| mu | 0.0501003370845 |
| sigma | 0.044955077738 |
| nu | 4.09292512785 |

### Test 7.3

| Parameter | Calculated value |
|---|---:|
| mu | 0 |
| sigma | 0.0506263037387 |
| nu | 4.66190876763 |
| Alpha | 0.0495878809086 |
| B1 | 1.17776320056 |
| B2 | 1.82695526253 |
| B3 | 3.05641231691 |

### Test 7.4

| Parameter | Calculated value |
|---|---:|
| AICC | -279.054620985 |

### Test 7.5

| Parameter | Calculated value |
|---|---:|
| mu | 0.0239433306427 |
| alpha | 52.9470625312 |
| beta | -12.245376486 |
| delta | 0.0574589011407 |

### Test 7.6

| Parameter | Calculated value |
|---|---:|
| mu | 0.0196595208849 |
| alpha | 46.9235803349 |
| beta | -8.23265464944 |
| delta | 0.0526078621407 |

The t scale is not its standard deviation. Error location in the regression is fixed to zero, while Alpha is estimated.
NIG parameters use (mu, alpha, beta, delta), with SciPy a=alpha*delta and b=beta*delta.

## Methods

1. Missing-data covariance/correlation: listwise or pairwise deletion, sample covariance divisor n-1.
2. EW estimators: normalized exponential weights and weighted mean; variance lambda 0.97 and correlation lambda 0.94 for the mixed estimator.
3. PSD repair: eigenvalue clipping and diagonal normalization, or Higham alternating projections with Dykstra correction.
4. PSD Cholesky: lower triangular factor with explicit handling of zero pivots.
5. Normal and PCA simulation: transform independent standard-normal draws by the factor; retain at least 99% variance for PCA.
6. Distribution fits: normal moment fit; constrained t MLE; t-error regression; three-parameter t AICc; NIG moment inversion and MLE.

Deterministic comparisons use rtol=1e-7, atol=1e-8. Numerical fits 7.2, 7.3 and 7.6 use rtol=5e-4, atol=1e-6.

Independent checks cover pandas agreement, hand-computed EW weights, translation invariance, repair eigenvalues and diagonals, singular Cholesky reconstruction, new-data and multi-seed simulation, PCA truncation, t parameter recovery, t regression, NIG moments/likelihood, invalid inputs, and a complete calculation chain.

## Reproduce

```bash
python -m pip install -r requirements.txt
python run_functional_tests.py
python -m pytest -q --junitxml=output/pytest-results.xml
python generate_answer.py
```

Environment: python 3.12.3, numpy 2.5.3, scipy 1.18.1, pandas 3.0.6.

See README.md for function conventions, data provenance, naming corrections, installation, and grading assumptions.
