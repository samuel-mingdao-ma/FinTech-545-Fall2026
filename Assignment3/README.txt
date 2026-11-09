Assignment 3 - Options, Option Risk, and Portfolio Construction
Samuel Mingdao Ma

STATUS
The calculations, charts and English answer draft are included.
Predict passages are explicitly AI-assisted analytical drafts for review,
not reconstructed or authenticated original student forecasts.
The student_predictions.json fields are intentionally null. If genuine
student notes exist, they can be supplied there as review comments; the
report does not assume an earlier chronology merely because text is entered.
Review the Predict passages, model judgments and the marked fee opinion
before submission. No GitHub push or Canvas submission is performed by scripts.
The original prediction-draft records are preserved. Q4's quantitative
50%-90% illustrative weight range was added during review after fitted
results were available; results/q4_review_draft.json records this revision.
It does not establish an original student prediction before optimization.

SETUP (Python 3.12)
Run these commands from the Assignment3 folder:
  python3.12 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
  python -m pip install --no-deps -e .

REPRODUCE EVERY NUMERICAL TABLE AND FIGURE
  python run_assignment.py --simulations 1000000 --tree-n 500 --bootstrap 300 --seed 545

INDEPENDENT VERIFICATION
  python verify_results.py
  python audit_q2.py
  python audit_q3.py --simulations 1000000 --seed 1545
  python -m pytest -q

BUILD THE ANSWER PDF
  python build_report.py --output Assignment3_Answers.pdf

Run a single problem, for example:
  python run_assignment.py --problem 2 --tree-n 500
Individual runs update their own results; the full command above should be
used before rebuilding a complete report. No internet download is needed.
The written narrative is tied to the supplied data and documented baseline
(N500, 1,000,000 paths, seed545, bootstrap300). The report builder rejects
other numerical settings rather than attach baseline prose to custom tables.
If changing the data or interpretation, review and update build_report.py.

FILES
  Assignment3_Answers.pdf   Complete numerical/written answer draft
  assignment3/             Installable reusable Python numerical library
    pricing.py             Own GBSM, analytical Greeks and IV inversion
    american.py            Own dividend-event, non-recombining CRR tree
    common.py              Moments, MLE, OLS, VaR/ES and statistical conventions
    q1.py ... q5.py         Clearly labeled question/subpart workflows
    reporting.py           PDF layout helpers
  data/                    Seven original instructor CSV files, unchanged
  results/                 CSV tables, JSON summaries and audit evidence
  figures/                 Nine report figures
  tests/                   Independent numerical and integration tests
  run_assignment.py        Main computational entry point
  verify_results.py        Independent reconstruction of results
  audit_q2.py              Stable divided-difference check of small-bump gamma
  audit_q3.py              Fresh-seed MC and finite-difference Greek audit
  build_report.py          Rebuilds the answer draft from saved results
  student_predictions.json Optional student-supplied review comments

NUMERICAL CONVENTIONS
1. Arithmetic returns. Loss VaR/ES are positive numbers. Supplied return
   files are not differenced again; only the ABC price history is converted.
2. Moment tables use sample variance/SD (n-1) and population standardized
   skewness/excess kurtosis. Normal fitting uses the MLE divisor n.
3. Historical VaR/ES use the instructor's Week05/RiskStats.jl convention:
   average the 1-based floor(n*alpha) and ceil(n*alpha) order statistics,
   then average all P&Ls at or below that threshold. The Q4 fractional
   empirical tail-mass alternative is separately labeled, not mixed in.
4. Q1: specified valuation date 2026-11-13, expiry 2026-12-18, calendar/365.
   Vega per absolute sigma and per one vol point differ by100. Missing IV
   means the individual quote is outside admissible model price bounds.
5. Q2: CRR steps coincide exactly with both dividends (N multiple of5).
   Distinct post-cash-dividend origins are preserved; only dividend-free
   subtrees recombine. American continuation/exercise are compared at all
   nodes, including dividend boundaries. Extreme ex-div equity is floored
   at zero; supplied inputs have negligible pricing effect from this floor.
   Matched-lattice European values are used for exercise premiums, avoiding
   confusing lattice-versus-GBSM numerical error with exercise value.
6. Q3: 255 trading days/year, ten-day horizon, fixed stated IVs, zero forecast
   means. Holdings are options on ONE share each, with shorts kept signed.
   Normal and t are both simulated even when AICc favors t. Ten daily
   arithmetic returns are compounded; remaining option life is reduced by
   ten days. Daily returns <=-100% raise, never silently clip/resample.
   Delta-normal uses sample daily SD and a sqrt(10) Normal shock. Simulated
   delta-only approximations use the exact compounded terminal price, so
   these are distinct calculations. Theta is dollars/model-year.
7. Q4: monthly rf=(1.03)^(1/12)-1; annual Sharpe=sqrt(12)*monthly Sharpe.
   VaR/ES retain monthly horizon. All240 supplied months are retained.
   Bootstrap is IID-month resampling, not a guarantee of future regime
   stability. Three unrestricted resamples have no finite positively
   oriented normalized tangency; failures are recorded rather than imputed.
8. Q5: factors and RF are divided by100 once; fund returns already decimals.
   Date alignment is explicit. The fund sample is1996-10-01..2001-09-28.
   The factor files end2024-09-30; the recent-ten-year premium window is
   2014-10-01..2024-09-30. The momentum stress searches the ENTIRE file.
   Expected annual returns use255*daily arithmetic expectation; geometric
   sample annual return is prod(1+daily_return)^(255/n)-1 and is separately
   labeled. Stress is incremental factor shock, excluding alpha/RF drift.
9. Regression errors use classical homoskedastic OLS SEs. Gaussian AICc
   counts all intercepts/slopes plus residual scale. This gives Q5 k=3/5/6.
10. Monte Carlo SEs use20 batches and condition on fitted parameters, fixed
    IVs and static holdings. Tree discretization, parameter uncertainty,
    model selection, liquidity and future regime changes are not included.

SOURCE
Original instructor files:
  https://github.com/dompazz/FinTech-545-Fall2026/tree/5958d11b51960c18be98f6df2b97c4e80670fb2d/Assignments/Assignment3
Input SHA256 and versions: results/run_metadata.json.
No external or previous-student solution files are needed.

SUBMISSION REQUIREMENTS FROM THE ASSIGNMENT
Put the answer PDF, all code and this README in Assignment3 in your own
repository. After completing the required review, respond "done" in Canvas.
The source PDF does not specify a due date; check Canvas for that date.
