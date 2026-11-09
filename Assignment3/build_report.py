"""Build the English answer draft from reproducible numerical result files."""
import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"fintech545-assignment3-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir())/"fintech545-assignment3-cache"))

import numpy as np
import pandas as pd

from assignment3.reporting import Report

ROOT = Path(__file__).resolve().parent
PCT = "{:.3%}"
MONEY = "{:,.2f}"


def read_json(results, name):
    return json.loads((results/name).read_text())


def frame(results, name):
    return pd.read_csv(results/name)


def select(table, **criteria):
    result = table
    for key, value in criteria.items():
        if isinstance(value, float):
            result = result.loc[np.isclose(result[key], value, rtol=0, atol=1e-12)]
        else:
            result = result.loc[result[key] == value]
    return result


def prediction_note(report, problem, student):
    value = student.get(problem)
    if value:
        report.p("Student-supplied review comments (no pre-fit chronology is assumed):", "note")
        report.p(str(value))
    else:
        report.p("Predict passages below are AI-assisted analytical drafts for student review. They are not a claim that the student recorded these predictions before seeing model results.", "note")


def cover(report, results):
    report.p("Assignment 3", "title")
    report.p("Options, Option Risk, and Portfolio Construction", "subtitle")
    report.p("FinTech 545 - Samuel Mingdao Ma", "subtitle")
    report.p("Numerical results and written answer draft", "subtitle")
    report.h("Status and scope", 2)
    report.p("All five problems are calculated from the supplied files. The pricing formula, analytical Greeks and dividend-event binomial tree are implemented in the accompanying Python library. Tables and figures below are generated from the same reproducible run.")
    report.p("The Predict passages are explicitly labeled AI-assisted analytical drafts, not reconstructed student forecasts. Review these passages and the marked opinion before submission. The saved prediction JSON files are not an authenticated record of an unaided student's pre-fit judgments.")
    report.h("Conventions", 2)
    report.p("Returns are arithmetic. Variance and standard deviation in moment summaries use n-1; skewness and excess kurtosis use population standardized moments. Normal and t likelihood fits use their MLE scale parameters. Loss VaR and ES are reported as positive numbers.")
    report.p("Primary empirical VaR/ES follow the instructor's RiskStats.jl: average the floor(nα) and ceil(nα) 1-based P&L order statistics for the threshold, then average P&L at or below it for ES. Problem 4 also reports a fractional empirical tail-mass sensitivity because its 1% tail contains only two or three months.")
    report.p("Problem 1 uses the specified 13 November 2026 valuation date and calendar days/365. Problems 2 and 3 use the years/steps specified in their questions; Problem 3 uses 255 trading days. Problem 4 is monthly with annual Sharpe scaled by √12. Problem 5 uses 255 times a daily arithmetic expected return, stated as an annualization convention.")
    meta = read_json(results, "run_metadata.json")
    report.p(f"Final simulation: {meta['simulations']:,} paths per fitted distribution, seed {meta['seed']}. Main dividend tree: N={meta['tree_n']}. Monthly bootstrap: {meta['bootstrap']} replications.")
    report.h("Reproduction", 2)
    report.p("Run python run_assignment.py, python verify_results.py, python audit_q2.py, python audit_q3.py, python -m pytest -q, and python build_report.py. The README gives setup details and exact commands. No internet data are fetched by these commands.")
    report.p("Source: instructor Assignment3 directory at commit 5958d11b51960c18be98f6df2b97c4e80670fb2d. The original data are preserved under data/; input hashes and package versions are recorded in results/run_metadata.json.", "note")


def question1(report, results, figures, student):
    report.page()
    report.h("1. Implied Volatility from a Dirty Chain")
    summary = read_json(results, "q1_summary.json")
    parameters = summary["parameters"]
    report.p("Data: problem1.csv. S=182.40; r=4.25%; continuous dividend yield q=0.60%. Valuation 13 November 2026; expiry 18 December 2026. T=35/365=0.09589041. Prices below are dollars per share; IVs are annualized.")
    prediction_note(report, "q1", student)
    parity = frame(results, "q1_parity.csv")
    flagged = parity.loc[parity.flag].iloc[0]
    report.part("1(a)", "Before IV inversion, check parity using midpoint prices and the specified spread band.",
                "For each strike I compute gap=Cmid-Pmid-[S exp(-qT)-K exp(-rT)] and band=(Cask-Cbid)+(Pask-Pbid). The flag uses |gap|>band, exactly the question's full-spread rule, not a half-spread band. The following table is a quote diagnostic before inversion.")
    report.table(parity, ["Strike", "gap", "band", "flag"],
                 labels={"Strike":"K", "gap":"Parity gap ($)", "band":"Spread band ($)", "flag":"Flag"},
                 formats={"Strike":"{:.0f}", "gap":"{:.6f}", "band":"{:.4f}"}, widths=[.7,1.5,1.5,.8])
    report.p(f"Only K={flagged.Strike:.0f} is flagged: gap={flagged.gap:.6f}, compared with band={flagged.band:.4f}. Its call and put midpoints are mutually inconsistent with the stated carry relation. Parity alone cannot say which side is wrong and is not a guarantee of an executable arbitrage after all trading frictions.")
    report.part("1(b)", "What does exact parity imply about call and put IV at the same strike?",
                "Under the same S,K,T,r,q, the GBSM call and put prices differ by the parity term independently of volatility. Both prices increase with volatility when vega is positive. If one common volatility prices the call, parity then forces that same volatility to price the put. Thus their unique implied volatilities coincide. Passing a nonzero bid-ask tolerance is weaker than exact midpoint parity and does not require identical computed mid IVs.")
    report.part("1(c)", "Which IV estimates should be least reliable before inversion?",
                "The local inverse sensitivity is dσ/dV=1/vega. Far-out-of-the-money and deep-in-the-money options have little time-value sensitivity, so a cent of price noise can dominate their IV. The far call wing already has identical0.05 bid and 0.10 ask quotes at strikes215,220,225. That price floor is a warning before looking at a volatility curve.")
    report.part("1(d)", "Invert every bid, mid and ask; plot both call and put IV curves.",
                "I solve the self-written GBSM price equation with a bracketed numerical root search. European call bounds are max(0,S exp(-qT)-K exp(-rT))≤C<S exp(-qT); put bounds are max(0,K exp(-rT)-S exp(-qT))≤P<K exp(-rT). A price below its lower bound has no admissible nonnegative IV. The solver leaves that value unavailable rather than inventing a volatility.")
    quotes = frame(results, "q1_quotes_iv.csv")
    for kind in ["Call", "Put"]:
        sub = quotes.loc[quotes.Type == kind].copy()
        report.h(kind+" implied volatilities and inverse sensitivity", 2)
        report.table(sub, ["Strike", "IV_bid", "IV_mid", "IV_ask", "vega_per_vol_point", "cent_dIV_vol_points"],
                     labels={"Strike":"K", "IV_bid":"Bid IV", "IV_mid":"Mid IV", "IV_ask":"Ask IV", "vega_per_vol_point":"Vega $/vol pt", "cent_dIV_vol_points":"Vol pts /$0.01"},
                     formats={"Strike":"{:.0f}","IV_bid":"{:.3%}","IV_mid":"{:.3%}","IV_ask":"{:.3%}","vega_per_vol_point":"{:.5f}","cent_dIV_vol_points":"{:.4f}"},
                     widths=[.6,1,1,1,1.25,1.25])
    report.p("All 36 midpoint IVs and all 36 ask IVs exist. Nine bid IVs do not: calls at 140,145,150,155 and puts at 205,210,215,220,225. Each is below its own no-arbitrage lower bound. The n.a. entries are therefore quote/model feasibility results, not failures to search hard enough. A wide spread can put its bid below a theoretical bound even while its midpoint is feasible.")
    report.image(figures/"q1_smile.png", "Call and put midpoint IVs with bid-ask IV ranges. The spike around175 and the call quote-floor wing are separate issues.", max_height=320)
    report.part("1(e)", "Compute vega and the IV effect of a one-cent price change.",
                "The tables above use vega per one volatility point (σ change0.01). The corresponding linear cent effect, in vol points, is 100×0.01/vega_per_unit_sigma. Mid IVs are used to evaluate vega. For example the 215,220,225 calls move about 0.4429,0.4823,0.5195 vol points for one cent. The implementation also re-inverts Mid+0.01 as a nonlinear check; that is retained in q1_quotes_iv.csv.")
    report.image(figures/"q1_vega_sensitivity.png", "Low vega produces large inverse sensitivity. Dollar vega per absolute σ=1 differs by 100 from dollar vega per vol point.", max_height=280)
    translation = frame(results, "q1_parity_vol_translation.csv").loc[lambda d:d.flag].iloc[0]
    report.part("1(f)", "Does the parity anomaly appear in IV, and does vega explain its size?",
                f"Yes. At K=175 the call-minus-put IV gap is {translation.actual_gap_vol_points:.6f} vol points. Dividing the ${translation.parity_gap:.6f} parity residual by the vega at the average IV ({translation.reference_vega:.6f} per absolute σ) gives {translation.estimated_gap_vol_points:.6f} vol points. The small difference is the curvature of price in volatility; a constant-vega translation is a first-order approximation. The abnormal point is not evidence of a normal smooth smile feature.")
    wing = pd.DataFrame(summary["call_wing"])
    report.part("1(g)", "Interpret the call wing above 210 and explain why parity did not flag it.",
                "The 215,220,225 call midpoints are all 0.075 although their strikes keep rising. Inverting this quote floor forces IV upward: the mid IVs are 23.8225%,26.6888%,29.4473%. Their bid-ask IV intervals are wide and their cent sensitivities are large. I would treat this as a quotation-resolution/liquidity artifact, not infer stronger demand for remote upside protection from that curve alone. Parity only tests the call-put price relation against a spread band. Here the small call quote-floor errors fit inside the combined band, especially with the corresponding deep-in-the-money put spread. A quote can pass that consistency test and still be a poor IV measurement.")
    report.table(wing, ["Strike", "Bid", "Ask", "IV_bid", "IV_mid", "IV_ask"],
                 labels={"Strike":"K","IV_bid":"Bid IV","IV_mid":"Mid IV","IV_ask":"Ask IV"},
                 formats={"Strike":"{:.0f}","Bid":"{:.2f}","Ask":"{:.2f}","IV_bid":"{:.3%}","IV_mid":"{:.3%}","IV_ask":"{:.3%}"})
    clean = frame(results, "q1_clean_otm_smile.csv")
    report.part("1(h)", "Describe the smile after setting aside the two quote problems.",
                f"Using the OTM side, excluding the 175 inconsistent pair and the call quote-floor wing, IV falls from {clean.iloc[0].IV_mid:.3%} at K={clean.iloc[0].Strike:.0f} to {clean.iloc[-1].IV_mid:.3%} at K={clean.iloc[-1].Strike:.0f}. This is a pronounced downside skew or smirk, not a symmetric U. Low-strike puts carry a higher IV than near-ATM and moderate upside calls, consistent with relatively expensive demand for downside protection. That is a statement about relative option prices, not a proof of physical crash probabilities.")


def question2(report, results, figures, student):
    report.h("2. American Options and Discrete Dividends")
    summary = read_json(results, "q2_summary.json")
    n = summary["tree_n"]
    report.p(f"S=48.50, K=50, T=0.5 years, r=4.5%, σ=30%. Cash dividends are 0.25 at 0.10 years and 1.50 at 0.40 years. Primary prices use N={n}; every tested grid has N divisible by 5, placing both dividend events exactly on nodes.")
    prediction_note(report, "q2", student)
    report.p("The saved Q2 notes explicitly disclose that implementation benchmarks preceded their persistence. They must not be presented as a time-stamped student-before-fit record.", "note")
    conditions = frame(results, "q2_dividend_conditions.csv")
    report.part("2(a)", "Apply the dividend early-exercise condition to both payments.",
                "I compare Dn with K[1-exp(-r(T-tn))]. The first hurdle is 0.891948, above 0.25; the second is 0.224495, below 1.50. Thus the first dividend is too small for this call-exercise mechanism, whereas exercise just before the second dividend can be optimal for sufficiently high stock prices. The inequality does not say every stock-price state should be exercised.")
    report.table(conditions, labels={"time":"Dividend time", "dividend":"D ($)", "strike_interest":"Interest hurdle ($)", "condition_holds":"D > hurdle"})
    report.part("2(b)", "Predict the four dividend price changes and the early-exercise premiums.",
                "Analytical draft: cash paid out reduces the future ex-dividend share price. I expect both calls to be worth less and both puts more than in the no-dividend case. The large second payment makes the call's exercise privilege more useful, so I expect its exercise premium to rise. Falling stock value near a dividend makes waiting relatively attractive to the put holder, so I expect the put's exercise premium to shrink. These are predictions to check against matched European/American valuations, not substitutes for those calculations.")
    report.part("2(c)", "Price all four options, with and without dividends, and compare the PV-dividend GBSM shortcut.",
                "Without dividends the European reference is own-code GBSM and American values use CRR backward induction. With cash dividends, each distinct pre-dividend node starts a continuation subtree at max(S-D,0). Subtrees recombine only within a dividend-free interval; post-dividend origins are not merged. American nodes compare exercise with continuation, including just before and after payments. All four cash-dividend values therefore use the non-recombining event-tree model, not a shifted-spot shortcut.")
    prices = frame(results, "q2_prices.csv")
    report.table(prices, ["option", "no_dividend_value", "with_dividend_value", "pv_dividend_gbsm"],
                 labels={"option":"Option", "no_dividend_value":"No dividend ($)", "with_dividend_value":"Cash-dividend tree ($)", "pv_dividend_gbsm":"PV-dividend GBSM ($)"},
                 formats={"no_dividend_value":"{:.6f}","with_dividend_value":"{:.6f}","pv_dividend_gbsm":"{:.6f}"}, widths=[1.5,1,1.2,1.2])
    report.p(f"The dividend PV is {summary['dividends_present_value']:.6f}. The shortcut replaces S by S-PV(D) in GBSM; it understates each European cash-dividend tree value by about 0.103986 at N={n}. A fixed cash subtraction changes the path distribution differently from a constant proportional dividend yield. The stock is floored at zero only at extreme low ex-dividend nodes; the European cash-dividend parity residual is {summary['european_parity_error']:.3g}, negligible at these inputs.")
    report.part("2(d)", "Show convergence and include neighboring feasible step counts.",
                "The table includes adjacent feasible grids such as100/105 and 500/505. Values oscillate as strikes and exercise boundaries move relative to the lattice; increasing N does not produce monotone prices. Reported precision is computational precision, not an assertion that the fifth decimal is grid-independent.")
    convergence = frame(results, "q2_convergence.csv")
    report.table(convergence, ["n", "american_call_no_dividend", "american_call_with_dividend", "american_put_no_dividend", "american_put_with_dividend"],
                 labels={"n":"N", "american_call_no_dividend":"Call no div", "american_call_with_dividend":"Call cash div", "american_put_no_dividend":"Put no div", "american_put_with_dividend":"Put cash div"},
                 formats={c:"{:.6f}" for c in convergence.columns if c != "n"}, widths=[.5,1,1,1,1])
    report.image(figures/"q2_convergence.png", "Both neighboring-grid oscillation and the broad convergence envelope are retained.", max_height=280)
    report.part("2(e)", "Compute American delta and gamma using all four specified central bumps.",
                "For each bump h I use [V(S+h)-V(S-h)]/(2h) for delta and [V(S+h)-2V(S)+V(S-h)]/h² for gamma, holding the entire N-step model fixed. The required results are below; smaller diagnostic bumps are retained in the CSV for the stability discussion.")
    fd = frame(results, "q2_finite_difference_greeks.csv")
    report.table(fd.loc[fd.required_bump], ["option", "bump", "delta", "gamma"],
                 labels={"option":"American option", "bump":"h ($)"}, formats={"bump":"{:.3f}","delta":"{:.8f}","gamma":"{:.8f}"}, widths=[1.8,.6,1,1])
    report.part("2(f)", "Compute every option's sensitivity to each dividend amount.",
                "I bump one cash amount at a time, leaving its date and the other amount unchanged. The main central bump is 0.001 dollars; a0.01 bump is also exported and gives closely matching first derivatives. Units are dollars of option value per additional dollar of cash dividend.")
    sensitivities = frame(results, "q2_dividend_sensitivities.csv")
    report.table(sensitivities.loc[np.isclose(sensitivities.bump, .001)], ["option", "dividend_number", "dividend_time", "dV_dDividend"],
                 labels={"option":"Option", "dividend_number":"Payment", "dividend_time":"Time (years)", "dV_dDividend":"dV/dD"},
                 formats={"dV_dDividend":"{:.8f}"}, widths=[1.6,.7,.9,1])
    report.part("2(g)", "Report exercise premiums and the second-dividend call boundary.",
                "An exercise premium is American minus European under the same cash-flow model and same lattice. I use matched-tree European references here so the no-dividend call's small CRR-versus-GBSM discretization error is not mislabeled an exercise premium. The predictions are borne out: all calls fall and all puts rise with dividends; the call premium rises while the put premium falls.")
    premiums = frame(results, "q2_exercise_premiums.csv")
    report.table(premiums, ["kind", "dividends", "early_exercise_premium"],
                 labels={"kind":"Option", "dividends":"Dividend case", "early_exercise_premium":"Exercise premium ($)"},
                 formats={"early_exercise_premium":"{:.8f}"}, widths=[.7,1.5,1.2])
    boundary = summary["second_dividend_exercise_boundary"]
    reference = summary["second_dividend_continuous_boundary_reference"]
    report.p(f"Immediately before the second payment, the tree boundary is S≈{boundary:.6f}. Exercise is preferred above it. The continuation value at S-1.50 is compared with S-50, using the remaining 0.10-year grid. As an independent check, the no-further-dividend GBSM continuation root is {reference:.6f}; its small difference from the lattice boundary reflects tree discretization.")
    report.image(figures/"q2_exercise_boundary.png", "Pre-dividend exercise payoff and continuation value around the call exercise threshold.", max_height=240)
    report.part("2(h)", "Which finite-difference Greek is unreliable as the bump shrinks, and why?",
                "Gamma is the problem. A fixed binomial tree's value is piecewise linear in starting spot, with kinks from payoffs and exercise decisions. Within one segment the second difference tends to zero; across a kink it depends sharply on which states the bump crosses. At h=1 the call gamma is 0.0408754, but at 0.001 it is numerically zero even though the continuous-model option has curvature. Put gamma also changes materially: at h=10^-5 it is about 0.06656. This jump is predominantly a fixed-lattice kink effect, rather than floating-point cancellation; a stable divided-difference recursion retains essentially the same value. Subtracting nearly equal prices can add numerical noise at such small bumps, but does not explain the main jump here. Delta is a first derivative and is much more stable (about 0.505189 for the call and -0.548763 for the put). Taking h to zero before refining N does not recover a reliable continuous-model gamma.")
    report.image(figures/"q2_greeks_bump.png", "Required central bumps plus smaller diagnostics show why a smaller bump is not automatically more accurate.", max_height=260)
    report.part("2(i)", "Why do the two cash payments move the American call differently per dollar?",
                "The American call sensitivities are approximately -0.485060 for the first payment and -0.105495 for the second. Before the first payment the dividend is below the interest hurdle, so the holder generally remains exposed to that ex-dividend stock reduction. Before the second, sufficiently high stock states exercise. In those exercised states the extra dividend is not a reduction in a continuing option claim, which attenuates the option's sensitivity. The European call has no such escape: its derivatives,-0.441060 and -0.386337, are much closer. The dates, discounting and exercise region all matter; the two nominal cash amounts alone do not determine the per-dollar effect.")


def question3(report, results, figures, student):
    report.h("3. VaR of an Option Book")
    summary = read_json(results, "q3_summary.json")
    parameters = summary["parameters"]
    books = frame(results, "q3_book_greeks.csv")
    full = frame(results, "q3_full_revaluation_risk.csv")
    dn = frame(results, "q3_delta_normal_risk.csv")
    approximations = frame(results, "q3_approximation_risk.csv")
    flat = frame(results, "q3_flat_price_pnl.csv")
    comparison = frame(results, "q3_method_comparison.csv")
    report.p(f"Data:501 ABC prices, giving500 arithmetic daily returns, and four supplied position books. Last price S₀={parameters['spot']:.2f}; r=4.25%, no dividends. A year has 255 trading days, and the horizon is 10 trading days. Holdings count shares or options on one share; negative means short. There is no 100-share multiplier.")
    prediction_note(report, "q3", student)
    report.part("3(a)", "Use each book's Greeks to predict and rank delta-normal risk errors before simulation.",
                "The book Greeks are holding-weighted sums of own-code GBSM position Greeks. Theta is dollar decay per model year, so divide it by 255 for a trading day. The analytical prediction ranks expected absolute dollar error by initial |gamma|: ShortStraddle, HedgedShortCall, CoveredCall, RiskReversal. The first two have very little delta but substantial negative gamma. Short-gamma curvature suggests delta-normal will understate loss; the RiskReversal's small positive initial gamma suggests a conservative linear loss estimate locally, not necessarily after a large move across strikes.")
    report.table(books, ["Portfolio", "delta", "gamma", "theta", "theta_per_trading_day"],
                 labels={"Portfolio":"Book", "delta":"Delta (shares)", "gamma":"Gamma", "theta":"Theta ($/year)", "theta_per_trading_day":"Theta ($/day)"},
                 formats={"delta":"{:.6f}","gamma":"{:.6f}","theta":"{:.3f}","theta_per_trading_day":"{:.6f}"}, widths=[1.65,1,1,1,1])
    report.p("The hedges are approximate rather than exactly zero: ShortStraddle delta is -0.888844 shares and HedgedShortCall delta is -0.444422 shares. That small residual does not explain the large gamma risk.")
    report.part("3(b)", "Predict flat-price P&L and unusually large ES/VaR ratios.",
                "The initial theta signs predict a gain for all four books if ABC is unchanged for ten days. The RiskReversal's positive theta is small, so this prediction is less robust to approximation error than for the other books. I expect ShortStraddle and HedgedShortCall to have particularly high ES/VaR ratios: much of their central P&L is premium decay, while sufficiently large moves create convex losses. The 1.25 Normal benchmark is approximately ES/VaR at the 5% tail for zero-mean Normal P&L; it is not a universal ratio for nonzero means or the 1% tail.")
    report.part("3(c)", "Compute moments; fit Normal and Student t; compare AICc.",
                "Distribution fits are made to the original returns, without removing the sample mean first. The sample is nearly symmetric but has positive excess kurtosis. Normal has two fitted parameters; t has location, scale and degrees of freedom. The t scale is not its standard deviation: SD=scale×sqrt(nu/(nu-2)) when nu>2.")
    report.table(frame(results, "q3_moments.csv"), formats={"mean":"{:.7f}","variance":"{:.9f}","sd":"{:.7f}","skewness":"{:.6f}","excess_kurtosis":"{:.6f}"})
    fits = frame(results, "q3_distribution_fits.csv")
    report.table(fits, ["family", "mu", "sigma", "scale", "nu", "ll", "AICc"],
                 labels={"family":"Family","mu":"Location","sigma":"Normal σ","scale":"t scale","nu":"t nu","ll":"Log likelihood"},
                 formats={"mu":"{:.7g}","sigma":"{:.7f}","scale":"{:.7f}","nu":"{:.5f}","ll":"{:.4f}","AICc":"{:.4f}"}, widths=[.75,1.1,1,1,.8,1.1,1.1])
    normal_fit, t_fit = [select(fits, family=name).iloc[0] for name in ["normal", "t"]]
    report.p(f"The t fit has AICc lower by {normal_fit.AICc-t_fit.AICc:.6f}, so it is preferred among these two candidates. Its nu={t_fit.nu:.6f} gives materially heavier tails than a Normal. This selection does not replace the question's requirement to simulate both families. For forecasting, both fitted locations are reset to zero as instructed; the fitted scale and shape remain unchanged.")
    report.part("3(d)", "Compute10-day delta-normal VaR and ES at 5% and 1%.",
                "I use sd(P&L)=|book delta|×S₀×sample_daily_SD×sqrt(10), with zero expected P&L. For tail probability α, VaR=-sd×Φ⁻¹(α) and ES=sd×φ(Φ⁻¹(α))/α. The sample SD convention here differs slightly from the Normal MLE SD in part(c), and that convention is recorded rather than hidden.")
    report.table(dn, ["Portfolio", "alpha", "VaR", "ES"],
                 labels={"Portfolio":"Book","alpha":"Tail α","VaR":"VaR ($)","ES":"ES ($)"},
                 formats={"alpha":"{:.0%}","VaR":MONEY,"ES":MONEY}, widths=[1.6,.6,1,1])
    report.part("3(e)", "Simulate both return distributions and revalue the entire book.",
                f"For each family I generate {parameters['paths']:,} independent ten-day paths, with zero-mean daily returns, and compound S₁₀=S₀×Π(1+rⱼ). Every stock and option is repriced at S₁₀; option maturity is reduced from 30 to 20 trading days and each stated IV is held fixed. P&L is horizon value minus initial value. Both model families use the same underlying uniform draws for a lower-noise comparison. The empirical tail conventions are stated on the cover.")
    for family in ["normal", "t"]:
        report.h(family.upper()+" full revaluation", 2)
        report.table(select(full, family=family), ["Portfolio", "alpha", "mean_PnL", "VaR", "ES", "ES_VaR_ratio"],
                     labels={"Portfolio":"Book","alpha":"Tail α","mean_PnL":"Mean P&L ($)","VaR":"VaR ($)","ES":"ES ($)","ES_VaR_ratio":"ES/VaR"},
                     formats={"alpha":"{:.0%}","mean_PnL":MONEY,"VaR":MONEY,"ES":MONEY,"ES_VaR_ratio":"{:.3f}"}, widths=[1.6,.55,1,1,1,.7])
    report.p("Normal/t arithmetic returns are theoretically unbounded below -100%. The code raises if such a path occurs rather than silently clipping or resampling it. No such daily return occurred in either final simulation. This remains a model-support limitation, not proof that the unbounded models are globally valid stock-return models.")
    report.part("3(f)", "Plot full10-day P&L against terminal stock price with a delta-only line.",
                "Each panel holds the supplied IV fixed and reduces maturity by ten trading days. The full curve includes option curvature and passage of time; the delta line is the initial book delta times S₁₀-S₀. In the near-delta-neutral books a nearly flat straight line is a poor description of the downside branches.")
    report.image(figures/"q3_pnl_curves.png", "All four book payoff revaluations and initial delta-only approximations on matching axes.", max_height=370)
    report.part("3(g)", "Explain the nearly zero delta-normal estimates and test second-order approximations.",
                "ShortStraddle and HedgedShortCall have offsetting first-order positions, so their delta-normal VaR is only a few dollars. Delta-normal does not see negative gamma or the 10-day change in time value. The local expansion is ΔP≈δΔS+½Γ(ΔS)²+ΘΔt. Adding gamma recovers a loss mechanism but gamma alone omits the positive theta carry, while gamma+theta is still only a local, fixed-Greek expansion.")
    approx_parts = []
    for family in ["normal", "t"]:
        for book in ["ShortStraddle", "HedgedShortCall"]:
            sub = select(approximations, family=family, Portfolio=book, alpha=.01)
            sub = sub.loc[sub.approximation.isin(["delta_gamma", "delta_gamma_theta"])].copy()
            approx_parts.append(sub[["family", "Portfolio", "approximation", "VaR", "ES"]])
            row = select(full, family=family, Portfolio=book, alpha=.01).iloc[0]
            approx_parts.append(pd.DataFrame([dict(family=family, Portfolio=book, approximation="full revaluation", VaR=row.VaR, ES=row.ES)]))
    approx_table = pd.concat(approx_parts, ignore_index=True)
    report.table(approx_table, labels={"family":"Family","Portfolio":"Book","approximation":"Approximation","VaR":"1% VaR ($)","ES":"1% ES ($)"},
                 formats={"VaR":MONEY,"ES":MONEY}, widths=[.7,1.6,1.7,1,1])
    report.p("The second-order results do not reproduce full revaluation exactly. At large moves the Greeks themselves change, strikes are crossed, maturity falls by one third, and higher-order spot/time effects matter. In addition, the analytical delta-normal shock uses a Normal sqrt(10) approximation, whereas the simulated delta-only line is evaluated on the exact compounded terminal price. Those two delta-only risk numbers need not be identical even before introducing option gamma.")
    report.part("3(h)", "Compare mean P&L with the flat-price prediction and explain positive short-gamma means.",
                "All four flat-price P&Ls are positive, matching the theta-sign draft. For the three short-gamma books, simulated means are much smaller because curved losses offset much of the time decay. RiskReversal is different: its positive initial gamma can add expected P&L relative to the flat-price case, and its simulated mean is about $9 rather than the $3.73 unchanged-price gain. For a zero-mean price change, the second-order expected P&L is approximately½Γ Var(ΔS)+ΘΔt. It is a useful scale check, not an exact identity for full option revaluation.")
    report.table(flat, ["Portfolio", "flat_price_PnL", "theta_linear_PnL"],
                 labels={"Portfolio":"Book","flat_price_PnL":"Exact unchanged-S P&L ($)","theta_linear_PnL":"Initial theta×time ($)"},
                 formats={"flat_price_PnL":MONEY,"theta_linear_PnL":MONEY}, widths=[1.6,1.3,1.3])
    means = frame(results, "q3_mean_pnl_checks.csv")
    report.table(means, ["family", "Portfolio", "empirical_mean_PnL", "empirical_mean_SE", "second_order_expected_PnL"],
                 labels={"family":"Family","Portfolio":"Book","empirical_mean_PnL":"Full mean ($)","empirical_mean_SE":"MC SE ($)","second_order_expected_PnL":"Second-order mean ($)"},
                 formats={"empirical_mean_PnL":MONEY,"empirical_mean_SE":"{:.3f}","second_order_expected_PnL":MONEY}, widths=[.7,1.6,1,1,1.2])
    report.p("The short-gamma books' positive physical-model means are compensation for absorbing adverse large moves, not a risk-free return. Option premiums were marked at the supplied IVs, while realized variation is forecast from the return fits with zero drift. Time decay can win on average while rare, large losses dominate ES. Higher-order effects explain residual differences from the second-order mean even after Monte Carlo noise is considered.")
    report.part("3(i)", "Which risks move most when Normal is replaced with t, and why does5% VaR move least?",
                "The largest proportional changes occur in the 1% tail and in ES, especially for the near-delta-neutral short-gamma books. The Normal and t fits have similar overall daily variance, and ten-day compounding averages much of the central distribution. The 5% threshold therefore moves less than the average of the more remote tail; a t tail has more mass in extreme moves, where nonlinear option losses can be especially severe. A local positive gamma is not a guarantee in a remote tail: RiskReversal can move through its short-put strike and its exposure changes.")
    report.table(comparison, ["Portfolio", "alpha", "normal_full_VaR", "t_full_VaR", "normal_full_ES", "t_full_ES"],
                 labels={"Portfolio":"Book","alpha":"Tail α","normal_full_VaR":"Normal VaR","t_full_VaR":"t VaR","normal_full_ES":"Normal ES","t_full_ES":"t ES"},
                 formats={"alpha":"{:.0%}","normal_full_VaR":MONEY,"t_full_VaR":MONEY,"normal_full_ES":MONEY,"t_full_ES":MONEY}, widths=[1.6,.55,1,1,1,1])
    report.p("The Greek-based ordering of absolute5% VaR errors agrees with the Normal full-revaluation comparison: ShortStraddle, HedgedShortCall, CoveredCall, RiskReversal. The local direction call for RiskReversal works at 5%, but its 1% t full VaR exceeds its delta-normal estimate; that is a concrete limit of extrapolating initial gamma to a distant strike-crossing tail.")
    report.part("3(j)", "Reply to the risk manager in two sentences.",
                "The supplied hedge is only approximately delta-flat (-0.444 shares), and making delta exactly zero would still leave negative gamma, time decay and material loss under a larger stock move. We should manage the book with full-revaluation VaR/ES and tail scenarios, not call it risk-free because its first-order exposure is small.")
    report.h("Simulation reliability", 2)
    audit_path = results/"q3_independent_audit.json"
    if audit_path.exists():
        audit = read_json(results, audit_path.name)
        report.p(f"An independent {audit['independent_paths']:,}-path run at seed{audit['independent_seed']} checks all 32 VaR/ES estimates against combined batch Monte Carlo standard errors. The maximum absolute standardized difference is {audit['max_absolute_joint_MCSE_z']:.4f}, below the 5-MCSE audit threshold. A separate24-case finite-difference check of analytical Greeks across call/put and three carry conventions has maximum relative error {audit['max_Greek_relative_error']:.3g}.")
    report.p("The batch MC standard errors are simulation noise conditional on the fitted return models, fixed IVs and static holdings. They exclude parameter uncertainty, volatility changes, model selection, liquidity and dynamic rehedging. More paths do not repair those assumptions.")


def question4(report, results, figures, student):
    report.h("4. A Sharpe Ratio That Sells Insurance")
    summary = read_json(results, "q4_summary.json")
    asset_moments = frame(results, "q4_asset_moments.csv")
    portfolios = frame(results, "q4_portfolios.csv")
    risk = frame(results, "q4_portfolio_risk.csv")
    report.p("Data: 240 monthly total returns, used as supplied rather than converted from prices. EQ, BOND, STK and fully collateralized PUTW are the four risky assets. Annual risk-free return is 3%; its monthly equivalent is (1.03)^(1/12)-1=0.24662698%. Sharpe ratios are annualized by √12; risk below is one-month return loss, not a dollar or annual loss.")
    report.p("The supplied monthly dates run through 30 November 2026. These are the assignment's fixed data, not a live-market dataset; I retain all 240 observations rather than truncate them at today's date.", "note")
    prediction_note(report, "q4", student)
    report.part("4(a)", "Compute asset moments and Sharpe; inspect the influence of one month.",
                "PUTW is the conspicuous combination: high Sharpe with extreme negative skew and excess kurtosis. Its apparent smooth premium income is interrupted by unusually large losses. The moment table uses sample variance/SD and population standardized skewness/excess kurtosis.")
    report.table(asset_moments, ["asset", "mean", "variance", "sd", "skewness", "excess_kurtosis", "annual_sharpe"],
                 labels={"asset":"Asset","mean":"Mean/mo","variance":"Variance/mo","sd":"SD/mo","skewness":"Skew","excess_kurtosis":"Excess kurt.","annual_sharpe":"Annual SR"},
                 formats={"mean":"{:.3%}","variance":"{:.8f}","sd":"{:.3%}","skewness":"{:.3f}","excess_kurtosis":"{:.3f}","annual_sharpe":"{:.4f}"},
                 widths=[.7,1,1.1,1,.8,1,1])
    influence = frame(results, "q4_single_month_influence.csv")
    report.table(influence, ["asset", "month", "event_return", "variance_share", "fourth_moment_share", "excess_kurtosis_without_event"],
                 labels={"asset":"Asset","month":"Influential month","event_return":"Return","variance_share":"Variance share","fourth_moment_share":"4th-moment share","excess_kurtosis_without_event":"Excess kurt. without month"},
                 formats={"event_return":"{:.2%}","variance_share":"{:.2%}","fourth_moment_share":"{:.2%}","excess_kurtosis_without_event":"{:.3f}"}, widths=[.7,1.25,1,1,1.1,1.2])
    report.p("PUTW's June 2008 return is -16.7197%. Around the full-sample mean, that one observation contributes 45.95% of the sum of squared deviations and 80.61% of the sum of fourth powers. Even omitting it leaves excess kurtosis 39.87, so the tail issue is not merely a single removable typo. This is an influence diagnostic only; the observation is retained in every main optimization and risk number.")
    report.part("4(b)", "Before optimizing, predict PUTW's long-only tangency weight and its 1% ES.",
                read_json(results, "q4_review_draft.json")["text"])
    report.part("4(c)", "Find the three tangency portfolios and draw the frontier and capital market line.",
                "The unrestricted solution normalizes Σ⁻¹(μ-rf1) to sum to one. The other two solve the same Sharpe objective with budget sum one and nonnegative weights, adding PUTW≤20% in the third case. Each constrained solve uses thirteen starting points and agrees in objective to about 3×10^-14. An independent active-set convex-quadratic reformulation also verifies the reported tangencies.")
    report.table(portfolios, ["portfolio", "EQ", "BOND", "STK", "PUTW", "annual_sharpe"],
                 labels={"portfolio":"Portfolio","annual_sharpe":"Annual SR"},
                 formats={"EQ":"{:.3%}","BOND":"{:.3%}","STK":"{:.3%}","PUTW":"{:.3%}","annual_sharpe":"{:.6f}"}, widths=[1.6,1,1,1,1,1])
    report.p("The long-only optimum puts about 72.49% in PUTW and 26.63% in bonds, with a very small STK weight and no EQ. The cap changes the exposure materially, at the cost of a lower fitted-sample Sharpe. A lower in-sample Sharpe is not by itself evidence that the capped portfolio is a worse risk decision.")
    report.image(figures/"q4_frontier.png", "Long-only mean/SD frontier and the capital market line through its tangency; asset points are marked. Axes use monthly units.", max_height=320)
    report.part("4(d)", "Report all three portfolios' historical and Normal VaR/ES at 5% and 1%.",
                "The Normal comparator matches each portfolio's full-sample monthly mean and sample SD. The historical calculation preserves joint monthly outcomes by first forming the portfolio return each month; it does not add individual asset VaRs. All percentages below are losses relative to zero return.")
    report.table(risk, ["portfolio", "alpha", "historical_VaR", "historical_ES", "normal_VaR", "normal_ES"],
                 labels={"portfolio":"Portfolio","alpha":"Tail α","historical_VaR":"Historical VaR","historical_ES":"Historical ES","normal_VaR":"Normal VaR","normal_ES":"Normal ES"},
                 formats={"alpha":"{:.0%}","historical_VaR":"{:.3%}","historical_ES":"{:.3%}","normal_VaR":"{:.3%}","normal_ES":"{:.3%}"}, widths=[1.6,.55,1.1,1.1,1.1,1.1])
    long_tail = select(risk, portfolio="LongOnly", alpha=.01).iloc[0]
    report.p(f"For the long-only portfolio, historical 1% ES is {long_tail.historical_ES:.3%}, versus {long_tail.normal_ES:.3%} under the matched Normal: about{long_tail.historical_ES/long_tail.normal_ES:.2f} times as large. In contrast its historical 5% VaR is much smaller than the Normal 5% VaR. A large mass of relatively calm premium months and a few severe loss months cannot be summarized by the same Normal shape at both thresholds.")
    report.p("At 1% of 240 months, the course threshold includes only the two worst observations. These tail numbers are therefore statistically fragile. The alternate fractional-mass estimator is exported in q4_tail_convention_sensitivity.csv rather than silently substituted for the course convention.")
    report.part("4(e)", "Estimate full-sample CAPM and a separate downside-market beta.",
                "Both sides use monthly excess returns. The downside subset is EQ excess return below -5%, not EQ raw return below -5%. I report the requested full-sample alpha, its classical OLS t statistic, beta and R², and retain the separate-regime fit as a descriptive diagnostic.")
    capm = frame(results, "q4_capm.csv")
    report.table(capm, ["sample", "n", "alpha", "alpha_t", "beta", "rsquared"],
                 labels={"sample":"Months","alpha":"Alpha/mo","alpha_t":"Alpha t","beta":"EQ beta","rsquared":"R²"},
                 formats={"alpha":"{:.4%}","alpha_t":"{:.4f}","beta":"{:.6f}","rsquared":"{:.4f}"}, widths=[1.9,.5,1,1,1,1])
    report.image(figures/"q4_capm_regimes.png", "The full-sample slope and the slope in the 26 downside months summarize very different behavior.", max_height=280)
    report.part("4(f)", "What has the optimizer bought, and why can ES see what Sharpe misses?",
                "It has bought a large short-downside-insurance exposure, not a newly discovered low-risk bond substitute. Full cash collateral prevents borrowing within PUTW's strategy but does not remove its downside payout. Sharpe ranks only excess mean relative to total SD; it cannot distinguish a mostly smooth premium stream with rare large losses from a symmetric return distribution with the same first two moments.")
    report.p("ES averages the tail that matters rather than stopping at a quantile threshold. In the loss convention it is monotone, cash-additive, positively homogeneous and subadditive: worse scenario losses cannot lower risk, deterministic losses add dollar for dollar, scaling positions scales risk, and combining risks respects genuine diversification. Coherence is a structural property under a specified joint distribution, not a guarantee that a240-month estimate has captured all future crashes.")
    report.p("As a same-scenario illustration, the long-only portfolio's1% ES is 10.1793%, below the nonnegative weighted sum of stand-alone asset ES values,11.1054%. For the capped portfolio the corresponding figures are 5.9167% and 8.1297%. These comparisons use nonnegative weights; applying a signed-weight sum of long-asset ES to the unrestricted short position would not be a valid subadditivity check.")
    report.part("4(g)", "Does PUTW's CAPM alpha demonstrate skill?",
                "Not on this evidence. The full regression reports monthly alpha0.27723%, t=3.0819 and beta0.1848, while downside months have beta0.9428. The single constant-beta line averages a nonlinear state-dependent payoff. Premium income during calm markets appears as an intercept, while market sensitivity becomes much larger in bad states. A significant intercept in this misspecified linear model is not proof of skill independent of the embedded insurance exposure. The small26-month downside sample and classical OLS standard errors also limit inference.")
    report.part("4(h)", "How precisely are the mean inputs known, and which weights are trustworthy?",
                "Under an IID-month calculation, SE(mean)=sample_SD/sqrt(240). These are input-estimation errors, separate from portfolio risk. An optimizer can turn small mean differences into large or boundary weights; the resulting decimals should not be mistaken for known true allocations.")
    report.table(asset_moments, ["asset", "mean", "mean_standard_error"],
                 labels={"asset":"Asset","mean":"Mean monthly return","mean_standard_error":"SE of monthly mean"},
                 formats={"mean":"{:.4%}","mean_standard_error":"{:.4%}"}, widths=[1,1.5,1.5])
    bootstrap = frame(results, "q4_bootstrap_intervals.csv")
    report.table(bootstrap, ["portfolio", "asset", "n_success", "p05", "median", "p95"],
                 labels={"portfolio":"Portfolio","asset":"Asset","n_success":"Valid runs","p05":"5th pct. weight","median":"Median weight","p95":"95th pct. weight"},
                 formats={"p05":"{:.2%}","median":"{:.2%}","p95":"{:.2%}"}, widths=[1.5,.75,.7,1,1,1])
    report.p("In 300 IID monthly resamples, the long-only PUTW weight's 5th-95th percentile range is about 22.96%-99.83%. Three unrestricted resamples have no positively oriented finite budget-normalized tangency and are reported as such; the other 297 supply its interval. I would put more trust in the need for exposure limits and broad diversification than in the exact unrestricted weights or the apparent certainty of a 72% insurance allocation. The capped PUTW weight staying near 20% reflects the imposed cap, not certainty about an unconstrained true optimum. Bootstrap is conditional on this empirical sample, assumes exchangeable months and cannot invent an unseen crash regime.")


def question5(report, results, figures, student):
    report.h("5. Alpha or Factor Exposure?")
    summary = read_json(results, "q5_summary.json")
    stats_table = frame(results, "q5_initial_statistics.csv")
    models = frame(results, "q5_model_comparison.csv")
    coefficients = frame(results, "q5_regression_coefficients.csv")
    expected = frame(results, "q5_expected_returns.csv")
    report.p(f"The aligned fund sample has {summary['n']} dates, {summary['fund_sample_start']} through {summary['fund_sample_end']}. Fund returns are already decimals. Every Fama-French factor and RF percentage is divided by 100 exactly once; Mom's padded column name is trimmed and all files are joined by date.")
    report.p("All regressions include an intercept and use daily excess fund returns. CAPM uses Mkt-RF; FF3 adds SMB and HML; Carhart4 also adds Mom. Reported SEs are classical homoskedastic OLS SEs. AICc includes the residual scale parameter: k=3,5,6 respectively. Model comparisons use the identical aligned sample.")
    prediction_note(report, "q5", student)
    report.part("5(a)", "Compute excess-return statistics and CAPM: which fund initially appears more skilled?",
                "Under CAPM alone, FUND1 initially looks stronger: it has the larger excess return, Sharpe and estimated alpha, and its alpha t statistic is above 3. This is an initial model-relative judgment, not yet a conclusion about skill after omitted-factor exposures are checked. The table labels daily versus annualized quantities explicitly; Sharpe uses excess-return SD.")
    report.table(stats_table, ["fund", "mean_daily_excess", "daily_volatility", "daily_sharpe", "annual_sharpe", "capm_alpha_daily", "capm_alpha_t"],
                 labels={"fund":"Fund","mean_daily_excess":"Mean excess/day","daily_volatility":"Fund SD/day","daily_sharpe":"Daily SR","annual_sharpe":"Annual SR","capm_alpha_daily":"CAPM alpha/day","capm_alpha_t":"Alpha t"},
                 formats={"mean_daily_excess":"{:.4%}","daily_volatility":"{:.4%}","daily_sharpe":"{:.4f}","annual_sharpe":"{:.4f}","capm_alpha_daily":"{:.4%}","capm_alpha_t":"{:.4f}"}, widths=[.75,1.1,1.1,.8,.8,1.1,.8])
    report.part("5(b)", "Inspect CAPM residual correlations before adding the extra factors.",
                "FUND1's residual is strongly correlated with Mom and positively with HML, so I expect some CAPM alpha to disappear when those rewarded exposures are included. FUND2's residual relates mainly to SMB, whose sample premium is negative, so I expect its alpha to rise when that headwind is separated out. The saved draft called that rise 'substantial'; part(f) checks the magnitude rather than rewriting the draft after fitting.")
    correlations = frame(results, "q5_capm_residual_correlations.csv")
    report.table(correlations, ["fund", "factor", "residual_correlation", "sample_factor_premium"],
                 labels={"fund":"Fund","factor":"Factor","residual_correlation":"CAPM residual corr.","sample_factor_premium":"Sample daily premium"},
                 formats={"residual_correlation":"{:.6f}","sample_factor_premium":"{:.5%}"}, widths=[.8,.8,1.6,1.4])
    report.part("5(c)", "Fit CAPM, FF3 and Carhart4 for both funds and report all coefficients and SEs.",
                "Alphas below are daily decimals, and factor slopes are dimensionless. Every coefficient has its own standard error and t statistic. The regression-level table supplies alpha's t statistic, R², log likelihood and AICc; k counts all slopes, the intercept and the residual Normal scale.")
    for fund in ["FUND1", "FUND2"]:
        report.h(fund+" coefficient estimates", 2)
        report.table(select(coefficients, fund=fund), ["model", "term", "coefficient", "standard_error", "t_statistic"],
                     labels={"model":"Model","term":"Term","coefficient":"Coefficient","standard_error":"Standard error","t_statistic":"t statistic"},
                     formats={"coefficient":"{:.8f}","standard_error":"{:.8f}","t_statistic":"{:.4f}"}, widths=[1,.9,1.15,1.15,1])
    report.table(models, ["fund", "model", "alpha_t", "rsquared", "ll", "AICc"],
                 labels={"fund":"Fund","model":"Model","alpha_t":"Alpha t","rsquared":"R²","ll":"Log likelihood"},
                 formats={"alpha_t":"{:.4f}","rsquared":"{:.6f}","ll":"{:.3f}","AICc":"{:.3f}"}, widths=[.8,1,.8,.8,1,1])
    report.p("Carhart4 is strongly preferred for FUND1: it explains the conspicuous momentum-related residual and leaves essentially no alpha. For FUND2, FF3 has AICc lower than Carhart4 by about 1.806; its momentum coefficient is small and insignificant (t≈0.461). Part(d) nevertheless uses four-factor betas for both funds because that is explicitly requested.")
    report.image(figures/"q5_alpha_models.png", "Annualized arithmetic alpha and its sampling error change with the factor model. Alpha is not invariant to the chosen benchmark.", max_height=280)
    report.part("5(d)", "Use recent ten-year factor premia to compute annual expected returns, with and without alpha.",
                f"The factor files end at {summary['factor_file_end']}, so their most recent ten-year window is {summary['forecast_first_date']} through {summary['forecast_last_date']} ({summary['forecast_n']} dates), not the fund's 1996-2001 window and not ten years ending today. I use Eannual=255×[mean(RF)+β' mean(factors)], optionally adding 255×fitted daily alpha. These are arithmetic expected returns; I do not compound factor premia or confuse this measure with CAGR.")
    premia = frame(results, "q5_recent_factor_premia.csv")
    report.table(premia, ["factor", "mean_daily", "annual_arithmetic"],
                 labels={"factor":"Factor","mean_daily":"Daily average","annual_arithmetic":"Annual arithmetic"},
                 formats={"mean_daily":"{:.5%}","annual_arithmetic":"{:.4%}"}, widths=[1,1.5,1.5])
    report.table(expected, ["fund", "expected_annual_without_alpha", "expected_annual_with_alpha", "annual_fitted_alpha"],
                 labels={"fund":"Fund","expected_annual_without_alpha":"Expected annual, no alpha","expected_annual_with_alpha":"Expected annual, with alpha","annual_fitted_alpha":"Annual fitted alpha"},
                 formats={"expected_annual_without_alpha":"{:.4%}","expected_annual_with_alpha":"{:.4%}","annual_fitted_alpha":"{:.4%}"}, widths=[.8,1.5,1.5,1.1])
    report.p("The recent-premium window postdates the fund sample. These are hypothetical expected returns conditional on transporting the estimated betas to that premium regime, not forecasts that were available in 1996. Including alpha additionally assumes that an estimated model residual return persists.")
    report.part("5(e)", "Replay the worst momentum day as a one-factor stress on $10 million.",
                "The search uses the entire supplied momentum file. Its worst daily return is -18.44% on5 September 1939. I apply only this momentum factor shock; other factor shocks are zero. This is an incremental stress, so I do not add ordinary alpha or RF accrual to the shock loss. Loss=-10,000,000×beta_Mom×(-0.1844); standardized loss divides the shocked percentage loss by each fund's sample daily return SD.")
    stress = frame(results, "q5_momentum_stress.csv")
    report.table(stress, ["fund", "momentum_beta", "incremental_fund_return", "loss_dollars", "loss_in_daily_sigmas"],
                 labels={"fund":"Fund","momentum_beta":"Mom beta","incremental_fund_return":"Shock return","loss_dollars":"Loss on $10m","loss_in_daily_sigmas":"Daily SD multiples"},
                 formats={"momentum_beta":"{:.8f}","incremental_fund_return":"{:.4%}","loss_dollars":MONEY,"loss_in_daily_sigmas":"{:.4f}"}, widths=[.8,1,1,1.3,1.1])
    report.part("5(f)", "Reconcile the alpha changes using sample factor premia.",
                "FUND1's daily alpha falls from 0.000440660 under CAPM to 0.000296309 under FF3 and 0.000002938 under Carhart4, matching the predicted direction. FUND2 rises from 0.000244907 to 0.000272797 overall. Its direction is also as predicted, but the earlier draft's 'substantial' rise overstates the observed magnitude; momentum adds little once FF3 is present.")
    attribution = frame(results, "q5_alpha_attribution.csv")
    attribution = select(attribution, fund="FUND1", earlier_model="CAPM", later_model="Carhart4").copy()
    report.table(attribution, ["factor", "delta_beta", "sample_daily_premium", "contribution_to_alpha_drop_daily", "contribution_to_alpha_drop_annual"],
                 labels={"factor":"Factor","delta_beta":"Change in beta","sample_daily_premium":"Sample daily premium","contribution_to_alpha_drop_daily":"Daily alpha drop contribution","contribution_to_alpha_drop_annual":"Annual contribution"},
                 formats={"delta_beta":"{:.7f}","sample_daily_premium":"{:.5%}","contribution_to_alpha_drop_daily":"{:.7f}","contribution_to_alpha_drop_annual":"{:.4%}"}, widths=[.75,1,1.2,1.35,1.2])
    report.p("With an intercept, the fitted sample mean satisfies mean(excess fund return)=alpha+beta' mean(factors). Therefore the alpha drop equals the changes in all included beta-premium products, including the changed market beta. Momentum contributes about 7.8464 annual percentage points to FUND1's CAPM-to-Carhart alpha drop, HML2.7555, the market-beta adjustment0.5635, and SMB-0.0035. Their sum matches the actual drop of 11.1619 annual percentage points to numerical precision; it is not correct to attribute the whole change to momentum alone.")
    report.part("5(g)", "Interpret alpha through APT and state a fee opinion.",
                "Alpha is the mean return left after compensation for the risks represented by a particular factor set. APT does not make alpha model-independent: omitted systematic exposure can enter the intercept. Adding economically relevant factors can remove apparent abnormal performance without changing any realized fund return.")
    report.p("Opinion: I would be more willing to consider an alpha fee for FUND2 than for FUND1, using Carhart4 to check that momentum is not being sold as skill and FF3 as a parsimonious cross-check for FUND2. FUND1's four-factor alpha is effectively zero (t≈0.042), whereas FUND2 retains about 6.9563% annual arithmetic alpha with t≈2.150 under the specified classical errors. This is conditional evidence, not proof of persistent skill; fees, model omissions, multiple testing, regime changes and alternative error estimates require further scrutiny.")
    report.part("5(h)", "Compare factor-implied and realized returns and connect the result to portfolio optimization.",
                "The following comparison keeps annual arithmetic means separate from the geometric realization. FUND1's recent-premium expectation is much lower than its sample mean, even after adding its negligible four-factor alpha. FUND2's expectation with alpha is closer to its sample mean, while the no-alpha expectation is lower.")
    report.table(expected, ["fund", "expected_annual_without_alpha", "expected_annual_with_alpha", "realized_annual_arithmetic", "realized_annual_geometric"],
                 labels={"fund":"Fund","expected_annual_without_alpha":"Expected, no alpha","expected_annual_with_alpha":"Expected, with alpha","realized_annual_arithmetic":"Sample arithmetic/year","realized_annual_geometric":"Sample geometric/year"},
                 formats={c:"{:.4%}" for c in ["expected_annual_without_alpha","expected_annual_with_alpha","realized_annual_arithmetic","realized_annual_geometric"]}, widths=[.8,1.2,1.2,1.2,1.2])
    report.p("For FUND1, its sample rewarded momentum and value exposures much more strongly than the recent ten-year premium window. Extrapolating the fund's raw five-year mean would carry those regime-specific premia into an optimizer as though they were permanent alpha. A factor model separates estimated exposure from an explicit premium assumption and can reduce that mistake. It does not eliminate estimation risk: betas may change, future premia are unknown, and an optimizer such as Problem 4's will still magnify small errors in expected returns. Shrinkage, exposure limits and sensitivity analysis remain important.")


def code_and_sources(report, results):
    report.h("6. Code, Verification, and Reproduction")
    report.p("The submission includes an installable assignment3 package. pricing.py implements GBSM, analytic Greeks and IV inversion; american.py implements dividend-event CRR valuation. Each qN.py is labeled by question and subpart. common.py owns the moment, likelihood, regression and tail conventions so the problems use one documented statistical definition.")
    report.p("The README provides the environment setup and the exact calculation, audit, testing and PDF commands. All tables are CSVs, all charts are PNGs, and summaries/provenance are JSON under results/. The original seven CSV inputs are preserved without modification. student_predictions.json is intentionally blank because no original student notes were supplied.")
    verification_path = results/"verification.json"
    if verification_path.exists():
        verification = read_json(results, verification_path.name)
        passed = verification.get("passed", verification.get("passed_checks"))
        total = verification.get("total", verification.get("total_checks"))
        report.p(f"Independent result audit: {passed}/{total} checks passed. The audit includes independent quote repricing, cash-dividend checks, an alternative convex-QP tangency solution, direct regression reconstruction and raw-file window/unit checks.")
    report.p("The included test suite has 66 tests covering pricing identities and Greeks, a small exhaustive non-recombining tree, convergence, constraints, regressions, data units/dates and statistical conventions. The separate Q2 audit uses stable divided differences to distinguish lattice kinks from small-bump price-subtraction noise. The Q3 audit checks 32 tail-risk estimates with an independent simulation and 24 analytical Greeks with finite differences.")
    report.h("Important numerical limitations", 2)
    report.p("Tree prices oscillate with the grid and small-bump tree gamma can be unreliable. Empirical 1% ES from 240 months uses very few observations. Bootstrap, regression standard errors and Monte Carlo errors are conditional diagnostics, not comprehensive uncertainty bands. Fixed-IV/static-hedge modeling excludes volatility and rehedging risk. Normal/t daily arithmetic returns have an invalid-price region below -100%, although no such event occurred in the reported paths.")
    report.page()
    report.h("Sources", 2)
    report.p("1. Dominic Pazzula, Assignment3 - Options, Option Risk, and Portfolio Construction, released 8 November 2026. Instructor repository: github.com/dompazz/FinTech-545-Fall2026, Assignments/Assignment3, commit 5958d11b51960c18be98f6df2b97c4e80670fb2d.")
    report.p("2. Instructor Week06 and Week07 notes and library examples: GBSM, carry conventions, Greeks and discrete-dividend recursion. All submitted pricing routines are implemented in Python rather than delegated to a financial pricing black box.")
    report.p("3. Instructor Week05/RiskStats.jl: empirical VaR order-statistic and ES threshold-tail conventions. The separate fractional-tail calculation is labeled a sensitivity, not silently mixed into the primary numbers.")
    report.p("4. Fama-French market/SMB/HML/RF and momentum CSVs supplied alongside the assignment. Percent-to-decimal conversion, exact sample/date joins, the recent ten-year premium window and full-history momentum-stress search are recorded in the results.")


def build(results, figures, output):
    results, figures = Path(results), Path(figures)
    metadata = read_json(results, "run_metadata.json")
    primary = read_json(results, "q3_summary.json")["parameters"]
    if (read_json(results, "q2_summary.json")["tree_n"] != 500
            or primary["paths"] != 1000000 or primary["seed"] != 545
            or read_json(results, "q4_summary.json")["bootstrap_replications"] != 300):
        raise ValueError("The written narrative is for the supplied-data baseline: tree N500, 1m paths seed 545, bootstrap300. Run the documented full command or revise the narrative before a custom report.")
    student = read_json(ROOT, "student_predictions.json")
    report = Report(output)
    cover(report, results)
    question1(report, results, figures, student)
    question2(report, results, figures, student)
    question3(report, results, figures, student)
    question4(report, results, figures, student)
    question5(report, results, figures, student)
    code_and_sources(report, results)
    report.build()
    print("PDF:", Path(output).resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"results")
    parser.add_argument("--figures", type=Path, default=ROOT/"figures")
    parser.add_argument("--output", type=Path, default=ROOT/"Assignment3_Answers.pdf")
    args = parser.parse_args()
    build(args.results, args.figures, args.output)
