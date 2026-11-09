"""Problem 2(a-i): American options and discrete cash dividends."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

from .american import exercise_boundary, tree_price
from .common import write_json, write_predictions
from .pricing import price


SPOT, STRIKE, MATURITY = 48.50, 50., .5
RATE, VOLATILITY = .045, .30
DIVIDENDS = ((.10, .25), (.40, 1.50))


def _value(is_call, american, n, dividends, spot=SPOT):
    return tree_price(spot, STRIKE, MATURITY, RATE, VOLATILITY, n,
                      is_call=is_call, american=american, dividends=dividends)


def _premium(american_value, european_value):
    difference = american_value-european_value
    return 0. if abs(difference) < 1e-12 else difference


def run(data_dir, output_dir, figure_dir, **kwargs):
    """Create the complete numerical evidence for Q2 without changing raw data."""
    out, figures = Path(output_dir), Path(figure_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    n = int(kwargs.get("tree_n", 500))
    if n < 5 or n % 5:
        raise ValueError("Q2 tree_n must be a positive multiple of five")

    # Problem 2(a/b): separate analytical draft; not a claim of student authorship.
    conditions = [dict(time=t, dividend=d,
                       strike_interest=STRIKE*(1-np.exp(-RATE*(MATURITY-t))),
                       condition_holds=bool(d > STRIKE*(1-np.exp(-RATE*(MATURITY-t)))))
                  for t, d in DIVIDENDS]
    predictions = dict(
        chronology_note="AI-assisted analytical draft, not a verified student-before-fit record; implementation benchmarks predated persistence",
        dividend_conditions=conditions,
        a="The first dividend fails the given interest condition. The second passes it; sufficiently high stock prices can make exercising the call just before that payment optimal.",
        b=dict(EuropeanCall="lower", AmericanCall="lower", EuropeanPut="higher", AmericanPut="higher",
               call_early_exercise_premium="larger", put_early_exercise_premium="expected smaller: a forthcoming stock-price drop makes waiting more attractive to a put holder"),
    )
    write_predictions(out / "q2_predictions.json", predictions)
    pd.DataFrame(conditions).to_csv(out / "q2_dividend_conditions.csv", index=False)

    # Problem 2(c): GBSM Europeans without dividends; otherwise explicit trees.
    prices, premiums = [], []
    pv_dividends = sum(amount*np.exp(-RATE*time) for time, amount in DIVIDENDS)
    values = {}
    for is_call, kind in [(True, "Call"), (False, "Put")]:
        european_no_div = float(price(SPOT, STRIKE, MATURITY, RATE, RATE, VOLATILITY, is_call))
        european_no_div_tree = _value(is_call, False, n, ())
        american_no_div = _value(is_call, True, n, ())
        european_div = _value(is_call, False, n, DIVIDENDS)
        american_div = _value(is_call, True, n, DIVIDENDS)
        pv_approximation = float(price(SPOT-pv_dividends, STRIKE, MATURITY, RATE, RATE, VOLATILITY, is_call))
        values[kind] = dict(european_no_div=european_no_div, european_no_div_tree=european_no_div_tree,
                            american_no_div=american_no_div, european_div=european_div,
                            american_div=american_div, pv_dividend_gbsm=pv_approximation)
        prices.extend([
            dict(option=f"European{kind}", n=n, no_dividend_value=european_no_div,
                 with_dividend_value=european_div, no_dividend_method="GBSM",
                 with_dividend_method="non-recombining cash-dividend CRR tree",
                 pv_dividend_gbsm=pv_approximation,
                 tree_minus_pv_gbsm=european_div-pv_approximation),
            dict(option=f"American{kind}", n=n, no_dividend_value=american_no_div,
                 with_dividend_value=american_div, no_dividend_method="CRR tree",
                 with_dividend_method="non-recombining cash-dividend CRR tree",
                 pv_dividend_gbsm=np.nan, tree_minus_pv_gbsm=np.nan),
        ])
        # Matched-tree comparisons isolate exercise value from lattice error.
        premiums.extend([
            dict(kind=kind, dividends="none", n=n, american_value=american_no_div,
                 european_value_same_tree=european_no_div_tree,
                 early_exercise_premium=_premium(american_no_div,european_no_div_tree),
                 european_gbsm_reference=european_no_div,
                 american_minus_gbsm=american_no_div-european_no_div),
            dict(kind=kind, dividends="two cash payments", n=n, american_value=american_div,
                 european_value_same_tree=european_div,
                 early_exercise_premium=_premium(american_div,european_div),
                 european_gbsm_reference=np.nan, american_minus_gbsm=np.nan),
        ])
    pd.DataFrame(prices).to_csv(out / "q2_prices.csv", index=False)
    pd.DataFrame(premiums).to_csv(out / "q2_exercise_premiums.csv", index=False)

    # Problem 2(d): neighbouring feasible grids demonstrate lattice oscillation.
    standard_grid = [25, 50, 75, 100, 105, 150, 155, 200, 205, 300, 305, 400, 405, 500, 505]
    grids = sorted(set(v for v in standard_grid if v <= max(n+5, 105)) | {n, n+5, 100, 105})
    convergence = []
    for steps in grids:
        row = dict(n=steps)
        for is_call, kind in [(True, "call"), (False, "put")]:
            row[f"american_{kind}_no_dividend"] = _value(is_call, True, steps, ())
            row[f"american_{kind}_with_dividend"] = _value(is_call, True, steps, DIVIDENDS)
            row[f"european_{kind}_with_dividend"] = _value(is_call, False, steps, DIVIDENDS)
        convergence.append(row)
    convergence_frame = pd.DataFrame(convergence)
    convergence_frame.to_csv(out / "q2_convergence.csv", index=False)

    # Problem 2(e/h): shrinking spot bumps with the pricing grid held FIXED.
    greek_rows = []
    for is_call, kind in [(True, "Call"), (False, "Put")]:
        base = values[kind]["american_div"]
        for bump in [1., .1, .01, .001, .0001, .00001]:
            up = _value(is_call, True, n, DIVIDENDS, SPOT+bump)
            down = _value(is_call, True, n, DIVIDENDS, SPOT-bump)
            greek_rows.append(dict(option=f"American{kind}", n=n, bump=bump,
                                   required_bump=bool(bump >= .001),
                                   delta=(up-down)/(2*bump),
                                   gamma=(up-2*base+down)/(bump*bump)))
    greek_frame = pd.DataFrame(greek_rows)
    greek_frame.to_csv(out / "q2_finite_difference_greeks.csv", index=False)

    # Problem 2(f/i): dollars of option value per dollar of the specified dividend.
    sensitivities = []
    for is_call, kind in [(True, "Call"), (False, "Put")]:
        for american, style in [(False, "European"), (True, "American")]:
            for index, (time, cash) in enumerate(DIVIDENDS):
                for bump in [.01, .001]:
                    up_divs, down_divs = list(DIVIDENDS), list(DIVIDENDS)
                    up_divs[index], down_divs[index] = (time, cash+bump), (time, cash-bump)
                    up = _value(is_call, american, n, up_divs)
                    down = _value(is_call, american, n, down_divs)
                    sensitivities.append(dict(option=f"{style}{kind}", n=n,
                                               dividend_number=index+1, dividend_time=time,
                                               dividend_amount=cash, bump=bump,
                                               dV_dDividend=(up-down)/(2*bump)))
    sensitivity_frame = pd.DataFrame(sensitivities)
    sensitivity_frame.to_csv(out / "q2_dividend_sensitivities.csv", index=False)

    # Problem 2(g): exercise threshold immediately before the LAST dividend.
    last_time, last_cash = DIVIDENDS[-1]
    remaining_steps = int(round(n*(MATURITY-last_time)/MATURITY))
    boundary = exercise_boundary(STRIKE, MATURITY-last_time, RATE, VOLATILITY,
                                 remaining_steps, last_cash)
    continuous_boundary = float(optimize.brentq(
        lambda spot: spot-STRIKE-price(spot-last_cash, STRIKE, MATURITY-last_time,
                                       RATE, RATE, VOLATILITY, True),
        STRIKE, 2*STRIKE, xtol=1e-11,
    ))
    boundary_convergence = []
    for steps in grids:
        tail_steps = int(round(steps*(MATURITY-last_time)/MATURITY))
        boundary_convergence.append(dict(n=steps, tail_steps=tail_steps,
            continuous_gbsm_reference=continuous_boundary,
            critical_pre_dividend_stock=exercise_boundary(STRIKE, MATURITY-last_time, RATE,
                                                          VOLATILITY, tail_steps, last_cash)))
    pd.DataFrame(boundary_convergence).to_csv(out / "q2_exercise_boundary.csv", index=False)
    spot_grid = np.linspace(45., 65., 161)
    exercise = np.maximum(spot_grid-STRIKE, 0.)
    continuation = tree_price(np.maximum(spot_grid-last_cash, 0.), STRIKE, MATURITY-last_time,
                               RATE, VOLATILITY, remaining_steps, True, True)
    pd.DataFrame(dict(pre_dividend_stock=spot_grid, exercise_value=exercise,
                      continuation_value=continuation,
                      exercise_minus_continuation=exercise-continuation)).to_csv(
        out / "q2_boundary_values.csv", index=False)

    # Numerical identities independent of any prediction or written conclusion.
    forward_parity = SPOT-pv_dividends-STRIKE*np.exp(-RATE*MATURITY)
    parity_error = values["Call"]["european_div"]-values["Put"]["european_div"]-forward_parity
    if abs(parity_error) > 1e-8:
        raise AssertionError("Discrete-dividend European put-call parity mismatch")
    if any(row["early_exercise_premium"] < -1e-10 for row in premiums):
        raise AssertionError("American price fell below its matched European tree price")
    if abs(values["Call"]["american_no_div"]-values["Call"]["european_no_div_tree"]) > 1e-10:
        raise AssertionError("Non-dividend American call differs from its European CRR value")

    # Figures: main convergence, fixed-grid Greeks, and the exercise threshold.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, kind in zip(axes, ["call", "put"]):
        ax.plot(convergence_frame.n, convergence_frame[f"american_{kind}_no_dividend"], "o-", ms=3, label="No dividends")
        ax.plot(convergence_frame.n, convergence_frame[f"american_{kind}_with_dividend"], "s-", ms=3, label="Two dividends")
        ax.set(title=f"American {kind}", xlabel="Tree steps N", ylabel="Option value ($)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "q2_convergence.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for option, frame in greek_frame.groupby("option", sort=False):
        axes[0].plot(frame.bump, frame.delta, "o-", label=option)
        axes[1].plot(frame.bump, frame.gamma, "o-", label=option)
    for ax, greek in zip(axes, ["Delta", "Gamma"]):
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set(xlabel="Spot bump ($), decreasing to the right", ylabel=greek, title=f"{greek}, fixed N={n}")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "q2_greeks_bump.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.8, 4))
    ax.plot(spot_grid, exercise, label="Exercise just before dividend")
    ax.plot(spot_grid, continuation, label="Continue after $1.50 dividend")
    ax.axvline(boundary, color="black", ls="--", lw=1, label=f"Boundary ${boundary:.4f}")
    ax.set(xlabel="Stock price just before second dividend ($)", ylabel="Call value ($)", title=f"Exercise decision; N={n}")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "q2_exercise_boundary.png", dpi=180); plt.close(fig)

    summary = dict(
        inputs=dict(spot=SPOT, strike=STRIKE, maturity=MATURITY, risk_free_rate=RATE,
                    volatility=VOLATILITY, dividends=[dict(time=t, amount=d) for t,d in DIVIDENDS]),
        tree_n=n, dividends_present_value=pv_dividends, dividend_conditions=conditions,
        prices=values, exercise_premiums=premiums,
        central_difference_greeks=greek_rows,
        dividend_sensitivities=[row for row in sensitivities if row["bump"] == .001],
        second_dividend_exercise_boundary=boundary,
        second_dividend_continuous_boundary_reference=continuous_boundary,
        boundary_remaining_steps=remaining_steps, european_parity_error=parity_error,
        convergence_last_rows=convergence[-4:],
        conventions=dict(tree="Cash-dividend non-recombining CRR; recombination only within intervals",
                         exercise_timing="Compare before-dividend exercise with ex-dividend continuation; continuation also allows immediate post-dividend exercise",
                         low_price_policy="Stock equity is absorbing at zero and cash dividends cannot exceed it; only negligible-probability extreme nodes affected here",
                         premiums="Use American minus European on the same lattice to remove GBSM-vs-tree discretization error",
                         greeks="Central finite differences on a FIXED N grid; extra 1e-4 and 1e-5 bumps are diagnostic",
                         dividend_sensitivity="Central amount differences, primary bump0.001; bump0.01 exported as robustness check"),
    )
    write_json(out / "q2_summary.json", summary)
    return summary
