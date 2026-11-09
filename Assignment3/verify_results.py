"""Independent numerical audit of the saved Assignment 3 results.

This module does not import the production calculation routines. It uses an
independent GBSM expression, scalar recursive dividend trees, normal equations
for regression, active-set convex QPs for tangencies, and fresh-seed Monte
Carlo scenarios. Only results/verification.json is written.
"""
import argparse
import csv
from datetime import date
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
ASSETS = ["EQ", "BOND", "STK", "PUTW"]
FACTORS = ["Mkt-RF", "SMB", "HML", "Mom"]
MODELS = {"CAPM": ["Mkt-RF"], "FF3": ["Mkt-RF", "SMB", "HML"], "Carhart4": FACTORS}


def _gbsm(s, k, t, r, sigma, call=True, q=0.):
    """Independent explicit European expression, including deterministic limits."""
    s = np.asarray(s, float)
    sign = 1. if call else -1.
    if t <= 0:
        return np.maximum(sign*(s-k), 0.)
    if sigma <= 0:
        return np.exp(-r*t)*np.maximum(sign*(s*np.exp((r-q)*t)-k), 0.)
    z = (np.log(s/k)+(r-q+sigma*sigma/2)*t)/(sigma*np.sqrt(t))
    zz = z-sigma*np.sqrt(t)
    if call:
        return s*np.exp(-q*t)*stats.norm.cdf(z)-k*np.exp(-r*t)*stats.norm.cdf(zz)
    return k*np.exp(-r*t)*stats.norm.cdf(-zz)-s*np.exp(-q*t)*stats.norm.cdf(-z)


def _ordinary_tree(s, k, t, r, sigma, n, call, american):
    """One origin, one-dimensional backward induction; no production batching."""
    dt = t/n
    up = np.exp(sigma*np.sqrt(dt))
    probability = (np.exp(r*dt)-1/up)/(up-1/up)
    sign = 1. if call else -1.
    values = np.maximum(sign*(s*up**(2*np.arange(n+1)-n)-k), 0.)
    for j in range(n-1, -1, -1):
        values = np.exp(-r*dt)*(probability*values[1:]+(1-probability)*values[:-1])
        if american:
            values = np.maximum(values, np.maximum(sign*(s*up**(2*np.arange(j+1)-j)-k), 0.))
    return float(values[0])


def _cash_tree(s, k, t, r, sigma, n, call, american, events):
    """Independent unbatched recursion over each dividend-node subtree."""
    if not events:
        return _ordinary_tree(s, k, t, r, sigma, n, call, american)
    dt = t/n
    step, cash = events[0]
    up = np.exp(sigma*np.sqrt(dt))
    probability = (np.exp(r*dt)-1/up)/(up-1/up)
    sign = 1. if call else -1.
    node_spots = s*up**(2*np.arange(step+1)-step)
    later = [(when-step, amount) for when, amount in events[1:]]
    values = np.array([_cash_tree(max(spot-cash, 0.), k, t-step*dt, r, sigma,
                                 n-step, call, american, later) for spot in node_spots])
    if american:
        values = np.maximum(values, np.maximum(sign*(node_spots-k), 0.))
    for j in range(step-1, -1, -1):
        values = np.exp(-r*dt)*(probability*values[1:]+(1-probability)*values[:-1])
        if american:
            values = np.maximum(values, np.maximum(sign*(s*up**(2*np.arange(j+1)-j)-k), 0.))
    return float(values[0])


def _regression(y, regressors):
    """Direct normal equations and closed-form Gaussian likelihood."""
    y, x = np.asarray(y, float), np.asarray(regressors, float)
    if x.ndim == 1:
        x = x[:, None]
    design = np.column_stack([np.ones(len(y)), x])
    cross = design.T @ design
    beta = np.linalg.solve(cross, design.T @ y)
    residual = y-design @ beta
    sse = float(residual @ residual)
    n, p = design.shape
    standard_error = np.sqrt(np.diag(np.linalg.inv(cross))*sse/(n-p))
    ll = -n/2*(np.log(2*np.pi)+1+np.log(sse/n))
    k = p+1
    return dict(beta=beta, se=standard_error, t=beta/standard_error, residual=residual,
                n=n, k=k, rsquared=1-sse/np.sum((y-y.mean())**2), ll=ll,
                AICc=-2*ll+2*k+2*k*(k+1)/(n-k-1), BIC=-2*ll+k*np.log(n))


def _convex_tangency_qp(excess, covariance, cap=None):
    """Solve min y'Sigma y, excess'y=1, y>=0 by exhaustive active sets.

    If capped, y_PUTW <= cap*sum(y). Normalizing y by its sum recovers budget
    weights. Four assets permit checking every nonnegative face and both cap
    states exactly, instead of using the production Sharpe-ratio optimizer.
    """
    e, sigma = np.asarray(excess), np.asarray(covariance)
    candidates = []
    for count in range(1, len(e)+1):
        for free in itertools.combinations(range(len(e)), count):
            indices = np.array(free)
            sub = sigma[np.ix_(indices, indices)]
            for active_cap in ([False, True] if cap is not None else [False]):
                rows, rhs = [e[indices]], [1.]
                if active_cap:
                    rows.append(np.full(len(e), cap)[indices]-(indices == len(e)-1))
                    rhs.append(0.)
                a, b = np.asarray(rows), np.asarray(rhs)
                if np.linalg.matrix_rank(a) != len(rows):
                    continue
                inv_a = np.linalg.solve(sub, a.T)
                try:
                    reduced = inv_a @ np.linalg.solve(a @ inv_a, b)
                except np.linalg.LinAlgError:
                    continue
                y = np.zeros(len(e)); y[indices] = reduced
                if np.min(y) < -1e-8 or abs(e @ y-1) > 1e-8:
                    continue
                if cap is not None and y[-1]-cap*y.sum() > 1e-8:
                    continue
                candidates.append((float(y @ sigma @ y), y))
    if not candidates:
        raise AssertionError("No feasible active-set tangency QP solution")
    objective, y = min(candidates, key=lambda candidate: candidate[0])
    return y/y.sum(), objective


def _course_risk(pnl, alpha):
    x = np.sort(np.asarray(pnl, float))
    mass = len(x)*alpha
    low, high = max(1, int(np.floor(mass))), max(1, int(np.ceil(mass)))
    cutoff = .5*(x[low-1]+x[high-1])
    return -cutoff, -x[x <= cutoff].mean()


def _normal_risk(mean, sd, alpha):
    z = stats.norm.ppf(alpha)
    return -mean-sd*z, -mean+sd*stats.norm.pdf(z)/alpha


def _raw_factors(path):
    """Independent csv.DictReader parser; convert source percentages once."""
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            row = {str(name).strip(): value for name, value in row.items() if name is not None}
            stamp = str(row.get("Date", "")).strip()
            if len(stamp) != 8 or not stamp.isdigit():
                continue
            rows.append(dict(Date=pd.Timestamp(date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:]))),
                             **{name: float(value)/100 for name, value in row.items() if name != "Date"}))
    frame = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    if frame.Date.duplicated().any():
        raise AssertionError("Duplicate raw factor dates")
    return frame


def verify(data_dir=ROOT/"data", results_dir=ROOT/"results", audit_simulations=100000):
    data, results = Path(data_dir), Path(results_dir)
    checks = []
    started = time.perf_counter()

    def check(name, passed, details=None):
        item = dict(check=name, passed=bool(passed))
        if details is not None:
            item["details"] = details
        checks.append(item)
        if not passed:
            print("FAIL", name, details if details is not None else "")

    def close(name, actual, expected, atol=1e-10, rtol=1e-8):
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        finite = np.isfinite(a).all() and np.isfinite(b).all()
        good = a.shape == b.shape and finite and np.allclose(a, b, atol=atol, rtol=rtol)
        difference = float(np.max(np.abs(a-b))) if a.shape == b.shape and a.size and finite else None
        check(name, good, dict(max_absolute_difference=difference, atol=atol, rtol=rtol))

    def read(name):
        return pd.read_csv(results/name)

    metadata = json.loads((results/"run_metadata.json").read_text())
    check("Input manifest contains every original CSV exactly once",
          set(metadata["inputs"]) == {path.name for path in data.iterdir() if path.suffix.lower() == ".csv"})
    for name, digest in metadata["inputs"].items():
        check(f"Input integrity: {name}", hashlib.sha256((data/name).read_bytes()).hexdigest() == digest)

    # Problem 1: bounds, all quote inversions, parity, and sensitivity units.
    begin = len(checks)
    try:
        raw = pd.read_csv(data/"problem1.csv")
        quoted = read("q1_quotes_iv.csv").set_index(["Type", "Strike"])
        parity = read("q1_parity.csv").set_index("Strike")
        s, r, q = 182.40, .0425, .006
        t = (date(2026, 12, 18)-date(2026, 11, 13)).days/365
        call = raw.loc[raw.Type == "Call"].set_index("Strike").sort_index()
        put = raw.loc[raw.Type == "Put"].set_index("Strike").sort_index()
        call_mid, put_mid = (call.Bid+call.Ask)/2, (put.Bid+put.Ask)/2
        theoretical = s*np.exp(-q*t)-call.index.to_numpy()*np.exp(-r*t)
        gap = call_mid.to_numpy()-put_mid.to_numpy()-theoretical
        band = (call.Ask-call.Bid+put.Ask-put.Bid).to_numpy()
        check("Q1 all raw call/put quote rows preserved", len(quoted) == len(raw) and not quoted.index.duplicated().any())
        source_quotes = raw.assign(Mid=(raw.Bid+raw.Ask)/2).set_index(["Type", "Strike"])
        close("Q1 published bid/mid/ask prices preserve the source quotes",
              quoted.loc[source_quotes.index, ["Bid", "Mid", "Ask"]], source_quotes[["Bid", "Mid", "Ask"]])
        close("Q1 midpoint parity residuals", parity.loc[call.index, "gap"], gap)
        close("Q1 complete call-plus-put spread band", parity.loc[call.index, "band"], band)
        check("Q1 parity flags equal absolute gap versus full spread band",
              np.array_equal(parity.loc[call.index, "flag"].to_numpy(bool), abs(gap) > band))
        missing = []
        vega_actual, vega_expected, cent_actual, cent_expected = [], [], [], []
        for row in raw.itertuples(index=False):
            published = quoted.loc[(row.Type, row.Strike)]
            is_call = row.Type == "Call"
            prepaid, strike_pv = s*np.exp(-q*t), row.Strike*np.exp(-r*t)
            lower = max(prepaid-strike_pv, 0.) if is_call else max(strike_pv-prepaid, 0.)
            upper = prepaid if is_call else strike_pv
            close(f"Q1 {row.Type} K={row.Strike:g} no-arbitrage bounds",
                  [published.lower_bound, published.upper_bound], [lower, upper])
            for source, value in [("bid", row.Bid), ("mid", (row.Bid+row.Ask)/2), ("ask", row.Ask)]:
                volatility = published[f"IV_{source}"]
                if value < lower-1e-8 or value > upper+1e-8 or value >= upper-1e-8:
                    missing.append((row.Type, row.Strike, source))
                    reason = ("below_no_arbitrage_lower_bound" if value < lower-1e-8 else
                              "above_no_arbitrage_upper_bound" if value > upper+1e-8 else "no_finite_volatility_at_upper_bound")
                    check(f"Q1 {row.Type} K={row.Strike:g} {source} correctly unavailable",
                          pd.isna(volatility) and published[f"status_{source}"] == reason)
                else:
                    value_again = _gbsm(s, row.Strike, t, r, volatility, is_call, q)
                    close(f"Q1 {row.Type} K={row.Strike:g} {source} independent repricing",
                          value_again, value, atol=5e-8, rtol=0.)
            vol = float(published.IV_mid)
            bump = min(1e-5, vol/10)
            fd_vega = (_gbsm(s, row.Strike, t, r, vol+bump, is_call, q)
                       -_gbsm(s, row.Strike, t, r, vol-bump, is_call, q))/(2*bump)
            vega_actual.append(published.vega_per_unit_sigma); vega_expected.append(fd_vega)
            cent_actual.extend([published.vega_per_vol_point, published.cent_dIV_linear, published.cent_dIV_vol_points])
            cent_expected.extend([fd_vega/100, .01/fd_vega, 1/fd_vega])
        summary = json.loads((results/"q1_summary.json").read_text())
        reported_missing = {(v["Type"], v["Strike"], v["source"]) for v in summary["missing_IV_roots"]}
        check("Q1 missing-IV inventory exactly matches independent bounds", set(missing) == reported_missing,
              dict(expected_missing_roots=len(missing)))
        close("Q1 vegas match independent price finite differences", vega_actual, vega_expected, atol=2e-7)
        close("Q1 per-vol-point and one-cent sensitivity units", cent_actual, cent_expected, atol=2e-7)
        translated = read("q1_parity_vol_translation.csv")
        for row in translated.itertuples(index=False):
            iv_call, iv_put = quoted.loc[("Call", row.Strike), "IV_mid"], quoted.loc[("Put", row.Strike), "IV_mid"]
            close(f"Q1 K={row.Strike:g} price-to-volatility translation",
                  [row.estimated_IV_gap, row.actual_IV_gap, row.actual_gap_vol_points],
                  [parity.loc[row.Strike, "gap"]/row.reference_vega, iv_call-iv_put, 100*(iv_call-iv_put)])
    except Exception as error:
        check("Q1 audit completed without schema/runtime error", False, f"{type(error).__name__}: {error}")
    print(f"Q1: {sum(c['passed'] for c in checks[begin:])}/{len(checks)-begin} checks passed")

    # Problem 2: independent scalar recursion, parity and complete sensitivities.
    begin = len(checks)
    try:
        prices = read("q2_prices.csv").set_index("option")
        summary = json.loads((results/"q2_summary.json").read_text())
        s, k, t, r, vol = 48.5, 50., .5, .045, .30
        dividend_times, amounts = np.array([.1, .4]), np.array([.25, 1.5])
        pv = float(amounts @ np.exp(-r*dividend_times))
        n = int(summary["tree_n"])
        conditions = read("q2_dividend_conditions.csv")
        hurdles = k*(1-np.exp(-r*(t-dividend_times)))
        close("Q2 dividend-interest condition values", conditions.strike_interest, hurdles)
        check("Q2 dividend conditions use the stated inequality", np.array_equal(conditions.condition_holds, amounts > hurdles))
        for is_call, kind in [(True, "Call"), (False, "Put")]:
            close(f"Q2 no-dividend European {kind} independent GBSM",
                  prices.loc["European"+kind, "no_dividend_value"], _gbsm(s, k, t, r, vol, is_call))
            close(f"Q2 PV-dividend European {kind} approximation",
                  prices.loc["European"+kind, "pv_dividend_gbsm"], _gbsm(s-pv, k, t, r, vol, is_call))
            close(f"Q2 no-dividend American {kind} independent one-dimensional CRR",
                  prices.loc["American"+kind, "no_dividend_value"], _ordinary_tree(s, k, t, r, vol, n, is_call, True))
        close("Q2 dividend European put-call parity",
              prices.loc["EuropeanCall", "with_dividend_value"]-prices.loc["EuropeanPut", "with_dividend_value"],
              s-pv-k*np.exp(-r*t), atol=1e-8, rtol=0.)
        premiums = read("q2_exercise_premiums.csv")
        for row in premiums.itertuples(index=False):
            close(f"Q2 {row.kind} {row.dividends} matched-lattice exercise premium",
                  row.early_exercise_premium, row.american_value-row.european_value_same_tree)
            check(f"Q2 {row.kind} {row.dividends} nonnegative exercise premium", row.early_exercise_premium >= -1e-10)
        convergence = read("q2_convergence.csv").set_index("n")
        check("Q2 convergence includes neighbouring feasible N=100 and105", {100, 105}.issubset(convergence.index))
        check("Q2 every convergence grid hits both dividend dates", np.all(convergence.index.to_numpy() % 5 == 0))
        for reference_n in [int(convergence.index.min()), 100, 105]:
            events = [(int(round(.1/(t/reference_n))), .25), (int(round(.4/(t/reference_n))), 1.5)]
            for is_call, kind in [(True, "call"), (False, "put")]:
                independent = _cash_tree(s, k, t, r, vol, reference_n, is_call, True, events)
                close(f"Q2 American {kind} N={reference_n} independent unbatched cash-tree recursion",
                      convergence.loc[reference_n, f"american_{kind}_with_dividend"], independent, atol=2e-10)
        greeks = read("q2_finite_difference_greeks.csv")
        required = {(option, h) for option in ("AmericanCall", "AmericanPut") for h in (1., .1, .01, .001)}
        check("Q2 all eight requested fixed-grid Greek rows present",
              required.issubset(set(zip(greeks.option, greeks.bump))) and np.isfinite(greeks[["delta", "gamma"]]).all().all())
        sensitivity = read("q2_dividend_sensitivities.csv")
        main = sensitivity.loc[sensitivity.bump == .001].set_index(["option", "dividend_number"])
        expected = {(option, i) for option in prices.index for i in (1, 2)}
        check("Q2 four options times two dividend sensitivities present", set(main.index) == expected)
        for i, when in enumerate(dividend_times, 1):
            close(f"Q2 dividend {i} European derivative parity",
                  main.loc[("EuropeanCall", i), "dV_dDividend"]-main.loc[("EuropeanPut", i), "dV_dDividend"],
                  -np.exp(-r*when), atol=2e-7, rtol=0.)
        boundary = float(summary["second_dividend_exercise_boundary"])
        tail_n = int(round(n*(t-.4)/t))
        close("Q2 tree exercise-boundary independent continuation identity", boundary-k,
              _ordinary_tree(boundary-1.5, k, t-.4, r, vol, tail_n, True, False), atol=1e-9)
        reference = float(summary["second_dividend_continuous_boundary_reference"])
        close("Q2 continuous boundary independent GBSM identity", reference-k,
              _gbsm(reference-1.5, k, t-.4, r, vol, True), atol=1e-9)
    except Exception as error:
        check("Q2 audit completed without schema/runtime error", False, f"{type(error).__name__}: {error}")
    print(f"Q2: {sum(c['passed'] for c in checks[begin:])}/{len(checks)-begin} checks passed")

    # Problem 3: spot Greeks/curves and fresh, statistically compared scenarios.
    begin = len(checks)
    try:
        source = pd.read_csv(data/"problem3.csv")
        positions = pd.read_csv(data/"problem3_portfolios.csv")
        sample = source.ABC.to_numpy()[1:]/source.ABC.to_numpy()[:-1]-1
        spot, r, horizon, days = float(source.ABC.iloc[-1]), .0425, 10, 255
        book_names = list(positions.Portfolio.unique())
        books = read("q3_book_greeks.csv").set_index("Portfolio")
        fits = read("q3_distribution_fits.csv").set_index("family")
        full = read("q3_full_revaluation_risk.csv").set_index(["family", "Portfolio", "alpha"])
        mc = read("q3_monte_carlo_errors.csv").set_index(["family", "Portfolio", "alpha"])
        means = read("q3_mean_pnl_checks.csv").set_index(["family", "Portfolio"])
        terminal_checks = read("q3_terminal_price_checks.csv").set_index("family")
        summary = json.loads((results/"q3_summary.json").read_text())

        def book_value(name, spots, elapsed=0.):
            value = np.zeros_like(np.asarray(spots, float))
            for row in positions.loc[positions.Portfolio == name].itertuples(index=False):
                unit = spots if row.Type == "Stock" else _gbsm(spots, row.Strike, max((row.DaysToExpiry-elapsed)/days, 0.),
                                                               r, row.ImpliedVol, row.Type == "Call")
                value = value+row.Holding*unit
            return value

        close("Q3 raw arithmetic returns reproduced", read("q3_returns.csv").arithmetic_return, sample)
        raw_moments = read("q3_moments.csv").iloc[0]
        centered = sample-sample.mean(); m2 = np.mean(centered**2)
        close("Q3 return moments independently calculated",
              raw_moments[["mean", "variance", "sd", "skewness", "excess_kurtosis"]],
              [sample.mean(), sample.var(ddof=1), sample.std(ddof=1), np.mean(centered**3)/m2**1.5, np.mean(centered**4)/m2**2-3])
        for family, row in fits.iterrows():
            logpdf = stats.norm.logpdf(sample, row.mu, row.sigma) if family == "normal" else stats.t.logpdf(sample, row.nu, loc=row.mu, scale=row.scale)
            count = 2 if family == "normal" else 3
            ll = logpdf.sum(); aicc = -2*ll+2*count+2*count*(count+1)/(len(sample)-count-1)
            close(f"Q3 {family} likelihood and information criteria", [row.ll, row.AICc, row.BIC], [ll, aicc, -2*ll+count*np.log(len(sample))])
        close("Q3 Normal MLE mean and scale", fits.loc["normal", ["mu", "sigma"]], [sample.mean(), sample.std(ddof=0)])
        parameter_checks = summary["parameters"]
        check("Q3 forecast/time/contract conventions match the question",
              parameter_checks["return_forecast_mean"] == 0 and parameter_checks["fixed_IV"]
              and parameter_checks["option_contract_multiplier"] == 1 and parameter_checks["horizon_days"] == horizon
              and parameter_checks["trading_days"] == days and parameter_checks["paths"] == metadata["simulations"])
        check("Q3 both distributions and all book/tail rows present",
              set(full.index) == {(f, b, a) for f in ("normal", "t") for b in book_names for a in (.05, .01)} and len(full) == 4*len(book_names))
        check("Q3 finite complete VaR/ES and ES at least VaR",
              np.isfinite(full[["mean_PnL", "VaR", "ES"]]).all().all() and (full.ES >= full.VaR-1e-10).all())
        check("Q3 the1percent tail has at least the5percent VaR and ES",
              all((full.loc[(family, name, .01), ["VaR", "ES"]]
                   >= full.loc[(family, name, .05), ["VaR", "ES"]]-1e-10).all()
                  for family in ("normal", "t") for name in book_names))
        check("Q3 MCSE rows complete, positive and finite",
              set(mc.index) == set(full.index) and np.isfinite(mc).all().all() and (mc > 0).all().all())
        check("Q3 MCSE less than10 percent of reported risk scale",
              (mc.VaR_MCSE < .1*full.VaR).all() and (mc.ES_MCSE < .1*full.ES).all())
        for name in book_names:
            base = float(book_value(name, spot))
            h = .01
            delta = float((book_value(name, spot+h)-book_value(name, spot-h))/(2*h))
            gamma = float((book_value(name, spot+h)-2*base+book_value(name, spot-h))/h**2)
            time_bump = 1e-6
            theta = float((book_value(name, spot, time_bump*days)-book_value(name, spot, -time_bump*days))/(2*time_bump))
            close(f"Q3 {name} independent portfolio price", books.loc[name, "price"], base, atol=2e-8)
            close(f"Q3 {name} delta/gamma from independent price differences",
                  books.loc[name, ["delta", "gamma"]], [delta, gamma], atol=3e-5, rtol=3e-6)
            close(f"Q3 {name} theta from independent time difference", books.loc[name, "theta"], theta, atol=2e-5, rtol=2e-7)
        dn = read("q3_delta_normal_risk.csv").set_index(["Portfolio", "alpha"])
        flat = read("q3_flat_price_pnl.csv").set_index("Portfolio")
        curves = read("q3_pnl_curves.csv")
        for name in book_names:
            for alpha in (.05, .01):
                sd = abs(books.loc[name, "delta"])*spot*sample.std(ddof=1)*np.sqrt(horizon)
                var, es = _normal_risk(0., sd, alpha)
                close(f"Q3 {name} alpha={alpha:g} delta-normal risk", dn.loc[(name, alpha), ["sd_PnL", "VaR", "ES"]], [sd, var, es])
            close(f"Q3 {name} flat-price expiry-reduced P&L", flat.loc[name, "flat_price_PnL"],
                  book_value(name, spot, horizon)-book_value(name, spot), atol=2e-8)
            selected_curves = curves.loc[curves.Portfolio == name]
            points = selected_curves.terminal_price.to_numpy()
            ds = points-spot
            expected_curves = np.column_stack([
                book_value(name, points, horizon)-book_value(name, spot), books.loc[name, "delta"]*ds,
                books.loc[name, "delta"]*ds+.5*books.loc[name, "gamma"]*ds**2+books.loc[name, "theta"]*horizon/days,
            ])
            close(f"Q3 {name} full/delta/second-order P&L curves",
                  selected_curves[["full_PnL", "delta_only_PnL", "delta_gamma_theta_PnL"]], expected_curves, atol=2e-8)
        approximate = read("q3_approximation_risk.csv")
        check("Q3 all48 approximation-distribution-book-tail rows present",
              len(approximate) == 3*len(full) and set(zip(approximate.family, approximate.Portfolio, approximate.approximation, approximate.alpha))
              == {(f, b, label, a) for f in ("normal", "t") for b in book_names for label in ("delta_only", "delta_gamma", "delta_gamma_theta") for a in (.05, .01)})
        check("Q3 no invalid arithmetic-return paths and matching simulation counts",
              set(terminal_checks.index) == {"normal", "t"} and (terminal_checks.invalid_paths == 0).all()
              and (terminal_checks.paths == metadata["simulations"]).all() and (terminal_checks.paths > 0).all())
        rng = np.random.default_rng(int(metadata["seed"])+104729)
        uniforms = np.clip(rng.random((audit_simulations, horizon)), np.nextafter(0., 1.), np.nextafter(1., 0.))
        for family in ("normal", "t"):
            row = fits.loc[family]
            daily_variance = row.sigma**2 if family == "normal" else row.scale**2*row.nu/(row.nu-2)
            reference = terminal_checks.loc[family]
            theoretical_variance_spot = spot*spot*((1+daily_variance)**horizon-1)
            terminal_se = np.sqrt(theoretical_variance_spot/reference.paths)
            check(f"Q3 {family} published finite-draw terminal mean statistically consistent with zero return",
                  abs(reference.mean_terminal_price-spot) < 7*terminal_se,
                  dict(z=float((reference.mean_terminal_price-spot)/terminal_se)))
            check(f"Q3 {family} published simulated daily mean statistically consistent with zero",
                  abs(reference.daily_simulated_mean) < 7*np.sqrt(daily_variance/(reference.paths*horizon)))
            close(f"Q3 {family} terminal-price standard deviation scale", reference.terminal_price_sd,
                  np.sqrt(theoretical_variance_spot), atol=0., rtol=.05)
            daily = row.sigma*stats.norm.ppf(uniforms) if family == "normal" else row.scale*stats.t.ppf(uniforms, row.nu)
            check(f"Q3 {family} independent audit paths are admissible", not np.any(daily <= -1))
            terminal = spot*np.prod(1+daily, axis=1)
            for name in book_names:
                pnl = book_value(name, terminal, horizon)-book_value(name, spot)
                pub = full.loc[(family, name, .05)]
                se = float(pnl.std(ddof=1)/np.sqrt(audit_simulations))
                combined_mean_se = np.hypot(se, mc.loc[(family, name, .05), "mean_PnL_MCSE"])
                check(f"Q3 {family} {name} independent mean P&L within combined7MCSE",
                      abs(pnl.mean()-pub.mean_PnL) < 7*combined_mean_se)
                close(f"Q3 {family} {name} published mean MCSE consistency",
                      mc.loc[(family, name, .05), "mean_PnL_MCSE"], means.loc[(family, name), "empirical_mean_SE"])
                expected_second_order = books.loc[name, "theta"]*horizon/days+.5*books.loc[name, "gamma"]*theoretical_variance_spot
                close(f"Q3 {family} {name} theoretical second-order mean identity",
                      means.loc[(family, name), "second_order_expected_PnL"], expected_second_order)
                for alpha in (.05, .01):
                    var, es = _course_risk(pnl, alpha)
                    chunks = [_course_risk(batch, alpha) for batch in np.array_split(pnl, 20)]
                    independent_se = np.std(chunks, axis=0, ddof=1)/np.sqrt(20)
                    published = full.loc[(family, name, alpha)]
                    published_se = mc.loc[(family, name, alpha)]
                    check(f"Q3 {family} {name} alpha={alpha:g} independent VaR within combined7MCSE",
                          abs(var-published.VaR) < 7*np.hypot(independent_se[0], published_se.VaR_MCSE))
                    check(f"Q3 {family} {name} alpha={alpha:g} independent ES within combined7MCSE",
                          abs(es-published.ES) < 7*np.hypot(independent_se[1], published_se.ES_MCSE))
    except Exception as error:
        check("Q3 audit completed without schema/runtime error", False, f"{type(error).__name__}: {error}")
    print(f"Q3: {sum(c['passed'] for c in checks[begin:])}/{len(checks)-begin} checks passed")

    # Problem 4: convex-QP tangencies and direct sample-risk/regression checks.
    begin = len(checks)
    try:
        raw = pd.read_csv(data/"problem4.csv")
        x = raw[ASSETS].to_numpy()
        check("Q4 source has240 unique complete monthly observations",
              x.shape == (240, 4) and not raw.Month.duplicated().any() and np.isfinite(x).all())
        mean, covariance = x.mean(axis=0), np.cov(x, rowvar=False, ddof=1)
        rf = 1.03**(1/12)-1
        table = read("q4_portfolios.csv").set_index("portfolio")
        excess = mean-rf
        unrestricted = np.linalg.solve(covariance, excess); unrestricted /= unrestricted.sum()
        close("Q4 unrestricted tangency normal equations", table.loc["Unconstrained", ASSETS], unrestricted, atol=2e-8)
        for label, cap in [("LongOnly", None), ("PUTW20", .2)]:
            independent_weights, objective = _convex_tangency_qp(excess, covariance, cap)
            close(f"Q4 {label} independent convex active-set QP tangency",
                  table.loc[label, ASSETS], independent_weights, atol=3e-6, rtol=2e-5)
            check(f"Q4 {label} budget/nonnegative/cap feasibility",
                  abs(table.loc[label, ASSETS].sum()-1) < 1e-9 and table.loc[label, ASSETS].min() >= -1e-9
                  and (cap is None or table.loc[label, "PUTW"] <= cap+1e-9),
                  dict(independent_QP_variance=objective))
        close("Q4 covariance agrees with original monthly returns", read("q4_covariance.csv").iloc[:, 1:], covariance)
        moments = read("q4_asset_moments.csv").set_index("asset")
        for j, asset in enumerate(ASSETS):
            values = x[:, j]; central = values-values.mean(); second = np.mean(central**2)
            expected = [values.mean(), values.var(ddof=1), values.std(ddof=1), np.mean(central**3)/second**1.5,
                        np.mean(central**4)/second**2-3, values.std(ddof=1)/np.sqrt(len(values)),
                        np.sqrt(12)*(values.mean()-rf)/values.std(ddof=1)]
            close(f"Q4 {asset} moments and mean standard error",
                  moments.loc[asset, ["mean", "variance", "sd", "skewness", "excess_kurtosis", "mean_standard_error", "annual_sharpe"]], expected)
        risk = read("q4_portfolio_risk.csv").set_index(["portfolio", "alpha"])
        for label, portfolio in table.iterrows():
            weights = portfolio[ASSETS].to_numpy(float)
            pnl = x @ weights
            close(f"Q4 {label} portfolio moments and annualized Sharpe",
                  portfolio[["mean", "sd", "annual_sharpe"]],
                  [pnl.mean(), pnl.std(ddof=1), np.sqrt(12)*(pnl.mean()-rf)/pnl.std(ddof=1)])
            for alpha in (.05, .01):
                hist = _course_risk(pnl, alpha); normal = _normal_risk(pnl.mean(), pnl.std(ddof=1), alpha)
                close(f"Q4 {label} alpha={alpha:g} original-return historical/Normal risks",
                      risk.loc[(label, alpha), ["historical_VaR", "historical_ES", "normal_VaR", "normal_ES"]], [*hist, *normal])
        regressions = read("q4_capm.csv").set_index("sample")
        market, target = x[:, 0]-rf, x[:, 3]-rf
        for label, mask in [("AllMonths", np.ones(len(x), bool)), ("EQExcessBelowMinus5pct", market < -.05)]:
            fitted = _regression(target[mask], market[mask])
            close(f"Q4 {label} independent CAPM and downside-beta regression",
                  regressions.loc[label, ["n", "alpha", "alpha_t", "beta", "rsquared", "aicc"]],
                  [mask.sum(), fitted["beta"][0], fitted["t"][0], fitted["beta"][1], fitted["rsquared"], fitted["AICc"]])
        frontier = read("q4_frontier.csv")
        fw = frontier[ASSETS].to_numpy()
        close("Q4 frontier means and volatility trace original inputs",
              frontier[["mean", "sd"]], np.column_stack([fw @ mean, np.sqrt(np.einsum("ij,jk,ik->i", fw, covariance, fw))]))
        check("Q4 long-only frontier budget and bounds", np.allclose(fw.sum(axis=1), 1) and (fw >= -1e-9).all() and (fw <= 1+1e-9).all())
    except Exception as error:
        check("Q4 audit completed without schema/runtime error", False, f"{type(error).__name__}: {error}")
    print(f"Q4: {sum(c['passed'] for c in checks[begin:])}/{len(checks)-begin} checks passed")

    # Problem 5: raw factors, date joins, six normal-equation regressions and stress.
    begin = len(checks)
    try:
        funds = pd.read_csv(data/"problem5.csv", parse_dates=["Date"])
        ff = _raw_factors(data/"F-F_Research_Data_Factors_daily.CSV")
        mom = _raw_factors(data/"F-F_Momentum_Factor_daily.CSV")
        factors = ff.merge(mom, on="Date", validate="one_to_one")
        aligned = funds.merge(factors, on="Date", validate="one_to_one").sort_values("Date").reset_index(drop=True)
        published = pd.read_csv(results/"q5_aligned_sample.csv", parse_dates=["Date"])
        check("Q5 all fund dates aligned one-to-one without omissions",
              len(aligned) == len(funds) == len(published) and np.array_equal(aligned.Date, published.Date))
        close("Q5 raw source factor/RF percent conversion and fund decimals preserved",
              published.drop(columns="Date")[aligned.columns.drop("Date")], aligned.drop(columns="Date"), atol=1e-14)
        coefficients = read("q5_regression_coefficients.csv").set_index(["fund", "model", "term"]).sort_index()
        scores = read("q5_model_comparison.csv").set_index(["fund", "model"]).sort_index()
        check("Q5 exactly six regressions and all22 coefficient rows",
              set(scores.index) == {(fund, model) for fund in ("FUND1", "FUND2") for model in MODELS}
              and len(coefficients) == 2*sum(len(names)+1 for names in MODELS.values()))
        ref = {}
        for fund in ("FUND1", "FUND2"):
            ref[fund] = {}
            for model, factor_names in MODELS.items():
                fit = _regression(aligned[fund]-aligned.RF, aligned[factor_names])
                ref[fund][model] = fit
                labels = ["alpha", *factor_names]
                estimate = coefficients.loc[(fund, model)].loc[labels]
                close(f"Q5 {fund} {model} coefficients from independent normal equations", estimate.coefficient, fit["beta"], atol=1e-10)
                close(f"Q5 {fund} {model} conventional OLS standard errors", estimate.standard_error, fit["se"], atol=1e-12)
                close(f"Q5 {fund} {model} coefficient t statistics", estimate.t_statistic, fit["t"], atol=2e-8)
                close(f"Q5 {fund} {model} likelihood, R2 and parameter-counted criteria",
                      scores.loc[(fund, model), ["n", "k", "ll", "rsquared", "AICc", "BIC"]],
                      [fit["n"], fit["k"], fit["ll"], fit["rsquared"], fit["AICc"], fit["BIC"]])
        correlations = read("q5_capm_residual_correlations.csv").set_index(["fund", "factor"])
        for fund in ref:
            for factor in ("SMB", "HML", "Mom"):
                close(f"Q5 {fund} CAPM residual correlation with {factor}", correlations.loc[(fund, factor), "residual_correlation"],
                      np.corrcoef(ref[fund]["CAPM"]["residual"], aligned[factor])[0, 1])
        summary = json.loads((results/"q5_summary.json").read_text())
        end = factors.Date.max(); start = end-pd.DateOffset(years=10)
        window = factors.loc[(factors.Date > start) & (factors.Date <= end)]
        check("Q5 ten-year window anchored to factor file, not today or fund sample",
              summary["factor_file_end"] == str(end.date()) and summary["forecast_window_start_exclusive"] == str(start.date())
              and summary["forecast_first_date"] == str(window.Date.min().date())
              and summary["forecast_last_date"] == str(window.Date.max().date()) and summary["forecast_n"] == len(window))
        averages = window[[*FACTORS, "RF"]].mean()
        premia = read("q5_recent_factor_premia.csv").set_index("factor")
        annual_days = int(summary["annual_days"])
        close("Q5 recent daily and annual arithmetic factor premia", premia.loc[averages.index, ["mean_daily", "annual_arithmetic"]],
              np.column_stack([averages, annual_days*averages]))
        expected = read("q5_expected_returns.csv").set_index("fund")
        initial = read("q5_initial_statistics.csv").set_index("fund")
        stress = read("q5_momentum_stress.csv").set_index("fund")
        worst = mom.loc[mom.Mom.idxmin()]
        for fund in ref:
            alpha, beta = ref[fund]["Carhart4"]["beta"][0], ref[fund]["Carhart4"]["beta"][1:]
            daily_expected = averages.RF+beta @ averages[FACTORS].to_numpy()
            raw_return = aligned[fund].to_numpy()
            excess_return = raw_return-aligned.RF.to_numpy()
            close(f"Q5 {fund} initial excess-return/volatility/Sharpe statistics",
                  initial.loc[fund, ["mean_daily_return", "mean_daily_excess", "daily_volatility", "daily_excess_volatility", "daily_sharpe", "annual_sharpe", "capm_alpha_daily", "capm_alpha_t"]],
                  [raw_return.mean(), excess_return.mean(), raw_return.std(ddof=1), excess_return.std(ddof=1),
                   excess_return.mean()/excess_return.std(ddof=1), np.sqrt(annual_days)*excess_return.mean()/excess_return.std(ddof=1),
                   ref[fund]["CAPM"]["beta"][0], ref[fund]["CAPM"]["t"][0]])
            close(f"Q5 {fund} annual forecasts with/without alpha and realized return",
                  expected.loc[fund, ["expected_annual_without_alpha", "expected_annual_with_alpha", "annual_fitted_alpha", "realized_annual_arithmetic", "realized_annual_geometric"]],
                  [annual_days*daily_expected, annual_days*(daily_expected+alpha), annual_days*alpha,
                   annual_days*raw_return.mean(), np.expm1(np.log1p(raw_return).sum()*annual_days/len(raw_return))])
            shock = beta[-1]*worst.Mom
            check(f"Q5 {fund} worst momentum date searched over full file", stress.loc[fund, "worst_momentum_date"] == str(worst.Date.date()))
            close(f"Q5 {fund} momentum-only dollar/sigma stress",
                  stress.loc[fund, ["worst_momentum_return", "momentum_beta", "incremental_fund_return", "loss_dollars", "loss_in_daily_sigmas"]],
                  [worst.Mom, beta[-1], shock, -10_000_000*shock, -shock/raw_return.std(ddof=1)])
        contributions = read("q5_alpha_attribution.csv")
        sample_premia = aligned[FACTORS].mean()
        for row in contributions.itertuples(index=False):
            beta_before = dict(zip(MODELS[row.earlier_model], ref[row.fund][row.earlier_model]["beta"][1:]))
            beta_after = dict(zip(MODELS[row.later_model], ref[row.fund][row.later_model]["beta"][1:]))
            delta = beta_after[row.factor]-beta_before.get(row.factor, 0.)
            contribution = delta*sample_premia[row.factor]
            close(f"Q5 {row.fund} {row.earlier_model}->{row.later_model} {row.factor} alpha attribution",
                  [row.delta_beta, row.sample_daily_premium, row.contribution_to_alpha_drop_daily, row.contribution_to_alpha_drop_annual],
                  [delta, sample_premia[row.factor], contribution, annual_days*contribution], atol=1e-11)
        for (fund, earlier, later), group in contributions.groupby(["fund", "earlier_model", "later_model"]):
            close(f"Q5 {fund} {earlier}->{later} intercept attribution identity",
                  group.contribution_to_alpha_drop_daily.sum(), ref[fund][earlier]["beta"][0]-ref[fund][later]["beta"][0], atol=1e-12)
        attribution_checks = read("q5_alpha_attribution_checks.csv")
        expected_checks = []
        for row in attribution_checks.itertuples(index=False):
            actual_drop = ref[row.fund][row.earlier_model]["beta"][0]-ref[row.fund][row.later_model]["beta"][0]
            expected_checks.append([actual_drop, actual_drop, 0.])
        close("Q5 saved attribution-check table agrees with independent intercept identities",
              attribution_checks[["actual_alpha_drop_daily", "explained_alpha_drop_daily", "residual"]], expected_checks, atol=1e-12)
    except Exception as error:
        check("Q5 audit completed without schema/runtime error", False, f"{type(error).__name__}: {error}")
    print(f"Q5: {sum(c['passed'] for c in checks[begin:])}/{len(checks)-begin} checks passed")

    result = dict(passed=sum(item["passed"] for item in checks), total=len(checks),
                  elapsed_seconds=time.perf_counter()-started, independent_MC_paths=audit_simulations,
                  independent_MC_seed=int(metadata["seed"])+104729, checks=checks)
    (results/"verification.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"{result['passed']}/{result['total']} independent result checks passed in {result['elapsed_seconds']:.2f}s")
    if result["passed"] != result["total"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data")
    parser.add_argument("--results-dir", "--output-dir", type=Path, default=ROOT/"results")
    parser.add_argument("--audit-simulations", type=int, default=100000)
    args = parser.parse_args()
    if args.audit_simulations < 10000:
        parser.error("Use at least10000 independent audit paths")
    verify(args.data_dir, args.results_dir, args.audit_simulations)
