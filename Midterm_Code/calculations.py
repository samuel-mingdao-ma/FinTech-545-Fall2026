"""Numerical calculations for Problems 1, 2, and 3."""
import numpy as np
import pandas as pd
from scipy import interpolate, stats

from examkit import (
    fit_copulas, fit_distribution, fit_ols, historical_risk,
    repair_psd, returns, simulate_normal,
)
from examkit.validation import (
    validate_holdings, validate_missing_frame, validate_price_frame, validate_return_frame,
)

_NIG_INVERSE_CACHE = {}


def q1(data, out):
    """Pairwise correlation, marginal variances, PSD repair, and covariance."""
    x = validate_missing_frame(pd.read_csv(data / "problem1.csv"))
    names = list(x.columns)

    # Problem 1(a): use complete pairs separately for each correlation.
    r = x.corr().to_numpy()
    pd.DataFrame(r, columns=names, index=pd.Index(names, name="series")).to_csv(
        out / "q1a_pairwise_correlation.csv"
    )

    # Problem 1(b): each variance uses that series' full available history.
    variance = x.var(ddof=1)
    pd.DataFrame(dict(series=names, n=x.count().values, variance=variance.values)).to_csv(
        out / "q1b_variances.csv", index=False
    )

    # Problem 1(c): eigenvalues in ascending order.
    eig = np.linalg.eigvalsh(r)
    pd.DataFrame(dict(component=np.arange(1, len(eig) + 1), eigenvalue=eig)).to_csv(
        out / "q1c_eigenvalues.csv", index=False
    )

    # Problem 1(d): Higham's alternating projections with Dykstra correction.
    repaired = repair_psd(r, "higham") if eig.min() < -1e-10 else r.copy()
    pd.DataFrame(repaired, columns=names, index=pd.Index(names, name="series")).to_csv(
        out / "q1d_higham_correlation.csv"
    )
    pd.DataFrame(dict(component=np.arange(1, len(eig) + 1),
                      eigenvalue=np.linalg.eigvalsh(repaired))).to_csv(
        out / "q1d_higham_eigenvalues.csv", index=False
    )
    changes = []
    for i in range(len(names)):
        for j in range(i):
            changes.append(dict(
                series_i=names[i], series_j=names[j], original=r[i, j],
                repaired=repaired[i, j], change=repaired[i, j] - r[i, j],
                absolute_change=abs(repaired[i, j] - r[i, j]),
            ))
    pd.DataFrame(changes).sort_values("absolute_change", ascending=False).to_csv(
        out / "q1d_entry_changes.csv", index=False
    )

    # Problem 1(e): restore the full-history marginal variances.
    cov = repaired * np.sqrt(np.outer(variance, variance))
    pd.DataFrame(cov, columns=names, index=pd.Index(names, name="series")).to_csv(
        out / "q1e_covariance.csv"
    )
    for i in range(len(names)):
        for j in range(i):
            pair = x[[names[i], names[j]]].dropna().to_numpy()
            np.testing.assert_allclose(r[i, j], np.corrcoef(pair, rowvar=False)[0, 1], atol=1e-12)
    np.testing.assert_allclose(np.diag(cov), variance, atol=1e-12)
    assert np.linalg.eigvalsh(cov).min() > -1e-10


def q2(data, out):
    """Problem 2(a/b): OLS market model and Normal VaR/ES at 95% and 99%."""
    prices = pd.read_csv(data / "problem2.csv")
    r = returns(validate_price_frame(prices, required_assets=["SPY", "XYZ"]))
    fit = fit_ols(r.XYZ, r.SPY)
    intercept, beta = fit["coefficients"]
    errors = fit["residuals"]
    mu = intercept + beta * r.SPY.mean()
    residual_variance = errors.var(ddof=1)
    variance = beta**2 * r.SPY.var(ddof=1) + residual_variance
    sd = np.sqrt(variance)
    pd.DataFrame([dict(
        n=len(r), intercept=intercept, beta=beta, market_mean=r.SPY.mean(),
        market_variance=r.SPY.var(ddof=1), residual_variance=residual_variance,
        return_mean=mu, return_sd=sd, current_price=prices.XYZ.iloc[-1], shares=100,
    )]).to_csv(out / "q2_market_model.csv", index=False)

    rows = []
    for alpha in [.05, .01]:
        z = stats.norm.ppf(alpha)
        # Problem 2(a): positive-loss VaR in percentage points.
        var_percent = 100*(-mu-sd*z)
        # Problem 2(b): positive-loss ES in percentage points.
        es_percent = 100*(-mu+sd*stats.norm.pdf(z)/alpha)
        rows.append(dict(
            confidence=1-alpha, VaR_percent=var_percent, ES_percent=es_percent,
        ))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "q2_risk_percent.csv", index=False)

    sensitivity = []
    divisors = [
        ("n-1", errors.var(ddof=1)),
        ("regression_df_n-2", errors @ errors / (len(errors)-2)),
        ("MLE_n", errors.var(ddof=0)),
    ]
    for label, residual_var in divisors:
        vv = beta**2 * r.SPY.var(ddof=1) + residual_var
        for alpha in [.05, .01]:
            z = stats.norm.ppf(alpha)
            sensitivity.append(dict(
                residual_divisor=label, confidence=1-alpha,
                VaR_percent=100*(-mu-np.sqrt(vv)*z),
                ES_percent=100*(-mu+np.sqrt(vv)*stats.norm.pdf(z)/alpha),
            ))
    pd.DataFrame(sensitivity).to_csv(out / "q2_divisor_sensitivity.csv", index=False)
    np.testing.assert_allclose(variance, r.XYZ.var(ddof=1), rtol=1e-10)
    np.testing.assert_allclose(mu, r.XYZ.mean(), atol=1e-12)
    np.testing.assert_allclose(frame.VaR_percent, -100*stats.norm(mu, sd).ppf([.05, .01]))


def q3a(data, out):
    """Problem 3(a): fit all three candidates; output parameters and scores only."""
    prices = pd.read_csv(data / "problem3.csv")
    assets = [f"S{i}" for i in range(1, 11)]
    asset_prices = validate_price_frame(prices, required_assets=assets)
    r = validate_return_frame(returns(asset_prices))
    rows, fits = [], {}
    for stock in r:
        fits[stock] = {}
        for family in ["normal", "t", "nig"]:
            f = fit_distribution(r[stock], family, normal_sample_std=False)
            fits[stock][family] = f
            rows.append(dict(stock=stock, family=family, **f.parameters,
                             ll=f.ll, AICc=f.aicc, BIC=f.bic))
        print(stock, "candidate fits computed", flush=True)
    frame = pd.DataFrame(rows).fillna("")
    frame.to_csv(out / "q3a_all_fits.csv", index=False)
    comparison = pd.DataFrame(index=pd.Index(r.columns, name="stock"))
    for family in ["normal", "t", "nig"]:
        candidate = frame.loc[frame.family == family].set_index("stock")
        for criterion in ["ll", "AICc", "BIC"]:
            comparison[f"{family}_{criterion}"] = candidate[criterion]
    comparison.to_csv(out / "q3a_comparison.csv")
    r.to_csv(out / "q3_arithmetic_returns.csv", index=False)
    return prices, r, fits


def inverse_margin(f, u):
    """Normal/t quantiles and probability-checked NIG quantile interpolation."""
    u = np.clip(u, 1e-10, 1-1e-10)
    if f.name != "nig":
        return f.distribution.ppf(u), 0.
    key = tuple(sorted((name, float(value)) for name, value in f.parameters.items()))
    if key in _NIG_INVERSE_CACHE:
        fn, error = _NIG_INVERSE_CACHE[key]
        return fn(u), error
    grid = np.unique(np.r_[
        np.linspace(.0001, .9999, 5001), np.geomspace(1e-10, .001, 250),
        1-np.geomspace(1e-10, .001, 250),
    ])
    q = f.distribution.ppf(grid)
    if not np.isfinite(q).all():
        raise RuntimeError("Nonfinite NIG inverse CDF grid")
    fn = interpolate.PchipInterpolator(grid, q)
    check = np.unique(np.r_[
        np.linspace(.001, .999, 41), np.geomspace(1e-8, .001, 15),
        1-np.geomspace(1e-8, .001, 15),
    ])
    error = float(np.max(np.abs(f.distribution.cdf(fn(check))-check)))
    if error > 1e-5:
        raise RuntimeError(f"NIG inverse CDF error {error}")
    _NIG_INVERSE_CACHE[key] = (fn, error)
    return fn(u), error


def q3b(out, r, fits, choices):
    """Problem 3(b): fit both copulas using the supplied marginal choices."""
    if set(choices) != set(r.columns) or any(f not in ("normal", "t", "nig") for f in choices.values()):
        raise ValueError("Choices must give normal/t/nig for each stock S1 through S10")
    selected = [fits[s][choices[s]] for s in r]
    c = fit_copulas(r.to_numpy(), selected)
    pd.DataFrame([
        dict(copula="gaussian", nu="", **c["gaussian"]),
        dict(copula="t", nu=c["nu"], **c["t"]),
    ]).to_csv(out / "q3b_copula_scores.csv", index=False)
    pd.DataFrame([
        dict(copula="gaussian", nu="", ll=c["gaussian"]["ll"],
             AICc=c["gaussian"]["aicc"], BIC=c["gaussian"]["bic"]),
        dict(copula="t", nu=c["nu"], ll=c["t"]["ll"],
             AICc=c["t"]["aicc"], BIC=c["t"]["bic"]),
    ]).to_csv(out / "q3b_comparison.csv", index=False)
    pd.DataFrame(c["r"], columns=r.columns, index=pd.Index(r.columns, name="stock")).to_csv(
        out / "q3b_kendall_R.csv"
    )
    return c


def q3c(data, out, prices, r, fits, choices, n, seed, c, copula):
    """Problem 3(c): simulate only the supplied copula and marginal choices."""
    if copula not in ("gaussian", "t"):
        raise ValueError("Explicit copula required: gaussian or t")
    if set(choices) != set(r.columns) or any(f not in ("normal", "t", "nig") for f in choices.values()):
        raise ValueError("Choices must give normal/t/nig for every stock")
    selected = [fits[s][choices[s]] for s in r]
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 5000:
        raise ValueError("At least 5000 integer simulation draws required")
    holding = validate_holdings(pd.read_csv(data / "problem3_holdings.csv"), list(r.columns))
    values = prices[r.columns].iloc[-1].to_numpy() * holding.Holding.to_numpy()
    pd.DataFrame(dict(
        stock=r.columns, book=holding.Book, shares=holding.Holding,
        last_price=prices[r.columns].iloc[-1], current_value=values,
    )).to_csv(out / "q3c_positions.csv", index=False)
    rows, checks, marginal_checks, mc_rows = [], [], [], []
    nu = None if copula == "gaussian" else c["nu"]
    z = simulate_normal(c["r"], n, seed)
    if nu is None:
        u = stats.norm.cdf(z)
    else:
        z /= np.sqrt(np.random.default_rng(seed+1).chisquare(nu, n)/nu)[:, None]
        u = stats.t.cdf(z, nu)
    simulated = []
    for j, f in enumerate(selected):
        draw, error = inverse_margin(f, u[:, j])
        simulated.append(draw)
        checks.append(dict(copula=copula, stock=r.columns[j], cdf_interpolation_max_error=error))
    pnl = np.column_stack(simulated) * values
    groups = {s: pnl[:, j] for j, s in enumerate(r.columns)}
    groups.update({f"Book {b}": pnl[:, holding.Book.to_numpy() == b].sum(axis=1) for b in ["A", "B"]})
    groups["Total"] = pnl.sum(axis=1)
    np.testing.assert_allclose(groups["Book A"]+groups["Book B"], groups["Total"], atol=1e-8)
    for name, p in groups.items():
        risk = historical_risk(p, .05)
        rows.append(dict(copula=copula, position=name, VaR95_dollars=risk["var"], ES95_dollars=risk["es"]))
        batch_risk = [historical_risk(batch, .05) for batch in np.array_split(p, 20)]
        mc_rows.append(dict(
            copula=copula, position=name,
            VaR_MCSE=np.std([b["var"] for b in batch_risk], ddof=1)/np.sqrt(20),
            ES_MCSE=np.std([b["es"] for b in batch_risk], ddof=1)/np.sqrt(20),
        ))
    for j, f in enumerate(selected):
        q = f.distribution.ppf(.05)
        analytic = -values[j] * q
        se = abs(values[j])*np.sqrt(.05*.95/n)/f.distribution.pdf(q)
        observed = historical_risk(pnl[:, j], .05)["var"]
        marginal_checks.append(dict(
            copula=copula, stock=r.columns[j], analytic_VaR=analytic,
            simulated_VaR=observed, quantile_MCSE=se, z=(observed-analytic)/se,
        ))
    print(copula, "scenario simulated", n, "draws", flush=True)
    pd.DataFrame(rows).to_csv(out / "q3c_risk.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "q3c_nig_inverse_checks.csv", index=False)
    pd.DataFrame(marginal_checks).to_csv(out / "q3c_marginal_validation.csv", index=False)
    pd.DataFrame(mc_rows).to_csv(out / "q3c_monte_carlo_errors.csv", index=False)
