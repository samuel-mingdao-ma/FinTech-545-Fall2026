"""Problem 3(a-j): delta approximations and full ten-day book revaluation."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special, stats

from .common import fit_distribution, historical_risk, moments, normal_risk, write_json, write_predictions
from .pricing import greeks, price

RATE, TRADING_DAYS, HORIZON = .0425, 255, 10


def book_values(positions, spot, days_elapsed=0):
    """Value each position on one share; negative Holding is a short position."""
    spot = np.asarray(spot, float)
    if not np.isfinite(spot).all() or (spot <= 0).any():
        raise ValueError("Positive finite underlying prices required")
    values = {name: np.zeros_like(spot) for name in positions.Portfolio.unique()}
    for row in positions.itertuples(index=False):
        if row.Type == "Stock":
            value = spot
        elif row.Type in ("Call", "Put"):
            t = max((row.DaysToExpiry-days_elapsed)/TRADING_DAYS, 0.)
            value = price(spot, row.Strike, t, RATE, RATE, row.ImpliedVol, row.Type == "Call")
        else:
            raise ValueError(f"Unknown position type {row.Type}")
        values[row.Portfolio] = values[row.Portfolio]+row.Holding*value
    return values


def position_greeks(positions, spot):
    rows = []
    for row in positions.itertuples(index=False):
        if row.Type == "Stock":
            unit = dict(price=spot, delta=1., gamma=0., theta=0., vega=0.)
        else:
            unit = greeks(spot, row.Strike, row.DaysToExpiry/TRADING_DAYS, RATE, RATE,
                          row.ImpliedVol, row.Type == "Call")
        rows.append(dict(Portfolio=row.Portfolio, Type=row.Type, Holding=row.Holding,
                         Strike=None if row.Type == "Stock" else row.Strike,
                         **{name: row.Holding*value for name, value in unit.items()}))
    return pd.DataFrame(rows)


def run(data_dir, output_dir, figure_dir, n_sim=200000, seed=545, **kwargs):
    data_dir, out, figures = map(Path, (data_dir, output_dir, figure_dir))
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    if not isinstance(n_sim, int) or n_sim < 5000:
        raise ValueError("Use at least 5000 simulation paths")
    prices = pd.read_csv(data_dir / "problem3.csv")
    positions = pd.read_csv(data_dir / "problem3_portfolios.csv")
    if not np.isfinite(prices.ABC).all() or (prices.ABC <= 0).any():
        raise ValueError("Invalid ABC price history")
    s = float(prices.ABC.iloc[-1])
    returns = prices.ABC.pct_change().dropna().to_numpy()
    pd.DataFrame(dict(Date=prices.Date.iloc[1:].values, arithmetic_return=returns)).to_csv(
        out / "q3_returns.csv", index=False
    )

    # Problem 3(a/b): Greeks and first analytical draft precede return fitting/MC.
    position_table = position_greeks(positions, s)
    position_table.to_csv(out / "q3_position_greeks.csv", index=False)
    books = position_table.groupby("Portfolio", sort=False)[["price", "delta", "gamma", "theta", "vega"]].sum()
    books["dollar_delta"] = books.delta*s
    books["theta_per_trading_day"] = books.theta/TRADING_DAYS
    books.to_csv(out / "q3_book_greeks.csv")
    order = list(books.gamma.abs().sort_values(ascending=False).index)
    write_predictions(out / "q3_predictions.json", dict(
        a=dict(
            metric="Expected absolute dollar error, with nearly delta-neutral short-gamma books also having the largest relative error",
            predicted_order=order,
            reasoning="A delta-only model omits local curvature and time decay. Order the initial concern by absolute gamma. Local negative gamma suggests understated tail loss; a locally positive-gamma risk reversal can instead make the linear estimate conservative locally, but crossing its strikes can reverse that prediction in a far tail.",
            local_direction={name: ("understate from omitted short gamma" if row.gamma < 0 else
                                     "overstate locally from positive gamma; strike crossings may reverse it")
                             for name, row in books.iterrows()},
        ),
        b=dict(flat_price_sign={name: ("positive" if row.theta > 0 else "negative" if row.theta < 0 else "zero")
                                for name, row in books.iterrows()},
               expected_high_ES_VaR=[name for name in ("ShortStraddle", "HedgedShortCall") if name in books.index],
               reasoning="Theta suggests the flat-price sign; short convexity concentrates losses in unusually large moves, so the near-delta-neutral short-gamma books should have a non-Normal tail ratio."),
        theta_units="Dollar change per model year; divide by255 for one trading day",
    ))

    # Problem 3(c): raw-return moments and two MLE candidates; no sample demeaning.
    moment_values = moments(returns)
    pd.DataFrame([moment_values]).to_csv(out / "q3_moments.csv", index=False)
    fits = {family: fit_distribution(returns, family) for family in ("normal", "t")}
    fit_rows = [dict(family=family, **fit.parameters, ll=fit.ll, AICc=fit.aicc, BIC=fit.bic)
                for family, fit in fits.items()]
    pd.DataFrame(fit_rows).to_csv(out / "q3_distribution_fits.csv", index=False)
    selected = min(fits, key=lambda family: fits[family].aicc)
    if fits["t"].parameters["nu"] <= 1:
        raise ValueError("A fitted t with nu <= 1 cannot satisfy the specified zero finite expected return")

    # Problem 3(d): standard delta-normal method, zero mean, no theta drift.
    dn_rows = []
    for name, row in books.iterrows():
        sd_pnl = abs(row.delta)*s*moment_values["sd"]*np.sqrt(HORIZON)
        for alpha in (.05, .01):
            risk = normal_risk(0., sd_pnl, alpha)
            dn_rows.append(dict(Portfolio=name, alpha=alpha, mean_PnL=0., sd_PnL=sd_pnl,
                                VaR=risk["var"], ES=risk["es"]))
    dn = pd.DataFrame(dn_rows)
    dn.to_csv(out / "q3_delta_normal_risk.csv", index=False)

    current_values = book_values(positions, s)
    flat_values = book_values(positions, s, HORIZON)
    flat_table = pd.DataFrame([dict(Portfolio=name, current_value=float(current_values[name]),
                                    flat_price_PnL=float(flat_values[name]-current_values[name]),
                                    theta_linear_PnL=float(books.loc[name, "theta"]*HORIZON/TRADING_DAYS))
                               for name in books.index])
    flat_table.to_csv(out / "q3_flat_price_pnl.csv", index=False)

    # Problem 3(e): BOTH candidates, common uniforms, independent daily returns.
    # Replace fitted locations by zero, but retain each fitted scale/shape.
    uniforms = np.random.default_rng(seed).random((n_sim, HORIZON))
    uniforms = np.maximum(uniforms, np.nextafter(0., 1.))
    full_rows, approximate_rows, mc_rows, terminal_rows = [], [], [], []
    moment_checks = []
    simulated = {}
    for family, fitted in fits.items():
        if family == "normal":
            daily = fitted.parameters["sigma"]*special.ndtri(uniforms)
            daily_variance = fitted.parameters["sigma"]**2
        else:
            nu, scale = fitted.parameters["nu"], fitted.parameters["scale"]
            daily = scale*stats.t.ppf(uniforms, nu)
            daily_variance = scale*scale*nu/(nu-2) if nu > 2 else np.inf
        invalid = int(np.count_nonzero(np.any(daily <= -1, axis=1)))
        if invalid:
            raise ValueError(f"{family}: {invalid} paths contain arithmetic return <= -100%; do not silently truncate or discard them")
        terminal = s*np.prod(1+daily, axis=1)
        revalued = book_values(positions, terminal, HORIZON)
        ds = terminal-s
        simulated[family] = terminal
        terminal_rows.append(dict(family=family, paths=n_sim, invalid_paths=invalid,
                                  mean_terminal_price=float(terminal.mean()),
                                  terminal_price_sd=float(terminal.std(ddof=1)),
                                  daily_simulated_mean=float(daily.mean()),
                                  daily_simulated_sd=float(daily.std(ddof=1))))
        for name, row in books.iterrows():
            full = revalued[name]-current_values[name]
            delta_only = row.delta*ds
            delta_gamma = delta_only+.5*row.gamma*ds*ds
            delta_gamma_theta = delta_gamma+row.theta*HORIZON/TRADING_DAYS
            if np.isfinite(daily_variance):
                expected_variance_spot = s*s*((1+daily_variance)**HORIZON-1)
                approximate_expected = row.theta*HORIZON/TRADING_DAYS+.5*row.gamma*expected_variance_spot
            else:
                approximate_expected = None
            moment_checks.append(dict(family=family, Portfolio=name,
                                      empirical_mean_PnL=float(full.mean()),
                                      empirical_mean_SE=float(full.std(ddof=1)/np.sqrt(n_sim)),
                                      second_order_expected_PnL=approximate_expected,
                                      mean_delta_gamma_theta=float(delta_gamma_theta.mean())))
            for alpha in (.05, .01):
                risk = historical_risk(full, alpha)
                full_rows.append(dict(family=family, Portfolio=name, alpha=alpha,
                                      mean_PnL=float(full.mean()), VaR=risk["var"], ES=risk["es"],
                                      ES_VaR_ratio=risk["es"]/risk["var"] if risk["var"] > 0 else None))
                for label, approximation in (("delta_only", delta_only), ("delta_gamma", delta_gamma),
                                               ("delta_gamma_theta", delta_gamma_theta)):
                    ar = historical_risk(approximation, alpha)
                    approximate_rows.append(dict(family=family, Portfolio=name, approximation=label,
                                                 alpha=alpha, mean_PnL=float(approximation.mean()),
                                                 VaR=ar["var"], ES=ar["es"]))
                batch = [historical_risk(chunk, alpha) for chunk in np.array_split(full, 20)]
                mc_rows.append(dict(family=family, Portfolio=name, alpha=alpha,
                                    VaR_MCSE=float(np.std([v["var"] for v in batch], ddof=1)/np.sqrt(20)),
                                    ES_MCSE=float(np.std([v["es"] for v in batch], ddof=1)/np.sqrt(20)),
                                    mean_PnL_MCSE=float(full.std(ddof=1)/np.sqrt(n_sim))))
    full_table = pd.DataFrame(full_rows)
    approximate_table = pd.DataFrame(approximate_rows)
    full_table.to_csv(out / "q3_full_revaluation_risk.csv", index=False)
    approximate_table.to_csv(out / "q3_approximation_risk.csv", index=False)
    pd.DataFrame(mc_rows).to_csv(out / "q3_monte_carlo_errors.csv", index=False)
    pd.DataFrame(terminal_rows).to_csv(out / "q3_terminal_price_checks.csv", index=False)
    pd.DataFrame(moment_checks).to_csv(out / "q3_mean_pnl_checks.csv", index=False)

    # Problem 3(f): deterministic P&L curves versus terminal stock price.
    low = min(np.quantile(v, .001) for v in simulated.values())
    high = max(np.quantile(v, .999) for v in simulated.values())
    grid = np.linspace(max(.01*s, low*.98), high*1.02, 500)
    full_curves = book_values(positions, grid, HORIZON)
    curve_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), sharex=True)
    for ax, (name, row) in zip(axes.flat, books.iterrows()):
        pnl = full_curves[name]-current_values[name]
        dp = row.delta*(grid-s)
        dgt = dp+.5*row.gamma*(grid-s)**2+row.theta*HORIZON/TRADING_DAYS
        ax.plot(grid, pnl, color="#235789", lw=1.7, label="Full revaluation")
        ax.plot(grid, dp, color="#bd632f", ls="--", lw=1.3, label="Delta only")
        ax.plot(grid, dgt, color="#547a3d", ls=":", lw=1.3, label="Delta + gamma + theta")
        ax.axhline(0, color="grey", lw=.5)
        ax.axvline(s, color="grey", lw=.5)
        ax.set(title=name, ylabel="10-day P&L ($)", xlabel="ABC terminal price ($)")
        for i in range(len(grid)):
            curve_rows.append(dict(Portfolio=name, terminal_price=grid[i], full_PnL=pnl[i],
                                   delta_only_PnL=dp[i], delta_gamma_theta_PnL=dgt[i]))
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Q3: fixed-IV ten-day revaluation; central terminal-price range", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures / "q3_pnl_curves.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(curve_rows).to_csv(out / "q3_pnl_curves.csv", index=False)

    # Problem 3(g-i): numerical comparisons supporting reconciliation prose.
    comparisons = []
    for name in books.index:
        for alpha in (.05, .01):
            linear = dn.loc[(dn.Portfolio == name) & (dn.alpha == alpha)].iloc[0]
            normal = full_table.loc[(full_table.Portfolio == name) & (full_table.alpha == alpha) &
                                    (full_table.family == "normal")].iloc[0]
            student = full_table.loc[(full_table.Portfolio == name) & (full_table.alpha == alpha) &
                                     (full_table.family == "t")].iloc[0]
            comparisons.append(dict(Portfolio=name, alpha=alpha, delta_normal_VaR=linear.VaR,
                                    normal_full_VaR=normal.VaR, t_full_VaR=student.VaR,
                                    delta_normal_ES=linear.ES, normal_full_ES=normal.ES, t_full_ES=student.ES,
                                    normal_VaR_minus_delta_normal=normal.VaR-linear.VaR,
                                    t_minus_normal_VaR=student.VaR-normal.VaR,
                                    t_minus_normal_ES=student.ES-normal.ES))
    comparison_table = pd.DataFrame(comparisons)
    comparison_table.to_csv(out / "q3_method_comparison.csv", index=False)
    if len(full_table) != 16 or len(approximate_table) != 48 or len(dn) != 8:
        raise AssertionError("Missing required book/distribution/tail-level rows")
    if not np.isfinite(full_table[["mean_PnL", "VaR", "ES"]]).all().all():
        raise AssertionError("Nonfinite book risk result")
    if not (full_table.ES >= full_table.VaR-1e-10).all():
        raise AssertionError("ES below corresponding VaR")

    summary = dict(parameters=dict(spot=s, rate=RATE, trading_days=TRADING_DAYS, horizon_days=HORIZON,
                                   paths=n_sim, seed=seed, return_forecast_mean=0., fixed_IV=True,
                                   theta_units="dollars per model year", option_contract_multiplier=1),
                   moments=moment_values, distribution_fits=fit_rows, selected_family_AICc=selected,
                   book_greeks=books.reset_index().to_dict("records"),
                   delta_normal_risk=dn.to_dict("records"), full_revaluation_risk=full_rows,
                   approximation_risk=approximate_rows, flat_price_pnl=flat_table.to_dict("records"),
                   method_comparison=comparisons, mean_pnl_checks=moment_checks,
                   terminal_price_checks=terminal_rows, monte_carlo_errors=mc_rows,
                   figures=["q3_pnl_curves.png"],
                   conventions=dict(moments="sample variance; population standardized skewness/excess kurtosis",
                                    delta_normal_sigma="sample standard deviation",
                                    normal_fit_sigma="MLE standard deviation",
                                    historical_tail="course floor/ceil order average and threshold-mean ES",
                                    common_random_numbers=True,
                                    model_limit="Unbounded Normal/t arithmetic returns can theoretically be <=-100%; any observed such path raises rather than silently truncating"))
    write_json(out / "q3_summary.json", summary)
    return summary
