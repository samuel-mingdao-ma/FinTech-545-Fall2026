"""Problem 4: sample Sharpe optimization, tail risk, and regime-dependent beta."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

from .common import historical_risk, moments, normal_risk, ols, write_json, write_predictions

ASSETS = ["EQ", "BOND", "STK", "PUTW"]


def sharpe(weights, mean, covariance, risk_free):
    w = np.asarray(weights)
    variance = float(w @ covariance @ w)
    if variance <= 0:
        return np.nan
    return float(w @ (mean-risk_free)/np.sqrt(variance))


def _feasible_start(weights, upper):
    """Project a nonnegative seed onto the bounded budget simplex."""
    x = np.maximum(np.asarray(weights, float), 0)
    x = x/x.sum() if x.sum() else np.ones(len(x))/len(x)
    for _ in range(20):
        over = x > upper
        if not over.any():
            break
        x[over] = upper[over]
        available = np.maximum(upper-x, 0)
        shortfall = 1-x.sum()
        if shortfall <= 1e-14:
            break
        x += shortfall*available/available.sum()
    return x


def maximum_sharpe(mean, covariance, risk_free, upper=None, *, multistart=True, seed=545):
    """Normalized analytic unrestricted tangency, or bounded positive-Sharpe SLSQP.

    With positive excess return, Sharpe upper-level sets are convex; feasible
    multistart solutions and the first-order checks provide numerical checks.
    """
    mean, covariance = np.asarray(mean, float), np.asarray(covariance, float)
    excess = mean-risk_free
    if upper is None:
        raw = np.linalg.solve(covariance, excess)
        if raw.sum() <= 0:
            raise ValueError("No positive-return budget-normalized unrestricted tangency")
        return raw/raw.sum(), dict(method="analytic", successful_starts=1, objective=sharpe(raw/raw.sum(), mean, covariance, risk_free))
    upper = np.broadcast_to(np.asarray(upper, float), mean.shape)
    if np.any(upper < 0) or upper.sum() < 1:
        raise ValueError("Infeasible upper bounds")
    rng = np.random.default_rng(seed)
    starts = [np.ones(len(mean))/len(mean)]
    if multistart:
        starts += list(np.eye(len(mean)))+list(rng.dirichlet(np.ones(len(mean)), size=8))

    def objective(w):
        return -sharpe(w, mean, covariance, risk_free)

    def gradient(w):
        sd = np.sqrt(w @ covariance @ w)
        return -(excess/sd-(w @ excess)*(covariance @ w)/sd**3)

    candidates = []
    for start in starts:
        result = optimize.minimize(
            objective, _feasible_start(start, upper), jac=gradient, method="SLSQP",
            bounds=list(zip(np.zeros(len(mean)), upper)),
            constraints=[dict(type="eq", fun=lambda w: w.sum()-1, jac=lambda w: np.ones(len(w)))],
            options=dict(ftol=1e-13, maxiter=2000),
        )
        feasible = (abs(result.x.sum()-1) < 1e-8 and (result.x >= -1e-9).all()
                    and (result.x <= upper+1e-9).all())
        if result.success and feasible and np.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("No bounded tangency solve converged")
    best = min(candidates, key=lambda result: result.fun)
    return best.x, dict(method="SLSQP", successful_starts=len(candidates),
                        objective=-float(best.fun), objective_spread=float(np.ptp([result.fun for result in candidates])))


def efficient_frontier(mean, covariance, points=81):
    """Long-only upper efficient frontier, from global minimum variance upward."""
    mean, covariance = np.asarray(mean), np.asarray(covariance)
    n = len(mean)
    budget = dict(type="eq", fun=lambda w: w.sum()-1, jac=lambda w: np.ones(n))
    bounds = [(0., 1.)]*n
    result = optimize.minimize(lambda w: w @ covariance @ w, np.ones(n)/n,
                               jac=lambda w: 2*covariance @ w, method="SLSQP", bounds=bounds,
                               constraints=[budget], options=dict(ftol=1e-14, maxiter=2000))
    if not result.success:
        raise RuntimeError(result.message)
    start, end = float(result.x @ mean), float(mean.max())
    rows, seed = [], result.x
    for target in np.linspace(start, end, points):
        if abs(target-end) < 1e-13:
            w = np.eye(n)[mean.argmax()]
        else:
            result = optimize.minimize(
                lambda w: w @ covariance @ w, seed, jac=lambda w: 2*covariance @ w,
                method="SLSQP", bounds=bounds,
                constraints=[budget, dict(type="eq", fun=lambda w, t=target: w @ mean-t,
                                          jac=lambda w: mean)],
                options=dict(ftol=1e-14, maxiter=2000),
            )
            if not result.success:
                raise RuntimeError(f"Efficient frontier: {result.message}")
            w = result.x
        seed = w
        rows.append(dict(mean=float(w @ mean), sd=float(np.sqrt(w @ covariance @ w)),
                         **dict(zip(ASSETS, w))))
    return pd.DataFrame(rows)


def run(data_dir, output_dir, figure_dir, **kwargs):
    data_dir, out, figures = Path(data_dir), Path(output_dir), Path(figure_dir)
    out.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data_dir/"problem4.csv", parse_dates=["Month"])
    x = frame[ASSETS].to_numpy(float)
    if x.shape != (240, 4) or not np.isfinite(x).all() or frame.Month.duplicated().any():
        raise ValueError("Expected 240 complete monthly observations of four assets")
    rf = 1.03**(1/12)-1
    mean, covariance = x.mean(axis=0), np.cov(x, rowvar=False, ddof=1)

    # Problem 4(a): these moments are computed before portfolio optimization.
    statistics, influence = [], []
    for j, asset in enumerate(ASSETS):
        m = moments(x[:, j]); sr = np.sqrt(12)*(m["mean"]-rf)/m["sd"]
        statistics.append(dict(asset=asset, **m, annual_sharpe=sr, mean_standard_error=m["sd"]/np.sqrt(len(x))))
        deviations = x[:, j]-mean[j]
        event = int(np.argmax(abs(deviations)))
        remaining = np.delete(x[:, j], event)
        other = moments(remaining)
        influence.append(dict(
            asset=asset, month=str(frame.Month.iloc[event].date()), event_return=x[event, j],
            variance_share=deviations[event]**2/np.sum(deviations**2),
            fourth_moment_share=deviations[event]**4/np.sum(deviations**4),
            excess_kurtosis_full=m["excess_kurtosis"], excess_kurtosis_without_event=other["excess_kurtosis"],
            annual_sharpe_full=sr, annual_sharpe_without_event=np.sqrt(12)*(other["mean"]-rf)/other["sd"],
        ))
    stat_frame = pd.DataFrame(statistics)
    stat_frame.to_csv(out/"q4_asset_moments.csv", index=False)
    pd.DataFrame(influence).to_csv(out/"q4_single_month_influence.csv", index=False)
    pd.DataFrame(covariance, index=ASSETS, columns=ASSETS).to_csv(out/"q4_covariance.csv")

    # Problem 4(b): generate a retrospective AI-assisted illustrative draft.
    putw_sharpe = float(stat_frame.loc[stat_frame.asset == "PUTW", "annual_sharpe"].iloc[0])
    review_draft = (
        "Retrospective AI-assisted illustrative draft: no authenticated student pre-fit prediction was supplied. "
        f"PUTW has annual sample Sharpe {putw_sharpe:.4f}. For the requested pre-optimization economic judgment, "
        "an illustrative expectation is roughly 50%-90% PUTW (a majority holding) in the long-only tangency portfolio. "
        "Its regular option premium and strong sample risk-adjusted return give a mean-variance objective reasons "
        "to favor a large allocation, while asset covariances can shift the weight. "
        "This illustrative range was added during review after the fitted results were available. "
        "Its negatively skewed loss observations suggest historical 1% ES will exceed a matched Normal ES. "
        "This draft does not establish what the student predicted before fitting."
    )
    write_predictions(out/"q4_predictions.json", dict(b=review_draft))
    write_json(out/"q4_review_draft.json", dict(
        origin="Retrospective AI-assisted review after fitted results were available",
        original_draft_file="q4_predictions.json",
        putw_weight_range=[0.5, 0.9],
        text=review_draft,
    ))

    # Problems 4(c-d): three portfolios, with raw monthly-return risk.
    constraints = {"Unconstrained": None, "LongOnly": np.ones(4), "PUTW20": np.array([1., 1., 1., .2])}
    portfolios, optimizers = {}, {}
    weight_rows, risks, sensitivity = [], [], []
    asset_tail_es = {a: np.array([historical_risk(x[:, j], a)["es"] for j in range(4)]) for a in (.05, .01)}
    for label, upper in constraints.items():
        w, diagnostic = maximum_sharpe(mean, covariance, rf, upper)
        portfolios[label], optimizers[label] = w, diagnostic
        p = x @ w
        weight_rows.append(dict(portfolio=label, **dict(zip(ASSETS, w)), mean=float(p.mean()),
                                sd=float(p.std(ddof=1)), annual_sharpe=np.sqrt(12)*sharpe(w, mean, covariance, rf)))
        for alpha in (.05, .01):
            hist, normal = historical_risk(p, alpha), normal_risk(p.mean(), p.std(ddof=1), alpha)
            fractional = historical_risk(p, alpha, method="fractional")
            risks.append(dict(portfolio=label, alpha=alpha, mean=p.mean(), sd=p.std(ddof=1),
                              annual_sharpe=np.sqrt(12)*sharpe(w, mean, covariance, rf),
                              historical_VaR=hist["var"], historical_ES=hist["es"],
                              normal_VaR=normal["var"], normal_ES=normal["es"],
                              historical_tail_count=int((p <= -hist["var"]).sum()),
                              weighted_standalone_ES=float(w @ asset_tail_es[alpha])))
            sensitivity.append(dict(portfolio=label, alpha=alpha, course_VaR=hist["var"], course_ES=hist["es"],
                                    fractional_VaR=fractional["var"], fractional_ES=fractional["es"]))
    pd.DataFrame(weight_rows).to_csv(out/"q4_portfolios.csv", index=False)
    pd.DataFrame(risks).to_csv(out/"q4_portfolio_risk.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(out/"q4_tail_convention_sensitivity.csv", index=False)
    frontier = efficient_frontier(mean, covariance)
    frontier.to_csv(out/"q4_frontier.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(100*frontier.sd, 100*frontier["mean"], label="Long-only efficient frontier", color="#244a73")
    tangency = next(row for row in weight_rows if row["portfolio"] == "LongOnly")
    sigma = np.linspace(0, max(np.sqrt(np.diag(covariance)))*1.08, 150)
    line = rf+sharpe(portfolios["LongOnly"], mean, covariance, rf)*sigma
    ax.plot(100*sigma, 100*line, "--", color="#9b4d33", label="Capital market line")
    for j, name in enumerate(ASSETS):
        ax.scatter(100*np.sqrt(covariance[j, j]), 100*mean[j], color="#466b59")
        ax.annotate(name, (100*np.sqrt(covariance[j, j]), 100*mean[j]), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.scatter(100*tangency["sd"], 100*tangency["mean"], marker="*", s=100, color="#9b4d33", label="Long-only tangency")
    ax.set(xlabel="Monthly standard deviation (%)", ylabel="Mean monthly return (%)", title="Problem 4(c): long-only opportunity set")
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(figures/"q4_frontier.png", dpi=170); plt.close(fig)

    # Problem 4(e): the drawdown filter is on EQ EXCESS return, not raw EQ.
    market_excess, putw_excess = x[:, 0]-rf, x[:, 3]-rf
    full = ols(putw_excess, market_excess, ["EQ_excess"])
    downside = market_excess < -.05
    down = ols(putw_excess[downside], market_excess[downside], ["EQ_excess"])
    regressions = []
    for label, fit in [("AllMonths", full), ("EQExcessBelowMinus5pct", down)]:
        regressions.append(dict(sample=label, n=fit["n"], alpha=fit["coefficients"][0], alpha_t=fit["t_statistics"][0],
                                beta=fit["coefficients"][1], rsquared=fit["rsquared"], aicc=fit["aicc"]))
    pd.DataFrame(regressions).to_csv(out/"q4_capm.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.scatter(100*market_excess, 100*putw_excess, alpha=.45, s=15, color="#244a73", label="Other months")
    ax.scatter(100*market_excess[downside], 100*putw_excess[downside], color="#a14a35", s=25, label="EQ excess < -5%")
    xx = np.linspace(market_excess.min(), market_excess.max(), 100)
    ax.plot(100*xx, 100*(full["coefficients"][0]+full["coefficients"][1]*xx), color="#244a73", label="Full-sample fit")
    xx = np.linspace(market_excess.min(), -.05, 100)
    ax.plot(100*xx, 100*(down["coefficients"][0]+down["coefficients"][1]*xx), color="#a14a35", label="Downside fit")
    ax.set(xlabel="EQ monthly excess return (%)", ylabel="PUTW monthly excess return (%)", title="Problem 4(e): beta across market regimes")
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(figures/"q4_capm_regimes.png", dpi=170); plt.close(fig)

    # Problem 4(h): bootstrap input uncertainty, retaining all months in main estimates.
    n_boot, seed = int(kwargs.get("n_boot", 300)), int(kwargs.get("seed", 545))
    rng = np.random.default_rng(seed)
    bootstrap_rows, failures = [], []
    for b in range(n_boot):
        sample = x[rng.integers(0, len(x), len(x))]
        bm, bc = sample.mean(axis=0), np.cov(sample, rowvar=False, ddof=1)
        for label, upper in constraints.items():
            try:
                w, _ = maximum_sharpe(bm, bc, rf, upper, multistart=False, seed=seed+b)
                bootstrap_rows.append(dict(replication=b, portfolio=label, **dict(zip(ASSETS, w))))
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                failures.append(dict(replication=b, portfolio=label, reason=str(exc)))
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(out/"q4_bootstrap_weights.csv", index=False)
    pd.DataFrame(failures, columns=["replication", "portfolio", "reason"]).to_csv(out/"q4_bootstrap_failures.csv", index=False)
    intervals = []
    for label in constraints:
        values = bootstrap.loc[bootstrap.portfolio == label] if len(bootstrap) else pd.DataFrame()
        for asset in ASSETS:
            if len(values):
                q = values[asset].quantile([.05, .5, .95])
                intervals.append(dict(portfolio=label, asset=asset, n_success=len(values), p05=q.loc[.05], median=q.loc[.5], p95=q.loc[.95],
                                      sd=values[asset].std(ddof=1), zero_share=(values[asset].abs() < 1e-6).mean()))
    pd.DataFrame(intervals).to_csv(out/"q4_bootstrap_intervals.csv", index=False)
    summary = dict(
        n=len(x), assets=ASSETS, risk_free_annual=.03, risk_free_monthly=rf,
        annualization="Sharpe uses sqrt(12); plots and VaR/ES use monthly returns",
        data_start=str(frame.Month.min().date()), data_end=str(frame.Month.max().date()),
        moments=statistics, single_month_influence=influence, portfolios=weight_rows, risk=risks,
        capm=regressions, optimizer_diagnostics=optimizers,
        mean_se=stat_frame[["asset", "mean", "mean_standard_error"]].to_dict("records"),
        bootstrap_replications=n_boot, bootstrap_failures=len(failures), bootstrap_intervals=intervals,
        historical_risk_convention="course order-statistic average and threshold-tail mean; fractional-tail sensitivity separately reported",
    )
    write_json(out/"q4_summary.json", summary)
    return summary
