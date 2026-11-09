"""Problem 5: nested factor regressions, attribution, and momentum stress."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import ols, write_json, write_predictions

FACTORS = ["Mkt-RF", "SMB", "HML", "Mom"]
MODELS = {"CAPM": ["Mkt-RF"], "FF3": ["Mkt-RF", "SMB", "HML"], "Carhart4": FACTORS}


def read_factor_file(path):
    """Parse the provided daily files, preserving dates and converting percent once."""
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    if "Date" not in frame:
        raise ValueError("Factor file needs Date")
    # Non-date/footer rows would be excluded explicitly, not treated as zero returns.
    dates = frame.Date.astype(str).str.strip()
    keep = dates.str.fullmatch(r"\d{8}")
    frame = frame.loc[keep].copy()
    frame["Date"] = pd.to_datetime(dates.loc[keep], format="%Y%m%d", errors="raise")
    for name in frame.columns.drop("Date"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")/100
    if frame.Date.duplicated().any() or not np.isfinite(frame.drop(columns="Date").to_numpy()).all():
        raise ValueError("Factor dates must be unique and observations finite")
    return frame.sort_values("Date").reset_index(drop=True)


def aligned_data(data_dir):
    data = Path(data_dir)
    funds = pd.read_csv(data/"problem5.csv", parse_dates=["Date"])
    if funds.Date.duplicated().any() or not np.isfinite(funds[["FUND1", "FUND2"]].to_numpy()).all():
        raise ValueError("Fund dates must be unique and returns finite")
    ff = read_factor_file(data/"F-F_Research_Data_Factors_daily.CSV")
    momentum = read_factor_file(data/"F-F_Momentum_Factor_daily.CSV")
    factors = ff.merge(momentum, on="Date", how="inner", validate="one_to_one")
    sample = funds.merge(factors, on="Date", how="inner", validate="one_to_one").sort_values("Date")
    if len(sample) != len(funds):
        raise ValueError("Every fund date must have all factors and RF")
    if len(sample) < 10:
        raise ValueError("Insufficient aligned observations")
    return funds, factors, momentum, sample.reset_index(drop=True)


def recent_factor_window(factors, years=10):
    """The open-left, closed-right trailing window ends at the FILE'S final date."""
    end = factors.Date.max()
    start_exclusive = end-pd.DateOffset(years=years)
    window = factors.loc[(factors.Date > start_exclusive) & (factors.Date <= end)].copy()
    if window.empty:
        raise ValueError("Empty trailing factor window")
    return window, start_exclusive, end


def regression_tables(fund, model, fit):
    coefficients = [dict(fund=fund, model=model, term=name, coefficient=fit["coefficients"][j],
                         standard_error=fit["standard_errors"][j], t_statistic=fit["t_statistics"][j],
                         p_value=fit["p_values"][j]) for j, name in enumerate(fit["names"])]
    score = dict(fund=fund, model=model, n=fit["n"], k=fit["k"],
                 alpha_daily=fit["coefficients"][0], alpha_standard_error=fit["standard_errors"][0],
                 alpha_t=fit["t_statistics"][0], rsquared=fit["rsquared"],
                 ll=fit["ll"], AICc=fit["aicc"], BIC=fit["bic"])
    return coefficients, score


def run(data_dir, output_dir, figure_dir, **kwargs):
    out, figures = Path(output_dir), Path(figure_dir)
    out.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    funds, factors, momentum, sample = aligned_data(data_dir)
    sample.to_csv(out/"q5_aligned_sample.csv", index=False)
    annual_days = 255
    statistics, correlations, capm_fits = [], [], {}

    # Problems 5(a-b): CAPM is explicitly part of Predict, unlike later FF3/C4 fits.
    for fund in ("FUND1", "FUND2"):
        excess = sample[fund]-sample.RF
        fit = ols(excess, sample[["Mkt-RF"]], ["Mkt-RF"])
        capm_fits[fund] = fit
        statistics.append(dict(
            fund=fund, n=len(excess), mean_daily_return=sample[fund].mean(),
            mean_daily_excess=excess.mean(), daily_volatility=sample[fund].std(ddof=1),
            daily_excess_volatility=excess.std(ddof=1), daily_sharpe=excess.mean()/excess.std(ddof=1),
            annual_sharpe=np.sqrt(annual_days)*excess.mean()/excess.std(ddof=1),
            capm_alpha_daily=fit["coefficients"][0], capm_alpha_t=fit["t_statistics"][0],
        ))
        for factor in ("SMB", "HML", "Mom"):
            correlations.append(dict(fund=fund, factor=factor,
                                     residual_correlation=np.corrcoef(fit["residuals"], sample[factor])[0, 1],
                                     sample_factor_premium=sample[factor].mean()))
    pd.DataFrame(statistics).to_csv(out/"q5_initial_statistics.csv", index=False)
    correlation_frame = pd.DataFrame(correlations)
    correlation_frame.to_csv(out/"q5_capm_residual_correlations.csv", index=False)
    predictions = {}
    for fund in ("FUND1", "FUND2"):
        rows = correlation_frame.loc[correlation_frame.fund == fund]
        evidence = ", ".join(f"{row.factor}: residual correlation {row.residual_correlation:.3f}, sample daily premium {row.sample_factor_premium:.6f}"
                             for row in rows.itertuples())
        # This heuristic only uses the permitted CAPM residuals and raw factor data.
        signed_evidence = sum(row.residual_correlation*row.sample_factor_premium for row in rows.itertuples())
        direction = "decline" if signed_evidence > 0 else "rise"
        magnitude = "substantially" if rows.residual_correlation.abs().max() > .2 else "only modestly"
        predictions[fund] = (f"Before FF3 or four-factor fitting, {evidence}. I predict alpha will {direction} {magnitude} "
                             "when the omitted factors are included; the direction is a heuristic from residual association and factor premia, "
                             "not a result from the expanded regression.")
    write_predictions(out/"q5_predictions.json", predictions)

    # Problem 5(c): six regressions; k includes the Gaussian residual scale.
    fits, coefficient_rows, scores = {}, [], []
    for fund in ("FUND1", "FUND2"):
        fits[fund] = {}
        excess = sample[fund]-sample.RF
        for model, names in MODELS.items():
            fit = capm_fits[fund] if model == "CAPM" else ols(excess, sample[names], names)
            fits[fund][model] = fit
            rows, score = regression_tables(fund, model, fit)
            coefficient_rows.extend(rows); scores.append(score)
    pd.DataFrame(coefficient_rows).to_csv(out/"q5_regression_coefficients.csv", index=False)
    pd.DataFrame(scores).to_csv(out/"q5_model_comparison.csv", index=False)

    # Problem 5(d): forecast premia are from the factor file's trailing ten years.
    window, start_exclusive, end = recent_factor_window(factors)
    averages = window[[*FACTORS, "RF"]].mean()
    pd.DataFrame(dict(factor=averages.index, mean_daily=averages.values,
                      annual_arithmetic=annual_days*averages.values)).to_csv(out/"q5_recent_factor_premia.csv", index=False)
    expected_rows = []
    for fund in ("FUND1", "FUND2"):
        fit = fits[fund]["Carhart4"]
        alpha, beta = fit["coefficients"][0], fit["coefficients"][1:]
        daily_without = float(averages.RF+beta @ averages[FACTORS].to_numpy())
        r = sample[fund].to_numpy()
        expected_rows.append(dict(
            fund=fund, expected_annual_without_alpha=annual_days*daily_without,
            expected_annual_with_alpha=annual_days*(daily_without+alpha),
            annual_fitted_alpha=annual_days*alpha,
            realized_annual_arithmetic=annual_days*r.mean(),
            realized_annual_geometric=float(np.expm1(np.log1p(r).sum()*annual_days/len(r))),
            realized_total_compounded=float(np.expm1(np.log1p(r).sum())),
            sample_daily_sd=r.std(ddof=1),
        ))
    pd.DataFrame(expected_rows).to_csv(out/"q5_expected_returns.csv", index=False)

    # Problem 5(e): search the ENTIRE momentum file, not the forecast or fund window.
    worst = momentum.loc[momentum.Mom.idxmin()]
    stress_rows = []
    for fund in ("FUND1", "FUND2"):
        beta_mom = fits[fund]["Carhart4"]["coefficients"][4]
        shock_return = float(beta_mom*worst.Mom)
        sigma = sample[fund].std(ddof=1)
        stress_rows.append(dict(
            fund=fund, worst_momentum_date=str(worst.Date.date()), worst_momentum_return=worst.Mom,
            momentum_beta=beta_mom, incremental_fund_return=shock_return,
            position_value=10_000_000., loss_dollars=-10_000_000*shock_return,
            daily_sample_sd=sigma, loss_in_daily_sigmas=-shock_return/sigma,
        ))
    pd.DataFrame(stress_rows).to_csv(out/"q5_momentum_stress.csv", index=False)

    # Problem 5(f): the intercept identity decomposes the actual change in alpha.
    sample_premia = sample[FACTORS].mean()
    attribution, attribution_checks = [], []
    for fund in ("FUND1", "FUND2"):
        for earlier, later in [("CAPM", "FF3"), ("FF3", "Carhart4"), ("CAPM", "Carhart4")]:
            first, second = fits[fund][earlier], fits[fund][later]
            beta_first = dict(zip(MODELS[earlier], first["coefficients"][1:]))
            beta_second = dict(zip(MODELS[later], second["coefficients"][1:]))
            explained = 0.
            for factor in MODELS[later]:
                delta_beta = beta_second[factor]-beta_first.get(factor, 0.)
                contribution = delta_beta*sample_premia[factor]
                explained += contribution
                attribution.append(dict(fund=fund, earlier_model=earlier, later_model=later, factor=factor,
                                        delta_beta=delta_beta, sample_daily_premium=sample_premia[factor],
                                        contribution_to_alpha_drop_daily=contribution,
                                        contribution_to_alpha_drop_annual=annual_days*contribution))
            actual = first["coefficients"][0]-second["coefficients"][0]
            np.testing.assert_allclose(explained, actual, atol=1e-12)
            attribution_checks.append(dict(fund=fund, earlier_model=earlier, later_model=later,
                                            actual_alpha_drop_daily=actual, explained_alpha_drop_daily=explained,
                                            residual=actual-explained))
    pd.DataFrame(attribution).to_csv(out/"q5_alpha_attribution.csv", index=False)
    pd.DataFrame(attribution_checks).to_csv(out/"q5_alpha_attribution_checks.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
    colors = ["#244a73", "#65816f", "#a15339"]
    for ax, fund in zip(axes, ("FUND1", "FUND2")):
        values = [fits[fund][model]["coefficients"][0]*annual_days*100 for model in MODELS]
        errors = [1.96*fits[fund][model]["standard_errors"][0]*annual_days*100 for model in MODELS]
        ax.bar(list(MODELS), values, color=colors, alpha=.9)
        ax.errorbar(range(3), values, yerr=errors, fmt="none", ecolor="#333333", capsize=4)
        ax.axhline(0, color="#555555", lw=.8)
        ax.set(title=fund, ylabel="Annualized arithmetic alpha (%)")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Problem 5: fitted alpha and conventional 95% intervals")
    fig.tight_layout(); fig.savefig(figures/"q5_alpha_models.png", dpi=170); plt.close(fig)

    summary = dict(
        n=len(sample), fund_sample_start=str(sample.Date.min().date()), fund_sample_end=str(sample.Date.max().date()),
        factor_file_end=str(end.date()), forecast_window_start_exclusive=str(start_exclusive.date()),
        forecast_first_date=str(window.Date.min().date()), forecast_last_date=str(window.Date.max().date()), forecast_n=len(window),
        annual_days=annual_days, annualization="255 times daily arithmetic expected returns; Sharpe multiplied by sqrt(255); geometric sample return reported separately",
        unit_conversion="Fund returns already decimals; every factor and RF column divided by 100 exactly once",
        regression_covariance="Classical OLS homoskedastic standard errors, intercept plus all specified factor slopes; Gaussian AICc k includes sigma",
        stress_convention="Incremental momentum factor shock only: -$10m * beta_Mom * worst_Mom; other factor shocks are zero and alpha/RF drift excluded",
        initial_statistics=statistics, residual_correlations=correlations, regressions=scores,
        coefficients=coefficient_rows, expected_returns=expected_rows, stress=stress_rows,
        recent_factor_premia=averages.to_dict(), sample_factor_premia=sample_premia.to_dict(),
        alpha_attribution=attribution, alpha_attribution_checks=attribution_checks,
    )
    write_json(out/"q5_summary.json", summary)
    return summary
