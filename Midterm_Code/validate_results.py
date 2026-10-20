"""Stage-aware numerical checks; no model-family selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, stats

from examkit import Fit, information_criteria, parametric_risk

ROOT = Path(__file__).resolve().parent


def distribution(row):
    if row.family == "normal":
        return stats.norm(row.mu, row.sigma), dict(mu=row.mu, sigma=row.sigma)
    if row.family == "t":
        return stats.t(row.nu, loc=row.mu, scale=row.scale), dict(mu=row.mu, scale=row.scale, nu=row.nu)
    return (stats.norminvgauss(row.alpha*row.delta, row.beta*row.delta, loc=row.mu, scale=row.delta),
            dict(mu=row.mu, alpha=row.alpha, beta=row.beta, delta=row.delta))


def validate(out=ROOT / "results"):
    out = Path(out)
    summary = json.loads((out / "run_summary.json").read_text())
    fits = pd.read_csv(out / "q3a_all_fits.csv")
    returns = pd.read_csv(out / "q3_arithmetic_returns.csv")
    comparison = pd.read_csv(out / "q3a_comparison.csv").set_index("stock")
    checks = []

    def check(name, ok):
        checks.append(dict(check=name, passed=bool(ok)))
        print(("PASS " if ok else "FAIL ")+name)

    variances = pd.read_csv(out / "q1b_variances.csv").variance.to_numpy()
    cov = pd.read_csv(out / "q1e_covariance.csv", index_col=0).to_numpy()
    check("Q1 covariance preserves full-history variances and PSD",
          np.allclose(np.diag(cov), variances) and np.linalg.eigvalsh(cov).min() > -1e-12)
    stocks = summary["stocks"]
    check("All three marginal candidates exist for every stock",
          len(fits) == 3*len(stocks) and not fits.duplicated(["stock", "family"]).any()
          and set(zip(fits.stock, fits.family)) == {(s, f) for s in stocks for f in ("normal", "t", "nig")})
    for _, row in fits.iterrows():
        d, _ = distribution(row)
        ll = d.logpdf(returns[row.stock]).sum()
        criteria = information_criteria(row.ll, {"normal": 2, "t": 3, "nig": 4}[row.family], len(returns))
        check(f"{row.stock} {row.family} likelihood and criteria",
              np.isfinite([ll, row.ll, row.AICc, row.BIC]).all()
              and np.isclose(ll, row.ll, rtol=1e-8, atol=1e-7)
              and np.isclose(row.AICc, criteria["aicc"])
              and np.isclose(row.BIC, criteria["bic"]))
    for family in ("normal", "t", "nig"):
        candidate = fits.loc[fits.family == family].set_index("stock").loc[stocks]
        check(f"{family} comparison columns match fitted values",
              all(np.allclose(comparison.loc[stocks, f"{family}_{key}"], candidate[key])
                  for key in ("ll", "AICc", "BIC")))
    stage = summary["stage"]
    check("Recognized calculation stage", stage in ("marginal_fits", "copula_fits", "risk"))

    if stage == "marginal_fits":
        check("Fit-only run has no supplied choices or downstream outputs",
              "supplied_margins" not in summary and "supplied_copula" not in summary
              and not list(out.glob("q3b*.csv")) and not list(out.glob("q3c*.csv")))
    else:
        choices = summary["supplied_margins"]
        check("Complete supplied marginal mapping",
              set(choices) == set(stocks) and all(f in ("normal", "t", "nig") for f in choices.values()))
        scores = pd.read_csv(out / "q3b_copula_scores.csv").set_index("copula")
        table = pd.read_csv(out / "q3b_comparison.csv").set_index("copula")
        check("Both copula candidates are present", set(scores.index) == {"gaussian", "t"})
        check("Copula comparison table matches candidate scores",
              all(np.allclose(table.loc[["gaussian", "t"], label], scores.loc[["gaussian", "t"], key])
                  for label, key in (("ll", "ll"), ("AICc", "aicc"), ("BIC", "bic"))))
        if stage == "copula_fits":
            check("Copula-only run has no supplied copula or risk outputs",
                  "supplied_copula" not in summary and not list(out.glob("q3c*.csv")))
        else:
            copula = summary["supplied_copula"]
            check("Explicit valid copula and at least 5000 draws",
                  copula in ("gaussian", "t") and summary["draws"] >= 5000)
            positions = pd.read_csv(out / "q3c_positions.csv").set_index("stock")
            risk = pd.read_csv(out / "q3c_risk.csv")
            mc = pd.read_csv(out / "q3c_monte_carlo_errors.csv").set_index("position")
            check("Only the supplied copula is simulated", set(risk.copula) == {copula})
            risk = risk.set_index("position")
            check("All stock/book/total risk rows are present",
                  len(risk) == len(stocks)+3 and set(risk.index) == set(stocks) | {"Book A", "Book B", "Total"})
            benchmarks = []
            for stock, family in choices.items():
                row = fits.loc[(fits.stock == stock) & (fits.family == family)].iloc[0]
                d, params = distribution(row)
                amount = positions.loc[stock, "current_value"]
                q = d.ppf(.05)
                analytic_var = -amount*q
                if family in ("normal", "t"):
                    analytic_es = amount*parametric_risk(Fit(family, d, params, 0, 0, 0), .05)["es"]
                else:
                    analytic_es = -amount*integrate.quad(
                        lambda x: x*d.pdf(x), d.ppf(1e-12), q, epsabs=1e-12, limit=200
                    )[0]/.05
                observed, se = risk.loc[stock], mc.loc[stock]
                check(f"{stock} VaR agrees with analytic marginal",
                      abs(observed.VaR95_dollars-analytic_var) < 7*se.VaR_MCSE)
                check(f"{stock} ES agrees with independent tail expectation",
                      abs(observed.ES95_dollars-analytic_es) < 7*se.ES_MCSE)
                benchmarks.append(dict(stock=stock, analytic_VaR95=analytic_var, analytic_ES95=analytic_es))
            check("ES exceeds corresponding VaR in every risk row", (risk.ES95_dollars >= risk.VaR95_dollars).all())
            pd.DataFrame(benchmarks).to_csv(out / "analytic_marginal_benchmarks.csv", index=False)

    result = dict(passed=sum(c["passed"] for c in checks), total=len(checks), checks=checks)
    (out / "verification.json").write_text(json.dumps(result, indent=2)+"\n")
    if not all(c["passed"] for c in checks):
        raise SystemExit(1)
    print(f"{result['passed']}/{result['total']} checks passed")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    validate(args.output_dir)
