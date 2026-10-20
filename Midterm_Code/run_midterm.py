"""Fit candidate models; require supplied choices before downstream calculations."""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import optimize, stats

from calculations import q1, q2, q3a, q3b, q3c
from examkit import copula_loglik, information_criteria
from examkit.validation import load_marginal_choices

ROOT = Path(__file__).resolve().parent


def independent_checks(data, out):
    """Cross-check Q1 with constrained optimization and Q2 with linregress."""
    raw = pd.read_csv(out / "q1a_pairwise_correlation.csv", index_col=0).to_numpy()
    higham = pd.read_csv(out / "q1d_higham_correlation.csv", index_col=0).to_numpy()
    indices = np.triu_indices(len(raw), 1)

    def make(v):
        a = np.eye(len(raw))
        a[indices] = v
        a[(indices[1], indices[0])] = v
        return a

    independent = optimize.minimize(
        lambda v: np.sum((v-raw[indices])**2), np.zeros(len(indices[0])),
        jac=lambda v: 2*(v-raw[indices]), method="SLSQP",
        constraints=[dict(type="ineq", fun=lambda v: np.linalg.eigvalsh(make(v)))],
        options=dict(ftol=1e-13, maxiter=1000),
    )
    if not independent.success:
        raise AssertionError(independent.message)
    max_difference = float(np.max(np.abs(make(independent.x)-higham)))
    if max_difference > 1e-6:
        raise AssertionError("Higham independent optimization mismatch")
    source = pd.read_csv(data / "problem2.csv")
    ols = stats.linregress(source.SPY.pct_change().dropna(), source.XYZ.pct_change().dropna())
    reported = pd.read_csv(out / "q2_market_model.csv").iloc[0]
    np.testing.assert_allclose([ols.intercept, ols.slope], [reported.intercept, reported.beta], atol=1e-12)
    return dict(higham_vs_independent_max_abs_difference=max_difference,
                higham_frobenius_distance=float(np.linalg.norm(higham-raw)))


def copula_checks(c, out, n):
    """Export numerical profile checks without comparing or selecting families."""
    z = stats.norm.ppf(np.clip(c["u"], 1e-12, 1-1e-12))
    independent_ll = (-len(z)/2*np.linalg.slogdet(c["r"])[1]
                      -.5*np.sum(z*(z @ np.linalg.inv(c["r"])-z)))
    np.testing.assert_allclose(independent_ll, c["gaussian"]["ll"], rtol=1e-12, atol=1e-10)
    profile = [dict(nu=v, ll=copula_loglik(c["u"], c["r"], v))
               for v in [3, 4, 6, 10, 20, 40, 100, 200, 1000, 10000]]
    pd.DataFrame(profile).to_csv(out / "q3b_extended_nu_profile.csv", index=False)
    extended = optimize.minimize_scalar(
        lambda theta: -copula_loglik(c["u"], c["r"], 1/theta),
        bounds=(1e-7, .01), method="bounded", options={"xatol": 1e-12},
    )
    if not extended.success:
        raise RuntimeError(extended.message)
    pd.DataFrame([dict(copula="t_extended", nu=1/extended.x, ll=-extended.fun,
                       **information_criteria(-extended.fun, 1, n))]).to_csv(
        out / "q3b_extended_fit.csv", index=False
    )
    return dict(copula_nu=float(c["nu"]), gaussian_copula_ll=float(independent_ll),
                kendall_R_min_eigenvalue=float(np.linalg.eigvalsh(c["r"]).min()))


def run(data, out, draws=250000, seed=545, choices_path=None, copula=None):
    data, out = Path(data), Path(out)
    if not isinstance(draws, (int, np.integer)) or isinstance(draws, bool) or draws < 5000:
        raise ValueError("At least 5000 integer simulation draws required")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("A nonnegative integer random seed is required")
    if copula is not None and copula not in ("gaussian", "t"):
        raise ValueError("Copula must be gaussian or t")
    if copula is not None and choices_path is None:
        raise ValueError("--copula also requires a complete --choices JSON file")
    stocks = list(pd.read_csv(data / "problem3.csv", nrows=0).drop(columns="Date").columns)
    choices = load_marginal_choices(choices_path, stocks)
    if out.resolve() == data.resolve():
        raise ValueError("Output directory must differ from input directory")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("Use a new or empty --output-dir to avoid stale results")
    out.mkdir(parents=True, exist_ok=True)

    q1(data, out)
    q2(data, out)
    prices, r, fits = q3a(data, out)
    summary = dict(
        stage="marginal_fits", stocks=stocks,
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__, pandas=pd.__version__),
        **independent_checks(data, out),
    )
    if choices is not None:
        c = q3b(out, r, fits, choices)
        summary.update(stage="copula_fits", supplied_margins=choices,
                       **copula_checks(c, out, len(r)))
        if copula is not None:
            q3c(data, out, prices, r, fits, choices, draws, seed, c, copula)
            checks = pd.read_csv(out / "q3c_marginal_validation.csv")
            if checks.z.abs().max() > 7:
                raise AssertionError("Marginal simulated VaR outside 7-MCSE tolerance")
            summary.update(stage="risk", supplied_copula=copula, draws=draws, seed=seed,
                           max_marginal_VaR_z=float(checks.z.abs().max()))
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("Numeric output:", out)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--choices", type=Path, help="JSON with your marginal family for every stock")
    parser.add_argument("--copula", choices=["gaussian", "t"], default=None,
                        help="Your explicit copula choice; requires --choices")
    parser.add_argument("--draws", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=545)
    args = parser.parse_args()
    try:
        run(args.data_dir, args.output_dir, args.draws, args.seed, args.choices, args.copula)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.error(str(exc))
