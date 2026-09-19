"""Adapters from the instructor's CSV contract to the public risk545 functions.

Only this test harness reads expected-output files. Model functions accept data
and parameters and have no dependency on filenames or reference answers.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import risk545 as risk

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CASE_IDS = [f"1.{i}" for i in range(1, 5)] + [f"2.{i}" for i in range(1, 4)] + [f"3.{i}" for i in range(1, 5)] + ["4.1"] + [f"5.{i}" for i in range(1, 6)] + [f"7.{i}" for i in range(1, 7)]


def output_name(case):
    # The workbook uses a different spelling from actual CSVs for group 7.
    return f"testout{case.replace('.', '_')}.csv" if case.startswith("7.") else f"testout_{case}.csv"


def read(name, data_dir=DATA):
    return pd.read_csv(Path(data_dir) / name)


def matrix(name, data_dir=DATA):
    return read(name, data_dir).to_numpy(dtype=float)


def calculate(case, data_dir=DATA, seed=1234):
    """Calculate one case without reading that case's expected answer."""
    if case.startswith("1."):
        frame = read("test1.csv", data_dir)
        a = risk.missing_covariance(frame, pairwise=case in ("1.3", "1.4"), corr=case in ("1.2", "1.4"))
        return pd.DataFrame(a, columns=frame.columns)
    if case.startswith("2."):
        frame = read("test2.csv", data_dir)
        if case == "2.1":
            a = risk.ew_covariance(frame, 0.97)
        elif case == "2.2":
            a = risk.correlation(risk.ew_covariance(frame, 0.94))
        else:
            a = risk.mixed_ew_covariance(frame, 0.97, 0.94)
        return pd.DataFrame(a, columns=frame.columns)
    if case.startswith("3."):
        # Earlier expected outputs are the explicit inputs specified by Tests.xlsx.
        source = "testout_1.3.csv" if case in ("3.1", "3.3") else "testout_1.4.csv"
        frame = read(source, data_dir)
        repair = risk.near_psd if case in ("3.1", "3.2") else risk.higham_psd
        return pd.DataFrame(repair(frame), columns=frame.columns)
    if case == "4.1":
        frame = read("testout_3.1.csv", data_dir)
        return pd.DataFrame(risk.chol_psd(frame), columns=frame.columns)
    if case.startswith("5."):
        source = {"5.1": "test5_1.csv", "5.2": "test5_2.csv", "5.3": "test5_3.csv", "5.4": "test5_3.csv", "5.5": "test5_2.csv"}[case]
        frame = read(source, data_dir)
        if case == "5.5":
            draws = risk.simulate_pca(frame, 100000, explained=0.99, seed=seed)
        else:
            draws = risk.simulate_normal(frame, 100000, seed=seed, repair=risk.higham_psd if case == "5.4" else risk.near_psd)
        return pd.DataFrame(np.cov(draws, rowvar=False, ddof=1), columns=frame.columns)
    if case == "7.1":
        parameters = risk.fit_normal(matrix("test7_1.csv", data_dir)[:, 0])
    elif case in ("7.2", "7.4"):
        x = matrix("test7_2.csv", data_dir)[:, 0]
        parameters = risk.fit_t(x)
        if case == "7.4":
            ll = stats.t.logpdf(x, parameters["nu"], loc=parameters["mu"], scale=parameters["sigma"]).sum()
            parameters = {"AICC": risk.aicc(ll, len(x), 3)}
    elif case == "7.3":
        frame = read("test7_3.csv", data_dir)
        parameters = risk.fit_t_regression(frame["y"], frame.drop(columns="y"))
    elif case in ("7.5", "7.6"):
        x = matrix("test7_5.csv", data_dir)[:, 0]
        parameters = (risk.fit_nig_moments if case == "7.5" else risk.fit_nig_mle)(x)
    else:
        raise ValueError(f"Case {case} is outside this submission's scope")
    return pd.DataFrame([parameters])


def simulation_target(case, data_dir=DATA):
    if case == "5.1":
        return matrix("test5_1.csv", data_dir)
    if case == "5.2":
        return matrix("test5_2.csv", data_dir)
    if case in ("5.3", "5.4"):
        fn = risk.near_psd if case == "5.3" else risk.higham_psd
        return fn(matrix("test5_3.csv", data_dir))
    return risk.pca_target(matrix("test5_2.csv", data_dir), 0.99)


def compare(case, actual, expected, data_dir=DATA):
    if list(actual.columns) != list(expected.columns) or actual.shape != expected.shape:
        raise AssertionError("CSV dimensions or column order differ from reference")
    a, e = actual.to_numpy(float), expected.to_numpy(float)
    if not np.isfinite(a).all():
        raise AssertionError("Output includes NaN or infinity")
    result = {"max_abs_error": float(np.max(np.abs(a-e)))}
    if case.startswith("5."):
        target = simulation_target(case, data_dir)
        # Cov(S_ij) for Gaussian draws: Var(S_ij)=(S_ii*S_jj+S_ij^2)/(N-1).
        # Independent Python and Julia draws contribute two sampling variances.
        se = np.sqrt((np.outer(np.diag(target), np.diag(target)) + target**2)/99999)
        safe = np.maximum(se, np.finfo(float).tiny)
        z_target = np.max(np.abs(a-target)/safe)
        z_reference = np.max(np.abs(a-e)/(np.sqrt(2)*safe))
        relative_frobenius = np.linalg.norm(a-e, "fro")/np.linalg.norm(target, "fro")
        result.update(max_mc_z=float(z_target), max_reference_mc_z=float(z_reference), relative_frobenius=float(relative_frobenius))
        # Fixed a priori: 7 standard errors accounts for multiple entries/cases.
        if z_target > 7 or z_reference > 7 or relative_frobenius > 0.03:
            raise AssertionError(f"Monte Carlo covariance mismatch: {result}")
        result["criterion"] = "MC z <= 7; reference z <= 7; relative Frobenius <= 0.03"
    else:
        # Numerical optimizers differ across Julia/Ipopt and SciPy.
        rtol, atol = (5e-4, 1e-6) if case in ("7.2", "7.3", "7.6") else (1e-7, 1e-8)
        np.testing.assert_allclose(a, e, rtol=rtol, atol=atol)
        result["criterion"] = f"rtol={rtol:g}, atol={atol:g}"
    return result
