"""Regression checks for selected-group inputs and portfolio asset identifiers."""
from pathlib import Path
import shutil
import json

import numpy as np
import pandas as pd
import pytest

from risk545_ft2.tail_cases import build_cases

DATA = Path(__file__).resolve().parents[1] / "data"


def test_group8_uses_only_its_own_input_files(tmp_path):
    for name in ["test7_1.csv", "test7_2.csv"]:
        shutil.copyfile(DATA / name, tmp_path / name)
    cases, diagnostics = build_cases(tmp_path, groups={8})
    assert set(cases) == {f"8.{case}" for case in range(1, 7)}
    assert "9.1" not in diagnostics


def test_single_asset_portfolio_preserves_numeric_stock_identifier(tmp_path):
    returns = pd.DataFrame({"001": np.random.default_rng(10).normal(.001, .02, 300)})
    returns.to_csv(tmp_path / "test9_1_returns.csv", index=False)
    pd.DataFrame({"Stock": ["001"], "Holding": [10], "Starting Price": [20],
                  "Distribution": ["Normal"]}).to_csv(tmp_path / "test9_1_portfolio.csv", index=False)
    cases, diagnostic = build_cases(tmp_path, groups={9}, simulations=10000)
    result = cases["9.1"]
    assert set(cases) == {"9.1"}
    assert result.Stock.tolist() == ["001", "Total"]
    np.testing.assert_array_equal(result.iloc[0, 1:], result.iloc[1, 1:])
    assert diagnostic["9.1"]["spearman_correlation"] == [[1.0]]


def test_total_label_is_reserved_for_portfolio_aggregation(tmp_path):
    pd.DataFrame({"Total": [.01, -.02, .03]}).to_csv(tmp_path / "test9_1_returns.csv", index=False)
    pd.DataFrame({"Stock": ["Total"], "Holding": [10], "Starting Price": [20],
                  "Distribution": ["Normal"]}).to_csv(tmp_path / "test9_1_portfolio.csv", index=False)
    with pytest.raises(ValueError, match="reserved"):
        build_cases(tmp_path, groups={9})


def test_normal_boundary_t_simulation_has_finite_json_diagnostics(tmp_path):
    sample = np.random.default_rng(0).normal(size=100)
    for name in ["test7_1.csv", "test7_2.csv"]:
        pd.DataFrame({"x": sample}).to_csv(tmp_path / name, index=False)
    cases, diagnostic = build_cases(tmp_path, groups={8})
    assert diagnostic["8_fits"]["t"]["normal_limit"]
    assert diagnostic["8_fits"]["t"]["nu"] is None
    for frame in cases.values():
        assert np.isfinite(frame.to_numpy()).all()
    json.dumps(diagnostic, allow_nan=False)


def test_single_t_margin_normal_boundary_keeps_portfolio_risk_finite(tmp_path):
    pd.DataFrame({"001": np.random.default_rng(0).normal(size=100)}).to_csv(
        tmp_path / "test9_1_returns.csv", index=False)
    pd.DataFrame({"Stock": ["001"], "Holding": [10], "Starting Price": [20],
                  "Distribution": ["T"]}).to_csv(tmp_path / "test9_1_portfolio.csv", index=False)
    cases, diagnostic = build_cases(tmp_path, groups={9}, simulations=10000)
    assert np.isfinite(cases["9.1"].iloc[:, 1:].to_numpy()).all()
    json.dumps(diagnostic, allow_nan=False)


@pytest.mark.parametrize("groups", [[], [7], [8, 13], [True]])
def test_invalid_group_selection(groups, tmp_path):
    with pytest.raises(ValueError, match="groups"):
        build_cases(tmp_path, groups=groups)
