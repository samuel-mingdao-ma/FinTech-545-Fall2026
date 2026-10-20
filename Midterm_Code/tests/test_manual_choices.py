"""Test explicit-input gates with synthetic data, not examination choices."""
import json
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_midterm as runner
from validate_results import validate


@pytest.fixture
def synthetic_data(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(14)
    pd.DataFrame(rng.normal(size=(100, 3)), columns=["x1", "x2", "x3"]).to_csv(
        data / "problem1.csv", index=False
    )
    spy = rng.normal(.0001, .01, 250)
    xyz = .0002+1.3*spy+rng.normal(0, .02, 250)
    pd.DataFrame(dict(
        Date=np.arange(251), SPY=np.r_[100, 100*np.cumprod(1+spy)],
        XYZ=np.r_[80, 80*np.cumprod(1+xyz)],
    )).to_csv(data / "problem2.csv", index=False)
    names = [f"S{i}" for i in range(1, 11)]
    factor = rng.normal(0, .01, (250, 1))
    returns = factor+rng.normal(0, .01, (250, 10))
    prices = pd.DataFrame(np.vstack([np.ones(10)*100, 100*np.cumprod(1+returns, axis=0)]), columns=names)
    prices.insert(0, "Date", np.arange(251))
    prices.to_csv(data / "problem3.csv", index=False)
    pd.DataFrame(dict(Stock=names, Book=["A"]*5+["B"]*5, Holding=[10]*10)).to_csv(
        data / "problem3_holdings.csv", index=False
    )
    return data


def write_fixture_choices(path):
    # A fixed test fixture; not a decision based on the exam data or fitted scores.
    choices = {f"S{i}": "normal" for i in range(1, 11)}
    path.write_text(json.dumps(choices))
    return choices


def test_default_outputs_candidates_without_any_selection(synthetic_data, tmp_path):
    out = tmp_path / "fits"
    summary = runner.run(synthetic_data, out, draws=5000)
    assert summary["stage"] == "marginal_fits"
    assert "supplied_margins" not in summary and "supplied_copula" not in summary
    assert len(pd.read_csv(out / "q3a_all_fits.csv")) == 30
    assert len(pd.read_csv(out / "q3a_comparison.csv")) == 10
    assert not list(out.glob("q3b*")) and not list(out.glob("q3c*"))
    assert not (out / "selected_margins.json").exists()
    assert not (out / "q3a_model_selection.csv").exists()
    result = validate(out)
    assert result["passed"] == result["total"]


def test_explicit_margins_only_outputs_both_copulas(synthetic_data, tmp_path):
    path, out = tmp_path / "choices.json", tmp_path / "copulas"
    choices = write_fixture_choices(path)
    summary = runner.run(synthetic_data, out, draws=5000, choices_path=path)
    assert summary["stage"] == "copula_fits"
    assert summary["supplied_margins"] == choices
    assert "supplied_copula" not in summary
    assert set(pd.read_csv(out / "q3b_comparison.csv").copula) == {"gaussian", "t"}
    assert not list(out.glob("q3c*"))
    result = validate(out)
    assert result["passed"] == result["total"]


@pytest.mark.parametrize("copula", ["gaussian", "t"])
def test_manual_choices_are_not_overridden_by_scores(synthetic_data, tmp_path, monkeypatch, copula):
    path, out = tmp_path / "choices.json", tmp_path / "risk"
    choices = write_fixture_choices(path)
    original_q3b = runner.q3b
    seen = {}

    def fit_with_opposing_scores(out, r, fits, supplied):
        assert supplied == choices
        # Give the explicitly supplied marginal families worse scores than a rival.
        for stock in fits:
            fits[stock]["normal"].aicc = 100
            fits[stock]["t"].aicc = -100
        c = original_q3b(out, r, fits, supplied)
        other = "t" if copula == "gaussian" else "gaussian"
        c[copula]["aicc"], c[other]["aicc"] = 100, -100
        seen["margins"] = supplied.copy()
        return c

    original_q3c = runner.q3c

    def simulate_supplied(data, out, prices, r, fits, supplied, n, seed, c, supplied_copula):
        seen["copula"] = supplied_copula
        return original_q3c(data, out, prices, r, fits, supplied, n, seed, c, supplied_copula)

    monkeypatch.setattr(runner, "q3b", fit_with_opposing_scores)
    monkeypatch.setattr(runner, "q3c", simulate_supplied)
    summary = runner.run(synthetic_data, out, draws=5000, choices_path=path, copula=copula)
    assert seen == dict(margins=choices, copula=copula)
    assert summary["supplied_margins"] == choices and summary["supplied_copula"] == copula
    risk = pd.read_csv(out / "q3c_risk.csv")
    assert len(risk) == 13 and set(risk.copula) == {copula}
    assert not (out / "q3c_final_risk.csv").exists()
    result = validate(out)
    assert result["passed"] == result["total"]


@pytest.mark.parametrize("payload", [
    {}, {"S1": "normal"}, {"S1": None, "S2": "t"},
    {"S1": "normal", "S2": "invalid"},
    {"S1": "normal", "S2": "t", "extra": "nig"}, [],
])
def test_missing_or_invalid_choices_are_rejected(tmp_path, payload):
    path = tmp_path / "choices.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        runner.load_marginal_choices(path, ["S1", "S2"])


def test_empty_template_has_no_prefilled_family():
    template = json.loads((runner.ROOT / "marginal_choices_template.json").read_text())
    assert set(template) == {f"S{i}" for i in range(1, 11)}
    assert all(value is None for value in template.values())


def test_copula_requires_manual_margins(synthetic_data, tmp_path):
    out = tmp_path / "risk"
    with pytest.raises(ValueError, match="requires"):
        runner.run(synthetic_data, out, draws=5000, copula="t")
    assert not out.exists()


def test_nonempty_output_is_not_reused(synthetic_data, tmp_path):
    out = tmp_path / "old_results"
    out.mkdir()
    marker = out / "q3c_final_risk.csv"
    marker.write_text("old output")
    with pytest.raises(FileExistsError, match="empty"):
        runner.run(synthetic_data, out, draws=5000)
    assert marker.read_text() == "old output"


@pytest.mark.parametrize("copula", ["gaussian", "t"])
def test_cli_rejects_copula_without_choices(tmp_path, copula):
    out=tmp_path/"risk"
    result=subprocess.run(
        [sys.executable,str(runner.ROOT/"run_midterm.py"),"--copula",copula,"--output-dir",str(out)],
        capture_output=True,text=True,check=False,
    )
    assert result.returncode==2 and "requires" in result.stderr
    assert not out.exists()
