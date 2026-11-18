"""Run selection, saved-result consistency and publication recovery regressions."""

import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

import functionaltests2_runner as runner
from risk545_ft2 import portfolios
from risk545_ft2.validation import CASE_IDS, output_name
from verify_results import verify


GROUP_INPUTS = {
    8: ["test7_1.csv", "test7_2.csv"],
    9: ["test9_1_returns.csv", "test9_1_portfolio.csv"],
    10: ["test5_2.csv", "test5_3.csv", "test10_3_means.csv"],
    11: ["test11_1_returns.csv", "test11_1_weights.csv", "test11_2_factor_returns.csv",
         "test11_2_stock_returns.csv", "test11_2_weights.csv", "test11_2_beta.csv"],
}


def _minimal_inputs(tmp_path, group):
    data, reference = tmp_path / "data", tmp_path / "reference"
    data.mkdir()
    reference.mkdir()
    for name in GROUP_INPUTS[group]:
        shutil.copyfile(runner.ROOT / "data" / name, data / name)
    for case in CASE_IDS:
        if case.startswith(f"{group}."):
            name = output_name(case)
            shutil.copyfile(runner.ROOT / "reference" / name, reference / name)
    return data, reference


def _rewrite_json(output, name, content):
    runner.write_json(output / name, content)
    manifest_path = output / "output_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    runner.write_json(manifest_path, manifest)


@pytest.fixture
def saved_run(tmp_path):
    data, reference = _minimal_inputs(tmp_path, 11)
    output = tmp_path / "output"
    runner.run(data, reference, output, simulations=10000, groups=[11])
    return output, reference


@pytest.mark.parametrize("group", [8, 9, 10, 11])
def test_each_multigroup_builder_uses_only_selected_inputs(tmp_path, group):
    data, reference = _minimal_inputs(tmp_path, group)
    output = tmp_path / "output"
    report = runner.run(data, reference, output, simulations=10000, groups=[group])
    summary = verify(output, reference)
    expected = [case for case in CASE_IDS if case.startswith(f"{group}.")]
    assert report["all_passed"]
    assert report["groups"] == [group]
    assert summary.Test.tolist() == expected
    assert len(list(output.glob("testout*.csv"))) == 2 * len(expected)


def test_single_group_builder_retains_its_existing_signature(tmp_path, monkeypatch):
    data, reference = tmp_path / "data", tmp_path / "reference"
    data.mkdir()
    reference.mkdir()
    frames = {case: pd.DataFrame({"Value": [1.]}) for case in ("12.1", "12.2", "12.3")}
    for case, frame in frames.items():
        frame.to_csv(reference / output_name(case), index=False)

    def build(data_dir, *, simulations, seed):
        assert data_dir == data
        return frames, {}

    monkeypatch.setattr(runner.options, "build_cases", build)
    report = runner.run(data, reference, tmp_path / "output", groups=[12])
    assert report["all_passed"]


def test_custom_data_and_reference_remain_supported(tmp_path):
    data, reference = _minimal_inputs(tmp_path, 11)
    returns_path = data / "test11_1_returns.csv"
    returns = pd.read_csv(returns_path) * .9
    returns.to_csv(returns_path, index=False)
    cases, _ = portfolios.build_cases(data, groups=[11])
    for case, frame in cases.items():
        frame.to_csv(reference / output_name(case), index=False, float_format="%.17g")
    output = tmp_path / "output"
    assert runner.run(data, reference, output, groups=[11])["all_passed"]
    assert verify(output, reference).Status.tolist() == ["PASS", "PASS"]


def test_repeated_and_partial_runs_replace_prior_outputs(saved_run):
    output, reference = saved_run
    data = output.parent / "data"
    first = {path.name: path.read_bytes() for path in output.glob("testout*.csv")}
    runner.run(data, reference, output, simulations=10000, groups=[11])
    assert first == {path.name: path.read_bytes() for path in output.glob("testout*.csv")}
    # Replace with another complete group using only its own inputs/references.
    group10_root = output.parent / "group10"
    group10_root.mkdir()
    data, reference = _minimal_inputs(group10_root, 10)
    runner.run(data, reference, output, simulations=10000, groups=[10])
    assert not (output / "testout11_1.csv").exists()
    assert verify(output, reference).Test.tolist() == ["10.1", "10.2", "10.3", "10.4"]
    assert not list(output.parent.glob("ft2-previous-*"))


def test_failed_run_preserves_published_results(saved_run, monkeypatch):
    output, reference = saved_run
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    def fail(*args, **kwargs):
        raise ValueError("calculation failed")

    monkeypatch.setattr(runner.portfolios, "build_cases", fail)
    with pytest.raises(ValueError, match="calculation failed"):
        runner.run(output.parent / "data", reference, output, groups=[11])
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


@pytest.mark.parametrize("groups", [[], [7], [11.0], [11, 11]])
def test_saved_groups_cannot_claim_an_empty_or_invalid_run(saved_run, groups):
    output, reference = saved_run
    report = json.loads((output / "test_report.json").read_text())
    report["groups"] = groups
    _rewrite_json(output, "test_report.json", report)
    with pytest.raises(ValueError, match="[Gg]roups"):
        verify(output, reference)


@pytest.mark.parametrize("name", ["testout11_1.csv", "testout_11.1.csv", "test_report.json",
                                 "diagnostics.json", "test_summary.csv", ".functionaltests2-output"])
def test_manifest_requires_every_generated_artifact(saved_run, name):
    output, reference = saved_run
    manifest = json.loads((output / "output_manifest.json").read_text())
    del manifest[name]
    runner.write_json(output / "output_manifest.json", manifest)
    with pytest.raises(AssertionError, match="missing required output"):
        verify(output, reference)


def test_saved_partial_run_rejects_stale_other_group_outputs(saved_run):
    output, reference = saved_run
    (output / "testout8_1.csv").write_text("stale prior group\n")
    with pytest.raises(AssertionError, match="case coverage"):
        verify(output, reference)


def test_saved_alias_must_equal_canonical_output(saved_run):
    output, reference = saved_run
    path = output / "testout_11.1.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    manifest = json.loads((output / "output_manifest.json").read_text())
    manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    runner.write_json(output / "output_manifest.json", manifest)
    with pytest.raises(AssertionError, match="aliases differ"):
        verify(output, reference)


@pytest.mark.parametrize("field", ["passed", "total", "checks_passed", "checks_total", "all_passed"])
def test_saved_report_counters_must_match_recomputed_results(saved_run, field):
    output, reference = saved_run
    report = json.loads((output / "test_report.json").read_text())
    report[field] = False if field == "all_passed" else report[field] + 1
    _rewrite_json(output, "test_report.json", report)
    with pytest.raises(AssertionError, match=f"Saved report {field}"):
        verify(output, reference)


@pytest.mark.parametrize("change", ["missing", "value", "status"])
def test_saved_report_comparisons_must_match_recomputed_results(saved_run, change):
    output, reference = saved_run
    report = json.loads((output / "test_report.json").read_text())
    if change == "missing":
        report["comparisons"].pop()
    elif change == "value":
        report["comparisons"][0]["calculated"] += .1
    else:
        report["comparisons"][0]["passed"] = False
    _rewrite_json(output, "test_report.json", report)
    with pytest.raises(AssertionError, match="Saved report comparison"):
        verify(output, reference)


def test_saved_summary_must_match_recomputed_results(saved_run):
    output, reference = saved_run
    path = output / "test_summary.csv"
    frame = pd.read_csv(path, dtype={"Test": str})
    frame.loc[0, "Checks"] += 1
    frame.to_csv(path, index=False)
    manifest = json.loads((output / "output_manifest.json").read_text())
    manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    runner.write_json(output / "output_manifest.json", manifest)
    with pytest.raises(AssertionError, match="Saved summary"):
        verify(output, reference)


def test_saved_simulation_count_must_match_diagnostics(tmp_path):
    data, reference = _minimal_inputs(tmp_path, 8)
    output = tmp_path / "output"
    runner.run(data, reference, output, groups=[8])
    report = json.loads((output / "test_report.json").read_text())
    report["univariate_simulations"] = 20000
    _rewrite_json(output, "test_report.json", report)
    with pytest.raises(AssertionError, match="simulation count differs"):
        verify(output, reference)


def _publication_folders(tmp_path):
    output, staging = tmp_path / "output", tmp_path / "staging"
    output.mkdir()
    staging.mkdir()
    (output / ".functionaltests2-output").write_text("generated\n")
    (output / "old.csv").write_text("previous result\n")
    (staging / "new.csv").write_text("new result\n")
    return output, staging


def test_publish_restores_previous_results_after_failed_install(tmp_path, monkeypatch):
    output, staging = _publication_folders(tmp_path)
    rename = Path.rename

    def fail_install(path, target):
        if path == staging:
            raise OSError("install failed")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_install)
    with pytest.raises(OSError, match="install failed"):
        runner._publish(staging, output)
    assert (output / "old.csv").read_text() == "previous result\n"
    assert (staging / "new.csv").is_file()
    assert not list(tmp_path.glob("ft2-previous-*"))


def test_publish_cleans_up_after_failed_backup_move(tmp_path, monkeypatch):
    output, staging = _publication_folders(tmp_path)
    rename = Path.rename

    def fail_backup(path, target):
        if path == output:
            raise OSError("backup move failed")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_backup)
    with pytest.raises(OSError, match="backup move failed"):
        runner._publish(staging, output)
    assert (output / "old.csv").is_file()
    assert not list(tmp_path.glob("ft2-previous-*"))


def test_publish_preserves_recovery_copy_if_restoration_fails(tmp_path, monkeypatch):
    output, staging = _publication_folders(tmp_path)
    rename = Path.rename

    def fail_install_and_restore(path, target):
        if path == staging or path.parent.name.startswith("ft2-previous-"):
            raise OSError("rename failed")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_install_and_restore)
    with pytest.raises(OSError, match="recover them from"):
        runner._publish(staging, output)
    backups = list(tmp_path.glob("ft2-previous-*/output/old.csv"))
    assert len(backups) == 1
    assert backups[0].read_text() == "previous result\n"
