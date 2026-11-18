"""Recheck saved outputs, source hashes and teacher comparisons without simulation."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from risk545_ft2.validation import CASE_IDS, STOCHASTIC, normalize_groups, output_name, validate_cases

ROOT = Path(__file__).resolve().parent


def _case_files(case_ids):
    names = {output_name(case) for case in case_ids}
    names.update(f"testout_{case}.csv" for case in case_ids if int(case.split(".")[0]) <= 12)
    return names


def _check_manifest(output, manifest, expected_ids):
    required = _case_files(expected_ids) | {
        ".functionaltests2-output", "test_summary.csv", "test_report.json", "diagnostics.json",
    }
    if not isinstance(manifest, dict) or not required <= set(manifest):
        raise AssertionError("Completed-run manifest is missing required output files")
    # A partial run replaces the whole result directory. Old group outputs must
    # not be mistaken for results from the selected groups.
    present_cases = {name for name in _case_files(CASE_IDS) if (output / name).exists()}
    if present_cases != _case_files(expected_ids):
        raise AssertionError("Saved output case coverage differs from the reported groups")
    for name, digest in manifest.items():
        if not isinstance(name, str) or Path(name).name != name or not (output / name).is_file():
            raise AssertionError(f"Invalid or missing output manifest file: {name}")
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"Output differs from the completed-run manifest: {name}")
    for case in expected_ids:
        if int(case.split(".")[0]) <= 12 and manifest[f"testout_{case}.csv"] != manifest[output_name(case)]:
            raise AssertionError(f"Saved output aliases differ for case {case}")


def _check_report(metadata, summary, details):
    counts = dict(passed=int((summary.Status == "PASS").sum()), total=len(summary),
                  checks_passed=sum(record["passed"] for record in details), checks_total=len(details))
    for name, expected in counts.items():
        saved = metadata.get(name)
        if type(saved) is not int or saved != expected:
            raise AssertionError(f"Saved report {name} differs from recomputed results")
    if metadata.get("all_passed") is not bool((summary.Status == "PASS").all()):
        raise AssertionError("Saved report all_passed differs from recomputed results")
    saved_details = metadata.get("comparisons")
    if not isinstance(saved_details, list) or len(saved_details) != len(details):
        raise AssertionError("Saved report comparisons differ from recomputed results")
    for saved, computed in zip(saved_details, details):
        if not isinstance(saved, dict) or saved.keys() != computed.keys():
            raise AssertionError("Saved report comparison schema differs from recomputed results")
        for name, expected in computed.items():
            value = saved[name]
            if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                # CSV readers can differ in their last floating-point bit.
                matches = (isinstance(value, (int, float)) and not isinstance(value, bool)
                           and math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-12))
            else:
                matches = type(value) is type(expected) and value == expected
            if not matches:
                raise AssertionError(f"Saved report comparison {name} differs for case {computed['case']}")


def verify(output_dir=ROOT / "output", reference_dir=ROOT / "reference"):
    output, reference = Path(output_dir), Path(reference_dir)
    provenance = json.loads((ROOT / "SOURCE.json").read_text())
    for name, metadata in provenance["files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != metadata["sha256"]:
            raise AssertionError(f"Bundled instructor source changed: {name}")
    metadata = json.loads((output / "test_report.json").read_text())
    if not isinstance(metadata, dict) or not isinstance(metadata.get("groups"), list):
        raise ValueError("Saved report must contain a list of groups")
    groups = normalize_groups(metadata["groups"])
    if metadata["groups"] != groups:
        raise ValueError("Saved report groups must be unique and sorted")
    for name, minimum in (("simulations", 10000), ("seed", 0), ("univariate_simulations", 10000)):
        value = metadata.get(name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"Saved report {name} must be an integer >= {minimum}")
    if metadata.get("source_commit") != provenance["commit"]:
        raise AssertionError("Saved report source commit differs from the bundled instructor source")
    expected_ids = [case for case in CASE_IDS if int(case.split(".")[0]) in groups]
    manifest = json.loads((output / "output_manifest.json").read_text())
    _check_manifest(output, manifest, expected_ids)
    cases = {case: pd.read_csv(output / output_name(case), float_precision="round_trip") for case in expected_ids}
    diagnostic = json.loads((output / "diagnostics.json").read_text())
    summary, details = validate_cases(cases, diagnostic, reference)
    for case in set(expected_ids) & STOCHASTIC:
        count_name = "univariate_simulations" if case.startswith("8.") else "simulations"
        if diagnostic[case]["simulations"] != metadata[count_name]:
            raise AssertionError(f"Saved report simulation count differs from diagnostics for case {case}")
    _check_report(metadata, summary, details)
    saved_summary = pd.read_csv(output / "test_summary.csv", dtype={"Test": str}, float_precision="round_trip")
    try:
        pd.testing.assert_frame_equal(saved_summary, summary, check_dtype=False, rtol=1e-10, atol=1e-12)
    except AssertionError as error:
        raise AssertionError("Saved summary differs from recomputed results") from error
    if not all(record["passed"] for record in details):
        raise AssertionError("Saved numerical comparison failed")
    print(f"Source hashes and output hashes passed; {len(summary)}/{len(summary)} cases and {len(details)}/{len(details)} comparisons passed")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "reference")
    arguments = parser.parse_args()
    verify(arguments.output_dir, arguments.reference_dir)
