"""Recheck saved outputs, source hashes and teacher comparisons without simulation."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from risk545_ft2.validation import CASE_IDS, output_name, validate_cases

ROOT = Path(__file__).resolve().parent


def verify(output_dir=ROOT / "output", reference_dir=ROOT / "reference"):
    output, reference = Path(output_dir), Path(reference_dir)
    provenance = json.loads((ROOT / "SOURCE.json").read_text())
    for name, metadata in provenance["files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != metadata["sha256"]:
            raise AssertionError(f"Bundled instructor source changed: {name}")
    manifest = json.loads((output / "output_manifest.json").read_text())
    for name, digest in manifest.items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"Output differs from the completed-run manifest: {name}")
    metadata = json.loads((output / "test_report.json").read_text())
    expected_ids = [case for case in CASE_IDS if int(case.split(".")[0]) in metadata["groups"]]
    cases = {case: pd.read_csv(output / output_name(case)) for case in expected_ids}
    diagnostic = json.loads((output / "diagnostics.json").read_text())
    summary, details = validate_cases(cases, diagnostic, reference)
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
