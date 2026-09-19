"""Run all in-scope cases, export calculated CSVs, and report PASS/FAIL."""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from functional_cases import CASE_IDS, DATA, ROOT, calculate, compare, output_name, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.output_dir.resolve() == args.data_dir.resolve():
        parser.error("Output directory must not overwrite the instructor reference data")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASE_IDS:
        try:
            actual = calculate(case, args.data_dir, args.seed)
            actual.to_csv(args.output_dir / output_name(case), index=False)
            expected = read(output_name(case), args.data_dir)
            metrics = compare(case, actual, expected, args.data_dir)
            row = {"test": case, "status": "PASS", **metrics}
        except Exception as exc:
            row = {"test": case, "status": "FAIL", "error": str(exc)}
        rows.append(row)
        print(f"{case:>3}: {row['status']}" + (f" — {row['error']}" if "error" in row else ""))
    passed = sum(r["status"] == "PASS" for r in rows)
    report = {"passed": passed, "total": len(rows), "seed": args.seed,
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__},
              "scope": "1.1–1.4, 2.1–2.3, 3.1–3.4, 4.1, 5.1–5.5, 7.1–7.6; inferred first checkpoint, not confirmed Canvas scope",
              "cases": rows}
    (args.output_dir / "test_report.json").write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(args.output_dir / "test_summary.csv", index=False)
    print(f"\n{passed}/{len(rows)} instructor cases passed")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
