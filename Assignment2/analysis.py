"""Run all five Assignment 2 analyses and save reproducible results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from risk545.problem4 import solve_problem4
from risk545.problems12 import solve_problem1, solve_problem2
from risk545.problems35 import solve_problem3, solve_problem5


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
FIGURES = ROOT / "figures"


def main(seed: int = 545) -> dict:
    """Calculate every submitted number and recreate every submitted figure."""

    OUTPUT.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    results = {
        "metadata": {
            "seed": int(seed),
            "var_es_sign": "positive loss",
            "default_alpha": 0.05,
        },
        "problem1": solve_problem1(ROOT / "problem1.csv", FIGURES),
        "problem2": solve_problem2(ROOT / "problem2.csv", FIGURES),
        "problem3": solve_problem3(ROOT / "problem3.csv", FIGURES, seed=seed + 3),
        "problem4": solve_problem4(ROOT / "problem4.csv", FIGURES, seed=seed + 4),
        "problem5": solve_problem5(ROOT / "problem5.csv", FIGURES, seed=seed + 5),
    }
    output_path = OUTPUT / "results.json"
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output_path}")
    return results


if __name__ == "__main__":
    main()
