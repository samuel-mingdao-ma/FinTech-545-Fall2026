"""One-command entry point that reproduces all outputs and the final PDF."""

from __future__ import annotations

import argparse

from analysis import main as run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=545)
    args = parser.parse_args()
    run_analysis(seed=args.seed)
    from build_report import build

    build()


if __name__ == "__main__":
    main()
