"""One-command runner for all Assignment 1 outputs and the final PDF."""

from analysis import main as run_analysis


def main() -> None:
    run_analysis()
    from build_report import build as build_pdf

    build_pdf()


if __name__ == "__main__":
    main()
