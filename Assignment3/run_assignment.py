"""Reproduce Assignment 3 numerical results and figures."""
import argparse
import hashlib
import importlib
import os
import platform
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import scipy

from assignment3.common import write_json

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"fintech545-assignment3-matplotlib"))


def run(problem="all", data_dir=ROOT/"data", output_dir=ROOT/"results", figure_dir=ROOT/"figures",
        n_sim=1000000, tree_n=500, n_boot=300, seed=545):
    data_dir, output_dir, figure_dir = map(Path, (data_dir, output_dir, figure_dir))
    if n_sim < 10000 or tree_n < 5 or tree_n % 5 or n_boot < 0:
        raise ValueError("Need simulations >=10000, tree N a positive multiple of5, bootstrap >=0")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    chosen = list(range(1, 6)) if str(problem) == "all" else [int(problem)]
    metadata = dict(seed=seed, simulations=n_sim, tree_n=tree_n, bootstrap=n_boot,
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  pandas=pd.__version__, scipy=scipy.__version__),
                    source_repository="https://github.com/dompazz/FinTech-545-Fall2026/tree/5958d11b51960c18be98f6df2b97c4e80670fb2d/Assignments/Assignment3",
                    inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(data_dir.iterdir()) if p.suffix.lower()==".csv"})
    summaries = {}
    for number in chosen:
        started = time.perf_counter()
        module = importlib.import_module(f"assignment3.q{number}")
        print(f"Running Problem {number}", flush=True)
        summaries[f"q{number}"] = module.run(data_dir, output_dir, figure_dir,
                                            n_sim=n_sim, tree_n=tree_n, n_boot=n_boot, seed=seed)
        print(f"Problem {number} completed in {time.perf_counter()-started:.2f}s", flush=True)
    metadata["problems"] = chosen
    write_json(output_dir/"run_metadata.json", metadata)
    write_json(output_dir/"run_summaries.json", summaries)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=["all", "1", "2", "3", "4", "5"], default="all")
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results")
    parser.add_argument("--figure-dir", type=Path, default=ROOT/"figures")
    parser.add_argument("--simulations", type=int, default=1000000)
    parser.add_argument("--tree-n", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=545)
    args = parser.parse_args()
    run(args.problem, args.data_dir, args.output_dir, args.figure_dir,
        args.simulations, args.tree_n, args.bootstrap, args.seed)
