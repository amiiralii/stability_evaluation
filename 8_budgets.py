"""
8_budgets.py — Wrapper around 8.py to compare ezr vs causal_ezr
across different leaf node sizes (the.leaf = 3, 5, 7, 9).

Usage:
  # Run a single (dataset, leaf_size) combo — for SLURM parallel dispatch:
  python 8_budgets.py data/optimize/misc/auto93.csv results/8 5

  # Run all leaf sizes for one dataset (sequential):
  python 8_budgets.py data/optimize/misc/auto93.csv

  # Aggregate only (after all parallel jobs are done):
  python 8_budgets.py --aggregate results/8

Output per run:
  results/8/leaf_{size}/{dataset}.csv

Aggregated (after all runs):
  results/8/leaf_comparison/{dataset}.csv
"""
from ezr import *
import sys
import os
import glob

# Import 8.py's run function
from importlib import import_module
mod8 = import_module("8")

LEAF_SIZES = [3, 5, 7, 9]


def run_single(file_directory, base_out_dir, leaf_size):
    """Run 8.py for one dataset at one leaf size."""
    print(f"\n{'='*60}")
    print(f"  the.leaf = {leaf_size}  |  {os.path.basename(file_directory)}")
    print(f"{'='*60}")
    the.leaf = leaf_size
    out_dir  = os.path.join(base_out_dir, f"leaf_{leaf_size}")
    mod8.run(file_directory, out_dir=out_dir)


def run_all_leaves(file_directory, base_out_dir="results/8"):
    """Run 8.py for one dataset across ALL leaf sizes (sequential)."""
    for leaf in LEAF_SIZES:
        run_single(file_directory, base_out_dir, leaf)
    aggregate(base_out_dir)


def aggregate(base_out_dir="results/8"):
    """Aggregate all results/8/leaf_*/*.csv into results/8/leaf_comparison/*.csv."""
    agg_dir = os.path.join(base_out_dir, "leaf_comparison")
    os.makedirs(agg_dir, exist_ok=True)

    # Collect all dataset names across all leaf dirs
    all_datasets = set()
    for leaf in LEAF_SIZES:
        leaf_dir = os.path.join(base_out_dir, f"leaf_{leaf}")
        if os.path.isdir(leaf_dir):
            for f in os.listdir(leaf_dir):
                if f.endswith(".csv"):
                    all_datasets.add(f)

    if not all_datasets:
        print("No result CSVs found to aggregate.")
        return

    header = "leaf_size, trt, performance_error, stability_agreement, best_performance, best_stability"

    for ds_file in sorted(all_datasets):
        agg_lines = [header]
        for leaf in LEAF_SIZES:
            csv_path = os.path.join(base_out_dir, f"leaf_{leaf}", ds_file)
            if not os.path.exists(csv_path):
                continue
            with open(csv_path) as f:
                for i, line in enumerate(f):
                    if i == 0:
                        continue  # skip header
                    line = line.strip()
                    if line:
                        agg_lines.append(f"{leaf}, {line}")

        agg_path = os.path.join(agg_dir, ds_file)
        with open(agg_path, "w") as f:
            f.write("\n".join(agg_lines) + "\n")
        print(f"  Aggregated: {agg_path}")

    print(f"\nAggregation complete. Results in: {agg_dir}")


if __name__ == "__main__":
    # Mode 1: --aggregate only
    if len(sys.argv) >= 2 and sys.argv[1] == "--aggregate":
        base_dir = sys.argv[2] if len(sys.argv) > 2 else "results/8"
        aggregate(base_dir)

    # Mode 2: single (dataset, leaf_size) — for SLURM parallel dispatch
    elif len(sys.argv) >= 4:
        file_directory = sys.argv[1]
        out_dir        = sys.argv[2]
        leaf_size      = int(sys.argv[3])
        run_single(file_directory, out_dir, leaf_size)

    # Mode 3: all leaf sizes for one dataset (sequential)
    elif len(sys.argv) >= 2:
        file_directory = sys.argv[1]
        out_dir        = sys.argv[2] if len(sys.argv) > 2 else "results/8"
        run_all_leaves(file_directory, out_dir)

    else:
        run_all_leaves("data/optimize/misc/auto93.csv")
