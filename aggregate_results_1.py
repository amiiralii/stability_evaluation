#!/usr/bin/env python3
"""Aggregate output CSVs from stability_evaluation 1.2.py runs.

This script scans the multi-folder layout produced by 1.2.py:
  outputs/1/<mode>/sta_perf_record/
  outputs/1/<mode>/budget_win_summary/

It produces aggregated tables in:
  outputs/1/<mode>/aggregates/aggregated_stability.csv
  outputs/1/<mode>/aggregates/aggregated_performance_mae.csv
  outputs/1/<mode>/aggregates/aggregated_win_rate.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple


def _read_metric(path: Path, x_key: str, y_key: str) -> Tuple[List[str], List[float]]:
    """Return (x_values, y_values) for a specific metric in a CSV file."""
    x_vals: List[str] = []
    y_vals: List[float] = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        
        # Flexibly find the x_key (e.g. 'budget' or 'budget_%')
        actual_x = x_key
        if x_key not in fields:
            if x_key == "budget" and "budget_%" in fields:
                actual_x = "budget_%"
            elif x_key == "label" and "budget" in fields:
                 # sometimes label is called budget in win_summary?
                 pass 

        if actual_x not in fields or y_key not in fields:
            raise ValueError(f"Missing columns {actual_x} or {y_key} in {path.name}. Found: {fields}")

        for row in reader:
            x_vals.append(row[actual_x])
            y_vals.append(float(row[y_key]))

    return x_vals, y_vals


def _write_pivot(out_path: Path, x_labels: List[str], rows: List[Tuple[str, List[float]]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", *x_labels])
        for dataset, values in sorted(rows):
            w.writerow([dataset, *values])


def aggregate_metric(artifact_dir: Path, file_pattern: str, x_key: str, y_key: str, out_path: Path) -> bool:
    """Aggregate all files in artifact_dir matching pattern into out_path."""
    files = sorted([p for p in artifact_dir.glob(file_pattern) if p.is_file()])
    if not files:
        return False

    x_ref: List[str] | None = None
    data_rows: List[Tuple[str, List[float]]] = []

    for path in files:
        try:
            xs, ys = _read_metric(path, x_key, y_key)
            if x_ref is None:
                x_ref = xs
            elif xs != x_ref:
                # Basic validation: ensure budgets align across all datasets
                continue 
            
            # Use original dataset name (remove suffix)
            dataset = path.name.replace(file_pattern.replace("*", ""), "")
            data_rows.append((dataset, ys))
        except Exception as e:
            print(f"  [Error] Skipping {path.name}: {e}")

    if not data_rows or not x_ref:
        return False

    _write_pivot(out_path, x_ref, data_rows)
    return True


def aggregate_mode_dir(mode_dir: Path) -> List[str]:
    """Aggregate Stability, Performance, and Win Rate for a specific mode folder."""
    written = []
    agg_dir = mode_dir / "aggregates"
    
    # 1. Stability (from sta_perf_record)
    if aggregate_metric(
        mode_dir / "sta_perf_record", "*_sta_perf_record.csv", 
        "budget", "stability", agg_dir / "aggregated_stability.csv"
    ):
        written.append("aggregated_stability.csv")

    # 2. Performance (from sta_perf_record)
    if aggregate_metric(
        mode_dir / "sta_perf_record", "*_sta_perf_record.csv", 
        "budget", "performance_mae", agg_dir / "aggregated_performance_mae.csv"
    ):
        written.append("aggregated_performance_mae.csv")

    # 3. Win Rate (from budget_win_summary)
    # Note: win_summary uses 'budget' as X and 'win_rate' as Y
    if aggregate_metric(
        mode_dir / "budget_win_summary", "*_budget_win_summary.csv", 
        "budget", "win_rate", agg_dir / "aggregated_win_rate.csv"
    ):
        written.append("aggregated_win_rate.csv")

    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./results/1", help="Root outputs directory (default: ./results/1)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        print(f"Root not found: {root}")
        return 1

    # Mode dirs are like 'percent_budget' and 'num_budget'
    mode_dirs = [p for p in root.iterdir() if p.is_dir() and "budget" in p.name]
    
    if not mode_dirs:
        # Fallback to root if it looks like a mode dir itself
        if (root / "sta_perf_record").exists():
            mode_dirs = [root]
        else:
            print(f"No budget mode directories found under {root}")
            return 1

    for mode_dir in mode_dirs:
        print(f"Processing mode: {mode_dir.name}")
        files = aggregate_mode_dir(mode_dir)
        for f in files:
            print(f"  [Created] {mode_dir.name}/aggregates/{f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

