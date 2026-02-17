"""
Aggregate CSV results from results/5.1/ directory.

Each per-dataset CSV has columns:
  trt, performance_error, stability_agreement
where trt = kmeans or tree.

Output: 2 pivot tables in results/5.1/aggregates/
  aggregated_performance_error.csv
  aggregated_stability_agreement.csv

Each pivot table has datasets as rows and treatments (kmeans, tree) as columns.
"""

import os
import argparse
import pandas as pd
from pathlib import Path


def aggregate_results(input_dir: str, output_dir: str = None):
    """
    Reads all per-dataset CSV files from input_dir and produces
    one aggregated pivot table per metric column.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path / "aggregates"
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect only dataset CSVs (skip the aggregates folder)
    csv_files = sorted(
        p for p in input_path.glob("*.csv")
        if p.is_file()
    )

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} dataset CSV files")

    # ── Read all CSVs ──────────────────────────────────────────
    all_data = {}       # {dataset: {metric: {trt: value}}}
    metrics = None       # metric column names (everything except trt)

    for csv_file in csv_files:
        dataset_name = csv_file.stem
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()

        trt_col = df.columns[0]  # "trt"

        if metrics is None:
            metrics = list(df.columns[1:])

        all_data[dataset_name] = {}
        for _, row in df.iterrows():
            treatment = str(row[trt_col]).strip()
            for metric in metrics:
                if metric not in all_data[dataset_name]:
                    all_data[dataset_name][metric] = {}
                all_data[dataset_name][metric][treatment] = row[metric]

    # ── Determine treatment order ──────────────────────────────
    first_dataset = list(all_data.keys())[0]
    treatments = list(all_data[first_dataset][metrics[0]].keys())

    print(f"Treatments: {treatments}")
    print(f"Metrics:    {metrics}")
    print(f"Datasets:   {len(all_data)} total")

    # ── Write one pivot CSV per metric ─────────────────────────
    for metric in metrics:
        rows = []
        for dataset_name in sorted(all_data.keys()):
            row = {"dataset": dataset_name}
            for treatment in treatments:
                row[treatment] = all_data[dataset_name].get(metric, {}).get(treatment, None)
            rows.append(row)

        result_df = pd.DataFrame(rows)
        out_file = output_path / f"aggregated_{metric}.csv"
        result_df.to_csv(out_file, index=False)
        print(f"\n[Created] {out_file}")
        print(result_df.head(10).to_string(index=False))
        if len(result_df) > 10:
            print(f"  ... ({len(result_df)} datasets total)")


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate 5.1.py per-dataset CSVs into pivot tables."
    )
    ap.add_argument(
        "--input", default="results/5.1",
        help="Directory containing per-dataset CSVs (default: results/5.1)"
    )
    ap.add_argument(
        "--output", default=None,
        help="Output directory for aggregated CSVs (default: <input>/aggregates)"
    )
    args = ap.parse_args()

    aggregate_results(args.input, args.output)


if __name__ == "__main__":
    main()
