"""
Aggregate CSV results from results/2/ directory.
Each row represents a dataset, each column represents a treatment.
"""

import os
import pandas as pd
from pathlib import Path

def aggregate_results(input_dir: str, output_dir: str = None):
    """
    Reads all CSV files from input_dir and aggregates them.
    
    Each CSV file has treatments in first column and metrics in other columns.
    Output will have treatments as columns and datasets as rows.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    
    # Find all CSV files
    csv_files = sorted(input_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Read all CSV files
    all_data = {}
    metrics = None
    
    for csv_file in csv_files:
        dataset_name = csv_file.stem  # filename without extension
        df = pd.read_csv(csv_file)
        
        # Clean column names (strip whitespace)
        df.columns = df.columns.str.strip()
        
        # Get treatment column name (first column)
        trt_col = df.columns[0]
        
        # Get metric columns (all except first)
        if metrics is None:
            metrics = list(df.columns[1:])
        
        # Store data with treatment as key
        all_data[dataset_name] = {}
        for _, row in df.iterrows():
            treatment = row[trt_col].strip() if isinstance(row[trt_col], str) else row[trt_col]
            for metric in metrics:
                if metric not in all_data[dataset_name]:
                    all_data[dataset_name][metric] = {}
                all_data[dataset_name][metric][treatment] = row[metric]
    
    # Get all treatments (from first dataset, assuming all have same treatments)
    first_dataset = list(all_data.keys())[0]
    first_metric = metrics[0]
    treatments = list(all_data[first_dataset][first_metric].keys())
    
    print(f"Treatments: {treatments}")
    print(f"Metrics: {metrics}")
    print(f"Datasets: {list(all_data.keys())}")
    
    # Create aggregated DataFrames for each metric
    for metric in metrics:
        rows = []
        for dataset_name in all_data.keys():
            row = {"dataset": dataset_name}
            for treatment in treatments:
                row[treatment] = all_data[dataset_name][metric].get(treatment, None)
            rows.append(row)
        
        result_df = pd.DataFrame(rows)
        
        # Save to CSV
        output_file = output_path / f"aggregated_{metric}.csv"
        result_df.to_csv(output_file, index=False)
        print(f"\nSaved {metric} aggregation to: {output_file}")
        print(result_df.to_string(index=False))


if __name__ == "__main__":
    # Default paths
    input_dir = "results/2"
    output_dir = "results/2/aggregates"
    
    aggregate_results(input_dir, output_dir)

