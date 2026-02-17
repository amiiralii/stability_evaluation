import glob
import csv
import os

def summarize_datasets():
    """
    Summarize datasets from data/optimize/*/*.csv
    Output: dataset name, rows, objectives (columns ending with = or -), 
            features (columns not ending with +, y, or X)
    """
    csv_files = glob.glob("data/optimize/*/*.csv")
    
    results = []
    
    for filepath in sorted(csv_files):
        dataset_name = os.path.basename(filepath).replace('.csv', '')
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                
                # Count rows
                num_rows = sum(1 for _ in reader)
                
                # Count objectives (columns ending with = or -)
                num_objectives = sum(1 for col in header if col.endswith('+') or col.endswith('-'))
                
                # Count features (columns not ending with +, y, or X)
                num_features = sum(1 for col in header 
                                   if not col.endswith('+') 
                                   and not col.endswith('X')
                                   and not col.endswith('-'))
                
                results.append({
                    'dataset': dataset_name,
                    'rows': num_rows,
                    'features': num_features,
                    'objectives': num_objectives,
                })
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    import sys

    # Write to CSV if requested
    write_csv = len(sys.argv) > 1 and sys.argv[1].lower() == "csv"
    output_file = 'dataset_summary.csv'
    results_sorted = sorted(results, key=lambda x: x['dataset'])

    if write_csv:
        os.makedirs('results', exist_ok=True)
        csv_path = os.path.join('results', output_file)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['dataset', 'rows', 'objectives', 'features'])
            writer.writeheader()
            for row in results_sorted:
                writer.writerow(row)
        print(f"Summary written to {csv_path}")
    else:
        # Print header
        print(f"{'dataset':35s} {'rows':>8s} {'objectives':>11s} {'features':>9s}")
        print("-" * 68)
        for row in results_sorted:
            print(f"{row['dataset']:35s} {row['rows']:8d} {row['objectives']:11d} {row['features']:9d}")


    print(f"Total datasets: {len(results)}")

if __name__ == "__main__":
    summarize_datasets()
