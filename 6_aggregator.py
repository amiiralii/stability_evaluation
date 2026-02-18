import pandas as pd
import glob
import os
import re

def parse_csv_file(file_path):
    """
    Parse a single CSV file and extract all information into a dictionary.
    """
    data = {}
    
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    for line in lines:
        parts = [p.strip() for p in line.split(',')]
        
        if parts[0] == 'Performance':
            data['Performance_RMSE'] = float(parts[1])
            # Extract error list from the bracket format [0-20-15-...]
            error_str = parts[2].strip('[]')
            data['Performance_Error'] = error_str  # Keep as string, or convert to list
            # Also parse individual errors if needed
            error_values = [int(x) for x in error_str.split('-')]
            data['Performance_Error_List'] = error_values
        
        elif parts[0] == 'Stability':
            data['Stability_Agreement'] = int(parts[1])
            data['Stability_Std'] = float(parts[2])
        
        elif parts[0] == 'Dataset':
            data['Dataset'] = parts[1]
        
        elif parts[0] == 'Wins':
            data['Wins_Mean'] = float(parts[1])
            data['Wins_SD'] = float(parts[2])
        
        elif parts[0] == 'Dist8':
            data['Dist8'] = int(parts[1])
        
        elif parts[0] == 'Quant':
            data['Quant_Min'] = int(parts[1])
            data['Quant_Q1'] = int(parts[2])
            data['Quant_Median'] = int(parts[3])
            data['Quant_Q3'] = int(parts[4])
            data['Quant_Max'] = int(parts[5])
    
    # Add filename for reference
    data['Filename'] = os.path.basename(file_path)
    
    return data

def aggregate_results(results_dir='results/6'):
    """
    Aggregate all CSV files from the results directory into a single DataFrame.
    """
    # Find all CSV files
    csv_files = glob.glob(os.path.join(results_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return None
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Parse each file
    all_data = []
    for csv_file in sorted(csv_files):
        try:
            parsed_data = parse_csv_file(csv_file)
            all_data.append(parsed_data)
        except Exception as e:
            print(f"Error parsing {csv_file}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Reorder columns for better readability
    column_order = [
        'Filename',
        'Dataset',
        'Performance_RMSE',
        'Performance_Error',
        'Stability_Agreement',
        'Stability_Std',
        'Wins_Mean',
        'Wins_SD',
        'Dist8',
        'Quant_Min',
        'Quant_Q1',
        'Quant_Median',
        'Quant_Q3',
        'Quant_Max'
    ]
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    return df

if __name__ == '__main__':
    # Aggregate results
    df = aggregate_results('results/6')
    
    if df is not None:
        # Save to CSV
        output_file = 'results/6/aggregate/aggregated_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\nAggregated results saved to {output_file}")
        print(f"Total rows: {len(df)}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nColumn names: {list(df.columns)}")
    else:
        print("Failed to aggregate results")
