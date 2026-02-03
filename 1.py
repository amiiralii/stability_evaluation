import ezr
import random
import os
import matplotlib.pyplot as plt
from ezr import *
import sys
    
def split_with_optima_in_test(data, random_seed=42):
    """
    Identifies the global optimal row, removes it, splits the rest,
    and ensures the optimal row is in the test set.
    """
    random.seed(random_seed)

    rows = data.rows[:] # copy to avoid modifying original
    # Identify global optima
    best_row = distysort(data, rows)[0]
    
    # Remove it from the pool
    others = [r for r in rows if r is not best_row]
    
    # Shuffle and split
    others = shuffle(others)
    mid = len(others) // 2
    train_rows = others[:mid]
    test_rows = others[mid:]
    
    # Add optimal back to test
    test_rows.append(best_row)
    
    return clone(data, train_rows), clone(data, test_rows)

def plot_sta_perf(percentages, stability_results, performance_results, file_name, out_file):
    """
    Draws a chart with budget percentage as X-axis and dual Y-axes for 
    Stability and Performance.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = [p * 100 for p in percentages]

    # First Y-axis (Stability)
    color1 = 'tab:blue'
    ax1.set_xlabel('Budget Percentage (%)')
    ax1.set_ylabel('Stability (Agreement Ratio)', color=color1)
    ax1.plot(x, stability_results, marker='o', linestyle='-', color=color1, label='Stability')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 1.1)

    # Second Y-axis (Performance)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Performance (Accuracy Score)', color=color2)
    ax2.plot(x, performance_results, marker='s', linestyle='--', color=color2, label='Performance')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 110)

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='lower right')

    plt.title(f"Stability & Performance vs Budget\n({file_name})")
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"\nChart saved as {out_file}")
    plt.close()

def run_sta_perf_check(repeats=20, n_samples=100):
    """
    Runs both stability and performance metrics.
    """
    # Initialize dataset
    dataset = Data(csv(the.file))
    
    # Output setup
    out_dir = "sta_perf_outputs/"
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(the.file).split('.')[0]
    out_path = os.path.join(out_dir, f"{base_name}_sta_perf.png")
    
    # Column validation
    if not dataset.cols.y:
        if dataset.cols.klass:
            dataset.cols.y = [dataset.cols.klass]
        else:
            print(f"Error: No objective columns found in {the.file}")
            return

    # Win function setup (based on full dataset)
    distys = [ezr.disty(dataset, r) for r in dataset.rows]
    b4 = ezr.adds(distys)
    def win(v):
        return 100 * (1 - (v - b4.lo) / (b4.mu - b4.lo + 1e-32))

    # Global Optima Info
    best_row_global = distysort(dataset, dataset.rows)[0]
    best_disty_global = ezr.disty(dataset, best_row_global)
    print(f"Dataset: {the.file} | Rows: {len(dataset.rows)}")
    print(f"Global Best Disty: {best_disty_global:.4f} | Win Score: {win(best_disty_global):.2f}")

    # Split for Experiment
    train_data, test_data = split_with_optima_in_test(dataset)
    budget_source = train_data.rows
    
    percentages = [0.1, 0.3, 0.5, 0.7, 0.9]
    stability_results = []
    performance_results = []
    
    for pct in percentages:
        the.Budget = max(the.Any, round(len(budget_source) * pct))
        print(f"Budget: {int(pct*100)}% ({the.Budget})", end=" ... ", flush=True)
        
        trees = []
        errors = []
        
        for seed in range(1, repeats + 1):
            random.seed(seed)
            # Active learning and tree building
            labels = likely(clone(dataset, budget_source))
            tree = Tree(clone(dataset, labels))
            trees.append(tree)
            
            # --- Performance Metric: Tree-Suggested Best ---
            # Use tree to find best candidate in test_data 
            # The tree evaluates test rows based on the leaf they fall into (leaf.mu)
            best_candidate = min(test_data.rows, key=lambda r: ezr.treeLeaf(tree, r).mu)
            
            # Evaluate the actual quality of that candidate using global 'win' function
            actual_dist = ezr.disty(dataset, best_candidate)
            actual_win = win(actual_dist)
            
            # Error = Squared difference of the actual score from ideal (100)
            error = (actual_win - 100) ** 2
            errors.append(error)

        # 1. Stability (Agreement Ratio)
        random.seed(42)
        test_rows_subset = random.sample(test_data.rows, min(n_samples, len(test_data.rows)))
        agreements = 0
        for row in test_rows_subset:
            vals = [win(ezr.treeLeaf(t, row).mu) for t in trees]
            stat = ezr.adds(vals)
            cv = stat.sd / (stat.mu + 1e-32)
            if cv < 0.2: agreements += 1
        
        stability_results.append(agreements / len(test_rows_subset))

        # 2. Performance (Accuracy Score)
        mse = sum(errors) / len(errors)
        rmse = mse ** 0.5
        accuracy = max(0, 100 - rmse)
        performance_results.append(accuracy)
        
        print(f"Stab: {stability_results[-1]:.2f}, Perf: {performance_results[-1]:.2f}")

    # Plot
    plot_sta_perf(percentages, stability_results, performance_results, base_name, out_path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        the.file = sys.argv[1]
    else:
        # Default
        the.file = "data/optimize/systems/PostgreSQL.csv"
        
    run_sta_perf_check()


    # peformance = 100 - rmse(tree_suggested_best, best_row_global)
    # rmse = math.sqrt(mse)
    # oops