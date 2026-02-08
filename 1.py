import ezr
from stats import *
import sys
import random
import os

BUDGET_NUMS = [10, 20, 50, 100, 200]
BUDGET_PERCENTS = [0.1, 0.3, 0.5, 0.7, 0.9]

### Helper functions ###
def split_with_optima_in_test(data, random_seed=42):
    random.seed(random_seed)
    rows = data.rows[:]
    best_row = ezr.distysort(data, rows)[0]
    others = [r for r in rows if r is not best_row]
    others = ezr.shuffle(others)
    mid = len(others) // 2
    train_rows = others[:mid]
    test_rows = others[mid:]
    # Add optimal back to test
    test_rows.append(best_row)
    return ezr.clone(data, train_rows), ezr.clone(data, test_rows)


### main function ###
def run_sta_perf_check(ezr_the, use_budget_num, out_dir, random_seed=42, tree_num=20, test_sample_num=100, cv_threshold=0.2):
    # Steps: 
    # 1. For each budget: 
    #   a. build multiple trees with different seeds
    # 2. Stability check: 
    #   a. sample from test set
    #   b. for each row, get predictions from all trees
    #   c. compute agreement ratio
    #   d. store csv files for cross variations ({dataset_nmae}_budget_cv.csv and {dataset_name}_budget_stability.csv): 
    #       i. Tag, b1, ...bn, row_values; each cell under bn is the cv of the row
    # 3. Performance check: 
    #   a. pass all the rows from test set to each tree
    #   b. then we have 20 suggested optimal win score for each budget
    #       i. also, for each row, we have 20 * num_budget win scores
    #.      ii. we will give each row a id, and store the win scores under each budget (id, b1, ...bn, winners), winners determined by Top()
    #.  c. we will compute the performance of each budget as before with rmse
    # 4. Save the results of basic stability and performance into {dataset_name}_sta_perf_record.csv
    # 5. Save the result of cross variations into {dataset_name}_budget_cv.csv
    # 6. Save the result of performance by budget and rows into {dataset_name}_row_performance.csv
    # 7. If use_budget_nums, save to results/1/num_budget/ else results/1/percent_budget/ 
    # 8. For each dir, save a budget.csv, with b1, ...bn.
    
    # Win function setup (based on full dataset)

    dataset = ezr.Data(ezr.csv(ezr_the.file))
    if not dataset.cols.y:
        if dataset.cols.klass:
            dataset.cols.y = [dataset.cols.klass]
        else:
            print(f"Error: No objective columns found in {ezr_the.file}")
            return
    dataset.rows = ezr.shuffle(dataset.rows)

    # Win function setup (based on full dataset)
    distys = [ezr.disty(dataset, r) for r in dataset.rows]
    b4     = ezr.adds(distys)
    def win(v):
        return 100 * (1 - (v - b4.lo) / (b4.mu - b4.lo + 1e-32))
    
    train, test = split_with_optima_in_test(dataset, random_seed)
    
    budget_source = train.rows
    budget_range  = BUDGET_NUMS if use_budget_num else BUDGET_PERCENTS

    random.seed(random_seed)
    sampled_rows = random.sample(test.rows, min(test_sample_num, len(test.rows)))
    
    stability_results = []
    performance_results = []
    sampled_stability = {}

    # New: per-row budget winner tallies
    budget_labels = [f"b{i}" for i in range(len(budget_range))]
    budget_wins = {lbl: 0 for lbl in budget_labels}
    row_winners = {}
    
    for i, br in enumerate(budget_range): 
        ezr_the.Budget = max(ezr_the.Any, round(len(budget_source) * br)) if not use_budget_num else max(ezr_the.Any, br)
        pct = 100 * ezr_the.Budget / (len(train.rows) or 1)
        print(f"Budget: {ezr_the.Budget} | {pct:.1f}% of train", flush=True)
        ## generate trees for each budget
        trees = []
        for seed in range(1, tree_num + 1):
            random.seed(seed)
            labels = ezr.likely(train) if ezr_the.acq != "random" else train.rows[:ezr_the.Budget]
            tree   = ezr.Tree(ezr.clone(train, labels))
            trees.append(tree)

        ## stability check 
        agreements = 0
        for j, row in enumerate(sampled_rows):

            # use the disty rather than win score for stability check
            vals = [ezr.treeLeaf(tree, row).mu for tree in trees]
            stat = ezr.adds(vals)
            cv   = stat.sd / (stat.mu + 1e-32)
            if cv < cv_threshold: agreements += 1
            sampled_stability[j] = sampled_stability.get(j, {})
            sampled_stability[j][f"b{i}"] = cv
            if "raw_val" not in sampled_stability[j]: sampled_stability[j]["raw_val"] = row[:]
        
        stability_results.append(agreements / len(sampled_rows))

        # Store trees so we can do per-row comparisons later.
        # We'll stash them on a dict keyed by budget label.
        if i == 0:
            trees_by_budget = {}
        trees_by_budget[f"b{i}"] = trees

        # Optional: keep the old scalar performance metric placeholder
        # (we'll compute a clearer metric after per-row winner loop).
        performance_results.append(None)

    # --- Per-row winner analysis (your new idea) ---
    # For each test row, compare budgets by how well their 20 predictions
    # match the row's actual win score.
    for rid, row in enumerate(test.rows):
        actual = win(ezr.disty(dataset, row))
        errors_by_budget = {}
        for i, _br in enumerate(budget_range):
            lbl = f"b{i}"
            trees = trees_by_budget[lbl]
            preds = [win(ezr.treeLeaf(t, row).mu) for t in trees]
            # Use absolute error (lower is better) so `top()` can pick best budgets.
            errors_by_budget[lbl] = [abs(actual - p) for p in preds]

        winners = top(errors_by_budget, reverse=False)
        row_winners[rid] = sorted(winners)
        for w in winners:
            budget_wins[w] += 1

    # Summarize win-rates per budget
    total_rows = len(test.rows)
    win_rates = {lbl: budget_wins[lbl] / (total_rows or 1) for lbl in budget_labels}

    # Also compute a scalar "performance" per budget as mean absolute error across rows+trees
    # (this is separate from the top()-based win counting).
    for i, _br in enumerate(budget_range):
        lbl = f"b{i}"
        trees = trees_by_budget[lbl]
        all_abs_err = []
        for row in test.rows:
            actual = win(ezr.disty(dataset, row))
            for t in trees:
                pred = win(ezr.treeLeaf(t, row).mu)
                all_abs_err.append(abs(actual - pred))
        performance_results[i] = sum(all_abs_err) / (len(all_abs_err) or 1)

    # --- Outputs ---
    # Layout (so files from different datasets live together):
    #   {out_dir}/
    #     sta_perf_record/{dataset}_sta_perf_record.csv
    #     budget_cv/{dataset}_budget_cv.csv
    #     budget_win_summary/{dataset}_budget_win_summary.csv
    #     row_budget_winners/{dataset}_row_budget_winners.csv
    #     budgets/{dataset}_budgets.csv
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(ezr_the.file).split('.')[0]

    rec_dir = os.path.join(out_dir, "sta_perf_record")
    cv_dir = os.path.join(out_dir, "budget_cv")
    sum_dir = os.path.join(out_dir, "budget_win_summary")
    row_dir = os.path.join(out_dir, "row_budget_winners")
    bud_dir = os.path.join(out_dir, "budgets")
    for d in (rec_dir, cv_dir, sum_dir, row_dir, bud_dir):
        os.makedirs(d, exist_ok=True)

    # 1) budget list (per dataset)
    budgets_path = os.path.join(bud_dir, f"{base_name}_budgets.csv")
    with open(budgets_path, "w") as f:
        f.write("label,budget\n")
        for i, br in enumerate(budget_range):
            f.write(f"b{i},{br}\n")

    # 2) basic stability + scalar performance (MAE)
    record_path = os.path.join(rec_dir, f"{base_name}_sta_perf_record.csv")
    with open(record_path, "w") as f:
        f.write("budget,label,stability,performance_mae\n")
        for i, br in enumerate(budget_range):
            f.write(f"{br},b{i},{stability_results[i]:.6f},{performance_results[i]:.6f}\n")

    # 3) per-row winners + win-rates
    win_path = os.path.join(row_dir, f"{base_name}_row_budget_winners.csv")
    with open(win_path, "w") as f:
        f.write("row_id,winners\n")
        for rid in range(total_rows):
            f.write(f"{rid},{'|'.join(row_winners.get(rid, []))}\n")

    summary_path = os.path.join(sum_dir, f"{base_name}_budget_win_summary.csv")
    with open(summary_path, "w") as f:
        f.write("label,budget,win_count,win_rate\n")
        for i, br in enumerate(budget_range):
            lbl = f"b{i}"
            f.write(f"{lbl},{br},{budget_wins[lbl]},{win_rates[lbl]:.6f}\n")

    # 4) sampled stability CVs (one row per sampled row)
    cv_path = os.path.join(cv_dir, f"{base_name}_budget_cv.csv")
    with open(cv_path, "w") as f:
        header = ["row_id"] + [f"b{i}" for i in range(len(budget_range))] + ["raw_val"]
        f.write(",".join(header) + "\n")
        for rid, d in sampled_stability.items():
            cells = [str(rid)]
            for i in range(len(budget_range)):
                cells.append(str(d.get(f"b{i}", "")))
            cells.append("|".join(map(str, d.get("raw_val", []))))
            f.write(",".join(cells) + "\n")

    print(f"\nSaved: {record_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {win_path}")
    print(f"Saved: {cv_path}")
    print(f"Saved: {budgets_path}")


### running ### 
if __name__ == "__main__":

    ezr.the.file = sys.argv[1] if len(sys.argv) > 1 else "data/optimize/misc/auto93.csv"
    budget_mode = sys.argv[2] if len(sys.argv) > 2 else "percent"  # 'num' or 'percent'
    use_budget_num = (budget_mode == "num")
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "./outputs/1/"
    # Adjust output directory based on budget type
    output_dir = os.path.join(output_dir, "num_budget" if use_budget_num else "percent_budget")
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    run_sta_perf_check(ezr.the, use_budget_num, output_dir)