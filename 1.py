"""
1.3.py — Stability & Performance evaluation across budget levels.

For each budget, builds 20 trees. Then:
  1. Performance: For each repeat, uses tree to pick best rows from holdout
     and measures error vs true holdout-best. Uses top() across budgets
     to count per-repeat performance winners.
  2. Stability: For each test row, measures prediction sd across 20 trees.
     Uses top() across budgets to count per-row stability winners.

Output: one CSV per dataset under results/1.3/{dataset}.csv
  Columns: trt, performance_error, stability_agreement, best_performance, best_stability
"""
from ezr import *
from stats import *
import sys
import random
import os

BUDGET_NUMS = [10, 20, 50, 100, 200]

def run(file_directory, out_dir="results/1.3", repeats=20):
    all_data = Data(csv(file_directory))
    if not all_data.cols.y:
        if all_data.cols.klass:
            all_data.cols.y = [all_data.cols.klass]
        else:
            print(f"Error: No objective columns found in {file_directory}")
            return

    ys      = [disty(all_data, row) for row in all_data.rows]
    b4      = adds(ys)
    win     = lambda v: int(100 * (1 - (v - b4.lo) / (b4.mu - b4.lo)))
    b4_wins = adds([win(k) for k in ys])

    # =========================================================
    # Part 1: Performance
    # =========================================================
    # For each budget, run `repeats` train/holdout splits.
    # Each repeat: build tree, pick top-10 rows from holdout by tree,
    # measure actual best win among those vs true holdout best.
    # Collect error distributions per budget, then use top() to find winners.
    performance_error = {}
    error_dist = {}

    for budget in BUDGET_NUMS:
        the.Budget = budget
        mse = 0
        error = []
        for rand_seed in range(repeats):
            random.seed(rand_seed)
            shuffled_rows = random.sample(all_data.rows, len(all_data.rows))
            half = int(0.5 * len(all_data.rows))
            train    = clone(all_data, shuffled_rows[:half])
            holdout  = clone(all_data, shuffled_rows[half:])
            labels   = likely(train) if budget < len(train.rows) else train.rows[:budget]
            tree     = Tree(clone(train, labels))
            top_rows = sorted(
                [(treeLeaf(tree, row).mu, row) for row in holdout.rows],
                key=lambda x: x[0]
            )[:the.Check]
            ezr_perf = win(sorted([disty(all_data, row) for _, row in top_rows])[0])
            ref_opt  = win(min(disty(all_data, row) for row in holdout.rows))
            mse += abs(ezr_perf - ref_opt) ** 2
            error.append(ref_opt - ezr_perf)
        error_dist[budget]       = error
        performance_error[budget] = (mse / repeats) ** 0.5

    # Which budgets are statistically best for performance?
    best_performances = top(error_dist, Ks=0.9, Delta="medium")

    # =========================================================
    # Part 2: Stability
    # =========================================================
    # Single train/test split. For each budget build 20 trees.
    # For each test row, measure sd of predictions across trees.
    # Then use top() per-row to find which budget is most stable.
    all_data.rows = shuffle(all_data.rows)
    tests_size = min(100, int(len(all_data.rows) * 0.3))
    test  = clone(all_data, all_data.rows[:tests_size])
    train = clone(all_data, all_data.rows[tests_size:])

    stability_agreement = {}
    # For each test row, store the sd under each budget
    stability_comp = [{budget: 0 for budget in BUDGET_NUMS} for _ in range(len(test.rows))]

    for budget in BUDGET_NUMS:
        the.Budget = budget
        trees = []
        for rand_seed in range(repeats):
            the.seed = rand_seed
            random.seed(the.seed)
            labels = likely(train)
            tree   = Tree(clone(train, labels))
            trees.append(tree)

        agreement = 0
        for idx, row in enumerate(test.rows):
            outputs = [win(treeLeaf(tree, row).mu) for tree in trees]
            preds   = adds(outputs)
            stability_comp[idx][budget] = preds.sd
            if preds.sd < 0.35 * b4_wins.sd:
                agreement += 1

        stability_agreement[budget] = agreement * 100 // tests_size

    # Per-row stability winners via top()
    best_stability = {budget: 0 for budget in BUDGET_NUMS}
    for row_stability in stability_comp:
        bests_in_row = top({k: [v] for k, v in row_stability.items()}, Ks=0.9, Delta="medium")
        for m in bests_in_row:
            best_stability[m] += 1

    # =========================================================
    # Output
    # =========================================================
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(file_directory).split('.')[0]
    out_path  = os.path.join(out_dir, f"{base_name}.csv")

    header = "trt, performance_error, stability_agreement, best_performance, best_stability"
    print(header)
    lines = [header]
    for budget in BUDGET_NUMS:
        line = (
            f"{budget}, "
            f"{performance_error[budget]:.2f}, "
            f"{stability_agreement[budget]}, "
            f"{1 if budget in best_performances else 0}, "
            f"{best_stability[budget]}"
        )
        print(line)
        lines.append(line)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    file_directory = sys.argv[1] if len(sys.argv) > 1 else "data/optimize/misc/auto93.csv"
    output_dir     = sys.argv[2] if len(sys.argv) > 2 else "results/1.3"
    run(file_directory, output_dir)
