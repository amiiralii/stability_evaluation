"""
adaptive_8.py — EZR vs Causal EZR vs Adaptive Causal EZR

Three treatments, identical conditions (same splits, seeds, budget=50, likely() rows).

Treatments:
  ezr                 — standard Tree() from ezr.py (correlation-based splits)
  causal_ezr          — causalTree() with fixed the.leaf
  adaptive_causal_ezr — causalTree(adaptive_leaf=True):
                         leaf size scales up proportionally to compensate for
                         features removed by confounder detection.
                         Formula: ceil(the.leaf × original_features / remaining_features)
                         Clamped to [the.leaf, 2 × the.leaf].

Prediction:
  ezr:                treeLeaf(tree, row).mu
  causal_ezr:         causalTreeLeaf(data, tree, row).mu
  adaptive_causal_ezr: causalTreeLeaf(data, tree, row).mu

Part 1 — Performance (20 train/holdout splits per treatment):
  Pick top-Check holdout rows → best win → RMSE vs true holdout best.

Part 2 — Stability (single shared train/test split, 20 models per treatment):
  Build 20 models (different seeds).
  For each test row, 20 predictions → sd.
  sd < 0.35 * b4_wins.sd → stable.

Output: one CSV per dataset under results/adaptive_8/{dataset}.csv
  Columns: trt, performance_error, stability_agreement, best_performance, best_stability
"""
from ezr import *
from causal_tools import causalTree, causalTreeLeaf
from stats import *
import sys
import random
import os

TREATMENTS = ["ezr", "causal_ezr", "adaptive_causal_ezr"]


# =============================================================================
# Main experiment
# =============================================================================

def run(file_directory, out_dir="results/adaptive_8", repeats=20):
    all_data = Data(csv(file_directory))
    if not all_data.cols.y:
        if all_data.cols.klass:
            all_data.cols.y = [all_data.cols.klass]
        else:
            print(f"Error: No objective columns found in {file_directory}")
            return

    # Global baseline stats
    ys      = [disty(all_data, row) for row in all_data.rows]
    b4      = adds(ys)
    win     = lambda v: int(100 * (1 - (v - b4.lo) / (b4.mu - b4.lo)))
    b4_wins = adds([win(k) for k in ys])

    budget     = 50
    the.Budget = budget
    the.Check  = 10

    # =========================================================
    # Part 1: Performance (20 train/holdout splits per treatment)
    # =========================================================
    performance_error = {}
    error_dist        = {}   # {trt: [per-seed errors]} for top()

    for trt in TREATMENTS:
        mse    = 0
        errors = []
        for rand_seed in range(repeats):
            random.seed(rand_seed)
            the.seed      = rand_seed
            shuffled_rows = random.sample(all_data.rows, len(all_data.rows))
            half    = int(0.5 * len(all_data.rows))
            train   = clone(all_data, shuffled_rows[:half])
            holdout = clone(all_data, shuffled_rows[half:])

            # All treatments use the same labeled rows from likely()
            labels  = likely(train)
            sampled = labels[:budget]

            if trt == "ezr":
                tree     = Tree(clone(train, labels))
                top_rows = sorted(
                    [(treeLeaf(tree, row).mu, row) for row in holdout.rows],
                    key=lambda x: x[0]
                )[:the.Check]

            elif trt == "causal_ezr":
                sampled_data = clone(all_data, sampled)
                ctree        = causalTree(sampled_data, adaptive_leaf=False)
                top_rows     = sorted(
                    [(causalTreeLeaf(sampled_data, ctree, row).mu, row)
                     for row in holdout.rows],
                    key=lambda x: x[0]
                )[:the.Check]

            else:  # adaptive_causal_ezr
                sampled_data = clone(all_data, sampled)
                ctree        = causalTree(sampled_data, adaptive_leaf=True)
                top_rows     = sorted(
                    [(causalTreeLeaf(sampled_data, ctree, row).mu, row)
                     for row in holdout.rows],
                    key=lambda x: x[0]
                )[:the.Check]

            ezr_perf = win(sorted([disty(all_data, row) for _, row in top_rows])[0])
            ref_opt  = win(min(disty(all_data, row) for row in holdout.rows))
            err      = ref_opt - ezr_perf
            errors.append(err)
            mse     += abs(err) ** 2

        error_dist[trt]        = errors
        performance_error[trt] = (mse / repeats) ** 0.5

    # Statistical ranking of performance
    pooled_sd = adds([e for errs in error_dist.values() for e in errs]).sd
    best_performances = top(error_dist, Ks=0.9, Delta="medium", eps=pooled_sd * 0.35)

    # =========================================================
    # Part 2: Stability (single shared train/test split)
    # =========================================================
    random.seed(42)
    all_data.rows = shuffle(all_data.rows)
    half       = len(all_data.rows) // 2
    train_rows = all_data.rows[:half]
    test_pool  = all_data.rows[half:]
    tests_size = min(100, len(test_pool))
    test_rows  = test_pool[:tests_size]

    train = clone(all_data, train_rows)
    test  = clone(all_data, test_rows)

    # Build 20 models per treatment (shared labels across treatments)
    ezr_models      = []
    causal_models    = []  # fixed leaf
    adaptive_models  = []  # adaptive leaf

    for rand_seed in range(repeats):
        the.seed = rand_seed
        random.seed(rand_seed)
        labels  = likely(train)
        sampled = labels[:budget]

        # ezr: standard tree
        ezr_models.append(Tree(clone(train, labels)))

        # causal_ezr: fixed leaf
        sampled_data = clone(all_data, sampled)
        causal_models.append((sampled_data, causalTree(sampled_data, adaptive_leaf=False)))

        # adaptive_causal_ezr: adaptive leaf
        sampled_data2 = clone(all_data, sampled)
        adaptive_models.append((sampled_data2, causalTree(sampled_data2, adaptive_leaf=True)))

    stability_agreement = {}

    # Compute win-score predictions for every (treatment, model, test row)
    all_win_scores = {}
    for trt in TREATMENTS:
        per_row = []
        for row in test.rows:
            if trt == "ezr":
                win_scores = [win(treeLeaf(tree, row).mu)
                              for tree in ezr_models]
            elif trt == "causal_ezr":
                win_scores = [win(causalTreeLeaf(sdata, ctree, row).mu)
                              for sdata, ctree in causal_models]
            else:  # adaptive_causal_ezr
                win_scores = [win(causalTreeLeaf(sdata, ctree, row).mu)
                              for sdata, ctree in adaptive_models]
            per_row.append(win_scores)
        all_win_scores[trt] = per_row

    # Threshold-based stability agreement
    for trt in TREATMENTS:
        agreement = 0
        for win_scores in all_win_scores[trt]:
            if adds(win_scores).sd < 0.35 * b4_wins.sd:
                agreement += 1
        stability_agreement[trt] = agreement * 100 // tests_size

    # top()-based stability: per row, which treatment is more stable?
    best_stability = {trt: 0 for trt in TREATMENTS}
    for row_idx in range(tests_size):
        row_sd = {trt: adds(all_win_scores[trt][row_idx]).sd
                  for trt in TREATMENTS}
        pooled_sd = adds([sds for sds in row_sd.values()]).sd
        bests_in_row = top({k: [v] for k, v in row_sd.items()},
                          Ks=0.9, Delta="medium", eps=pooled_sd * 0.35)
        for trt in bests_in_row:
            best_stability[trt] += 1

    # =========================================================
    # Output
    # =========================================================
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(file_directory).split('.')[0]
    out_path  = os.path.join(out_dir, f"{base_name}.csv")

    header = "trt, performance_error, stability_agreement, best_performance, best_stability"
    print(header)
    lines = [header]
    for trt in TREATMENTS:
        bp = 1 if trt in best_performances else 0
        bs = best_stability[trt]
        line = f"{trt}, {performance_error[trt]:.2f}, {stability_agreement[trt]}, {bp}, {bs}"
        print(line)
        lines.append(line)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    file_directory = sys.argv[1] if len(sys.argv) > 1 else "data/optimize/misc/auto93.csv"
    output_dir     = sys.argv[2] if len(sys.argv) > 2 else "results/adaptive_8"
    run(file_directory, output_dir)
