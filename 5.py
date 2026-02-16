"""
5.py — K-Means Stability evaluation across budget levels.

Compares stability of K-Means clustering vs Trees (experiment 1.py).

For each budget:
  1. Split data 50/50 into train / test-pool.
  2. Sample 100 test rows from test-pool.
  3. Build 20 K-Means models, each from `budget` rows sampled from train.
     - Use distKpp for centroid initialization, then run Lloyd's iterations
       with ezr's distx (handles mixed types).
  4. For each test row, find its nearest centroid in each model,
     compute disty of that centroid -> win score.
     20 models -> 20 win scores per test row -> compute sd.
  5. Stability check: sd < 0.35 * b4_wins.sd.

Output: one CSV per dataset under results/5/{dataset}.csv
  Columns: trt, stability_agreement, best_stability
"""
from ezr import *
from stats import *
import sys
import random
import os

BUDGET_NUMS = [10, 20, 50, 100, 200]
LLOYD_ITERS = 20      # max iterations for Lloyd's algorithm


def kmeans(data, rows, k=None, max_iter=LLOYD_ITERS):
    """K-Means clustering using ezr's distx for mixed-type distance.

    Args:
        data:     ezr Data object (provides column metadata for distx).
        rows:     list of rows to cluster.
        k:        number of clusters.
        max_iter: maximum Lloyd iterations.

    Returns:
        centroids: list of k rows (actual data rows acting as medoids).
    """
    if k is None:
        k = max(2, len(rows) // the.leaf)
    if len(rows) <= k:
        return rows[:]

    # K-Means++ initialization via ezr's distKpp
    centroids = distKpp(data, rows=rows, k=k)

    for _ in range(max_iter):
        # Assign each row to nearest centroid
        clusters = [[] for _ in range(k)]
        for row in rows:
            dists = [distx(data, row, c) for c in centroids]
            nearest = dists.index(min(dists))
            clusters[nearest].append(row)

        # Update centroids: pick the medoid (row with min total dist to cluster)
        new_centroids = []
        for ci, cluster in enumerate(clusters):
            if not cluster:
                new_centroids.append(centroids[ci])  # keep old centroid
                continue
            # Medoid: the row in the cluster closest to all others
            best_row = min(
                cluster,
                key=lambda r: sum(distx(data, r, other) for other in cluster)
            )
            new_centroids.append(best_row)

        # Check convergence (centroids unchanged)
        if all(c1 is c2 for c1, c2 in zip(centroids, new_centroids)):
            break
        centroids = new_centroids

    return centroids


def predict_nearest(data, centroids, row):
    """Find the nearest centroid to `row` and return it."""
    return min(centroids, key=lambda c: distx(data, row, c))


def run(file_directory, out_dir="results/5", repeats=20):
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

    # =========================================================
    # Split: 50/50 train / test-pool, sample 100 test rows
    # =========================================================
    all_data.rows = shuffle(all_data.rows)
    half       = len(all_data.rows) // 2
    train_rows = all_data.rows[:half]
    test_pool  = all_data.rows[half:]
    tests_size = min(100, len(test_pool))
    test_rows  = test_pool[:tests_size]

    train = clone(all_data, train_rows)
    test  = clone(all_data, test_rows)

    # =========================================================
    # Stability via K-Means
    # =========================================================
    stability_agreement = {}
    # For each test row, store the sd under each budget
    stability_comp = [{budget: 0 for budget in BUDGET_NUMS} for _ in range(len(test.rows))]

    for budget in BUDGET_NUMS:
        k = max(2, budget // the.leaf)  # clusters = budget / leaf size

        # Build `repeats` K-Means models, each from a different sample
        models = []  # list of centroid lists
        for rand_seed in range(repeats):
            random.seed(rand_seed)
            sampled = random.sample(train.rows, min(budget, len(train.rows)))
            sample_data = clone(all_data, sampled)
            centroids = kmeans(sample_data, sampled, k=k)
            models.append(centroids)

        # For each test row, get 20 predictions -> win scores -> sd
        agreement = 0
        for idx, row in enumerate(test.rows):
            win_scores = []
            for centroids in models:
                nearest = predict_nearest(all_data, centroids, row)
                d = disty(all_data, nearest)
                win_scores.append(win(d))
            preds = adds(win_scores)
            stability_comp[idx][budget] = preds.sd
            if preds.sd < 0.35 * b4_wins.sd:
                agreement += 1

        stability_agreement[budget] = agreement * 100 // tests_size
        print(f"  Budget {budget}: agreement = {stability_agreement[budget]}%")

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

    header = "trt, stability_agreement, best_stability"
    print(header)
    lines = [header]
    for budget in BUDGET_NUMS:
        line = (
            f"{budget}, "
            f"{stability_agreement[budget]}, "
            f"{best_stability[budget]}"
        )
        print(line)
        lines.append(line)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    file_directory = sys.argv[1] if len(sys.argv) > 1 else "data/optimize/misc/auto93.csv"
    output_dir     = sys.argv[2] if len(sys.argv) > 2 else "results/5"
    run(file_directory, output_dir)
