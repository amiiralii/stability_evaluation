"""
5.py — K-Means Stability & Performance evaluation.

Uses K-Means clustering at fixed budget = 50.

Part 1 — Performance (20 train/holdout splits):
  1. 50/50 train / holdout split.
  2. Sample `budget` rows from train, build K-Means model.
  3. For each holdout row, find nearest centroid -> win score.
     Pick top-Check rows -> best win -> compare with true holdout best -> error.
  4. RMSE across repeats = performance_error.

Part 2 — Stability (single train/test split, 20 models):
  1. Sample 100 test rows.
  2. Build 20 K-Means models (different seeds).
  3. For each test row, 20 predictions -> sd.
  4. sd < 0.35 * b4_wins.sd -> stable.

Output: one CSV per dataset under results/5/{dataset}.csv
  Columns: performance_error, stability_agreement
"""
from ezr import *
from stats import *
import sys
import random
import os

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

    budget = 50
    the.Budget = budget

    # =========================================================
    # Part 1: Performance
    # =========================================================
    # 20 train/holdout splits.  Each repeat: sample `budget` rows from
    # train, build K-Means, find nearest centroid for holdout rows,
    # pick top-Check -> measure error vs true holdout best.
    mse = 0
    for rand_seed in range(repeats):
        random.seed(rand_seed)
        shuffled_rows = random.sample(all_data.rows, len(all_data.rows))
        half = int(0.5 * len(all_data.rows))
        train   = clone(all_data, shuffled_rows[:half])
        holdout = clone(all_data, shuffled_rows[half:])

        sampled = random.sample(train.rows, min(budget, len(train.rows)))
        k = max(2, budget // the.leaf)
        centroids = kmeans(clone(all_data, sampled), sampled, k=k)

        # For each holdout row, predict via nearest centroid
        scored = []
        for row in holdout.rows:
            nearest = predict_nearest(all_data, centroids, row)
            scored.append((disty(all_data, nearest), row))
        top_rows = sorted(scored, key=lambda x: x[0])[:the.Check]

        ezr_perf = win(sorted([disty(all_data, row) for _, row in top_rows])[0])
        ref_opt  = win(min(disty(all_data, row) for row in holdout.rows))
        mse += abs(ezr_perf - ref_opt) ** 2

    performance_error = (mse / repeats) ** 0.5

    # =========================================================
    # Part 2: Stability
    # =========================================================
    # Single train/test split. Build 20 K-Means models.
    # For each test row, measure sd of predictions across models.
    all_data.rows = shuffle(all_data.rows)
    half       = len(all_data.rows) // 2
    train_rows = all_data.rows[:half]
    test_pool  = all_data.rows[half:]
    tests_size = min(100, len(test_pool))
    test_rows  = test_pool[:tests_size]

    train = clone(all_data, train_rows)
    test  = clone(all_data, test_rows)

    k = max(2, budget // the.leaf)

    # Build `repeats` K-Means models, each from a different sample
    models = []
    for rand_seed in range(repeats):
        random.seed(rand_seed)
        sampled = random.sample(train.rows, min(budget, len(train.rows)))
        centroids = kmeans(clone(all_data, sampled), sampled, k=k)
        models.append(centroids)

    # For each test row, get 20 predictions -> win scores -> sd
    agreement = 0
    for row in test.rows:
        win_scores = []
        for centroids in models:
            nearest = predict_nearest(all_data, centroids, row)
            d = disty(all_data, nearest)
            win_scores.append(win(d))
        preds = adds(win_scores)
        if preds.sd < 0.35 * b4_wins.sd:
            agreement += 1

    stability_agreement = agreement * 100 // tests_size

    # =========================================================
    # Output
    # =========================================================
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(file_directory).split('.')[0]
    out_path  = os.path.join(out_dir, f"{base_name}.csv")

    header = "performance_error, stability_agreement"
    print(header)
    line = f"{performance_error:.2f}, {stability_agreement}"
    print(line)
    lines = [header, line]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    file_directory = sys.argv[1] if len(sys.argv) > 1 else "data/optimize/misc/auto93.csv"
    output_dir     = sys.argv[2] if len(sys.argv) > 2 else "results/5"
    run(file_directory, output_dir)
