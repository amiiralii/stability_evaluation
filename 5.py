"""
5.1.py — K-Means vs Tree: Stability & Performance comparison.

Runs both K-Means and ezr Trees under identical conditions
(same splits, same seeds, fixed budget = 50) so they can be
directly compared.

Treatments: kmeans, tree

Part 1 — Performance (20 train/holdout splits per treatment):
  K-Means: sample budget rows -> build K-Means -> nearest centroid prediction.
  Tree:    likely(train) -> build Tree -> treeLeaf prediction.
  For each holdout row, pick top-Check -> best win -> error vs true best.

Part 2 — Stability (single train/test split, 20 models per treatment):
  Build 20 models (different seeds).
  For each test row, 20 predictions -> sd.
  sd < 0.35 * b4_wins.sd -> stable.

Output: one CSV per dataset under results/5.1/{dataset}.csv
  Columns: trt, performance_error, stability_agreement
"""
from ezr import *
from stats import *
import sys
import random
import os

LLOYD_ITERS = 20      # max iterations for Lloyd's algorithm
TREATMENTS  = ["kmeans", "tree"]


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
            best_row = min(
                cluster,
                key=lambda r: sum(distx(data, r, other) for other in cluster)
            )
            new_centroids.append(best_row)

        if all(c1 is c2 for c1, c2 in zip(centroids, new_centroids)):
            break
        centroids = new_centroids

    return centroids


def predict_nearest(data, centroids, row):
    """Find the nearest centroid to `row` and return it."""
    return min(centroids, key=lambda c: distx(data, row, c))


def run(file_directory, out_dir="results/5.1", repeats=20):
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
    performance_error = {}

    for trt in TREATMENTS:
        mse = 0
        for rand_seed in range(repeats):
            random.seed(rand_seed)
            shuffled_rows = random.sample(all_data.rows, len(all_data.rows))
            half = int(0.5 * len(all_data.rows))
            train   = clone(all_data, shuffled_rows[:half])
            holdout = clone(all_data, shuffled_rows[half:])

            if trt == "kmeans":
                sampled = random.sample(train.rows, min(budget, len(train.rows)))
                k = max(2, budget // the.leaf)
                centroids = kmeans(clone(all_data, sampled), sampled, k=k)
                # Score holdout rows by nearest centroid
                scored = []
                for row in holdout.rows:
                    nearest = predict_nearest(all_data, centroids, row)
                    scored.append((disty(all_data, nearest), row))
                top_rows = sorted(scored, key=lambda x: x[0])[:the.Check]
            else:  # tree
                the.seed = rand_seed
                labels = likely(train) if budget < len(train.rows) else train.rows[:budget]
                tree   = Tree(clone(train, labels))
                top_rows = sorted(
                    [(treeLeaf(tree, row).mu, row) for row in holdout.rows],
                    key=lambda x: x[0]
                )[:the.Check]

            ezr_perf = win(sorted([disty(all_data, row) for _, row in top_rows])[0])
            ref_opt  = win(min(disty(all_data, row) for row in holdout.rows))
            mse += abs(ezr_perf - ref_opt) ** 2

        performance_error[trt] = (mse / repeats) ** 0.5

    # =========================================================
    # Part 2: Stability
    # =========================================================
    # Single train/test split, shared across treatments.
    all_data.rows = shuffle(all_data.rows)
    half       = len(all_data.rows) // 2
    train_rows = all_data.rows[:half]
    test_pool  = all_data.rows[half:]
    tests_size = min(100, len(test_pool))
    test_rows  = test_pool[:tests_size]

    train = clone(all_data, train_rows)
    test  = clone(all_data, test_rows)

    stability_agreement = {}

    for trt in TREATMENTS:
        if trt == "kmeans":
            k = max(2, budget // the.leaf)
            models = []
            for rand_seed in range(repeats):
                random.seed(rand_seed)
                sampled = random.sample(train.rows, min(budget, len(train.rows)))
                centroids = kmeans(clone(all_data, sampled), sampled, k=k)
                models.append(centroids)

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

        else:  # tree
            trees = []
            for rand_seed in range(repeats):
                the.seed = rand_seed
                random.seed(the.seed)
                labels = likely(train)
                tree   = Tree(clone(train, labels))
                trees.append(tree)

            agreement = 0
            for row in test.rows:
                outputs = [win(treeLeaf(tree, row).mu) for tree in trees]
                preds   = adds(outputs)
                if preds.sd < 0.35 * b4_wins.sd:
                    agreement += 1

        stability_agreement[trt] = agreement * 100 // tests_size

    # =========================================================
    # Output
    # =========================================================
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(file_directory).split('.')[0]
    out_path  = os.path.join(out_dir, f"{base_name}.csv")

    header = "trt, performance_error, stability_agreement"
    print(header)
    lines = [header]
    for trt in TREATMENTS:
        line = f"{trt}, {performance_error[trt]:.2f}, {stability_agreement[trt]}"
        print(line)
        lines.append(line)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    file_directory = sys.argv[1] if len(sys.argv) > 1 else "data/optimize/misc/auto93.csv"
    output_dir     = sys.argv[2] if len(sys.argv) > 2 else "results/5.1"
    run(file_directory, output_dir)
