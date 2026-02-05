from ezr import *
from stats import *

file_directory = sys.argv[1] if len(sys.argv) > 1 else "data/optimize/misc/auto93.csv"
repeats = 20

all_data = Data(csv(file_directory))

b4   = adds(disty(all_data,row) for row in all_data.rows)
win  = lambda v: int(100*(1 - (v - b4.lo)/(b4.mu - b4.lo)))
the.Check   = 10
the.Budget  = 50

treatments = ["near", "xploit", "xplor", "bore", "random"]

performace_metric = {}
error_dist = {}
for acquisition in treatments:
    the.acq     = acquisition
    mse = 0
    error = []
    for rand_seed in range(repeats):
        the.seed    = rand_seed
        random.seed(the.seed)
        shuffled_rows = random.sample(all_data.rows, len(all_data.rows))
        half = int(0.5 * len(all_data.rows))
        train, holdout = clone(all_data, shuffled_rows[:half]), clone(all_data, shuffled_rows[half:])
        labels = likely(train) if acquisition != "random" else train.rows[: the.Budget]
        tree   = Tree(clone(train, labels))
        top_rows = sorted( [(treeLeaf(tree, row).mu, row) for row in holdout.rows], key=lambda x: x[0])[:the.Check]
        ezr_performace = win( sorted([disty(all_data, row) for _, row in top_rows])[0] )
        # print("Suggested row win:\t", ezr_performace)
        # print("Referenced Optimal:\t", win(min(disty(all_data, row) for row in holdout.rows)))
        referenced_optima = win(min(disty(all_data, row) for row in holdout.rows))
        mse += abs(ezr_performace - referenced_optima) ** 2
        error.append(referenced_optima - ezr_performace)
    error_dist[acquisition] = error
    performace_metric[acquisition] = (mse / repeats) ** 0.5
# print("Performance =", performace_metric)
# print(error_dist)
# sample = {
#     "A" : [1,2,3,4,6,6],
#     "B" : [60,64,30,40,60,60]
# }
# print(top(error_dist, Ks=0.9, Delta="medium"))

all_data.rows = shuffle(all_data.rows)
tests_size = 100
test, train = clone(all_data, all_data.rows[:tests_size]), clone(all_data, all_data.rows[tests_size:])
the.Check   = 10
the.Budget  = 50

stability_metric = {}
for acquisition in treatments:
    the.acq     = acquisition
    trees = []
    for rand_seed in range(repeats):
        the.seed    = rand_seed
        random.seed(the.seed)
        labels = likely(train)
        tree   = Tree(clone(train, labels))
        trees.append(tree)

    aggreement = 0
    for row in test.rows:
        preds = adds( [win(treeLeaf(tree,row).mu) for tree in trees] )

        # Avoid division by zero or near-zero mean; use an absolute sd threshold when mean is close to zero
        if abs(preds.mu) > 1e-6:
            if preds.sd / abs(preds.mu) < 0.2:
                aggreement += 1
        else:
            if preds.sd < 0.2:  # You may want to adjust 0.2 to an appropriate sd threshold for your data
                aggreement += 1

    stability_metric[acquisition] = aggreement
# print("Stability,", stability_metric)

print("trt, performance, stability")
for trt in treatments:
    print(f"{trt}, {performace_metric[trt]:.2f}, {stability_metric[trt]}")
