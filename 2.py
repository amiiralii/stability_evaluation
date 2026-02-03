from ezr import *


file_directory = sys.argv[1] if len(sys.argv) > 1 else "data/optimize/misc/auto93.csv"
repeats = 20

all_data = Data(csv(file_directory))

b4   = adds(disty(all_data,row) for row in all_data.rows)
win  = lambda v: int(100*(1 - (v - b4.lo)/(b4.mu - b4.lo)))
the.Check   = 10
the.Budget  = 50

performace_metric = {}
for acquisition in ["near", "xploit", "xplor", "bore", "random"]:
    the.acq     = acquisition
    mse = 0
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
        mse += abs(ezr_performace - win(min(disty(all_data, row) for row in holdout.rows))) ** 2
    performace_metric[acquisition] = mse / repeats
print("Performance =", performace_metric)


all_data.rows = shuffle(all_data.rows)
tests_size = 100
test, train = clone(all_data, all_data.rows[:tests_size]), clone(all_data, all_data.rows[tests_size:])
the.Check   = 10
the.Budget  = 50

stability_metric = {}
for acquisition in ["near", "xploit", "xplor", "bore", "random"]:
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
        preds = adds( [treeLeaf(tree,row).mu for tree in trees] )
        if preds.sd / preds.mu < 0.15:  aggreement += 1

    stability_metric[acquisition] = aggreement
print("Stability =", stability_metric)
