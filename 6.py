from ezr import *
from stats import *

file_directory = sys.argv[1] if len(sys.argv) > 1 else "data/optimize/misc/auto93.csv"
repeats = 20

all_data = Data(csv(file_directory))
ys      = [disty(all_data,row) for row in all_data.rows]
b4      = adds(ys)
win     = lambda v: int(100*(1 - (v - b4.lo)/(b4.mu - b4.lo)))
b4_wins = adds([win(k) for k in ys])

the.Check   = 10
the.Budget  = 50
the.acq     = "near"

## Performance : RMSE(one value), error([]*20)
mse = 0
error = []
for rand_seed in range(repeats):
    the.seed    = rand_seed
    random.seed(the.seed)
    shuffled_rows = random.sample(all_data.rows, len(all_data.rows))
    half = int(0.5 * len(all_data.rows))
    train, holdout = clone(all_data, shuffled_rows[:half]), clone(all_data, shuffled_rows[half:])
    labels = likely(train)
    tree   = Tree(clone(train, labels))
    top_rows = sorted( [(treeLeaf(tree, row).mu, row) for row in holdout.rows], key=lambda x: x[0])[:the.Check]
    ezr_performace = win( sorted([disty(all_data, row) for _, row in top_rows])[0] )
    # print("Suggested row win:\t", ezr_performace)
    # print("Referenced Optimal:\t", win(min(disty(all_data, row) for row in holdout.rows)))
    referenced_optima = win(min(disty(all_data, row) for row in holdout.rows))
    mse += abs(ezr_performace - referenced_optima) ** 2
    error.append(referenced_optima - ezr_performace)
rmse = (mse / repeats) ** 0.5

# -----------------------------

all_data.rows = shuffle(all_data.rows)
tests_size = min(100, int(len(all_data.rows) * 0.3))
test, train = clone(all_data, all_data.rows[:tests_size]), clone(all_data, all_data.rows[tests_size:])
the.Check   = 10
the.Budget  = 50
the.acq     = "near"

trees = []
for rand_seed in range(repeats):
    the.seed    = rand_seed
    random.seed(the.seed)
    labels = likely(train)
    tree   = Tree(clone(train, labels))
    trees.append(tree)

## Stability : aggreement(one value out of 100), std(sd of predictions)
aggreement = 0
for idx, row in enumerate(test.rows):
    outputs = [win(treeLeaf(tree,row).mu) for tree in trees]
    preds = adds( outputs )
    std = preds.sd
    if preds.sd < 0.35 * b4_wins.sd:
        aggreement += 1

aggreement = (aggreement * 100) // tests_size


# -----------------------------

print(f"Performance, {rmse:.2f}, [{'-'.join(str(x) for x in error)}]")
print(f"Stability, {aggreement}, {std:.2f}")
print(f"Dataset, {file_directory.split("/")[-1][:-4]}")

wins = [win(y) for y in sorted(ys)]
b4_wins = adds([win(k) for k in sorted(ys)])
print(f"Wins, {b4_wins.mu:.2f}, {b4_wins.sd:.2f}")
print(f"Dist8, {(100 * sum([1 for w in wins if w > 80])) // len(wins)}")
print(f"Quant, {wins[0]}, {wins[len(wins) // 4]}, {wins[len(wins) // 2]}, {wins[3*len(wins) // 4]}, {wins[-1]}")