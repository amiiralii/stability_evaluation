from ezr import *

file_directory = sys.argv[1] if len(sys.argv) > 1 else "data/optimize/misc/auto93.csv"
data = Data(csv(file_directory))
b4   = adds(disty(data,row) for row in data.rows)
win  = lambda v: int(100*(1 - (v - b4.lo)/(b4.mu - b4.lo)))


data.rows = shuffle(data.rows)
train, holdout = data.rows[10:], data.rows[:10]
labels = likely(clone(data, train))
tree   = Tree(clone(data, labels))

print("Test row to predict: ", holdout[0])
print("Actual d2h: ", disty(data, holdout[0]))
print("Actual win: ", win(disty(data, holdout[0])))
print("Predicted d2h: ", treeLeaf(tree, holdout[0]).mu)
print("Predicted win: ", win(treeLeaf(tree, holdout[0]).mu) )


train, holdout = data.rows[10:], data.rows[:10]
test_row = holdout[0]
mse = 0
for i in range(10):
    the.seed = i
    train = shuffle(train)
    labels = likely(clone(data, train))
    tree   = Tree(clone(data, labels))
    print(f"--------{i}------")
    print("Actual win:\t", win(disty(data, test_row)))
    print("Predicted win:\t", win(treeLeaf(tree, test_row).mu) )
    mse += abs(win(disty(data, test_row)) - win(treeLeaf(tree, test_row).mu)) ** 2
print(f"Performance = {mse/10}")