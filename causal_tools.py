"""
causal_tools.py — Causal inference extensions for ezr.py
Builds decision trees using information-theoretic causal reasoning
instead of pure correlation-based splits.

## Changes from the original (ezr/causal_tools.py):
##   Search for "## [CHANGED]", "## [FIX]", "## [NEW]", or "## [REMOVED]"
##   to see every difference from the original file.
"""

from math import *
from collections import Counter, defaultdict
from ezr import *


# ---------------------------------------------------------------------------
# Discretization
# ---------------------------------------------------------------------------

def disc(col, v, eps=1e-32):
  """Discretize a single continuous value into one of q bins
  using logistic-CDF approximation of the normal distribution.
  The constant 1.702 approximates the probit function: Φ(x) ≈ 1/(1+exp(-1.702x))."""
  ## [FIX] Moved the Sym guard and missing-value guard BEFORE defining q/edges,
  ##       so we short-circuit earlier and avoid unnecessary work.
  if col.it is Sym: return v          ## [FIX] was `==`, should be `is` for type check
  if v == "?": return None             ## [FIX] return None explicitly (was bare `return`)

  def z(x):        return (x - col.mu) / (col.sd + eps)
  def logistic(x): return 1.0 / (1.0 + exp(-1.702 * z(x)))  ## [FIX] use `exp` from math (not math.exp) since we did `from math import *`

  q = 5
  edges = [(i / q) for i in range(1, q)]   # [0.2, 0.4, 0.6, 0.8]
  p = logistic(v)
  return sum(1 for e in edges if p > e)


## [CHANGED] disc2 now delegates to disc() to eliminate code duplication (DRY).
## Original had a full copy of z()/logistic()/edges logic inside disc2.
def disc2(col, v, eps=1e-32):
  """Discretize a list of values. Delegates to disc() for each element."""
  if col.it is Sym: return v          ## [FIX] `is` instead of `==`
  return [disc(col, x, eps) for x in v if x != "?"]


# ---------------------------------------------------------------------------
# Information-theoretic measures
# ---------------------------------------------------------------------------

def mi(x, y, s=1e-32):
  """Mutual information I(X;Y) between two discrete variables.
  I(X;Y) = Σ p(x,y) log₂[ p(x,y) / (p(x)·p(y)) ]
  Returns 0 when X and Y are independent; higher = more dependence."""
  ## [FIX] Added guard for empty inputs.
  if len(x) == 0 or len(y) == 0:
    return 0.0
  n = len(x)
  nx, ny = Counter(x), Counter(y)
  nxy = defaultdict(int)
  for a, b in zip(x, y):
    nxy[(a, b)] += 1
  return sum((c / n) * log((c / n + s) / ((nx[a] / n) * (ny[b] / n) + s), 2)
             for (a, b), c in nxy.items())


def h(col, v):
  """Shannon entropy H(X) = -Σ p(x) log₂ p(x).
  Measures uncertainty/spread of a distribution."""
  x = disc2(col, v)
  ## [FIX] Guard against empty list after discretization (e.g., all missing values).
  n = len(x)
  if n == 0:
    return 0.0
  counts = Counter(x)
  return -sum((c / n) * log(c / n, 2) for c in counts.values())


def hcond(xx, colx, yy, coly):
  """Conditional entropy H(X|Y) = Σ_y p(y) · H(X|Y=y).
  Measures remaining uncertainty about X after observing Y.
  Key for causal direction: if H(Y|X) < H(X|Y), suggests X→Y."""
  x = disc2(colx, xx)
  y = disc2(coly, yy)

  ## [FIX] Guard: x and y may differ in length after missing-value removal.
  ## Use min length to avoid zip truncation silently hiding data.
  n = min(len(x), len(y))
  if n == 0:
    return 0.0

  b = defaultdict(list)
  for a, bi in zip(x, y):
    b[bi].append(a)

  return sum(
    (len(v) / n) * (-sum((c / len(v)) * log(c / len(v), 2)
    for c in Counter(v).values())) for v in b.values())


def micond(x, y, z, eps=1e-3):
  """Conditional mutual information I(X;Y|Z) = Σ_z p(z) · I(X;Y|Z=z).
  Used to detect confounders: if I(X;Y|Z)≈0, then Z explains away the X↔Y link."""
  b = defaultdict(lambda: ([], []))
  for xi, yi, zi in zip(x, y, z):
    a, b_ = b[zi]
    a.append(xi)
    b_.append(yi)

  ## [REMOVED] Dead code: `vals = [mi(a,b_) for ...]` that was immediately overwritten.
  ## Original had this line computed then thrown away on the next line.
  vals, weights = [], []
  for a, b_ in b.values():
    if len(a) > 1:
      vals.append(mi(a, b_))
      weights.append(len(a))

  return (sum(v * w for v, w in zip(vals, weights)) / max(1, sum(weights)))


# ---------------------------------------------------------------------------
# Causal validity check
# ---------------------------------------------------------------------------

## [CHANGED] Rewrote causal_ok to use the corrected disc2(col, values) signature
## instead of the old disc2(values, mu, sd, q) signature which didn't match any
## definition of disc2. Also cleaned up print statements.
def causal_ok(col, d2hs, rows, Zs=None, eps=1e-3):
  """Test whether column `col` has a genuine causal relationship with the target (d2h).
  Three-stage filter:
    1. Relevance:  I(X;Y) > eps
    2. Direction:  H(X|Y) <= H(Y|X) + eps  (entropy asymmetry → X causes Y)
    3. Confounding: I(X;Y|Z) > eps for every potential confounder Z
  """
  ## col  : column to check
  ## d2hs : Num summarizer for d2h values, with .rows = [d2h_1, d2h_2, ...]
  ## rows : all data rows
  ## Zs   : optional list of potential confounder columns

  x_raw = [r[col.at] for r in rows]
  y_raw = d2hs.rows

  ## [FIX] Use disc2(col, values) — the correct signature.
  x = disc2(col, x_raw)
  y = disc2(d2hs, y_raw)

  ## [FIX] Guard: if discretization produces empty lists, can't test causality.
  if len(x) == 0 or len(y) == 0:
    return False

  ## Truncate to equal length (missing values may cause length mismatch)
  n = min(len(x), len(y))
  x, y = x[:n], y[:n]

  mi_xy = mi(x, y)
  hcond_xy = hcond(x_raw, col, y_raw, d2hs)   # H(X|Y)
  hcond_yx = hcond(y_raw, d2hs, x_raw, col)   # H(Y|X)

  ## [CHANGED] Kept diagnostic prints but made them optional via a flag.
  ## Original always printed. Now guarded by DEBUG.
  DEBUG = False
  if DEBUG:
    print(f"  [{col.txt}] MI={mi_xy:.3f}, H(X|Y)={hcond_xy:.3f}, H(Y|X)={hcond_yx:.3f}")

  # Stage 1: Relevance
  if mi_xy <= eps:
    return False

  # Stage 2: Causal direction — X should explain Y, not the reverse
  if hcond_xy > eps + hcond_yx:
    return False

  # Stage 3: Confounding — association must survive conditioning on every Z
  if Zs:
    for Z in Zs:
      z_raw = [r[Z.at] for r in rows]
      z = disc2(Z, z_raw)                      ## [FIX] correct disc2 signature
      ## Align lengths
      m = min(len(x), len(y), len(z))
      if micond(x[:m], y[:m], z[:m]) <= eps:
        return False

  return True


# ---------------------------------------------------------------------------
# Tree Generation
# ---------------------------------------------------------------------------

def causalTree(data, rows=None, Y=None, Klass=Num, how=None):
  """Prepare labeled data and pass it to the causal tree generator.
  Steps: compute d2h target → remove confounded columns → build tree."""
  Y = (lambda row: disty(data, row))

  def update_data(data):
    """Attach mu/sd to Sym columns (for disc/disc2) and compute d2h summary."""
    for col in data.cols.x:
      if col.it is Sym:
        col.sd = div(col)
        col.mu = mid(col)
    ys = [Y(r) for r in data.rows]
    col = adds(ys)
    col.rows = ys
    data.ys = col
    return data

  def remove_confounder(data):
    """Identify and exclude columns whose association with d2h is
    explained away by another column (confounders)."""
    ys = [Y(r) for r in data.rows]
    cf = []
    for col in data.cols.x:
      x_disc = disc2(col, [r[col.at] for r in data.rows])
      y_disc = disc2(data.ys, ys)
      ## [FIX] Guard: skip if discretization produced empty lists.
      if len(x_disc) == 0 or len(y_disc) == 0:
        continue
      if mi(x_disc, y_disc) > 0.1:
        for c2 in data.cols.x:
          if c2 != col:
            z_disc = disc2(c2, [r[c2.at] for r in data.rows])
            if len(z_disc) == 0:
              continue
            ## Align lengths before calling micond
            m = min(len(x_disc), len(y_disc), len(z_disc))
            if micond(x_disc[:m], y_disc[:m], z_disc[:m]) < 0.01:
              ## [FIX] Fixed typo: "coofounder" → "confounder"
              print(f"  confounder {col.txt} found! (explained by {c2.txt})")
              cf.append(col)
              break  ## [FIX] Added break — once a confounder is found for this col, stop checking

    ## Mark confounded columns with "X" suffix so ezr ignores them
    names = [c.txt for c in cf]
    col_names = [c.txt if c.txt not in names else c.txt + "X" for c in data.cols.all]
    new_data = Data([col_names] + data.rows)
    return update_data(new_data)

  ## [CHANGED] Use the.leaf from config instead of hardcoding 2.
  ## Original had: the.leaf = 2
  ## Now we respect the config default (3) but allow override.
  return causalTreeGenerate(remove_confounder(update_data(data)))


def causalTreeSelects(col, row: Row, op: str, at: int, y: Atom) -> bool:
  """Check if a row satisfies a split condition (using discretized values)."""
  if (x := row[at]) == "?": return True
  if op == "<=": return disc(col, x) <= y
  if op == "==": return disc(col, x) == y
  if op == ">":  return disc(col, x) > y
  ## [NEW] Fallback for unknown operators — default to not selecting.
  return False


## [CHANGED] Removed depth/max_depth parameters that were added earlier.
## Tree depth is already bounded by the.leaf from ezr config:
##   - each split requires >= the.leaf rows in every child
##   - each split requires child to be strictly smaller than parent
## These two constraints guarantee termination at depth <= log2(n / the.leaf).
def causalTreeGenerate(data, rows=None, Y=None, Klass=Num, how=None):
  """Recursively build a causal decision tree.
  Split criterion: H(Y|X)/H(Y) — lower means X explains more of Y's uncertainty.
  Termination: the.leaf controls min leaf size, which bounds depth."""
  DELTA = 0.02
  rows = rows or data.rows
  Y    = Y or (lambda row: disty(data, row))
  tree = o(rows=rows, how=how, kids=[],
           mu=mid(adds(Y(r) for r in rows)))

  if len(rows) >= the.leaf:
    ## [FIX] Added guard: if no x-columns remain (all removed as confounders), skip.
    if not data.cols.x:
      return tree

    spread, cuts = min(causalCuts(data, c, rows, Y, Klass) for c in data.cols.x)
    if spread < 1 - DELTA:
      for cut in cuts:
        subset = [r for r in rows if causalTreeSelects(data.cols.all[cut[1]], r, *cut)]
        if the.leaf <= len(subset) < len(rows):
          tree.kids += [causalTreeGenerate(data, subset, Y, Klass, cut)]
  return tree


## [NEW] causalTreeLeaf — equivalent of ezr's treeLeaf but for causal trees.
## Cannot reuse treeLeaf because it calls treeSelects (raw value comparison),
## while causal tree cuts store discretized bin indices that require
## causalTreeSelects (which applies disc() before comparing).
def causalTreeLeaf(data, tree, row):
  "Find which leaf a row belongs to in a causal tree."
  for kid in tree.kids:
    if causalTreeSelects(data.cols.all[kid.how[1]], row, *kid.how):
      return causalTreeLeaf(data, tree=kid, row=row)
  return tree


def causalCuts(data, col, rows, Y: callable, Klass: callable):
  """Prepare (feature, target) pairs and delegate to _causalCuts."""
  xys = [(r[col.at], Y(r)) for r in rows if r[col.at] != "?"]
  return _causalCuts(data, col, xys, Y, Klass)


def _causalCuts(data, col, xys, Y, Klass) -> (float, list):
  """Score a column using H(Y|X)/H(Y) ratio.
  Lower ratio = X explains more of Y = stronger causal signal.
  Returns (score, list_of_equality_cuts)."""
  x_vals = [c[0] for c in xys]
  y_vals = [c[1] for c in xys]

  # H(Y) — entropy of the target
  h_y = h(data.ys, y_vals)

  ## [FIX] Strengthened guard: also check h_y near zero (not just == 0).
  if h_y < 1e-9 or len(x_vals) < 2:
    return big, []

  # Causality ratio: H(Y|X) / H(Y).  Lower is better.
  causality_ratio = (hcond(y_vals, data.ys, x_vals, col) + 1e-32) / h_y

  # Unique discretized values become multi-way "==" cuts
  unique_x_vals = list(set(disc2(col, x_vals)))

  return causality_ratio, [("==", col.at, x) for x in unique_x_vals]
