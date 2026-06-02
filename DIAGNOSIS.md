# Deterministic Numerical Policy

`benchmark-fragility` is designed to return the same result for the same score matrix
regardless of row order, column order, or NumPy summation changes.

This document records the numerical conventions that make that guarantee
explicit.

## Breakdown Point

The breakdown-point ratio is the smallest fraction of tasks that must be
removed before the higher-mean model no longer leads.

For each winner-loser pair:

1. Scores are oriented so larger is better.
2. Tasks are sorted by descending winner advantage, `winner - loser`.
3. Equal advantages are tie-broken by task label.
4. Tasks are removed in that order.
5. Breakdown occurs when the kept-task mean satisfies
   `mean_winner <= mean_loser`.

The `<=` rule is intentional: an exact tie no longer supports a strict win.

## Mean Calculation

Subset means are computed with `math.fsum(values) / len(values)`.

This avoids dependence on NumPy's pairwise summation order. The difference is
most visible on discrete or rounded leaderboards, where two subset means can be
mathematically equal at the breakdown boundary.

## Task Subsampling

`robustness.by_datasets` draws task labels from `sorted(index)` instead of from
the input row order.

With a fixed seed, the sampled task set is therefore a function of:

- the task labels
- the seed
- the requested sample size

It is not a function of the caller's row ordering.

## Reproduce The Breakdown Convention

```python
import math
from itertools import combinations

import numpy as np
import pandas as pd

df = pd.read_csv("tests/data/MMLU.csv", index_col=0)
cols = df.mean().sort_values(ascending=False).head(20).index
sub = df[cols]
k = len(sub.index)


def fmean(values):
    return math.fsum(values) / len(values)


def bp(winner, loser):
    order = sorted(
        range(k),
        key=lambda i: (-(winner.values[i] - loser.values[i]), str(winner.index[i])),
    )
    w = [winner.values[i] for i in order]
    l = [loser.values[i] for i in order]
    for removed in range(1, k + 1):
        if not w[removed:]:
            return 1.0
        if fmean(w[removed:]) <= fmean(l[removed:]):
            return removed / k
    return 1.0


values = []
for a, b in combinations(cols, 2):
    winner, loser = (sub[a], sub[b]) if fmean(sub[a]) > fmean(sub[b]) else (sub[b], sub[a])
    values.append(bp(winner, loser))

print(round(np.mean(values) * 100, 4))
```
