"""Deterministic pairwise diagnostics.

Three metrics quantify whether a higher-mean model truly beats another:

- ``win_rate``        — fraction of tasks the winner wins (consistency).
- ``cohens_d``        — standardized mean difference (magnitude / effect size).
- ``breakdown_point`` — smallest fraction of tasks to remove (most-favorable
                        first) before the winner no longer leads (stability).

All inputs are 1-D float arrays aligned by task, **already oriented so that
larger is better** (callers negate ``lower_better`` matrices upstream).

Determinism
-----------
``breakdown_point`` is computed so the result does not depend on NumPy's float
summation order or on the input row ordering:

1. Subset means are compared with :func:`math.fsum` (order-independent exact
   summation) instead of ``numpy.mean`` (pairwise summation whose result can
   flip an exact tie between NumPy versions).
2. Tasks are removed in a fully deterministic order: by descending advantage,
   ties broken by task label — never by the array's incoming order.
3. The winner is declared broken when ``mean_winner <= mean_loser``. A tie no
   longer supports a strict win.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = ["fsum_mean", "win_rate", "cohens_d", "breakdown_point"]

_STD_EPS = 1e-10


def fsum_mean(values: Sequence[float]) -> float:
    """Order-independent exact mean via :func:`math.fsum`.

    Unlike ``numpy.mean``/``sum`` (pairwise summation), ``math.fsum`` returns a
    correctly-rounded result regardless of element order, so two equal multisets
    always compare equal — the property that makes the breakdown point stable.
    """
    n = len(values)
    if n == 0:
        raise ValueError("fsum_mean of an empty sequence")
    return math.fsum(values) / n


def win_rate(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Return ``(win, loss, tie)`` fractions of ``a`` vs ``b`` across tasks.

    Pure elementwise counts — already deterministic (no summation of floats).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    wins = int(np.count_nonzero(a > b))
    losses = int(np.count_nonzero(a < b))
    ties = n - wins - losses
    return wins / n, losses / n, ties / n


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d_z = mean(a-b) / std(a-b, ddof=1).

    Returns 0.0 when the paired differences have ~zero variance (degenerate
    effect size). Uses :func:`fsum_mean` for the mean so the value is stable.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    mean_diff = fsum_mean(diff)
    # sample std around the fsum mean, computed order-independently
    sq = [(x - mean_diff) ** 2 for x in diff]
    var = math.fsum(sq) / (len(diff) - 1) if len(diff) > 1 else 0.0
    std_diff = math.sqrt(var)
    if std_diff < _STD_EPS:
        return 0.0
    return mean_diff / std_diff


def breakdown_point(winner: np.ndarray, loser: np.ndarray,
                    labels: Sequence | None = None) -> float:
    """Deterministic breakdown-point ratio in ``(0, 1]``.

    Greedily remove the winner's most-favorable tasks (largest ``winner-loser``
    advantage first) until ``mean(winner) <= mean(loser)`` over the kept tasks;
    return ``removed / k``. Returns ``1.0`` if the winner never falls behind.

    Inputs are oriented so larger is better. ``labels`` (defaults to positional
    indices) provide the deterministic tie-break for equal advantages.
    """
    w = np.asarray(winner, dtype=float)
    l = np.asarray(loser, dtype=float)
    k = len(w)
    if labels is None:
        labels = list(range(k))
    labels = list(labels)

    advantage = w - l
    # descending advantage; ties broken by label string -> input-order independent
    order = sorted(range(k), key=lambda i: (-advantage[i], str(labels[i])))
    w_sorted = [w[i] for i in order]
    l_sorted = [l[i] for i in order]

    for removed in range(1, k + 1):
        keep_w = w_sorted[removed:]
        keep_l = l_sorted[removed:]
        if not keep_w:
            return 1.0
        if fsum_mean(keep_w) <= fsum_mean(keep_l):  # tie => breakdown
            return removed / k
    return 1.0
