"""Robustness sweeps: how fragility moves with the analysis scope.

- :func:`by_models` — vary the number of top models compared.
- :func:`by_datasets` — fix the top-``top_k`` models, vary the number of tasks
  via a **deterministic** random subsample.

Determinism note (subsample)
----------------------------
``numpy.random.Generator.choice(df.index, n)`` draws by array *position*, so the
selected tasks depend on the matrix's row order. We instead draw from
``sorted(index)``, making the subsample a function of (labels, seed, n) only —
independent of how the caller ordered the rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import analyze
from .thresholds import Thresholds

__all__ = ["by_models", "by_datasets"]

_DEFAULT_KS = (10, 20, 30, 40, 50, 60)
_DEFAULT_NS = (7, 17, 27, 37, 47, 57)


def _row(label_key: str, label_val, result) -> dict:
    s = result.summary
    return {label_key: label_val, "magnitude": s["magnitude"],
            "consistency": s["consistency"], "stability": s["stability"],
            "fragility": s["fragility"]}


def by_models(matrix: pd.DataFrame, ks=_DEFAULT_KS, *,
              direction: str = "higher_better",
              thresholds: Thresholds | None = None) -> pd.DataFrame:
    """Fragility/violation rates for each top-``k`` in ``ks`` (skips k > n_models)."""
    n = matrix.shape[1]
    rows = [_row("n_models", k, analyze(matrix, direction=direction, top_k=k,
                                        thresholds=thresholds))
            for k in ks if k <= n]
    return pd.DataFrame(rows)


def by_datasets(matrix: pd.DataFrame, ns=_DEFAULT_NS, *, seed: int = 42,
                top_k: int = 20, direction: str = "higher_better",
                thresholds: Thresholds | None = None) -> pd.DataFrame:
    """Fragility/violation rates for deterministic ``n``-task subsamples.

    The top-``top_k`` models are fixed first; each ``n`` draws ``n`` task labels
    from ``sorted(index)`` with ``default_rng(seed)`` (order-independent).
    """
    oriented = matrix if direction == "higher_better" else -matrix
    # fix the model set on the full task set, then subsample tasks
    from .core import _ranked_models
    models = _ranked_models(oriented, top_k)
    sub_models = matrix[models]
    sorted_index = sorted(matrix.index, key=str)

    rows = []
    for n in ns:
        if n > len(sorted_index):
            continue
        rng = np.random.default_rng(seed)
        chosen = rng.choice(np.array(sorted_index, dtype=object), size=n, replace=False)
        sub = sub_models.loc[list(chosen)]
        res = analyze(sub, direction=direction, top_k=top_k, thresholds=thresholds)
        rows.append(_row("n_datasets", n, res))
    return pd.DataFrame(rows)
