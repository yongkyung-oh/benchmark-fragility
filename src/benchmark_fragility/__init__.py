"""benchmark-fragility — deterministic fragility diagnostics for ML benchmark leaderboards.

Quantifies, for the top model pairs, whether a higher-mean model is a *robust*
winner: consistent (win rate), meaningful (Cohen's d), and stable to task removal
(breakdown point), OR-combined into a fragility flag.

Results are deterministic across NumPy versions and input row/column orderings —
see ``DIAGNOSIS.md``.

>>> from benchmark_fragility import analyze, load_matrix
>>> res = analyze(load_matrix("MMLU.csv"), direction="higher_better", top_k=20)
>>> res.summary["fragility"]
"""
from __future__ import annotations

from .core import Result, analyze
from .io import load_matrix
from .metrics import breakdown_point, cohens_d, fsum_mean, win_rate
from .robustness import by_datasets, by_models
from .thresholds import Thresholds

__version__ = "0.1.0"
__all__ = [
    "analyze", "Result", "Thresholds", "load_matrix",
    "by_models", "by_datasets",
    "win_rate", "cohens_d", "breakdown_point", "fsum_mean",
]
