"""The ``analyze`` engine: per-pair diagnostics + summary over the top models.

Pipeline (all steps order-independent → deterministic):

1. Orient so larger is better (negate the matrix when ``direction='lower_better'``).
2. Rank models by their exact (``fsum``) mean score and keep the top ``top_k``;
   ties broken by model label.
3. For every unordered pair, the winner is the higher-mean model (label tie-break);
   compute win rate, Cohen's d, breakdown point, violation flags, and fragility.
4. Aggregate violation/fragility rates into a summary.
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd

from .metrics import breakdown_point, cohens_d, fsum_mean, win_rate
from .thresholds import Thresholds

__all__ = ["analyze", "Result"]


class Result:
    """Container for :func:`analyze` output: ``.pairs`` (DataFrame) + ``.summary`` (dict)."""

    __slots__ = ("pairs", "summary", "models", "n_tasks")

    def __init__(self, pairs: pd.DataFrame, summary: dict, models: list, n_tasks: int):
        self.pairs = pairs
        self.summary = summary
        self.models = models
        self.n_tasks = n_tasks

    def __repr__(self) -> str:
        s = self.summary
        return (f"Result(models={len(self.models)}, tasks={self.n_tasks}, "
                f"pairs={s['n_pairs']}, fragility={s['fragility']:.2f}%)")


def _ranked_models(oriented: pd.DataFrame, top_k: int) -> list:
    """Top-``top_k`` model labels by exact mean, descending; ties by label."""
    scores = {m: fsum_mean(oriented[m].values) for m in oriented.columns}
    order = sorted(scores, key=lambda m: (-scores[m], str(m)))
    return order[: min(top_k, len(order))]


def analyze(matrix: pd.DataFrame, *, direction: str = "higher_better",
            top_k: int = 20, thresholds: Thresholds | None = None,
            wilcoxon: bool = False) -> Result:
    """Run fragility diagnostics over the top ``top_k`` models of ``matrix``.

    Parameters
    ----------
    matrix : DataFrame
        ``tasks x models`` numeric scores.
    direction : {'higher_better', 'lower_better'}
        Whether larger scores are better.
    top_k : int
        Number of top models to compare (all if fewer are present).
    thresholds : Thresholds, optional
        Violation cut-offs (defaults to :class:`Thresholds`).
    wilcoxon : bool
        If True and SciPy is installed, add a ``wilcoxon_p`` column (reporting
        only; never affects violations). Silently skipped without SciPy.
    """
    if direction not in ("higher_better", "lower_better"):
        raise ValueError(f"direction must be higher_better|lower_better, got {direction!r}")
    thresholds = thresholds or Thresholds()

    oriented = matrix if direction == "higher_better" else -matrix
    models = _ranked_models(oriented, top_k)
    sub = oriented[models]
    labels = list(sub.index)

    means = {m: fsum_mean(sub[m].values) for m in models}
    wilcoxon_fn = _maybe_wilcoxon() if wilcoxon else None

    rows = []
    for a, b in combinations(models, 2):
        winner, loser = (a, b) if (means[a], str(b)) > (means[b], str(a)) else (b, a)
        sw = sub[winner].values
        sl = sub[loser].values

        wr, _, _ = win_rate(sw, sl)
        d = cohens_d(sw, sl)
        bp = breakdown_point(sw, sl, labels=labels)
        v_cons, v_mag, v_stab, fragile = thresholds.violations(wr, d, bp)

        row = {
            "winner": winner, "loser": loser,
            "win_rate": wr, "cohens_d": d, "bp_ratio": bp,
            "viol_consistency": v_cons, "viol_magnitude": v_mag,
            "viol_stability": v_stab, "fragile": fragile,
        }
        if wilcoxon_fn is not None:
            row["wilcoxon_p"] = wilcoxon_fn(sw, sl)
        rows.append(row)

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        raise ValueError(
            f"need at least 2 models to form pairs; got {len(models)} "
            f"(top_k={top_k}, matrix has {matrix.shape[1]} model column(s))"
        )
    summary = {
        "n_models": len(models),
        "n_tasks": len(labels),
        "n_pairs": len(pairs),
        "direction": direction,
        "magnitude": 100.0 * pairs["viol_magnitude"].mean(),
        "consistency": 100.0 * pairs["viol_consistency"].mean(),
        "stability": 100.0 * pairs["viol_stability"].mean(),
        "fragility": 100.0 * pairs["fragile"].mean(),
        "thresholds": {"tau_w": thresholds.tau_w, "tau_d": thresholds.tau_d,
                       "tau_b": thresholds.tau_b},
    }
    return Result(pairs, summary, models, len(labels))


def _maybe_wilcoxon():
    """Return a ``(w, l) -> p`` function if SciPy is importable, else ``None``."""
    try:
        from scipy import stats  # noqa: PLC0415
    except ImportError:
        return None

    def _p(w, l):
        try:
            return float(stats.wilcoxon(w, l).pvalue)
        except ValueError:
            return 1.0

    return _p
