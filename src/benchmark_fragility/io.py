"""Loading score matrices.

A score matrix is a CSV with **rows = tasks** and **columns = models**; the
first column is the task label (index). Values are numeric scores; non-numeric
cells become NaN.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["load_matrix"]


def load_matrix(path: str | Path) -> pd.DataFrame:
    """Read a ``tasks x models`` score matrix from ``path``.

    The first CSV column is used as the task index; remaining columns are coerced
    to numeric. Fully-empty rows/columns (a task or model with no numeric score)
    are dropped so they cannot silently distort the diagnostics.
    """
    df = pd.read_csv(path, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.dropna(axis=1, how="all").dropna(axis=0, how="all")
