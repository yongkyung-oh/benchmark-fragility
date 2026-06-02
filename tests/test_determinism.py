"""The headline guarantee: results are independent of row/column order.

(Cross-NumPy-version identity is enforced separately by running this same suite
under multiple NumPy versions; see README "Determinism".)
"""
import numpy as np
import pytest

from benchmark_fragility import analyze, load_matrix

BENCHES = ["MMLU", "Tab_B", "TSFM_MAE"]
DIRECTION = {"MMLU": "higher_better", "Tab_B": "lower_better", "TSFM_MAE": "lower_better"}


def _summary(df, direction):
    s = analyze(df, direction=direction, top_k=20).summary
    return (s["magnitude"], s["consistency"], s["stability"], s["fragility"])


@pytest.mark.parametrize("bench", BENCHES)
def test_row_shuffle_invariant(data_dir, bench):
    m = load_matrix(data_dir / f"{bench}.csv")
    base = _summary(m, DIRECTION[bench])
    for seed in (1, 7, 99):
        shuffled = m.sample(frac=1, random_state=seed)
        assert _summary(shuffled, DIRECTION[bench]) == base


@pytest.mark.parametrize("bench", BENCHES)
def test_column_shuffle_invariant(data_dir, bench):
    m = load_matrix(data_dir / f"{bench}.csv")
    base = _summary(m, DIRECTION[bench])
    cols = list(np.random.RandomState(3).permutation(m.columns))
    assert _summary(m[cols], DIRECTION[bench]) == base


@pytest.mark.parametrize("bench", BENCHES)
def test_matches_frozen_expected(data_dir, expected, bench):
    m = load_matrix(data_dir / f"{bench}.csv")
    s = analyze(m, direction=DIRECTION[bench], top_k=20).summary
    e = expected[bench]
    for key in ("magnitude", "consistency", "stability", "fragility"):
        assert s[key] == pytest.approx(e[key], abs=1e-6)
    assert s["n_pairs"] == e["n_pairs"]
