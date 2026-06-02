import numpy as np
import pandas as pd
import pytest

from benchmark_fragility import analyze


def test_lower_better_equals_negated_higher_better():
    rng = np.random.RandomState(0)
    m = pd.DataFrame(rng.rand(12, 6), columns=[f"m{i}" for i in range(6)],
                     index=[f"t{i}" for i in range(12)])
    lower = analyze(m, direction="lower_better", top_k=4).summary
    higher = analyze(-m, direction="higher_better", top_k=4).summary
    for key in ("magnitude", "consistency", "stability", "fragility", "n_pairs"):
        assert lower[key] == higher[key]


def test_invalid_direction():
    m = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 1.0]})
    with pytest.raises(ValueError):
        analyze(m, direction="bigger")


def test_top_k_caps_at_available_models():
    m = pd.DataFrame(np.eye(3), columns=["a", "b", "c"], index=["t0", "t1", "t2"])
    res = analyze(m, top_k=20)
    assert res.summary["n_models"] == 3
    assert res.summary["n_pairs"] == 3


def test_fewer_than_two_models_raises():
    m = pd.DataFrame(np.eye(3), columns=["a", "b", "c"], index=["t0", "t1", "t2"])
    with pytest.raises(ValueError, match="at least 2 models"):
        analyze(m, top_k=1)
    single = pd.DataFrame({"only": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="at least 2 models"):
        analyze(single)
