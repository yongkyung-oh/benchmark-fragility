import numpy as np
import pandas as pd

from benchmark_fragility import load_matrix


def test_load_matrix_drops_all_nan_columns_and_rows(tmp_path):
    csv = tmp_path / "m.csv"
    pd.DataFrame(
        {"good1": [0.1, 0.2, np.nan], "empty": [np.nan, np.nan, np.nan],
         "good2": [0.3, 0.4, np.nan]},
        index=["t0", "t1", "t_empty"],
    ).to_csv(csv)
    m = load_matrix(csv)
    assert "empty" not in m.columns          # all-NaN model dropped
    assert "t_empty" not in m.index          # all-NaN task dropped
    assert list(m.columns) == ["good1", "good2"]


def test_load_matrix_idempotent_on_numeric(tmp_path):
    csv = tmp_path / "n.csv"
    pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=["x", "y"]).to_csv(csv)
    m = load_matrix(csv)
    assert m.shape == (2, 2)
    assert m.loc["x", "a"] == 1.0
