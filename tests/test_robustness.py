import pytest

from benchmark_fragility import by_datasets, by_models, load_matrix


def test_by_models_columns_and_monotone_shrink(data_dir):
    m = load_matrix(data_dir / "MMLU.csv")
    df = by_models(m, direction="higher_better")
    assert list(df.columns) == ["n_models", "magnitude", "consistency",
                                "stability", "fragility"]
    assert df["n_models"].tolist() == [10, 20, 30, 40, 50, 60]


def test_by_datasets_deterministic_under_row_shuffle(data_dir):
    m = load_matrix(data_dir / "MMLU.csv")
    base = by_datasets(m, direction="higher_better", seed=42)
    shuffled = by_datasets(m.sample(frac=1, random_state=5),
                           direction="higher_better", seed=42)
    # subsample draws from sorted(index) -> identical despite row reordering
    assert base.equals(shuffled)


def test_by_datasets_seed_changes_result(data_dir):
    m = load_matrix(data_dir / "MMLU.csv")
    a = by_datasets(m, seed=42)
    b = by_datasets(m, seed=1)
    assert not a.equals(b)


def test_by_models_skips_oversized_k(data_dir):
    m = load_matrix(data_dir / "TSFM_MAE.csv")  # 14 models
    df = by_models(m, ks=(10, 20, 30), direction="lower_better")
    assert df["n_models"].tolist() == [10]  # 20, 30 skipped
