import json

import pandas as pd

from benchmark_fragility import analyze, load_matrix
from benchmark_fragility.cli import main


def test_cli_analyze_json(data_dir, capsys):
    rc = main(["analyze", str(data_dir / "MMLU.csv"), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_pairs"] == 190
    assert "fragility" in out


def test_cli_robustness(data_dir, capsys):
    rc = main(["robustness", str(data_dir / "MMLU.csv")])
    assert rc == 0
    assert "By #models" in capsys.readouterr().out


def test_wilcoxon_optional(data_dir):
    """wilcoxon=True adds a column iff scipy is present; never affects fragility."""
    m = load_matrix(data_dir / "MMLU.csv")
    base = analyze(m, top_k=20).summary["fragility"]
    res = analyze(m, top_k=20, wilcoxon=True)
    assert res.summary["fragility"] == base
    try:
        import scipy  # noqa: F401
        assert "wilcoxon_p" in res.pairs.columns
    except ImportError:
        assert "wilcoxon_p" not in res.pairs.columns
