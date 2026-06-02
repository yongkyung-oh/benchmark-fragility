<div align="center">

# benchmark-fragility

**Deterministic pairwise robustness metrics for benchmark leaderboards.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Metrics](https://img.shields.io/badge/metrics-effect%20size%20%C2%B7%20win%20rate%20%C2%B7%20breakdown-555)

</div>

---

> **Part of the [SOTA](https://github.com/yongkyung-oh/SOTA) package** — the standalone metrics library behind *Position: State-of-the-Art Claims Require State-of-the-Art Evidence* ([arXiv:2605.17273](https://arxiv.org/abs/2605.17273)).

**TL;DR:** `benchmark-fragility` checks whether a benchmark leaderboard winner is
robust across tasks, not just ahead on the mean. It diagnoses top-model pairs
with effect size, task-level win rate, and breakdown-point stability.

**Companion packages**

| Package | Reproduces |
|---|---|
| **`benchmark-fragility`** _(this package)_ | reusable fragility metrics — Cohen's *d* · win rate · breakdown-point |
| [`sota-evidence`](https://github.com/yongkyung-oh/SOTA/tree/main/sota-evidence) | Table 1 — MMLU fragility |
| [`sota-counter`](https://github.com/yongkyung-oh/SOTA/tree/main/sota-counter) | Figure 1 + Appendix A — SOTA mentions |

---

## Background

Benchmark leaderboards usually rank models by average score, but a mean-score
lead can be fragile when it is small, inconsistent across tasks, or dependent on
a small subset of tasks. This package makes those cases explicit by measuring:

- **Magnitude:** whether the top-pair gap is small under Cohen's *d*.
- **Consistency:** whether the higher-ranked model wins enough tasks.
- **Stability:** whether the ranking survives task removal under the
  breakdown-point ratio.

`benchmark-fragility` analyzes a `tasks × models` score matrix. It ranks models
by mean score, compares the top model pairs, and reports how often the
higher-ranked model is weak under three checks:

| Property | Metric | Violation rule | Default |
|---|---|---:|---:|
| Magnitude | Cohen's *d* | `d <= tau_d` | `0.20` |
| Consistency | Win rate across tasks | `WR <= tau_w` | `0.60` |
| Stability | Breakdown-point ratio | `BP <= tau_b` | `0.20` |

The fragility rate is the percentage of top-model pairs that violate at least
one rule.

## 🚀 Installation

```bash
pip install -e .
```

The distribution name is `benchmark-fragility`; the Python import package is
`benchmark_fragility`.

Optional extras:

```bash
pip install -e ".[stats]"   # optional Wilcoxon p-values
pip install -e ".[test]"    # test dependencies
```

## 💻 Usage

```bash
benchmark-fragility analyze scores.csv --direction higher_better --top-k 20
benchmark-fragility robustness scores.csv --direction lower_better
```

The short `fragility` command is also installed as a convenience alias.

```python
from benchmark_fragility import Thresholds, analyze, load_matrix

matrix = load_matrix("scores.csv")
result = analyze(matrix, direction="higher_better", top_k=20)

result.summary
result.pairs
```

Custom thresholds:

```python
analyze(
    matrix,
    thresholds=Thresholds(tau_w=0.75, tau_d=0.50, tau_b=0.30),
)
```

## 📥 Input

Input CSV files must have tasks as rows and models as columns. The first column
is used as the task index.

```text
,model_a,model_b,model_c
task_1,0.82,0.80,0.76
task_2,0.74,0.77,0.71
```

Use `direction="higher_better"` for accuracy-like metrics and
`direction="lower_better"` for error metrics such as WER, RMSE, or log loss.

## 🎯 Determinism

Results are designed to be stable under row/column reordering and across NumPy
summation changes. The implementation uses order-independent means, explicit
tie handling, and deterministic task ordering in the breakdown-point metric.

See [DIAGNOSIS.md](DIAGNOSIS.md) for the exact numerical policy.

## 🧪 Testing

```bash
pip install -e ".[test]"
pytest
```

## 📄 Citation

```bibtex
@misc{oh_sota_2026,
  title     = {Position: {State}-of-the-{Art} {Claims} {Require} {State}-of-the-{Art} {Evidence}},
  author    = {Oh, YongKyung},
  year      = 2026,
  publisher = {arXiv},
  doi       = {10.48550/arXiv.2605.17273}
}
```
