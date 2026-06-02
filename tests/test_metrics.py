import math

import numpy as np
import pytest

from benchmark_fragility import breakdown_point, cohens_d, fsum_mean, win_rate


def test_fsum_mean_matches_exact():
    assert fsum_mean([0.1, 0.2, 0.3]) == math.fsum([0.1, 0.2, 0.3]) / 3


def test_fsum_mean_order_independent():
    vals = [0.7703703703703704, 0.77, 0.4962962962962963, 0.1, 0.9]
    assert fsum_mean(vals) == fsum_mean(list(reversed(vals)))


def test_fsum_mean_empty_raises():
    with pytest.raises(ValueError):
        fsum_mean([])


def test_win_rate_counts():
    a = np.array([1.0, 0.0, 1.0, 1.0])
    b = np.array([0.0, 1.0, 1.0, 0.0])  # win, loss, tie, win
    w, l, t = win_rate(a, b)
    assert (w, l, t) == (0.5, 0.25, 0.25)
    assert abs(w + l + t - 1.0) < 1e-12


def test_cohens_d_zero_variance_guard():
    a = np.array([0.5, 0.6, 0.7])
    assert cohens_d(a, a) == 0.0  # identical -> zero diff variance


def test_cohens_d_sign_and_value():
    a = np.array([0.8, 0.7, 0.9, 0.6, 0.5])
    b = np.array([0.7, 0.7, 0.6, 0.6, 0.4])
    diff = a - b
    expected = fsum_mean(diff) / np.std(diff, ddof=1)
    assert cohens_d(a, b) == pytest.approx(expected, rel=1e-9)


def test_breakdown_single_outlier():
    # winner leads on mean only because of task 0; removing it flips -> 1/5
    w = np.array([0.9, 0.5, 0.5, 0.5, 0.5])
    l = np.array([0.3, 0.55, 0.55, 0.55, 0.55])
    assert breakdown_point(w, l) == pytest.approx(0.2)


def test_breakdown_never_flips_is_one():
    w = np.array([0.9, 0.9, 0.9])
    l = np.array([0.1, 0.1, 0.1])
    assert breakdown_point(w, l) == 1.0


def test_breakdown_exact_tie_counts_as_breakdown():
    # after removing the top task, the two means are EXACTLY equal -> breakdown (<=)
    w = np.array([1.0, 0.4, 0.6])
    l = np.array([0.0, 0.5, 0.5])  # keep={1,2}: mean_w=0.5, mean_l=0.5 -> tie
    assert breakdown_point(w, l) == pytest.approx(1 / 3)


def test_breakdown_label_tiebreak_is_order_independent():
    # equal advantages on several tasks; result must not depend on input order
    w = np.array([0.6, 0.6, 0.6, 0.9])
    l = np.array([0.5, 0.5, 0.5, 0.1])
    labels = ["a", "b", "c", "d"]
    perm = [2, 0, 3, 1]
    bp1 = breakdown_point(w, l, labels=labels)
    bp2 = breakdown_point(w[perm], l[perm], labels=[labels[i] for i in perm])
    assert bp1 == bp2
