import numpy as np

from floodit.evaluate.metrics import (
    bootstrap_ci,
    brier,
    pr_auc,
    recall_at_precision,
    roc_auc,
    threshold_at_precision,
)


def test_pr_auc_perfect_separation_is_one():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert pr_auc(y, p) == 1.0


def test_roc_auc_perfect_separation_is_one():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(y, p) == 1.0


def test_brier_zero_for_perfect_confident_predictions():
    y = np.array([0, 1])
    p = np.array([0.0, 1.0])
    assert brier(y, p) == 0.0


def test_bootstrap_ci_brackets_point_estimate_and_is_ordered():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    lo, mid, hi = bootstrap_ci(y, p, metric=pr_auc, n=200, seed=0)
    assert lo <= mid <= hi


def test_bootstrap_ci_is_deterministic_with_seed():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = rng.random(300)
    a = bootstrap_ci(y, p, metric=pr_auc, n=100, seed=7)
    b = bootstrap_ci(y, p, metric=pr_auc, n=100, seed=7)
    assert a == b


def test_threshold_at_precision_recovers_separating_threshold():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.4, 0.6, 0.9])
    thr = threshold_at_precision(y, p, target=0.6)
    # at thr, predicting p>=thr should reach precision >= 0.6
    pred = p >= thr
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    assert precision >= 0.6


def test_recall_at_precision_in_unit_interval():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    r = recall_at_precision(y, p, target=0.6)
    assert 0.0 <= r <= 1.0
