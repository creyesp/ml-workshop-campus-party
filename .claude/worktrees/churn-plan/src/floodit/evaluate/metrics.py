"""Evaluation metrics for the churn task (spec §4, §5).

Primary metric is PR-AUC (average precision) on the churn/positive class.
Bootstrap CIs give the spread needed for the non-overlapping-CI decision rule
(spec §5). Threshold helpers support the cost-based operating point (spec §4).
"""
from typing import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from floodit import config


def pr_auc(y_true, y_prob) -> float:
    """Average precision (area under the precision-recall curve)."""
    return float(average_precision_score(y_true, y_prob))


def roc_auc(y_true, y_prob) -> float:
    return float(roc_auc_score(y_true, y_prob))


def brier(y_true, y_prob) -> float:
    return float(brier_score_loss(y_true, y_prob))


def bootstrap_ci(
    y_true,
    y_prob,
    metric: Callable = pr_auc,
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = config.RANDOM_SEED,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for ``metric``.

    Returns ``(low, median, high)`` at the ``alpha`` two-sided level. Resamples
    row indices with replacement; folds that end up single-class are skipped.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    n_obs = len(y_true)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(metric(yt, y_prob[idx]))
    scores = np.sort(scores)
    lo = float(np.quantile(scores, alpha / 2))
    mid = float(np.quantile(scores, 0.5))
    hi = float(np.quantile(scores, 1 - alpha / 2))
    return lo, mid, hi


def threshold_at_precision(y_true, y_prob, target: float = 0.60) -> float:
    """Smallest probability threshold whose precision is >= ``target``.

    Lower thresholds admit more positives (higher recall) while holding the
    precision floor. Falls back to 1.0 if no threshold reaches ``target``.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision/recall have len(thresholds)+1; align by dropping the last point.
    precision = precision[:-1]
    thresholds = np.asarray(thresholds)
    ok = precision >= target
    if not ok.any():
        return 1.0
    return float(thresholds[ok].min())


def recall_at_precision(y_true, y_prob, target: float = 0.60) -> float:
    """Recall achievable while holding precision >= ``target``."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ok = precision >= target
    if not ok.any():
        return 0.0
    return float(recall[ok].max())
