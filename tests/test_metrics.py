"""Tests de las métricas puras."""

from __future__ import annotations

import numpy as np
from src.train.metrics import (
    classification_metrics,
    confusion_matrix_dict,
    point_metrics,
    probabilistic_metrics,
)


def test_perfect_prediction_scores_one():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    scores = point_metrics(y_true, y_pred)
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["f1"] == 1.0


def test_probabilistic_metrics_range():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])
    scores = probabilistic_metrics(y_true, y_proba)
    assert scores["roc_auc"] == 1.0
    assert 0.0 <= scores["average_precision"] <= 1.0


def test_confusion_matrix_counts():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    cm = confusion_matrix_dict(y_true, y_pred)
    assert cm == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 0,
        "true_positive": 2,
    }


def test_classification_metrics_keys():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.2, 0.8, 0.6, 0.7])
    result = classification_metrics(y_true, y_pred, y_proba)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"]:
        assert key in result
