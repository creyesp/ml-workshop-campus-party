"""
Métricas de clasificación puras.

Cada función recibe arrays de numpy/pandas y devuelve valores o diccionarios simples,
sin efectos secundarios ni dependencia del pipeline. Se separan las métricas de
clase (``point``) de las probabilísticas (``probabilistic``) para poder reportarlas y
compararlas de forma independiente.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Métricas basadas en la predicción de clase (umbral ya aplicado)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def probabilistic_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Métricas basadas en la probabilidad estimada de churn."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }


def confusion_matrix_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Matriz de confusión como diccionario serializable."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float | dict[str, int]]:
    """Combina métricas de clase, probabilísticas y matriz de confusión."""
    return {
        **point_metrics(y_true, y_pred),
        **probabilistic_metrics(y_true, y_proba),
        "confusion_matrix": confusion_matrix_dict(y_true, y_pred),
    }
