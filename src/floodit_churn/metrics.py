"""Métricas de evaluación del modelo de churn.

Migrado de ``notebooks/src/utils.py`` reemplazando las APIs de sklearn removidas
(``metrics.plot_confusion_matrix`` / ``plot_roc_curve``) por las clases
``*Display`` vigentes en scikit-learn >= 1.2.
"""
from __future__ import annotations

import numpy as np
from sklearn import metrics


def classification_metrics(y_true, y_proba, threshold: float = 0.5) -> dict[str, float]:
    """Métricas clave para comparar/monitorear el modelo de churn.

    Usa métricas threshold-agnósticas (ROC-AUC, PR-AUC) más las dependientes
    del umbral (precision/recall/f1) evaluadas al ``threshold`` dado.
    """
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "roc_auc": float(metrics.roc_auc_score(y_true, y_proba)),
        "pr_auc": float(metrics.average_precision_score(y_true, y_proba)),
        "precision": float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


def precision_recall_vs_threshold(model, x, y, ax=None):
    """Curva de precision y recall en función del umbral de probabilidad."""
    import matplotlib.pyplot as plt

    proba = model.predict_proba(x)[:, 1]
    thrs = np.arange(0, 1, 0.01)
    precision = [metrics.precision_score(y, proba > k, zero_division=0) for k in thrs]
    recall = [metrics.recall_score(y, proba > k, zero_division=0) for k in thrs]

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(thrs, precision, label="precision")
    ax.plot(thrs, recall, label="recall")
    ax.set(xlabel="probability threshold", ylabel="score")
    ax.legend()
    return ax


def plot_metric_curves(model, x, y, ax=None):
    """Panel con matriz de confusión, precision/recall vs umbral, PR y ROC."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax = ax.flatten()
    metrics.ConfusionMatrixDisplay.from_estimator(model, x, y, ax=ax[0])
    precision_recall_vs_threshold(model, x, y, ax=ax[1])
    metrics.PrecisionRecallDisplay.from_estimator(model, x, y, ax=ax[2])
    metrics.RocCurveDisplay.from_estimator(model, x, y, ax=ax[3])
    ax[0].set(title="confusion_matrix")
    ax[1].set(title="Precision Recall vs threshold", xlim=(0, 1), ylim=(0, 1))
    ax[2].set(title="Precision Recall curve", xlim=(0, 1), ylim=(0, 1))
    ax[3].set(title="ROC curve", xlim=(0, 1), ylim=(0, 1))
    for i in (1, 2, 3):
        ax[i].grid(True, alpha=0.5, linestyle="--")
    return ax
