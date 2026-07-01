"""
Evaluación de modelos.

Orquesta el cálculo de métricas puras (``metrics.py``) sobre los conjuntos de train y
test a partir de un pipeline entrenado. No define fórmulas de métricas: solo aplica el
modelo y delega el cálculo.
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from src.config.settings import get_settings
from src.train.metrics import classification_metrics
from src.train.training import predict_label, predict_proba


def evaluate_split(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float | None = None,
) -> dict:
    """Calcula todas las métricas para un único conjunto (train o test)."""
    y_true = target.to_numpy()
    y_proba = predict_proba(model, features)
    y_pred = predict_label(model, features, threshold=threshold)
    return classification_metrics(y_true, y_pred, y_proba)


def evaluate_model(
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float | None = None,
) -> dict[str, dict]:
    """
    Evalúa el modelo en train y test.

    Returns:
        ``{"train": {...métricas...}, "test": {...métricas...}}``.
    """
    if threshold is None:
        threshold = get_settings().training.decision_threshold
    results = {
        "train": evaluate_split(model, x_train, y_train, threshold),
        "test": evaluate_split(model, x_test, y_test, threshold),
    }
    if get_settings().verbose:
        print(
            f"[eval] test ROC-AUC={results['test']['roc_auc']:.4f} "
            f"F1={results['test']['f1']:.4f}"
        )
    return results
