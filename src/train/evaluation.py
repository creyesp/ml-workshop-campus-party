"""
Módulo de evaluación de modelos y visualización.
Calcula conjuntos de métricas completas y genera gráficos de diagnóstico
(curvas ROC, PR, matrices de confusión, etc.) que guarda en disco.
"""

import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from src.train.metrics import (
    compute_accuracy,
    compute_brier_score,
    compute_confusion_matrix,
    compute_f1,
    compute_log_loss,
    compute_mae,
    compute_mse,
    compute_precision,
    compute_r2,
    compute_recall,
    compute_rmse,
    compute_roc_auc,
)


def evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
) -> dict[str, Any]:
    """
    Evalúa predicciones de clasificación binaria y retorna un diccionario de métricas.

    Args:
        y_true (np.ndarray): Etiquetas reales.
        y_pred (np.ndarray): Predicciones puntuales de clase (0 o 1).
        y_prob (np.ndarray, opcional): Probabilidades estimadas para la clase 1.

    Returns:
        Dict[str, Any]: Diccionario con las métricas calculadas.
    """
    metrics_dict = {
        "accuracy": compute_accuracy(y_true, y_pred),
        "precision": compute_precision(y_true, y_pred),
        "recall": compute_recall(y_true, y_pred),
        "f1_score": compute_f1(y_true, y_pred),
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred),
    }

    if y_prob is not None:
        metrics_dict["roc_auc"] = compute_roc_auc(y_true, y_prob)
        metrics_dict["log_loss"] = compute_log_loss(y_true, y_prob)
        metrics_dict["brier_score"] = compute_brier_score(y_true, y_prob)

    return metrics_dict


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """
    Evalúa predicciones de regresión y retorna un diccionario de métricas.

    Args:
        y_true (np.ndarray): Valores reales.
        y_pred (np.ndarray): Predicciones puntuales.

    Returns:
        Dict[str, Any]: Diccionario con las métricas calculadas.
    """
    return {
        "mse": compute_mse(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "r2_score": compute_r2(y_true, y_pred),
    }


def save_classification_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    output_path: str,
) -> None:
    """
    Genera y guarda en disco una imagen con gráficos de diagnóstico para clasificación.

    Args:
        y_true (np.ndarray): Etiquetas reales.
        y_pred (np.ndarray): Predicciones puntuales.
        y_prob (np.ndarray): Probabilidades estimadas para la clase 1.
        output_path (str): Ruta completa para guardar la imagen.
    """
    # Crear carpeta contenedora si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # 1. Matriz de Confusión
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        ax=axes[0],
        cmap=plt.cm.Blues,
        colorbar=False,
    )
    axes[0].set_title("Matriz de Confusión")

    # 2. Curva ROC
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=axes[1])
    axes[1].set_title("Curva ROC")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Curva Precision-Recall
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=axes[2])
    axes[2].set_title("Curva Precision-Recall")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    # 4. Distribución de Probabilidades Predichas por Clase
    axes[3].hist(
        y_prob[y_true == 0],
        bins=30,
        alpha=0.5,
        label="Clase 0 (No Churn)",
        color="blue",
    )
    axes[3].hist(
        y_prob[y_true == 1],
        bins=30,
        alpha=0.5,
        label="Clase 1 (Churn)",
        color="red",
    )
    axes[3].set_xlabel("Probabilidad de Churn")
    axes[3].set_ylabel("Frecuencia")
    axes[3].set_title("Distribución de Probabilidades")
    axes[3].legend()
    axes[3].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_regression_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
) -> None:
    """
    Genera y guarda en disco gráficos de diagnóstico para regresión.

    Args:
        y_true (np.ndarray): Valores reales.
        y_pred (np.ndarray): Predicciones puntuales.
        output_path (str): Ruta completa para guardar la imagen.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Gráfico de Dispersión Real vs Predicho
    axes[0].scatter(y_true, y_pred, alpha=0.5, color="teal")
    # Línea ideal y=x
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    axes[0].set_xlabel("Valores Reales")
    axes[0].set_ylabel("Predicciones")
    axes[0].set_title("Real vs Predicho")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Histograma de Residuos
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=30, color="orange", edgecolor="black", alpha=0.7)
    axes[1].axvline(0, color="red", linestyle="--", lw=2)
    axes[1].set_xlabel("Residuos (Real - Predicho)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribución de Residuos")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
