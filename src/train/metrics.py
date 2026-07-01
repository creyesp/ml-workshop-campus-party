"""
Módulo de métricas puras.
Contiene funciones puras e independientes para calcular métricas de evaluación
tanto puntuales como probabilísticas (para clasificación y regresión).
"""

import numpy as np
from sklearn import metrics

# =====================================================================
# METRICAS DE CLASIFICACION
# =====================================================================

# --- Métricas Puntuales (Point Classification) ---


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula la exactitud (accuracy)."""
    return float(metrics.accuracy_score(y_true, y_pred))


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula la precisión (precision)."""
    return float(metrics.precision_score(y_true, y_pred, zero_division=0))


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula la sensibilidad (recall)."""
    return float(metrics.recall_score(y_true, y_pred, zero_division=0))


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula la puntuación F1 (F1-score)."""
    return float(metrics.f1_score(y_true, y_pred, zero_division=0))


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list:
    """Calcula y retorna la matriz de confusión estructurada como lista de listas."""
    cm = metrics.confusion_matrix(y_true, y_pred)
    return cm.tolist()


# --- Métricas Probabilísticas (Probabilistic Classification) ---


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calcula el área bajo la curva ROC (ROC-AUC)."""
    try:
        return float(metrics.roc_auc_score(y_true, y_prob))
    except ValueError:
        return 0.5


def compute_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calcula la pérdida logarítmica (Log Loss)."""
    try:
        return float(metrics.log_loss(y_true, y_prob, labels=[0, 1]))
    except ValueError:
        return float("nan")


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calcula el Brier Score (mide la calibración de probabilidades)."""
    return float(metrics.brier_score_loss(y_true, y_prob))


# =====================================================================
# METRICAS DE REGRESION
# =====================================================================

# --- Métricas Puntuales (Point Regression) ---


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el error cuadrático medio (MSE)."""
    return float(metrics.mean_squared_error(y_true, y_pred))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula la raíz del error cuadrático medio (RMSE)."""
    return float(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el error absoluto medio (MAE)."""
    return float(metrics.mean_absolute_error(y_true, y_pred))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el coeficiente de determinación R2."""
    return float(metrics.r2_score(y_true, y_pred))


# --- Métricas Probabilísticas (Probabilistic Regression - Quantiles) ---


def compute_pinball_loss(y_true: np.ndarray, y_quantile: np.ndarray, alpha: float) -> float:
    """
    Calcula la pérdida Pinball (Pinball loss) para evaluar la calidad de un cuantil.

    Args:
        y_true (np.ndarray): Valores reales.
        y_quantile (np.ndarray): Predicción del cuantil.
        alpha (float): Nivel del cuantil (ej. 0.1, 0.9).

    Returns:
        float: Pérdida pinball.
    """
    diff = y_true - y_quantile
    loss = np.where(diff >= 0, alpha * diff, (alpha - 1) * diff)
    return float(np.mean(loss))


def compute_interval_coverage(
    y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray
) -> float:
    """
    Calcula la proporción de valores reales que caen dentro del intervalo de predicción.

    Args:
        y_true (np.ndarray): Valores reales.
        y_lower (np.ndarray): Predicción del cuantil inferior.
        y_upper (np.ndarray): Predicción del cuantil superior.

    Returns:
        float: Cobertura del intervalo (0 a 1).
    """
    covered = (y_true >= y_lower) & (y_true <= y_upper)
    return float(np.mean(covered))


def compute_interval_width(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """
    Calcula el ancho promedio de los intervalos de predicción.

    Args:
        y_lower (np.ndarray): Predicción del cuantil inferior.
        y_upper (np.ndarray): Predicción del cuantil superior.

    Returns:
        float: Ancho promedio.
    """
    return float(np.mean(y_upper - y_lower))
