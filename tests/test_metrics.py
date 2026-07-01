"""
Pruebas unitarias para el módulo de métricas puras.
Verifica que las funciones de evaluación retornen valores dentro de rangos esperados.
"""

import numpy as np
import pytest
from src.train.metrics import (
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_confusion_matrix,
    compute_roc_auc,
    compute_log_loss,
    compute_brier_score,
    compute_mse,
    compute_rmse,
    compute_mae,
    compute_r2,
    compute_pinball_loss,
    compute_interval_coverage,
    compute_interval_width,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def perfect_binary():
    """Predicción perfecta de clasificación binaria."""
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([0.05, 0.95, 0.05, 0.95, 0.95])
    return y_true, y_pred, y_prob


@pytest.fixture()
def random_binary():
    """Predicción aleatoria para probar robustez de funciones."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100)
    y_pred = rng.integers(0, 2, size=100)
    y_prob = rng.uniform(0, 1, size=100)
    return y_true, y_pred, y_prob


@pytest.fixture()
def regression_data():
    """Datos simples de regresión."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    return y_true, y_pred


# ── Clasificación Puntual ─────────────────────────────────────────────────────

def test_compute_accuracy_perfect(perfect_binary):
    y_true, y_pred, _ = perfect_binary
    assert compute_accuracy(y_true, y_pred) == 1.0


def test_compute_accuracy_range(random_binary):
    y_true, y_pred, _ = random_binary
    acc = compute_accuracy(y_true, y_pred)
    assert 0.0 <= acc <= 1.0


def test_compute_precision_perfect(perfect_binary):
    y_true, y_pred, _ = perfect_binary
    assert compute_precision(y_true, y_pred) == 1.0


def test_compute_recall_perfect(perfect_binary):
    y_true, y_pred, _ = perfect_binary
    assert compute_recall(y_true, y_pred) == 1.0


def test_compute_f1_perfect(perfect_binary):
    y_true, y_pred, _ = perfect_binary
    assert compute_f1(y_true, y_pred) == 1.0


def test_compute_confusion_matrix_shape(perfect_binary):
    y_true, y_pred, _ = perfect_binary
    cm = compute_confusion_matrix(y_true, y_pred)
    assert len(cm) == 2
    assert len(cm[0]) == 2


def test_compute_confusion_matrix_perfect(perfect_binary):
    y_true, y_pred, _ = perfect_binary
    cm = compute_confusion_matrix(y_true, y_pred)
    # Sin falsos positivos ni falsos negativos
    assert cm[0][1] == 0
    assert cm[1][0] == 0


# ── Clasificación Probabilística ──────────────────────────────────────────────

def test_compute_roc_auc_perfect(perfect_binary):
    y_true, _, y_prob = perfect_binary
    assert compute_roc_auc(y_true, y_prob) == pytest.approx(1.0, abs=1e-6)


def test_compute_roc_auc_range(random_binary):
    y_true, _, y_prob = random_binary
    auc = compute_roc_auc(y_true, y_prob)
    assert 0.0 <= auc <= 1.0


def test_compute_log_loss_perfect(perfect_binary):
    y_true, _, y_prob = perfect_binary
    # Log loss perfecto debería ser muy bajo
    assert compute_log_loss(y_true, y_prob) < 0.2


def test_compute_brier_score_range(random_binary):
    y_true, _, y_prob = random_binary
    score = compute_brier_score(y_true, y_prob)
    assert 0.0 <= score <= 1.0


# ── Regresión ─────────────────────────────────────────────────────────────────

def test_compute_mse_positive(regression_data):
    y_true, y_pred = regression_data
    assert compute_mse(y_true, y_pred) >= 0.0


def test_compute_rmse_geq_mae(regression_data):
    """RMSE siempre es >= MAE (por desigualdad de medias cuadráticas)."""
    y_true, y_pred = regression_data
    assert compute_rmse(y_true, y_pred) >= compute_mae(y_true, y_pred)


def test_compute_r2_perfect():
    y_true = np.array([1.0, 2.0, 3.0])
    assert compute_r2(y_true, y_true) == pytest.approx(1.0, abs=1e-9)


# ── Regresión Probabilística ──────────────────────────────────────────────────

def test_compute_pinball_loss_q50():
    """Pinball loss al 50% es equivalente a MAE / 2."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 1.5, 3.5])
    pinball = compute_pinball_loss(y_true, y_pred, alpha=0.5)
    assert pinball >= 0.0


def test_compute_interval_coverage_full():
    """Si los intervalos son infinitamente amplios, la cobertura es 1."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_lower = np.full(3, -1e9)
    y_upper = np.full(3, 1e9)
    assert compute_interval_coverage(y_true, y_lower, y_upper) == 1.0


def test_compute_interval_width():
    y_lower = np.array([0.0, 1.0, 2.0])
    y_upper = np.array([1.0, 2.0, 3.0])
    assert compute_interval_width(y_lower, y_upper) == pytest.approx(1.0)
