import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    result = {
        "accuracy": round(accuracy_score(y_true, y_pred), 6),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 6),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 6),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 6),
    }
    if y_proba is not None:
        result["roc_auc"] = round(roc_auc_score(y_true, y_proba), 6)
        result["log_loss"] = round(log_loss(y_true, y_proba), 6)
    return result


def probabilistic_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: list[float] | None = None,
) -> dict[str, float]:
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    result = {
        "roc_auc": round(roc_auc_score(y_true, y_proba), 6),
        "log_loss": round(log_loss(y_true, y_proba), 6),
    }
    for thr in thresholds:
        pred = (y_proba >= thr).astype(int)
        result[f"precision_at_{thr:.1f}"] = round(
            precision_score(y_true, pred, zero_division=0), 6
        )
        result[f"recall_at_{thr:.1f}"] = round(
            recall_score(y_true, pred, zero_division=0), 6
        )
        result[f"f1_at_{thr:.1f}"] = round(f1_score(y_true, pred, zero_division=0), 6)
    return result


def classification_report_dict(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = classification_metrics(y_true, y_pred, y_proba)
    metrics["confusion_matrix"] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics
