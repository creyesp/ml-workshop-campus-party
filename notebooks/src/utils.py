import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import learning_curve


def precission_recall_vs_thr(model, x, y, ax=None):
    pred = model.predict_proba(x)[:, 1]

    thrs = np.arange(0, 1, 0.01)
    precission = [
        metrics.precision_score(y, pred > k, zero_division=0) for k in thrs
    ]
    recall = [metrics.recall_score(y, pred > k, zero_division=0) for k in thrs]

    if ax is None:
        f, ax = plt.subplots()
    ax.plot(thrs, precission, label="precission")
    ax.plot(thrs, recall, label="recall")
    ax.set(xlabel="probability threshold", ylabel="score")
    ax.legend()

    return ax


def plot_metric_curves(model, x, y, ax=None):
    if ax is None:
        f, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax = ax.flatten()
    metrics.plot_confusion_matrix(model, x, y, ax=ax[0])
    precission_recall_vs_thr(model, x, y, ax=ax[1])
    metrics.plot_precision_recall_curve(model, x, y, ax=ax[2])
    metrics.plot_roc_curve(model, x, y, ax=ax[3])
    ax[0].set(title="confusion_matrix")
    ax[1].set(title="Precision Recall vs threshold", xlim=(0, 1), ylim=(0, 1))
    ax[2].set(title="Precision Recall curve", xlim=(0, 1), ylim=(0, 1))
    ax[3].set(title="ROC curve", xlim=(0, 1), ylim=(0, 1))
    ax[1].grid(True, alpha=0.5, linestyle="--")
    ax[2].grid(True, alpha=0.5, linestyle="--")
    ax[3].grid(True, alpha=0.5, linestyle="--")

    return ax
