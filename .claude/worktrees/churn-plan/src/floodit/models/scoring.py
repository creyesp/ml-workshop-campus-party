"""Cross-validated out-of-fold scoring (shared by baseline/threshold/xgb runs).

Each training row is predicted exactly once, when it sits in the validation
fold. Pooling these out-of-fold probabilities gives a single held-out vector to
bootstrap a PR-AUC CI from (spec §5) without reusing the frozen test set.
"""
import numpy as np
import pandas as pd

from floodit.models.split import stratified_kfold


def oof_proba(model_factory, df: pd.DataFrame, y, n_splits: int = 5):
    """Return ``(y, oof_prob)`` aligned to ``df`` row order.

    ``model_factory`` is a zero-arg callable returning a fresh, unfitted
    estimator with ``predict_proba``. A fresh model is fit per fold.
    """
    y = np.asarray(y)
    oof = np.full(len(df), np.nan)
    for tr, va in stratified_kfold(df, y, n_splits=n_splits):
        model = model_factory()
        model.fit(df.iloc[tr], y[tr])
        oof[va] = model.predict_proba(df.iloc[va])[:, 1]
    assert not np.isnan(oof).any(), "some rows never landed in a validation fold"
    return y, oof
