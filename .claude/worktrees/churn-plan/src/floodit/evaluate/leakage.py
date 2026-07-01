"""Leakage audit for the ``cnt_*`` features (spec §7 H3, §8).

Why this is the tripwire: ``churned`` is derived from ``user_last_engagement``
(global MAX of user_engagement events), while the ``cnt_*`` features count
events with ``timestamp <= first+24h`` (the observation window, which closes at
prediction time). That time-boxing is what makes the features legitimate — this
module quantifies whether any feature nevertheless encodes the label.

Two checks:
- ``single_feature_auc``: univariate ROC-AUC of a feature used alone as a score.
  A value near 1.0 means the feature almost determines the label.
- ``ablation_drop``: cross-validated PR-AUC with the feature neutralised vs. the
  full model. A large drop means the model leans heavily on that one feature.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from floodit import config
from floodit.evaluate.metrics import pr_auc
from floodit.models.split import stratified_kfold


def single_feature_auc(df: pd.DataFrame, feature: str, y) -> float:
    """ROC-AUC of one numeric feature used directly as the score (raw, unsigned)."""
    return float(roc_auc_score(y, df[feature].values))


def directed_auc(df: pd.DataFrame, feature: str, y) -> float:
    """ROC-AUC folded to [0.5, 1] so anti-correlated features are comparable."""
    auc = single_feature_auc(df, feature, y)
    return max(auc, 1.0 - auc)


def _cv_pr_auc(model_factory, df: pd.DataFrame, y, n_splits: int = 5) -> float:
    """Mean out-of-fold PR-AUC for a fresh pipeline per fold (sequential)."""
    y = np.asarray(y)
    scores = []
    for tr, va in stratified_kfold(df, y, n_splits=n_splits):
        model = model_factory()
        model.fit(df.iloc[tr], y[tr])
        prob = model.predict_proba(df.iloc[va])[:, 1]
        scores.append(pr_auc(y[va], prob))
    return float(np.mean(scores))


def ablation_drop(model_factory, df: pd.DataFrame, y, feature: str) -> dict:
    """CV PR-AUC with ``feature`` neutralised (set constant) vs. the full model.

    Setting the column to a constant makes StandardScaler emit zeros for it, so
    the feature carries no signal without changing the pipeline structure.
    """
    full = _cv_pr_auc(model_factory, df, y)
    ablated_df = df.copy()
    ablated_df[feature] = 0
    ablated = _cv_pr_auc(model_factory, ablated_df, y)
    return {"feature": feature, "full": full, "ablated": ablated, "delta": full - ablated}


def audit_all(model_factory, df: pd.DataFrame, y) -> pd.DataFrame:
    """Per-feature univariate (directed) AUC and ablation delta for all counts."""
    rows = []
    for feat in config.NUMERICAL_COLUMNS:
        drop = ablation_drop(model_factory, df, y, feat)
        rows.append(
            {
                "feature": feat,
                "univariate_auc": single_feature_auc(df, feat, y),
                "directed_auc": directed_auc(df, feat, y),
                "cv_pr_auc_full": drop["full"],
                "cv_pr_auc_ablated": drop["ablated"],
                "ablation_delta": drop["delta"],
            }
        )
    return pd.DataFrame(rows).sort_values("directed_auc", ascending=False).reset_index(drop=True)
