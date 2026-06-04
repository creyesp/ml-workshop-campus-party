"""Segment-level metric reporting (spec §8 fairness flag).

A formal fairness audit is out of scope, but disparities must be *visible*:
this reports n, prevalence, and PR-AUC per group of a categorical attribute
(e.g. country_name, device_os) so the growth team can see them.
"""
import numpy as np
import pandas as pd

from floodit.evaluate.metrics import pr_auc


def segment_metrics(df: pd.DataFrame, y_true, y_prob, by: str) -> pd.DataFrame:
    """One row per value of ``by`` with n, prevalence, and PR-AUC.

    PR-AUC is NaN for segments that contain a single class (undefined there).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rows = []
    for value, idx in df.groupby(by, dropna=False).indices.items():
        yt = y_true[idx]
        yp = y_prob[idx]
        ap = pr_auc(yt, yp) if len(np.unique(yt)) == 2 else np.nan
        rows.append(
            {
                "segment": value,
                "n": len(idx),
                "prevalence": float(yt.mean()),
                "pr_auc": ap,
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
