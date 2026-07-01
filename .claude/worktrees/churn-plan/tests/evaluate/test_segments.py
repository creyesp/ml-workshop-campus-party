import numpy as np

from floodit.data.load import load_dataset
from floodit.evaluate.segments import segment_metrics


def test_segment_metrics_one_row_per_group():
    df = load_dataset("test", verify=False)
    rng = np.random.default_rng(0)
    y = df["churned"].values
    p = rng.random(len(df))
    out = segment_metrics(df, y, p, by="device_os")
    assert set(out.columns) >= {"segment", "n", "prevalence", "pr_auc"}
    assert out["n"].sum() == len(df)
    assert out["segment"].nunique(dropna=False) == out.shape[0]


def test_segment_metrics_skips_single_class_pr_auc():
    df = load_dataset("test", verify=False).head(50).copy()
    y = np.zeros(50, dtype=int)  # single class -> pr_auc undefined
    p = np.linspace(0, 1, 50)
    out = segment_metrics(df, y, p, by="device_os")
    # pr_auc should be NaN where a segment has only one class
    assert out["pr_auc"].isna().any()
