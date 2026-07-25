import numpy as np

from floodit.data.load import load_dataset
from floodit.models.baseline import logreg_baseline
from floodit.models.scoring import oof_proba


def test_oof_covers_every_row_once():
    df = load_dataset("train", verify=False)
    y = df["churned"].values
    yy, oof = oof_proba(logreg_baseline, df, y, n_splits=5)
    assert len(oof) == len(df)
    assert not np.isnan(oof).any()
    assert ((oof >= 0) & (oof <= 1)).all()
    assert np.array_equal(yy, y)
