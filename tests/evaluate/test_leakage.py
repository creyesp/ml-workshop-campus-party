import numpy as np
import pandas as pd

from floodit.evaluate.leakage import directed_auc, single_feature_auc


def _toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    # leaky: equals the label; noise: independent
    df = pd.DataFrame(
        {
            "leaky": y.astype(float) + rng.normal(0, 0.01, n),
            "noise": rng.normal(0, 1, n),
        }
    )
    return df, y


def test_single_feature_auc_in_unit_interval():
    df, y = _toy()
    assert 0.0 <= single_feature_auc(df, "noise", y) <= 1.0


def test_leaky_feature_has_near_perfect_directed_auc():
    df, y = _toy()
    assert directed_auc(df, "leaky", y) > 0.95


def test_noise_feature_directed_auc_near_half():
    df, y = _toy()
    assert directed_auc(df, "noise", y) < 0.65
