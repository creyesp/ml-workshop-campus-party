import numpy as np

from floodit import config
from floodit.data.load import load_dataset
from floodit.features.preprocess import build_preprocessor


def test_preprocessor_transforms_all_rows():
    df = load_dataset("train", verify=False)
    pre = build_preprocessor()
    X = pre.fit_transform(df)
    assert X.shape[0] == len(df)


def test_preprocessor_drops_identity_and_timing_columns():
    df = load_dataset("train", verify=False)
    pre = build_preprocessor().fit(df)
    names = list(pre.get_feature_names_out())
    for ignored in config.IGNORE_COLUMNS:
        assert all(ignored not in str(n) for n in names), f"{ignored} leaked into features"


def test_preprocessor_handles_unknown_categories_at_transform():
    df = load_dataset("train", verify=False)
    pre = build_preprocessor().fit(df)
    novel = df.head(5).copy()
    novel.loc[:, "country_name"] = "Atlantis"  # unseen category
    X = pre.transform(novel)  # must not raise
    assert X.shape[0] == 5


def test_numeric_features_scaled_zero_mean():
    df = load_dataset("train", verify=False)
    pre = build_preprocessor().fit(df)
    X = pre.transform(df)
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    # first 11 columns are the scaled numeric block -> ~zero mean
    assert np.allclose(X[:, : len(config.NUMERICAL_COLUMNS)].mean(axis=0), 0, atol=1e-6)
