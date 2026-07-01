import numpy as np

from floodit_churn.features import MostCommonCategories, build_preprocessor


def test_preprocessor_fit_transform(sample_df):
    pre = build_preprocessor()
    out = pre.fit_transform(sample_df)
    assert out.shape[0] == len(sample_df)


def test_preprocessor_drop_categorical(sample_df):
    pre = build_preprocessor(categorical=False)
    out = pre.fit_transform(sample_df)
    # Sólo columnas numéricas (11) al descartar categóricas.
    assert out.shape[1] == 11


def test_most_common_does_not_mutate_input():
    X = np.array([["a"], ["a"], ["a"], ["b"], ["c"]], dtype=object)
    original = X.copy()
    mcc = MostCommonCategories(thr=0.6).fit(X)
    out = mcc.transform(X)
    # La entrada no debe mutar in-place.
    assert np.array_equal(X, original)
    # Las categorías raras se colapsan a "other".
    assert "other" in out
